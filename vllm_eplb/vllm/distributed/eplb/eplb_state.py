# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Expert parallelism load balancer (EPLB) metrics and states.

# Glossary

- **Logical Expert**: An expert that is part of the model's logical structure.
  It holds a set of weights and is replicated across multiple physical
  experts.
- **Redundant Expert**: To achieve load balancing, for some popular logical
  experts, we create additional copies of the expert weights. During inference,
  each of these copies can be routed to by the same set of tokens.
- **Physical Expert**: An expert that is instantiated on a specific device.
  It is a replica of a logical expert and can be rearranged across devices.
  I.e., one logical expert may have multiple sets of weights initialized on
  different devices, and each of these sets is a physical expert.
- **Local Physical Expert**: A physical expert that is instantiated on the
  current device.

For example: DeepSeek-R1 has 256 logical experts, so each MoE layer
has 256 sets of linear layer weights in the model parameters. If we add 32
redundant experts, DeepSeek-R1 will have 256 + 32 = 288 physical experts in
total. And when deploying, we'll have 288 sets of linear layer weights for each
MoE layer. If we have 32 EP ranks, then each GPU will hold 288 / 32 = 9 local
physical experts.
"""

import threading
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.distributed import ProcessGroup, all_reduce

from vllm.config import ModelConfig, ParallelConfig
from vllm.distributed.parallel_state import (
    get_ep_group,
    get_node_count,
    in_the_same_node_as,
)
from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import MixtureOfExperts

from .async_worker import start_async_worker
from .policy import EPLB_POLICIES, AbstractEplbPolicy, DefaultEplbPolicy
from .rebalance_execute import (
    RecvMetadata,
    move_from_buffer,
    rearrange_expert_weights_inplace,
)

logger = init_logger(__name__)


@dataclass
class EplbStats:
    """
    Model stats used in EPLB rebalancing algorithm.
    """

    global_expert_load_window: torch.Tensor
    """
    Experts load window.
    Shape: (window_size, num_moe_layers, num_physical_experts)
    """
    num_replicas: int
    """
    Number of physical experts.
    """
    num_groups: int
    """
    Number of expert groups.
    """
    num_nodes: int
    """
    Number of nodes.
    """
    num_gpus: int
    """
    Number of GPUs.
    """


@dataclass
class EplbModelState:
    """EPLB metrics."""

    physical_to_logical_map: torch.Tensor
    """
    Mapping from physical experts to logical experts.

    Shape: (num_moe_layers, num_physical_experts)

    # Example

    For a 2-layer MoE model with 6 physical experts and 4 logical experts on 3
    EP ranks, the mapping could look like this:

    ```
    [[0, 1, 2, 3, 0, 1],
     [0, 2, 0, 1, 0, 3]]
    ```
    """
    logical_to_physical_map: torch.Tensor
    """
    Mapping from logical experts to physical experts.

    This is a sparse matrix, where -1 indicates no mapping.

    Shape: (num_moe_layers, num_logical_experts, num_redundant_experts + 1)

    # Example

    For a 2-layer MoE model with 6 physical experts and 4 logical experts on 3
    EP ranks, the mapping could look like this:

    ```
    [[[0, 4, -1],
      [1, 5, -1],
      [2, -1, -1],
      [3, -1, -1]],
     [[0, 2, 4],
      [3, -1, -1],
      [1, -1, -1],
      [5, -1, -1]]]
    ```
    """
    logical_replica_count: torch.Tensor
    """
    Number of replicas for each logical expert.
    This is exactly the non-`-1` count in the `logical_to_physical_map`.

    Shape: (num_moe_layers, num_logical_experts)

    # Example
    For a 2-layer MoE model with 6 physical experts and 4 logical experts on 3
    EP ranks, the count could look like this:

    ```
    [[2, 2, 1, 1],
     [3, 1, 1, 1]]
    """

    expert_load_pass: torch.Tensor
    """
    Expert load during this forward pass. 
    We use the token count each expert processes as the load.

    Shape: (num_moe_layers, num_physical_experts)
    """
    expert_load_window: torch.Tensor
    """
    A sliding window of expert load.

    Shape: (window_size, num_moe_layers, num_physical_experts)

    NOTE: The expert_load_view now records load for all physical experts
    rather than just local experts. This ensures consistent load statistics
    across different dispatch methods (naive all-to-all, DeepEP).
    The recorded load will be multiplied by dp_size when using naive all-to-all
    due to each DP rank contributing the same token set to the calculation.
    See:
    https://github.com/vllm-project/vllm/pull/22167#pullrequestreview-3086143856
    """
    eplb_algorithm: str
    """Algorithm for load estimation."""
    eplb_ema_alpha: float
    """EMA decay factor. Only used when eplb_algorithm="ema"."""
    expert_load_ema: torch.Tensor | None
    """
    EMA-smoothed expert load. Only allocated when eplb_algorithm="ema".
    Shape: (num_moe_layers, num_physical_experts)
    """
    expert_load_logical_ema: torch.Tensor | None
    """
    EMA-smoothed logical expert load.
    Only allocated when eplb_algorithm="fgate-hybrid-cache".
    Shape: (num_moe_layers, num_logical_experts)
    """
    expert_load_fgate: torch.Tensor | None
    """
    fgate predicted expert load per step (logical expert space).
    Only allocated when eplb_algorithm in
    ("fgate", "fgate-v2", "fgate-peer-cache", "fgate-hybrid-cache").
    Shape: (num_moe_layers, num_logical_experts)
    """
    expert_load_fgate_window: torch.Tensor | None
    """
    Sliding window for fgate predicted load (logical expert space).
    Only allocated when eplb_algorithm in
    ("fgate", "fgate-v2", "fgate-peer-cache").
    Shape: (window_size, num_moe_layers, num_logical_experts)
    """
    expert_load_fgate_window_step: int
    """Current step in the fgate sliding window."""
    model_name: str
    model: MixtureOfExperts
    expert_buffer: list[torch.Tensor]
    """
    The buffer to store the expert weights during transfer.
    """
    buffer_lock: threading.Lock
    """
    The lock to protect the expert buffer.
    """
    buffer_ready_event: torch.cuda.Event | None
    """
    CUDA event recorded when the async worker finishes filling the buffer.
    The main thread waits on this before consuming the buffer.
    """
    buffer_consumed_event: torch.cuda.Event | None
    """
    CUDA event recorded after the main thread finishes consuming the buffer.
    The async worker waits on this before writing to the buffer again.
    """
    window_ready_event: torch.cuda.Event | None
    """
    CUDA event recorded after all-reduce and clone on the main thread.
    The async worker waits on this before accessing global_expert_load_window.
    """
    ep_buffer_ready: int
    """
    The flag indicates whether the expert buffer is ready for transfer.
    0 or 1.
    """
    layer_to_transfer: int
    """
    The layer index to transfer in async mode.
    """
    rebalanced: bool
    """
    The flag indicates whether the experts rebalance have been computed.
    """
    pending_global_ready_check: bool
    """
    Whether the async EPLB needs to poll peers for buffer readiness.
    """
    eplb_stats: EplbStats | None
    """
    EPLB stats for the model.
    """
    is_unchanged: np.ndarray
    """
    intermediate variable between `move_to_buffer` and `move_to_workspace`.
    The size is same as the num of physical experts in the current layer.
    """
    is_received_locally: np.ndarray
    """
    intermediate variable between `move_to_buffer` and `move_to_workspace`.
    The size is same as the num of physical experts in the current layer.
    """
    recv_metadata: RecvMetadata
    """
    intermediate variable between `move_to_buffer` and `move_to_workspace`.
    """
    cuda_device_index: int | None
    """
    CUDA device index for the async EPLB worker thread.
    """
    new_physical_to_logical_map: torch.Tensor | None = None
    """
    intermediate variable between `move_to_buffer` and `move_to_workspace`.
    the size is same as physical_to_logical_map
    """
    new_logical_to_physical_map: torch.Tensor | None = None
    """
    intermediate variable between `move_to_buffer` and `move_to_workspace`.
    the size is same as logical_to_physical_map
    """
    new_logical_replica_count: torch.Tensor | None = None
    """
    intermediate variable between `move_to_buffer` and `move_to_workspace`.
    the size is same as logical_replica_count
    """
    peer_cache_static_physical_to_logical_map: torch.Tensor | None = None
    """
    For fgate-peer-cache, the immutable primary-slot logical layout.
    Dynamic slots are patched on top of this base layout.
    """
    peer_cache_dynamic_physical_ids: torch.Tensor | None = None
    """
    For fgate-peer-cache, the physical expert ids reserved as dynamic slots.
    Shape: (num_dynamic_slots,)
    """
    peer_cache_home_rank: torch.Tensor | None = None
    """
    For fgate-peer-cache, the primary/home rank for each logical expert.
    Shape: (num_logical_experts,)
    """
    peer_cache_primary_physical_to_logical_map: torch.Tensor | None = None
    """
    For fgate-hybrid-cache, the immutable primary/home logical layout without
    any redundant overlay slots.
    """
    peer_cache_static_physical_ids: torch.Tensor | None = None
    """
    For fgate-hybrid-cache, the physical expert ids reserved as EMA-managed
    static redundant slots.
    """
    peer_cache_dynamic_bank_physical_ids: torch.Tensor | None = None
    """
    For fgate-hybrid-cache, the physical expert ids reserved as the two
    double-buffered dynamic banks.
    Shape: (2, num_bank_slots)
    """
    peer_cache_active_dynamic_bank_idx: torch.Tensor | None = None
    """
    For fgate-hybrid-cache, the active dynamic bank index for each layer.
    Shape: (num_moe_layers,)
    """


class EplbState:
    """
    EplbState of each expert parallel model. Key is the model config hash.
    """

    def __init__(self, parallel_config: ParallelConfig, device: torch.device):
        self.parallel_config = parallel_config
        self.device = device
        self.model_states: dict[str, EplbModelState] = {}
        self.policy: type[AbstractEplbPolicy] = DefaultEplbPolicy
        """
        Selected EPLB algorithm class
        """
        self.expert_load_window_step: int = 0
        """
        Current step in the sliding window.

        Different from `expert_rearrangement_step`, 
        each EP rank may have its own `expert_load_window_step`.
        """
        self.expert_load_window_size: int = 0
        """
        Size of the expert load sliding window.
        This is a constant and is taken from the config.
        """
        self.expert_rearrangement_step: int = 0
        """
        Steps after last rearrangement.
        Will trigger a rearrangement if it exceeds the threshold.

        NOTE: Keep in mind that all EP ranks need to have the same
        `expert_rearrangement_step` value to ensure synchronization.
        Otherwise, the rearrangement will hang at collective
        communication calls.
        """
        self.expert_rearrangement_step_interval: int = 0
        """
        Interval for expert rearrangement steps.
        This is a constant and is taken from the config.
        """
        self.is_async: bool = False
        """
        The flag indicates whether the EPLB is running in async mode.
        """
        self.rearrange_event = threading.Event()
        """
        Event to signal when a new rearrangement is needed for the async thread.
        """
        self.async_worker: threading.Thread | None = None
        """
        Background thread handling async transfers.
        """
        self.cuda_device_index: int | None = None
        """
        CUDA device index for the async EPLB worker thread.
        """
        self.num_valid_physical_experts: int = 0
        """
        Number of valid physical experts.
        This is the number of physical experts that are
        actually mapped to logical experts. In elastic EP,
        newly started EP ranks may not have physical experts
        mapped yet.
        """
        if self.device.type == "cuda":
            self.cuda_device_index = self.device.index
            if self.cuda_device_index is None and torch.cuda.is_available():
                self.cuda_device_index = torch.accelerator.current_device_index()

    @staticmethod
    def build_initial_global_physical_to_logical_map(
        num_routed_experts: int,
        num_redundant_experts: int,
        ep_size: int | None = None,
        balance_redundant: bool = False,
    ) -> Sequence[int]:
        """
        Build an initial expert arrangement using the following structure:
        [original routed experts, redundant experts]

        Returns:
            physical_to_logical_map (Sequence[int]): A list of integers,
                where each integer is the index of the logical expert
                that the corresponding physical expert maps to.
        """
        if (
            balance_redundant
            and ep_size is not None
            and ep_size > 0
            and num_redundant_experts > 0
            and (num_routed_experts + num_redundant_experts) % ep_size == 0
        ):
            num_physical_experts = num_routed_experts + num_redundant_experts
            num_local_physical_experts = num_physical_experts // ep_size
            base_per_rank = [
                num_routed_experts // ep_size
                + int(rank < (num_routed_experts % ep_size))
                for rank in range(ep_size)
            ]
            redundant_per_rank = [
                num_local_physical_experts - base_slots
                for base_slots in base_per_rank
            ]

            global_physical_to_logical_map: list[int] = []
            next_logical = 0
            for rank in range(ep_size):
                rank_map = list(range(next_logical, next_logical + base_per_rank[rank]))
                next_logical += base_per_rank[rank]
                for slot_idx in range(redundant_per_rank[rank]):
                    rank_map.append((slot_idx * ep_size + rank) % num_routed_experts)
                global_physical_to_logical_map.extend(rank_map)

            if len(global_physical_to_logical_map) == num_physical_experts:
                return global_physical_to_logical_map

        global_physical_to_logical_map = list(range(num_routed_experts))
        global_physical_to_logical_map += [
            i % num_routed_experts for i in range(num_redundant_experts)
        ]
        return global_physical_to_logical_map

    @staticmethod
    def build_logical_mapping_from_physical(
        physical_to_logical_map: torch.Tensor,
        num_logical_experts: int,
        max_slots_per_logical_expert: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logical_to_physical_map = torch.full(
            (num_logical_experts, max_slots_per_logical_expert),
            -1,
            device=physical_to_logical_map.device,
            dtype=torch.long,
        )
        logical_replica_count = torch.zeros(
            (num_logical_experts,),
            device=physical_to_logical_map.device,
            dtype=torch.long,
        )
        for i in range(physical_to_logical_map.shape[0]):
            logical_idx = physical_to_logical_map[i]
            logical_to_physical_map[logical_idx, logical_replica_count[logical_idx]] = i
            logical_replica_count[logical_idx] += 1
        return logical_to_physical_map, logical_replica_count

    @staticmethod
    def get_peer_cache_dynamic_physical_ids(
        num_logical_experts: int,
        num_physical_experts: int,
        ep_size: int,
    ) -> torch.Tensor:
        num_local_physical_experts = num_physical_experts // ep_size
        base_per_rank = [
            num_logical_experts // ep_size
            + int(rank < (num_logical_experts % ep_size))
            for rank in range(ep_size)
        ]
        dynamic_ids: list[int] = []
        for rank, base_slots in enumerate(base_per_rank):
            rank_start = rank * num_local_physical_experts
            dynamic_ids.extend(
                range(rank_start + base_slots, rank_start + num_local_physical_experts)
            )
        return torch.tensor(dynamic_ids, dtype=torch.long)

    @staticmethod
    def get_hybrid_peer_cache_physical_ids(
        num_logical_experts: int,
        num_physical_experts: int,
        ep_size: int,
        num_static_redundant_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dynamic_ids = EplbState.get_peer_cache_dynamic_physical_ids(
            num_logical_experts=num_logical_experts,
            num_physical_experts=num_physical_experts,
            ep_size=ep_size,
        )
        if dynamic_ids.numel() == 0:
            return (
                torch.empty((0,), dtype=torch.long),
                torch.empty((2, 0), dtype=torch.long),
            )

        if (
            num_static_redundant_experts < 0
            or num_static_redundant_experts > dynamic_ids.numel()
        ):
            raise ValueError(
                "Invalid hybrid peer-cache static slot count: "
                f"{num_static_redundant_experts} for {dynamic_ids.numel()} redundant "
                "slots."
            )

        num_static_per_rank = num_static_redundant_experts // max(1, ep_size)
        dynamic_ids_per_rank = dynamic_ids.numel() // max(1, ep_size)
        remaining_dynamic_per_rank = dynamic_ids_per_rank - num_static_per_rank
        if remaining_dynamic_per_rank < 0 or remaining_dynamic_per_rank % 2 != 0:
            raise ValueError(
                "Hybrid peer-cache requires the dynamic slots on each EP rank "
                "to be splittable into two equal banks."
            )

        static_ids: list[int] = []
        bank0_ids: list[int] = []
        bank1_ids: list[int] = []
        for rank in range(max(1, ep_size)):
            rank_dynamic = dynamic_ids[
                rank * dynamic_ids_per_rank : (rank + 1) * dynamic_ids_per_rank
            ]
            static_end = num_static_per_rank
            bank_mid = static_end + remaining_dynamic_per_rank // 2
            static_ids.extend(rank_dynamic[:static_end].tolist())
            bank0_ids.extend(rank_dynamic[static_end:bank_mid].tolist())
            bank1_ids.extend(rank_dynamic[bank_mid:].tolist())

        return (
            torch.tensor(static_ids, dtype=torch.long),
            torch.stack(
                (
                    torch.tensor(bank0_ids, dtype=torch.long),
                    torch.tensor(bank1_ids, dtype=torch.long),
                )
            ),
        )

    @staticmethod
    def build_logical_mapping_from_physical_subset(
        physical_to_logical_map: torch.Tensor,
        included_physical_ids: torch.Tensor,
        num_logical_experts: int,
        max_slots_per_logical_expert: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logical_to_physical_map = torch.full(
            (num_logical_experts, max_slots_per_logical_expert),
            -1,
            device=physical_to_logical_map.device,
            dtype=torch.long,
        )
        logical_replica_count = torch.zeros(
            (num_logical_experts,),
            device=physical_to_logical_map.device,
            dtype=torch.long,
        )
        for physical_idx in included_physical_ids.detach().cpu().tolist():
            logical_idx = int(physical_to_logical_map[physical_idx].item())
            if logical_idx < 0:
                continue
            replica_idx = int(logical_replica_count[logical_idx].item())
            logical_to_physical_map[logical_idx, replica_idx] = int(physical_idx)
            logical_replica_count[logical_idx] += 1
        return logical_to_physical_map, logical_replica_count

    def validate_ep_configuration(self, new_model: MixtureOfExperts):
        """
        Validate that the expert parallel configuration of
        the new model is the same as the existing models.
        """
        if len(self.model_states) > 0:
            model = next(iter(self.model_states.values())).model
            if (
                model.num_routed_experts != new_model.num_routed_experts
                or model.num_redundant_experts != new_model.num_redundant_experts
                or model.num_physical_experts != new_model.num_physical_experts
                or model.num_logical_experts != new_model.num_logical_experts
                or model.num_expert_groups != new_model.num_expert_groups
            ):
                raise RuntimeError(
                    "Model: {} "
                    "with config {} "
                    "{} {} {} {} "
                    "mismatch with new model {} "
                    "with config {} "
                    "{} {} {} {}".format(
                        type(model),
                        model.num_routed_experts,
                        model.num_redundant_experts,
                        model.num_physical_experts,
                        model.num_logical_experts,
                        model.num_expert_groups,
                        type(new_model),
                        new_model.num_routed_experts,
                        new_model.num_redundant_experts,
                        new_model.num_physical_experts,
                        new_model.num_logical_experts,
                        new_model.num_expert_groups,
                    )
                )

    @staticmethod
    def _logical_expert_load_from_physical(
        physical_load: torch.Tensor,
        physical_to_logical_map: torch.Tensor,
        num_logical_experts: int,
    ) -> torch.Tensor:
        logical_load = torch.zeros(
            physical_load.shape[0],
            num_logical_experts,
            dtype=torch.float32,
            device=physical_load.device,
        )
        logical_load.scatter_add_(
            dim=-1,
            index=physical_to_logical_map.long(),
            src=physical_load.float(),
        )
        return logical_load

    def _build_hybrid_peer_cache_runtime_logical_mapping(
        self,
        model_state: EplbModelState,
        physical_to_logical_map: torch.Tensor,
        active_dynamic_bank_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert model_state.peer_cache_primary_physical_to_logical_map is not None
        assert model_state.peer_cache_static_physical_ids is not None
        assert model_state.peer_cache_dynamic_bank_physical_ids is not None

        num_physical_experts = physical_to_logical_map.shape[1]
        base_mask = torch.ones(
            (num_physical_experts,),
            dtype=torch.bool,
            device=physical_to_logical_map.device,
        )
        dynamic_ids = model_state.peer_cache_dynamic_bank_physical_ids.reshape(-1).long()
        if dynamic_ids.numel() > 0:
            base_mask[dynamic_ids] = False
        static_ids = model_state.peer_cache_static_physical_ids.long()
        if static_ids.numel() > 0:
            base_mask[static_ids] = True
        base_physical_ids = torch.nonzero(base_mask, as_tuple=False).flatten()

        max_slots_per_logical_expert = model_state.logical_to_physical_map.shape[-1]
        logical_to_physical_map = torch.full_like(model_state.logical_to_physical_map, -1)
        logical_replica_count = torch.zeros_like(model_state.logical_replica_count)

        for layer_idx in range(model_state.model.num_moe_layers):
            included_ids = base_physical_ids
            active_bank = int(active_dynamic_bank_idx[layer_idx].item())
            active_ids = model_state.peer_cache_dynamic_bank_physical_ids[active_bank].long()
            if active_ids.numel() > 0:
                included_ids = torch.cat((included_ids, active_ids))
            (
                layer_logical_to_physical,
                layer_logical_replica_count,
            ) = self.build_logical_mapping_from_physical_subset(
                physical_to_logical_map=physical_to_logical_map[layer_idx].long(),
                included_physical_ids=included_ids,
                num_logical_experts=model_state.model.num_logical_experts,
                max_slots_per_logical_expert=max_slots_per_logical_expert,
            )
            logical_to_physical_map[layer_idx].copy_(layer_logical_to_physical)
            logical_replica_count[layer_idx].copy_(layer_logical_replica_count)

        return logical_to_physical_map, logical_replica_count

    @staticmethod
    def _assign_peer_cache_dynamic_slots(
        desired_logical_ids: np.ndarray,
        current_logical_ids: np.ndarray,
        slot_ranks: np.ndarray,
        home_ranks: np.ndarray,
    ) -> np.ndarray:
        assigned = np.full_like(current_logical_ids, -1)
        remaining = desired_logical_ids.tolist()

        for slot_idx, logical_id in enumerate(current_logical_ids.tolist()):
            if logical_id in remaining:
                assigned[slot_idx] = logical_id
                remaining.remove(logical_id)

        for slot_idx in range(assigned.shape[0]):
            if assigned[slot_idx] != -1:
                continue
            preferred_idx = None
            for idx, logical_id in enumerate(remaining):
                if home_ranks[logical_id] != slot_ranks[slot_idx]:
                    preferred_idx = idx
                    break
            if preferred_idx is None:
                preferred_idx = 0
            assigned[slot_idx] = remaining.pop(preferred_idx)

        return assigned

    def _build_peer_cache_mapping(
        self,
        model_state: EplbModelState,
        global_expert_load_window: torch.Tensor,
        num_ranks: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert model_state.peer_cache_static_physical_to_logical_map is not None
        assert model_state.peer_cache_dynamic_physical_ids is not None
        assert model_state.peer_cache_home_rank is not None

        if not torch.count_nonzero(global_expert_load_window):
            return (
                model_state.physical_to_logical_map.clone(),
                model_state.logical_to_physical_map.clone(),
                model_state.logical_replica_count.clone(),
            )

        static_map = model_state.peer_cache_static_physical_to_logical_map
        dynamic_ids = model_state.peer_cache_dynamic_physical_ids.long()
        new_physical_to_logical_map = static_map.clone()
        num_local_physical_experts = model_state.model.num_local_physical_experts
        slot_ranks = (
            dynamic_ids.cpu().numpy() // max(1, num_local_physical_experts)
        ).astype(np.int64)
        home_ranks = model_state.peer_cache_home_rank.cpu().numpy().astype(np.int64)

        desired_phy2log_np, _, _ = DefaultEplbPolicy.replicate_experts(
            global_expert_load_window.float().cpu().numpy(),
            model_state.model.num_physical_experts,
        )
        desired_dynamic_np = desired_phy2log_np[
            :, model_state.model.num_logical_experts :
        ].astype(np.int64)
        current_dynamic_np = (
            model_state.physical_to_logical_map[:, dynamic_ids].cpu().numpy().astype(np.int64)
        )

        for layer_idx in range(model_state.model.num_moe_layers):
            assigned = self._assign_peer_cache_dynamic_slots(
                desired_logical_ids=desired_dynamic_np[layer_idx],
                current_logical_ids=current_dynamic_np[layer_idx],
                slot_ranks=slot_ranks,
                home_ranks=home_ranks,
            )
            new_physical_to_logical_map[layer_idx, dynamic_ids] = torch.from_numpy(
                assigned
            ).to(new_physical_to_logical_map.device)

        max_slots_per_logical_expert = model_state.logical_to_physical_map.shape[-1]
        new_logical_to_physical_map = torch.full_like(model_state.logical_to_physical_map, -1)
        new_logical_replica_count = torch.zeros_like(model_state.logical_replica_count)
        for layer_idx in range(model_state.model.num_moe_layers):
            (
                layer_logical_to_physical,
                layer_logical_replica_count,
            ) = self.build_logical_mapping_from_physical(
                physical_to_logical_map=new_physical_to_logical_map[layer_idx].long(),
                num_logical_experts=model_state.model.num_logical_experts,
                max_slots_per_logical_expert=max_slots_per_logical_expert,
            )
            new_logical_to_physical_map[layer_idx].copy_(layer_logical_to_physical)
            new_logical_replica_count[layer_idx].copy_(layer_logical_replica_count)

        return (
            new_physical_to_logical_map,
            new_logical_to_physical_map,
            new_logical_replica_count,
        )

    def _build_hybrid_peer_cache_static_mapping(
        self,
        model_state: EplbModelState,
        global_logical_ema_load: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert model_state.peer_cache_static_physical_ids is not None
        assert model_state.peer_cache_active_dynamic_bank_idx is not None
        assert model_state.peer_cache_home_rank is not None

        new_physical_to_logical_map = model_state.physical_to_logical_map.clone()
        static_ids = model_state.peer_cache_static_physical_ids.long()
        if static_ids.numel() == 0 or not torch.count_nonzero(global_logical_ema_load):
            return (
                new_physical_to_logical_map,
                model_state.logical_to_physical_map.clone(),
                model_state.logical_replica_count.clone(),
            )

        num_logical_experts = model_state.model.num_logical_experts
        desired_phy2log_np, _, _ = DefaultEplbPolicy.replicate_experts(
            global_logical_ema_load.float().cpu().numpy(),
            num_logical_experts + static_ids.numel(),
        )
        desired_static_np = desired_phy2log_np[:, num_logical_experts:].astype(np.int64)
        current_static_np = (
            model_state.physical_to_logical_map[:, static_ids].cpu().numpy().astype(np.int64)
        )

        num_local_physical_experts = model_state.model.num_local_physical_experts
        slot_ranks = (
            static_ids.cpu().numpy() // max(1, num_local_physical_experts)
        ).astype(np.int64)
        home_ranks = model_state.peer_cache_home_rank.cpu().numpy().astype(np.int64)

        for layer_idx in range(model_state.model.num_moe_layers):
            assigned = self._assign_peer_cache_dynamic_slots(
                desired_logical_ids=desired_static_np[layer_idx],
                current_logical_ids=current_static_np[layer_idx],
                slot_ranks=slot_ranks,
                home_ranks=home_ranks,
            )
            new_physical_to_logical_map[layer_idx, static_ids] = torch.from_numpy(
                assigned
            ).to(new_physical_to_logical_map.device)

        (
            new_logical_to_physical_map,
            new_logical_replica_count,
        ) = self._build_hybrid_peer_cache_runtime_logical_mapping(
            model_state=model_state,
            physical_to_logical_map=new_physical_to_logical_map,
            active_dynamic_bank_idx=model_state.peer_cache_active_dynamic_bank_idx,
        )
        return (
            new_physical_to_logical_map,
            new_logical_to_physical_map,
            new_logical_replica_count,
        )

    def _build_hybrid_peer_cache_fast_refresh(
        self,
        model_state: EplbModelState,
        global_predicted_load: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        assert model_state.peer_cache_static_physical_ids is not None
        assert model_state.peer_cache_dynamic_bank_physical_ids is not None
        assert model_state.peer_cache_active_dynamic_bank_idx is not None
        assert model_state.peer_cache_home_rank is not None

        new_physical_to_logical_map = model_state.physical_to_logical_map.clone()
        new_active_dynamic_bank_idx = (
            model_state.peer_cache_active_dynamic_bank_idx.clone()
        )
        layer_changed_slots = torch.zeros(
            (model_state.model.num_moe_layers,),
            dtype=torch.long,
            device=self.device,
        )
        layer_swap_mask = torch.zeros(
            (model_state.model.num_moe_layers,),
            dtype=torch.bool,
            device=self.device,
        )

        bank_ids = model_state.peer_cache_dynamic_bank_physical_ids.long()
        if bank_ids.numel() == 0 or not torch.count_nonzero(global_predicted_load):
            return (
                new_physical_to_logical_map,
                model_state.logical_to_physical_map.clone(),
                model_state.logical_replica_count.clone(),
                new_active_dynamic_bank_idx,
                layer_swap_mask,
                layer_changed_slots,
            )

        num_logical_experts = model_state.model.num_logical_experts
        num_local_physical_experts = model_state.model.num_local_physical_experts
        static_ids = model_state.peer_cache_static_physical_ids.long()
        static_logicals = (
            model_state.physical_to_logical_map[:, static_ids]
            if static_ids.numel() > 0
            else None
        )
        home_ranks = model_state.peer_cache_home_rank.cpu().numpy().astype(np.int64)
        slot_ranks = [
            (bank_ids[bank_idx].cpu().numpy() // max(1, num_local_physical_experts)).astype(
                np.int64
            )
            for bank_idx in range(2)
        ]

        for layer_idx in range(model_state.model.num_moe_layers):
            dynamic_weight = global_predicted_load[layer_idx].clone()
            if static_logicals is not None:
                covered_static = torch.unique(static_logicals[layer_idx].long())
                covered_static = covered_static[covered_static >= 0]
                if covered_static.numel() > 0:
                    dynamic_weight[covered_static] = 0
            if not torch.count_nonzero(dynamic_weight):
                continue

            desired_phy2log_np, _, _ = DefaultEplbPolicy.replicate_experts(
                dynamic_weight.unsqueeze(0).float().cpu().numpy(),
                num_logical_experts + bank_ids.shape[1],
            )
            desired_dynamic_ids = desired_phy2log_np[0, num_logical_experts:].astype(
                np.int64
            )

            active_bank = int(model_state.peer_cache_active_dynamic_bank_idx[layer_idx].item())
            standby_bank = 1 - active_bank
            active_ids = bank_ids[active_bank]
            standby_ids = bank_ids[standby_bank]

            active_current = (
                model_state.physical_to_logical_map[layer_idx, active_ids]
                .cpu()
                .numpy()
                .astype(np.int64)
            )
            active_assigned = self._assign_peer_cache_dynamic_slots(
                desired_logical_ids=desired_dynamic_ids,
                current_logical_ids=active_current,
                slot_ranks=slot_ranks[active_bank],
                home_ranks=home_ranks,
            )
            if np.array_equal(active_assigned, active_current):
                continue

            standby_current = (
                model_state.physical_to_logical_map[layer_idx, standby_ids]
                .cpu()
                .numpy()
                .astype(np.int64)
            )
            standby_assigned = self._assign_peer_cache_dynamic_slots(
                desired_logical_ids=desired_dynamic_ids,
                current_logical_ids=standby_current,
                slot_ranks=slot_ranks[standby_bank],
                home_ranks=home_ranks,
            )
            changed_slots = int(np.count_nonzero(standby_assigned != standby_current))
            if changed_slots > 0:
                new_physical_to_logical_map[layer_idx, standby_ids] = torch.from_numpy(
                    standby_assigned
                ).to(new_physical_to_logical_map.device)
            layer_changed_slots[layer_idx] = changed_slots
            layer_swap_mask[layer_idx] = True
            new_active_dynamic_bank_idx[layer_idx] = standby_bank

        (
            new_logical_to_physical_map,
            new_logical_replica_count,
        ) = self._build_hybrid_peer_cache_runtime_logical_mapping(
            model_state=model_state,
            physical_to_logical_map=new_physical_to_logical_map,
            active_dynamic_bank_idx=new_active_dynamic_bank_idx,
        )
        return (
            new_physical_to_logical_map,
            new_logical_to_physical_map,
            new_logical_replica_count,
            new_active_dynamic_bank_idx,
            layer_swap_mask,
            layer_changed_slots,
        )

    def _run_peer_cache_fast_path(
        self,
        is_dummy: bool,
        is_profile: bool,
    ) -> None:
        MAX_FAST_REFRESH_LAYERS_PER_STEP = 1
        if is_profile:
            self.rearrange(is_profile=True)
            return

        ep_group = get_ep_group().device_group
        if is_dummy:
            for model_state in self.model_states.values():
                if model_state.expert_load_fgate is not None:
                    model_state.expert_load_fgate.zero_()
                model_state.expert_load_pass.zero_()
            return

        predicted_loads: list[torch.Tensor] = []
        model_states = list(self.model_states.values())
        for model_state in model_states:
            assert model_state.expert_load_fgate is not None
            predicted_loads.append(model_state.expert_load_fgate.clone())
            model_state.expert_load_fgate.zero_()
            model_state.expert_load_pass.zero_()

        global_predicted_loads = self._allreduce_list(predicted_loads)
        fast_refresh_plan: list[
            tuple[
                EplbModelState,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                int,
                int,
            ]
        ] = []

        for model_state, global_predicted_load in zip(
            model_states, global_predicted_loads
        ):
            if not torch.count_nonzero(global_predicted_load):
                continue

            (
                new_physical_to_logical_map,
                new_logical_to_physical_map,
                new_logical_replica_count,
            ) = self._build_peer_cache_mapping(
                model_state=model_state,
                global_expert_load_window=global_predicted_load,
                num_ranks=ep_group.size(),
            )
            assert model_state.peer_cache_dynamic_physical_ids is not None
            dynamic_ids = model_state.peer_cache_dynamic_physical_ids.long()
            layer_changed_slots = (
                model_state.physical_to_logical_map[:, dynamic_ids]
                != new_physical_to_logical_map[:, dynamic_ids]
            ).sum(dim=1)
            changed_layers = torch.nonzero(layer_changed_slots > 0, as_tuple=False).flatten()
            total_changed_slots = int(layer_changed_slots.sum().item())
            if total_changed_slots == 0:
                continue

            if changed_layers.numel() > MAX_FAST_REFRESH_LAYERS_PER_STEP:
                changed_scores = torch.stack(
                    (
                        layer_changed_slots[changed_layers].float(),
                        global_predicted_load[changed_layers].amax(dim=1),
                    ),
                    dim=1,
                )
                sort_index = torch.argsort(
                    changed_scores[:, 0] * 1000.0 + changed_scores[:, 1],
                    descending=True,
                )
                changed_layers = changed_layers[
                    sort_index[:MAX_FAST_REFRESH_LAYERS_PER_STEP]
                ]

            selected_changed_slots = int(
                layer_changed_slots[changed_layers].sum().item()
            )
            fast_refresh_plan.append(
                (
                    model_state,
                    new_physical_to_logical_map,
                    new_logical_to_physical_map,
                    new_logical_replica_count,
                    changed_layers,
                    selected_changed_slots,
                    total_changed_slots,
                )
            )

        if not fast_refresh_plan:
            return

        is_main_rank = ep_group.rank() == 0
        if is_main_rank:
            total_changed_slots = sum(item[5] for item in fast_refresh_plan)
            total_pending_slots = sum(item[6] for item in fast_refresh_plan)
            total_dynamic_slots = sum(
                int(
                    item[0].peer_cache_dynamic_physical_ids.numel()
                    * item[0].model.num_moe_layers
                )
                for item in fast_refresh_plan
                if item[0].peer_cache_dynamic_physical_ids is not None
            )
            logger.info(
                "fgate-peer-cache immediate refresh: selected_changed_slots=%d, "
                "pending_changed_slots=%d/%d",
                total_changed_slots,
                total_pending_slots,
                total_dynamic_slots,
            )

        for (
            model_state,
            new_physical_to_logical_map,
            new_logical_to_physical_map,
            new_logical_replica_count,
            changed_layers,
            _,
            _,
        ) in fast_refresh_plan:
            rearrange_expert_weights_inplace(
                model_state.physical_to_logical_map[changed_layers],
                new_physical_to_logical_map[changed_layers],
                [model_state.model.expert_weights[int(layer)] for layer in changed_layers],
                ep_group,
                is_profile=False,
            )
            model_state.physical_to_logical_map[changed_layers].copy_(
                new_physical_to_logical_map[changed_layers]
            )
            model_state.logical_to_physical_map[changed_layers].copy_(
                new_logical_to_physical_map[changed_layers]
            )
            model_state.logical_replica_count[changed_layers].copy_(
                new_logical_replica_count[changed_layers]
            )

    def _step_hybrid_peer_cache(
        self,
        is_dummy: bool = False,
        is_profile: bool = False,
        log_stats: bool = False,
    ) -> None:
        ep_group = get_ep_group().device_group
        if is_profile:
            self.rearrange(is_profile=True)
            return

        if (
            log_stats
            and self.expert_rearrangement_step
            % self.parallel_config.eplb_config.log_balancedness_interval
            == 0
        ):
            expert_load_pass_list = self._sync_load_pass()
            for expert_load_pass, eplb_model_state in zip(
                expert_load_pass_list, self.model_states.values()
            ):
                num_tokens_per_rank = (
                    expert_load_pass.reshape(
                        expert_load_pass.shape[0], ep_group.size(), -1
                    )
                    .sum(dim=-1)
                    .float()
                )
                avg_tokens_tensor = num_tokens_per_rank.mean(dim=0).sum(dim=0)
                max_tokens_tensor = num_tokens_per_rank.max(dim=0).values.sum(dim=0)
                avg_tokens, max_tokens = torch.stack(
                    [avg_tokens_tensor, max_tokens_tensor]
                ).tolist()
                balancedness = avg_tokens / max_tokens if max_tokens > 0 else 0.0
                if ep_group.rank() == 0:
                    logger.info(
                        "EPLB step: %d for model %s: avg_tokens=%.2f, "
                        "max_tokens=%d, balancedness=%.4f, "
                        "steps until the next rearrangement: %d",
                        self.expert_rearrangement_step,
                        eplb_model_state.model_name,
                        avg_tokens,
                        max_tokens,
                        balancedness,
                        self.expert_rearrangement_step_interval
                        - self.expert_rearrangement_step,
                    )

        if is_dummy:
            for model_state in self.model_states.values():
                if model_state.expert_load_fgate is not None:
                    model_state.expert_load_fgate.zero_()
                model_state.expert_load_pass.zero_()
            self.expert_rearrangement_step += 1
            if self.expert_rearrangement_step >= self.expert_rearrangement_step_interval:
                self.expert_rearrangement_step = 0
                self.rearrange()
            return

        predicted_loads: list[torch.Tensor] = []
        model_states = list(self.model_states.values())
        for model_state in model_states:
            assert model_state.expert_load_logical_ema is not None
            logical_load = self._logical_expert_load_from_physical(
                physical_load=model_state.expert_load_pass,
                physical_to_logical_map=model_state.physical_to_logical_map,
                num_logical_experts=model_state.model.num_logical_experts,
            )
            alpha = model_state.eplb_ema_alpha
            model_state.expert_load_logical_ema.mul_(1.0 - alpha).add_(
                logical_load, alpha=alpha
            )
            assert model_state.expert_load_fgate is not None
            predicted_loads.append(model_state.expert_load_fgate.clone())
            model_state.expert_load_fgate.zero_()
            model_state.expert_load_pass.zero_()

        global_predicted_loads = self._allreduce_list(predicted_loads)
        fast_refresh_plan: list[
            tuple[
                EplbModelState,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]
        ] = []

        for model_state, global_predicted_load in zip(
            model_states, global_predicted_loads
        ):
            (
                new_physical_to_logical_map,
                new_logical_to_physical_map,
                new_logical_replica_count,
                new_active_dynamic_bank_idx,
                layer_swap_mask,
                layer_changed_slots,
            ) = self._build_hybrid_peer_cache_fast_refresh(
                model_state=model_state,
                global_predicted_load=global_predicted_load,
            )
            if not torch.any(layer_swap_mask):
                continue
            fast_refresh_plan.append(
                (
                    model_state,
                    new_physical_to_logical_map,
                    new_logical_to_physical_map,
                    new_logical_replica_count,
                    new_active_dynamic_bank_idx,
                    layer_changed_slots,
                )
            )

        if fast_refresh_plan and ep_group.rank() == 0:
            total_changed_slots = sum(int(item[5].sum().item()) for item in fast_refresh_plan)
            total_swapped_layers = sum(
                int(
                    torch.count_nonzero(
                        item[4] != item[0].peer_cache_active_dynamic_bank_idx
                    ).item()
                )
                for item in fast_refresh_plan
                if item[0].peer_cache_active_dynamic_bank_idx is not None
            )
            logger.info(
                "fgate-hybrid-cache immediate refresh: swapped_layers=%d, "
                "changed_standby_slots=%d",
                total_swapped_layers,
                total_changed_slots,
            )

        for (
            model_state,
            new_physical_to_logical_map,
            new_logical_to_physical_map,
            new_logical_replica_count,
            new_active_dynamic_bank_idx,
            layer_changed_slots,
        ) in fast_refresh_plan:
            changed_layers = torch.nonzero(layer_changed_slots > 0, as_tuple=False).flatten()
            if changed_layers.numel() > 0:
                rearrange_expert_weights_inplace(
                    model_state.physical_to_logical_map[changed_layers],
                    new_physical_to_logical_map[changed_layers],
                    [
                        model_state.model.expert_weights[int(layer)]
                        for layer in changed_layers
                    ],
                    ep_group,
                    is_profile=False,
                )
            swap_layers = torch.nonzero(
                new_active_dynamic_bank_idx != model_state.peer_cache_active_dynamic_bank_idx,
                as_tuple=False,
            ).flatten()
            if swap_layers.numel() == 0:
                continue
            model_state.physical_to_logical_map[swap_layers].copy_(
                new_physical_to_logical_map[swap_layers]
            )
            model_state.peer_cache_active_dynamic_bank_idx[swap_layers].copy_(
                new_active_dynamic_bank_idx[swap_layers]
            )
            model_state.logical_to_physical_map[swap_layers].copy_(
                new_logical_to_physical_map[swap_layers]
            )
            model_state.logical_replica_count[swap_layers].copy_(
                new_logical_replica_count[swap_layers]
            )

        self.expert_rearrangement_step += 1
        if self.expert_rearrangement_step >= self.expert_rearrangement_step_interval:
            self.expert_rearrangement_step = 0
            self.rearrange()

    def add_model(
        self,
        model: MixtureOfExperts,
        model_config: ModelConfig,
    ):
        """
        Build the initial EPLB state.
        """
        self.validate_ep_configuration(model)
        self.is_async = self.parallel_config.eplb_config.use_async

        ep_group = get_ep_group().device_group
        ep_size = ep_group.size()
        eplb_algorithm = self.parallel_config.eplb_config.algorithm
        physical_to_logical_map_list = (
            EplbState.build_initial_global_physical_to_logical_map(
                model.num_routed_experts,
                model.num_redundant_experts,
                ep_size=ep_size,
                balance_redundant=(
                    eplb_algorithm in ("fgate-peer-cache", "fgate-hybrid-cache")
                ),
            )
        )
        physical_to_logical_map = torch.tensor(
            physical_to_logical_map_list,
            device=self.device,
        )
        # Assuming 8 GPUs per node, this supports up to
        # (1023 + 1) / 8 = 128 nodes for now.
        # TODO(rui): make this configurable
        MAX_EXPERT_REDUNDANCY = 1023
        assert model.num_redundant_experts <= MAX_EXPERT_REDUNDANCY, (
            f"num_redundant_experts {model.num_redundant_experts} "
            f"must be less than or equal to {MAX_EXPERT_REDUNDANCY}"
        )
        max_slots_per_logical_expert = MAX_EXPERT_REDUNDANCY + 1
        logical_to_physical_map, logical_replica_count = (
            EplbState.build_logical_mapping_from_physical(
                physical_to_logical_map=physical_to_logical_map.long(),
                num_logical_experts=model.num_logical_experts,
                max_slots_per_logical_expert=max_slots_per_logical_expert,
            )
        )

        # Duplicate initial mapping for all layers
        physical_to_logical_map = (
            physical_to_logical_map.unsqueeze(0)
            .expand(
                model.num_moe_layers,
                -1,
            )
            .contiguous()
        )
        logical_to_physical_map = (
            logical_to_physical_map.unsqueeze(0)
            .expand(
                model.num_moe_layers,
                -1,
                -1,
            )
            .contiguous()
        )
        logical_replica_count = (
            logical_replica_count.unsqueeze(0)
            .expand(
                model.num_moe_layers,
                -1,
            )
            .contiguous()
        )

        expert_load_pass = torch.zeros(
            (model.num_moe_layers, model.num_physical_experts),
            dtype=torch.int32,
            device=self.device,
        )
        eplb_ema_alpha = self.parallel_config.eplb_config.ema_alpha
        self.expert_load_window_size = self.parallel_config.eplb_config.window_size
        expert_load_window = torch.zeros(
            (
                self.expert_load_window_size,
                model.num_moe_layers,
                model.num_physical_experts,
            ),
            dtype=torch.int32,
            device=self.device,
        )

        expert_load_ema: torch.Tensor | None = None
        if eplb_algorithm == "ema":
            expert_load_ema = torch.zeros(
                (model.num_moe_layers, model.num_physical_experts),
                dtype=torch.float32,
                device=self.device,
            )

        expert_load_logical_ema: torch.Tensor | None = None
        if eplb_algorithm == "fgate-hybrid-cache":
            expert_load_logical_ema = torch.zeros(
                (model.num_moe_layers, model.num_logical_experts),
                dtype=torch.float32,
                device=self.device,
            )

        expert_load_fgate: torch.Tensor | None = None
        expert_load_fgate_window: torch.Tensor | None = None
        if eplb_algorithm in (
            "fgate",
            "fgate-v2",
            "fgate-peer-cache",
            "fgate-hybrid-cache",
        ):
            expert_load_fgate = torch.zeros(
                (model.num_moe_layers, model.num_logical_experts),
                dtype=torch.float32,
                device=self.device,
            )
        if eplb_algorithm in ("fgate", "fgate-v2", "fgate-peer-cache"):
            expert_load_fgate_window = torch.zeros(
                (
                    self.expert_load_window_size,
                    model.num_moe_layers,
                    model.num_logical_experts,
                ),
                dtype=torch.float32,
                device=self.device,
            )

        # Set the initial progress of rearrangement to 3/4
        eplb_step_interval = self.parallel_config.eplb_config.step_interval
        self.expert_rearrangement_step = max(
            0, eplb_step_interval - eplb_step_interval // 4
        )
        self.expert_rearrangement_step_interval = eplb_step_interval

        policy_type = self.parallel_config.eplb_config.policy
        self.policy = EPLB_POLICIES[policy_type]
        logger.debug("Selected EPLB policy: %s", policy_type)
        if ep_group.rank() == 0:
            logger.info(
                "EPLB initialized for model %s: algorithm=%s, policy=%s, "
                "window_size=%d, step_interval=%d, ema_alpha=%s",
                model_config.model,
                eplb_algorithm,
                policy_type,
                self.expert_load_window_size,
                eplb_step_interval,
                eplb_ema_alpha,
            )

        expert_buffer = [torch.empty_like(w) for w in model.expert_weights[0]]
        peer_cache_dynamic_physical_ids = None
        peer_cache_static_physical_to_logical_map = None
        peer_cache_home_rank = None
        peer_cache_primary_physical_to_logical_map = None
        peer_cache_static_physical_ids = None
        peer_cache_dynamic_bank_physical_ids = None
        peer_cache_active_dynamic_bank_idx = None
        if eplb_algorithm == "fgate-peer-cache":
            if ep_group.rank() == 0 and self.parallel_config.data_parallel_size > 1:
                logger.warning(
                    "fgate-peer-cache immediate per-step refresh is disabled when "
                    "data_parallel_size=%d because request-local decode steps are "
                    "not synchronized across EP ranks; falling back to periodic "
                    "peer-cache refresh controlled by window_size/step_interval.",
                    self.parallel_config.data_parallel_size,
                )
            peer_cache_dynamic_physical_ids = (
                EplbState.get_peer_cache_dynamic_physical_ids(
                    model.num_logical_experts,
                    model.num_physical_experts,
                    ep_size,
                ).to(self.device)
            )
            peer_cache_static_physical_to_logical_map = physical_to_logical_map.clone()
            primary_logical_to_physical = logical_to_physical_map[:, :, 0]
            num_local_physical_experts = model.num_local_physical_experts
            peer_cache_home_rank = (
                primary_logical_to_physical[0] // num_local_physical_experts
            ).to(self.device)
        elif eplb_algorithm == "fgate-hybrid-cache":
            if ep_group.rank() == 0 and self.parallel_config.data_parallel_size > 1:
                logger.warning(
                    "fgate-hybrid-cache immediate per-step refresh is disabled when "
                    "data_parallel_size=%d because request-local decode steps are "
                    "not synchronized across EP ranks; falling back to periodic "
                    "hybrid peer-cache refresh controlled by step_interval.",
                    self.parallel_config.data_parallel_size,
                )
            static_redundant = (
                self.parallel_config.eplb_config.resolved_static_redundant_experts()
            )
            (
                peer_cache_static_physical_ids,
                peer_cache_dynamic_bank_physical_ids,
            ) = EplbState.get_hybrid_peer_cache_physical_ids(
                model.num_logical_experts,
                model.num_physical_experts,
                ep_size,
                static_redundant,
            )
            peer_cache_primary_physical_to_logical_map = physical_to_logical_map.clone()
            peer_cache_dynamic_physical_ids = (
                peer_cache_dynamic_bank_physical_ids.reshape(-1).to(self.device)
            )
            peer_cache_static_physical_ids = peer_cache_static_physical_ids.to(
                self.device
            )
            peer_cache_dynamic_bank_physical_ids = (
                peer_cache_dynamic_bank_physical_ids.to(self.device)
            )
            peer_cache_active_dynamic_bank_idx = torch.zeros(
                (model.num_moe_layers,),
                dtype=torch.long,
                device=self.device,
            )
            primary_logical_to_physical = logical_to_physical_map[:, :, 0]
            num_local_physical_experts = model.num_local_physical_experts
            peer_cache_home_rank = (
                primary_logical_to_physical[0] // num_local_physical_experts
            ).to(self.device)

            hybrid_base_mask = torch.ones(
                (model.num_physical_experts,),
                dtype=torch.bool,
                device=self.device,
            )
            hybrid_base_mask[peer_cache_dynamic_physical_ids.long()] = False
            hybrid_base_mask[peer_cache_static_physical_ids.long()] = True
            hybrid_base_physical_ids = torch.nonzero(
                hybrid_base_mask, as_tuple=False
            ).flatten()
            hybrid_logical_to_physical_map = torch.full_like(
                logical_to_physical_map, -1
            )
            hybrid_logical_replica_count = torch.zeros_like(logical_replica_count)
            for layer_idx in range(model.num_moe_layers):
                included_ids = hybrid_base_physical_ids
                active_ids = peer_cache_dynamic_bank_physical_ids[0].long()
                if active_ids.numel() > 0:
                    included_ids = torch.cat((included_ids, active_ids))
                (
                    layer_logical_to_physical,
                    layer_logical_replica_count,
                ) = EplbState.build_logical_mapping_from_physical_subset(
                    physical_to_logical_map=physical_to_logical_map[layer_idx].long(),
                    included_physical_ids=included_ids,
                    num_logical_experts=model.num_logical_experts,
                    max_slots_per_logical_expert=max_slots_per_logical_expert,
                )
                hybrid_logical_to_physical_map[layer_idx].copy_(
                    layer_logical_to_physical
                )
                hybrid_logical_replica_count[layer_idx].copy_(
                    layer_logical_replica_count
                )
            logical_to_physical_map = hybrid_logical_to_physical_map
            logical_replica_count = hybrid_logical_replica_count

        model.set_eplb_state(
            expert_load_pass,
            logical_to_physical_map,
            logical_replica_count,
            expert_load_fgate=expert_load_fgate,
            fgate_skip_prefill=(
                eplb_algorithm in (
                    "fgate-v2",
                    "fgate-peer-cache",
                    "fgate-hybrid-cache",
                )
            ),
        )

        model_state = EplbModelState(
            physical_to_logical_map=physical_to_logical_map,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
            expert_load_pass=expert_load_pass,
            expert_load_window=expert_load_window,
            eplb_algorithm=eplb_algorithm,
            eplb_ema_alpha=eplb_ema_alpha,
            expert_load_ema=expert_load_ema,
            expert_load_logical_ema=expert_load_logical_ema,
            expert_load_fgate=expert_load_fgate,
            expert_load_fgate_window=expert_load_fgate_window,
            expert_load_fgate_window_step=0,
            model_name=model_config.model,
            model=model,
            expert_buffer=expert_buffer,
            buffer_lock=threading.Lock(),
            buffer_ready_event=None,
            buffer_consumed_event=None,
            window_ready_event=None,
            ep_buffer_ready=0,
            layer_to_transfer=0,
            rebalanced=False,
            pending_global_ready_check=False,
            eplb_stats=None,
            is_unchanged=np.array([]),
            is_received_locally=np.array([]),
            recv_metadata=RecvMetadata(
                recv_primary_mask=np.array([]),
                recv_count=0,
                recv_expert_ids=np.array([]),
                recv_dst_rows=np.array([]),
            ),
            cuda_device_index=self.cuda_device_index,
            new_physical_to_logical_map=None,
            new_logical_to_physical_map=None,
            new_logical_replica_count=None,
            peer_cache_static_physical_to_logical_map=peer_cache_static_physical_to_logical_map,
            peer_cache_dynamic_physical_ids=peer_cache_dynamic_physical_ids,
            peer_cache_home_rank=peer_cache_home_rank,
            peer_cache_primary_physical_to_logical_map=peer_cache_primary_physical_to_logical_map,
            peer_cache_static_physical_ids=peer_cache_static_physical_ids,
            peer_cache_dynamic_bank_physical_ids=peer_cache_dynamic_bank_physical_ids,
            peer_cache_active_dynamic_bank_idx=peer_cache_active_dynamic_bank_idx,
        )
        self.model_states[model_config.compute_hash()] = model_state
        self.num_valid_physical_experts = model.num_physical_experts

    def step(
        self,
        is_dummy: bool = False,
        is_profile: bool = False,
        log_stats: bool = False,
    ) -> None:
        """
        Step the EPLB state.

        Args:
            is_dummy (bool): If `True`, this is a dummy step and the load
                metrics recorded in this forward pass will not count.
                Defaults to `False`.
            is_profile (bool): If `True`, perform a dummy rearrangement
                with maximum communication cost. This is used in
                `profile_run` to reserve enough memory
                for the communication buffer.
            log_stats (bool): If `True`, log the expert load metrics.

        # Stats
            The metrics are all summed up across layers.
            - `avg_tokens`: The average load across ranks.
            - `max_tokens`: The maximum load across ranks.
            - `balancedness`: The ratio of average load to maximum load.
        """
        ep_group = get_ep_group().device_group
        if is_profile:
            self.rearrange(is_profile=True)
            return

        if is_dummy:
            # Do not record load metrics for dummy steps
            for eplb_model_state in self.model_states.values():
                eplb_model_state.expert_load_pass.zero_()

        if (
            log_stats
            and self.expert_rearrangement_step
            % self.parallel_config.eplb_config.log_balancedness_interval
            == 0
        ):
            # Sync the expert load pass for each model (main and drafter).
            # expert_load_pass: (num_moe_layers, num_physical_experts)
            expert_load_pass_list = self._sync_load_pass()
            ep_group = get_ep_group().device_group
            for expert_load_pass, eplb_model_state in zip(
                expert_load_pass_list, self.model_states.values()
            ):
                # num_tokens_per_rank: (num_moe_layers, num_ranks)
                num_tokens_per_rank = (
                    expert_load_pass.reshape(
                        expert_load_pass.shape[0], ep_group.size(), -1
                    )
                    .sum(dim=-1)
                    .float()
                )

                # Compute balancedness ratio:
                # for each layer:
                #   (mean load across ranks) / (max load across ranks)
                avg_tokens_tensor = num_tokens_per_rank.mean(dim=0).sum(dim=0)
                max_tokens_tensor = num_tokens_per_rank.max(dim=0).values.sum(dim=0)

                # Just to make type checker happy
                tokens_tensors: list[float] = torch.stack(
                    [avg_tokens_tensor, max_tokens_tensor]
                ).tolist()
                avg_tokens, max_tokens = tokens_tensors
                balancedness = avg_tokens / max_tokens if max_tokens > 0 else 0.0

                if ep_group.rank() == 0:
                    logger.info(
                        "EPLB step: %d for model %s: avg_tokens=%.2f, "
                        "max_tokens=%d, balancedness=%.4f, "
                        "steps until the next rearrangement: %d",
                        self.expert_rearrangement_step,
                        eplb_model_state.model_name,
                        avg_tokens,
                        max_tokens,
                        balancedness,
                        self.expert_rearrangement_step_interval
                        - self.expert_rearrangement_step,
                    )

        if (
            self.parallel_config.data_parallel_size == 1
            and self.model_states
            and all(
                eplb_model_state.eplb_algorithm == "fgate-hybrid-cache"
                for eplb_model_state in self.model_states.values()
            )
        ):
            self._step_hybrid_peer_cache(
                is_dummy=is_dummy,
                is_profile=is_profile,
                log_stats=False,
            )
            return

        if any(
            eplb_model_state.eplb_algorithm == "fgate-peer-cache"
            for eplb_model_state in self.model_states.values()
        ) and self.parallel_config.data_parallel_size == 1:
            self.expert_rearrangement_step += 1
            self._run_peer_cache_fast_path(
                is_dummy=is_dummy,
                is_profile=is_profile,
            )
            return

        # Update load estimation
        if not is_dummy:
            advance_swm_window = False
            for eplb_model_state in self.model_states.values():
                if eplb_model_state.eplb_algorithm == "ema":
                    assert eplb_model_state.expert_load_ema is not None
                    alpha = eplb_model_state.eplb_ema_alpha
                    eplb_model_state.expert_load_ema.mul_(1.0 - alpha).add_(
                        eplb_model_state.expert_load_pass.float(), alpha=alpha
                    )
                elif eplb_model_state.eplb_algorithm in (
                    "fgate",
                    "fgate-v2",
                    "fgate-peer-cache",
                ):
                    assert eplb_model_state.expert_load_fgate is not None
                    assert eplb_model_state.expert_load_fgate_window is not None
                    eplb_model_state.expert_load_fgate_window[
                        eplb_model_state.expert_load_fgate_window_step
                    ] = eplb_model_state.expert_load_fgate.clone()
                    eplb_model_state.expert_load_fgate_window_step += 1
                    if (
                        eplb_model_state.expert_load_fgate_window_step
                        >= self.expert_load_window_size
                    ):
                        eplb_model_state.expert_load_fgate_window_step = 0
                    eplb_model_state.expert_load_fgate.zero_()
                elif eplb_model_state.eplb_algorithm == "fgate-hybrid-cache":
                    assert eplb_model_state.expert_load_logical_ema is not None
                    assert eplb_model_state.expert_load_fgate is not None
                    logical_load = self._logical_expert_load_from_physical(
                        physical_load=eplb_model_state.expert_load_pass,
                        physical_to_logical_map=eplb_model_state.physical_to_logical_map,
                        num_logical_experts=eplb_model_state.model.num_logical_experts,
                    )
                    alpha = eplb_model_state.eplb_ema_alpha
                    eplb_model_state.expert_load_logical_ema.mul_(1.0 - alpha).add_(
                        logical_load, alpha=alpha
                    )
                    eplb_model_state.expert_load_fgate.zero_()
                else:
                    eplb_model_state.expert_load_window[self.expert_load_window_step] = (
                        eplb_model_state.expert_load_pass.clone()
                    )
                    advance_swm_window = True
                eplb_model_state.expert_load_pass.zero_()

            if advance_swm_window:
                self.expert_load_window_step += 1
                if self.expert_load_window_step >= self.expert_load_window_size:
                    self.expert_load_window_step = 0

        # Step the expert rearrangement step
        # Note that even if this is a dummy step, we still increment the
        # rearrangement step and perform rearrangement to ensure all ranks are
        # performing collective communication.
        self.expert_rearrangement_step += 1

        if self.is_async:
            for eplb_model_state in self.model_states.values():
                all_ranks_buffer_ready = False
                if eplb_model_state.pending_global_ready_check:
                    all_ranks_buffer_ready = self._all_ranks_buffer_ready(
                        eplb_model_state
                    )
                if eplb_model_state.ep_buffer_ready and all_ranks_buffer_ready:
                    self.move_to_workspace(
                        model_state=eplb_model_state,
                        ep_group=ep_group,
                        is_profile=is_profile,
                    )

        if self.expert_rearrangement_step >= self.expert_rearrangement_step_interval:
            if self.is_async and any(
                eplb_model_state.rebalanced
                for eplb_model_state in self.model_states.values()
            ):
                # Still performing asynchronous rearrangement
                return
            self.expert_rearrangement_step = 0
            self.rearrange()

    def rearrange(
        self,
        is_profile: bool = False,
        rank_mapping: dict[int, int] | None = None,
    ) -> torch.Tensor | None:
        """
        Rearrange the experts according to the current load.

        Args:
            is_profile (bool): If `True`, perform a dummy rearrangement.
                This is used in `profile_run` to reserve enough memory,
                no memory movement will be performed. Default is False.
            rank_mapping (dict[int, int] | None): The rank mapping
                when scaling is done in EEP.
        """

        ep_group = get_ep_group().device_group
        ep_rank = ep_group.rank()

        start_event = None
        end_event = None
        is_main_rank = ep_rank == 0
        if is_main_rank:
            if not self.is_async or is_profile:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            logger.info(
                "Rearranging experts %s %s...",
                "(async mode)" if self.is_async else "sync mode",
                "(profile)" if is_profile else "",
            )

        # Map the physical expert load to global logical experts
        global_expert_load_windows = []
        for eplb_model_state in self.model_states.values():
            if eplb_model_state.eplb_algorithm == "ema":
                assert eplb_model_state.expert_load_ema is not None
                logical_expert_load = torch.zeros(
                    eplb_model_state.model.num_moe_layers,
                    eplb_model_state.model.num_logical_experts,
                    dtype=torch.float32,
                    device=eplb_model_state.expert_load_ema.device,
                )
                logical_expert_load.scatter_add_(
                    dim=-1,
                    index=eplb_model_state.physical_to_logical_map.long(),
                    src=eplb_model_state.expert_load_ema,
                )
                global_expert_load_window = logical_expert_load
            elif eplb_model_state.eplb_algorithm in (
                "fgate",
                "fgate-v2",
                "fgate-peer-cache",
            ):
                assert eplb_model_state.expert_load_fgate_window is not None
                global_expert_load_window = (
                    eplb_model_state.expert_load_fgate_window.sum(dim=0).float()
                )
            elif eplb_model_state.eplb_algorithm == "fgate-hybrid-cache":
                assert eplb_model_state.expert_load_logical_ema is not None
                global_expert_load_window = eplb_model_state.expert_load_logical_ema
            else:
                expert_load_window = eplb_model_state.expert_load_window[
                    :, :, : self.num_valid_physical_experts
                ]
                logical_expert_load_window = torch.zeros(
                    self.expert_load_window_size,
                    eplb_model_state.model.num_moe_layers,
                    eplb_model_state.model.num_logical_experts,
                    dtype=eplb_model_state.expert_load_window.dtype,
                    device=eplb_model_state.expert_load_window.device,
                )
                logical_expert_load_window.scatter_add_(
                    dim=-1,
                    index=eplb_model_state.physical_to_logical_map[
                        :, : self.num_valid_physical_experts
                    ]
                    .unsqueeze(0)
                    .expand_as(expert_load_window)
                    .long(),
                    src=expert_load_window,
                )

                global_expert_load_window = logical_expert_load_window.sum(dim=0)
            global_expert_load_windows.append(global_expert_load_window)
        # Perform all-reduce to get the expert load across all ranks for each model
        global_expert_load_windows = self._allreduce_list(global_expert_load_windows)

        # TODO(bowen): Treat differently for prefill and decode nodes
        eplb_model_state = next(iter(self.model_states.values()))
        model = eplb_model_state.model
        num_replicas = model.num_physical_experts
        num_groups = model.num_expert_groups

        if rank_mapping is not None and len(rank_mapping) == ep_group.size():
            # NOTE(yongji): scale down, we need to rebalance the experts on
            # remaining GPUs, transfer the experts while we haven't shutdown
            # the GPUs to be released.
            coordinator = get_ep_group()
            assert isinstance(coordinator, StatelessGroupCoordinator)
            tcp_store_group = coordinator.tcp_store_group
            num_nodes = _node_count_with_rank_mapping(tcp_store_group, rank_mapping)
            num_gpus = sum(new_rank != -1 for new_rank in rank_mapping.values())
            num_replicas = (
                num_replicas // ep_group.size() * num_gpus
            )  # handle num replicas change
        else:
            num_nodes = get_node_count()
            num_gpus = ep_group.size()

        if any(
            eplb_model_state.eplb_algorithm
            in ("fgate-peer-cache", "fgate-hybrid-cache")
            for eplb_model_state in self.model_states.values()
        ) and num_nodes != 1:
            raise RuntimeError(
                "fgate peer-cache algorithms currently only support same-node EPLB "
                f"(got num_nodes={num_nodes})."
            )

        if num_gpus % num_nodes != 0:
            num_nodes = 1
            logger.warning_once(
                f"num_gpus % num_nodes != 0, "
                "not using hierarchical rearrangement algorithm.\n"
                f"{num_gpus=}, {num_nodes=}"
            )

        # Get new expert mappings
        for eplb_model_state, global_expert_load_window in zip(
            self.model_states.values(), global_expert_load_windows
        ):
            if not self.is_async or is_profile:
                # Get new expert mappings for the model
                if eplb_model_state.eplb_algorithm == "fgate-peer-cache":
                    (
                        new_physical_to_logical_map,
                        new_logical_to_physical_map,
                        new_logical_replica_count,
                    ) = self._build_peer_cache_mapping(
                        model_state=eplb_model_state,
                        global_expert_load_window=global_expert_load_window,
                        num_ranks=num_gpus,
                    )
                    if (
                        is_main_rank
                        and not is_profile
                        and eplb_model_state.peer_cache_dynamic_physical_ids is not None
                    ):
                        dynamic_ids = eplb_model_state.peer_cache_dynamic_physical_ids.long()
                        changed_slots = (
                            eplb_model_state.physical_to_logical_map[:, dynamic_ids]
                            != new_physical_to_logical_map[:, dynamic_ids]
                        ).sum().item()
                        logger.info(
                            "fgate-peer-cache refresh for model %s: "
                            "changed_dynamic_slots=%d/%d",
                            eplb_model_state.model_name,
                            changed_slots,
                            int(dynamic_ids.numel() * eplb_model_state.model.num_moe_layers),
                        )
                elif eplb_model_state.eplb_algorithm == "fgate-hybrid-cache":
                    (
                        new_physical_to_logical_map,
                        new_logical_to_physical_map,
                        new_logical_replica_count,
                    ) = self._build_hybrid_peer_cache_static_mapping(
                        model_state=eplb_model_state,
                        global_logical_ema_load=global_expert_load_window,
                    )
                    if (
                        is_main_rank
                        and not is_profile
                        and eplb_model_state.peer_cache_static_physical_ids is not None
                    ):
                        static_ids = eplb_model_state.peer_cache_static_physical_ids.long()
                        changed_slots = (
                            eplb_model_state.physical_to_logical_map[:, static_ids]
                            != new_physical_to_logical_map[:, static_ids]
                        ).sum().item()
                        logger.info(
                            "fgate-hybrid-cache static refresh for model %s: "
                            "changed_static_slots=%d/%d",
                            eplb_model_state.model_name,
                            changed_slots,
                            int(static_ids.numel() * eplb_model_state.model.num_moe_layers),
                        )
                else:
                    (
                        new_physical_to_logical_map,
                        new_logical_to_physical_map,
                        new_logical_replica_count,
                    ) = self.policy.rebalance_experts(
                        global_expert_load_window,
                        num_replicas,
                        num_groups,
                        num_nodes,
                        num_gpus,
                        eplb_model_state.physical_to_logical_map,
                    )

                # Update expert weights
                rearrange_expert_weights_inplace(
                    eplb_model_state.physical_to_logical_map,
                    new_physical_to_logical_map,
                    eplb_model_state.model.expert_weights,
                    ep_group,
                    is_profile,
                    rank_mapping,
                )

                if not is_profile:
                    if (
                        eplb_model_state.physical_to_logical_map.shape[1]
                        != new_physical_to_logical_map.shape[1]
                    ):
                        eplb_model_state.physical_to_logical_map = (
                            new_physical_to_logical_map.to(
                                eplb_model_state.physical_to_logical_map.device
                            )
                        )
                    else:
                        eplb_model_state.physical_to_logical_map.copy_(
                            new_physical_to_logical_map
                        )
                    max_physical_slots = new_logical_to_physical_map.shape[-1]
                    assert (
                        max_physical_slots
                        <= eplb_model_state.logical_to_physical_map.shape[-1]
                    )
                    new_logical_to_physical_map = torch.nn.functional.pad(
                        new_logical_to_physical_map,
                        (
                            0,
                            eplb_model_state.logical_to_physical_map.shape[-1]
                            - max_physical_slots,
                        ),
                        value=-1,
                    )
                    eplb_model_state.logical_to_physical_map.copy_(
                        new_logical_to_physical_map
                    )
                    eplb_model_state.logical_replica_count.copy_(
                        new_logical_replica_count
                    )
                if is_main_rank:
                    assert start_event is not None
                    assert end_event is not None
                    end_event.record()
                    end_event.synchronize()
                    gpu_elapsed = start_event.elapsed_time(end_event) / 1000.0
                    logger.info(
                        "Rearranged experts %s in %.2f s.",
                        " (profile) " if is_profile else " ",
                        gpu_elapsed,
                    )
            else:
                eplb_model_state.eplb_stats = EplbStats(
                    # We copy the tensor to snapshot the global_expert_load_window
                    # on the main thread so that async worker can access it safely
                    # while the main thread is running.
                    global_expert_load_window=global_expert_load_window.clone(),
                    num_replicas=num_replicas,
                    num_groups=num_groups,
                    num_nodes=num_nodes,
                    num_gpus=num_gpus,
                )
                # Record event after clone to signal async worker
                # that load stats data is ready
                sync_event = torch.cuda.Event()
                sync_event.record()
                eplb_model_state.window_ready_event = sync_event

                eplb_model_state.rebalanced = True
                eplb_model_state.layer_to_transfer = 0
                eplb_model_state.pending_global_ready_check = True
        # Signal async thread to start transferring layers
        if self.is_async and (not is_profile):
            self.rearrange_event.set()
        return None

    def start_async_loop(
        self,
        rank_mapping: dict[int, int] | None = None,
        is_profile: bool = False,
    ):
        if not self.is_async:
            return
        if self.async_worker is None:
            self.async_worker = start_async_worker(
                self,
                is_profile=is_profile,
            )

    def _update_layer_mapping_from_new(
        self, model_state: EplbModelState, layer: int
    ) -> None:
        if (
            model_state.new_physical_to_logical_map is None
            or model_state.new_logical_to_physical_map is None
            or model_state.new_logical_replica_count is None
        ):
            return

        target_device = model_state.physical_to_logical_map.device
        new_physical = model_state.new_physical_to_logical_map
        # If the number of physical experts has changed, then the new map needs to
        # be copied synchronously to avoid a race condition with the async worker
        if model_state.physical_to_logical_map.shape[1] != new_physical.shape[1]:
            model_state.physical_to_logical_map = new_physical.to(target_device)
        else:
            model_state.physical_to_logical_map[layer].copy_(
                new_physical[layer].to(target_device, non_blocking=True)
            )

        logical_device = model_state.logical_to_physical_map.device
        new_logical = model_state.new_logical_to_physical_map[layer].to(logical_device)
        max_slots = model_state.logical_to_physical_map.shape[-1]
        slot_delta = max_slots - new_logical.shape[-1]
        if slot_delta > 0:
            new_logical = torch.nn.functional.pad(
                new_logical, (0, slot_delta), value=-1
            )
        model_state.logical_to_physical_map[layer].copy_(new_logical)

        replica_device = model_state.logical_replica_count.device
        model_state.logical_replica_count[layer].copy_(
            model_state.new_logical_replica_count[layer].to(replica_device)
        )

    def _all_ranks_buffer_ready(self, model_state: EplbModelState) -> bool:
        parallel_state = get_ep_group()
        cpu_group = getattr(parallel_state, "cpu_group", None)
        if cpu_group is not None and cpu_group.size() > 1:
            flag = torch.tensor(
                (int(model_state.ep_buffer_ready),), dtype=torch.int32, device="cpu"
            )
            all_reduce(flag, group=cpu_group)
            return int(flag.item()) == cpu_group.size()

        device_group = parallel_state.device_group
        if device_group.size() <= 1:
            return bool(model_state.ep_buffer_ready)

        device = getattr(
            parallel_state, "device", model_state.physical_to_logical_map.device
        )
        flag = torch.tensor(
            (int(model_state.ep_buffer_ready),), dtype=torch.int32, device=device
        )
        all_reduce(flag, group=device_group)
        return int(flag.item()) == device_group.size()

    def move_to_workspace(
        self,
        model_state: EplbModelState,
        ep_group: ProcessGroup,
        is_profile: bool = False,
    ):
        # We call move_to_workspace only when ep_buffer_ready is 1.
        # It means we only need to wait for the lock for a short time.
        max_retries = 6  # 1 minute max
        retries = 0
        while not model_state.buffer_lock.acquire(blocking=True, timeout=10.0):
            retries += 1
            if retries >= max_retries:
                raise RuntimeError(
                    f"Rank {ep_group.rank()}: buffer_lock timeout after "
                    "{max_retries * 10}s"
                )
            logger.warning(
                "Rank %d: EPLB buffer_lock acquire failed, retrying (%d/%d)",
                ep_group.rank(),
                retries,
                max_retries,
            )
        try:
            assert model_state.new_physical_to_logical_map is not None
            device_index = model_state.cuda_device_index or self.cuda_device_index
            if model_state.buffer_ready_event is not None and device_index is not None:
                stream = torch.cuda.current_stream(device=device_index)
                stream.wait_event(model_state.buffer_ready_event)
                model_state.buffer_ready_event = None
            expert_weights = model_state.model.expert_weights[
                model_state.layer_to_transfer
            ]
            expert_weights_buffer = model_state.expert_buffer
            new_indices = model_state.new_physical_to_logical_map[
                model_state.layer_to_transfer
            ].numpy()
            move_from_buffer(
                expert_weights=expert_weights,
                expert_weights_buffers=expert_weights_buffer,
                is_unchanged=model_state.is_unchanged,
                is_received_locally=model_state.is_received_locally,
                recv_metadata=model_state.recv_metadata,
                new_indices=new_indices,
                ep_rank=ep_group.rank(),
            )
            # Record event after consuming buffer to signal async thread
            # that it's safe to overwrite the intermediate buffer
            consumed_event = torch.cuda.Event()
            consumed_event.record()
            model_state.buffer_consumed_event = consumed_event

            transferred_layer = model_state.layer_to_transfer
            self._update_layer_mapping_from_new(model_state, transferred_layer)
            # After the main thread consumes, advance layer_to_transfer
            model_state.layer_to_transfer += 1
            model_state.ep_buffer_ready = 0
            logger.debug(
                "model %s successfully move_to_workspace layer %d",
                model_state.model_name,
                transferred_layer,
            )
            if model_state.layer_to_transfer >= model_state.model.num_moe_layers:
                self.post_eplb(model_state, is_profile)
                model_state.rebalanced = False
                model_state.layer_to_transfer = 0
                model_state.pending_global_ready_check = False
                logger.info(
                    "finish async transfer for model %s rank %d layer %d",
                    model_state.model_name,
                    ep_group.rank(),
                    model_state.model.num_moe_layers,
                )

        finally:
            try:
                model_state.buffer_lock.release()
            except Exception as e:
                logger.error(
                    "Rank %d: buffer_lock release failed in move_to_workspace: %s",
                    ep_group.rank(),
                    str(e),
                )

    def post_eplb(self, model_state: EplbModelState, is_profile: bool = False) -> None:
        assert model_state.new_physical_to_logical_map is not None
        assert model_state.new_logical_to_physical_map is not None
        assert model_state.new_logical_replica_count is not None

        model_state.new_physical_to_logical_map = None
        model_state.new_logical_to_physical_map = None
        model_state.new_logical_replica_count = None

    def _allreduce_list(self, tensor_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        All-reduce a list of tensors.
        """
        if len(tensor_list) == 1:
            all_reduce(tensor_list[0], group=get_ep_group().device_group)
            return tensor_list
        assert all(t.dim() == 2 for t in tensor_list), "All tensors must be 2D."
        assert all(t.shape[1] == tensor_list[0].shape[1] for t in tensor_list), (
            "All tensors must have the same shape[1]."
        )
        # Concatenate, all_reduce, then unpack to original shapes.
        # We assume all tensors are 2D and shape[1] (num_physical_experts)
        # is the same across all models.
        shapes = [t.shape for t in tensor_list]
        concat_tensor = torch.cat(tensor_list, dim=0)

        ep_group = get_ep_group().device_group
        all_reduce(concat_tensor, group=ep_group)

        all_reduce_list = []
        offset = 0
        for shape in shapes:
            all_reduce_list.append(concat_tensor[offset : offset + shape[0], :])
            offset += shape[0]
        return all_reduce_list

    def _sync_load_pass(self) -> list[torch.Tensor]:
        """
        Sync the expert load pass across all ranks for log stats.
        Doesn't update the expert load pass in eplb_model_state.
        """
        load_pass_list = []
        for eplb_model_state in self.model_states.values():
            load_pass_list.append(eplb_model_state.expert_load_pass.clone())
        return self._allreduce_list(load_pass_list)

    @classmethod
    def from_mapping(
        cls,
        model: MixtureOfExperts,
        model_config: ModelConfig,
        device: torch.device,
        parallel_config: ParallelConfig,
        expanded_physical_to_logical: torch.Tensor,
        num_valid_physical_experts: int,
    ) -> "EplbState":
        eplb_state = cls(
            parallel_config=parallel_config,
            device=device,
        )
        eplb_state.add_model(
            model=model,
            model_config=model_config,
        )
        eplb_state.num_valid_physical_experts = num_valid_physical_experts
        num_moe_layers = expanded_physical_to_logical.shape[0]
        num_physical_experts = expanded_physical_to_logical.shape[1]
        eplb_model_state = eplb_state.model_states[model_config.compute_hash()]
        eplb_model_state.physical_to_logical_map.copy_(expanded_physical_to_logical)

        if eplb_model_state.eplb_algorithm == "fgate-hybrid-cache":
            assert eplb_model_state.peer_cache_active_dynamic_bank_idx is not None
            (
                logical_to_physical_map,
                logical_replica_count,
            ) = eplb_state._build_hybrid_peer_cache_runtime_logical_mapping(
                model_state=eplb_model_state,
                physical_to_logical_map=eplb_model_state.physical_to_logical_map,
                active_dynamic_bank_idx=eplb_model_state.peer_cache_active_dynamic_bank_idx,
            )
            eplb_model_state.logical_to_physical_map.copy_(logical_to_physical_map)
            eplb_model_state.logical_replica_count.copy_(logical_replica_count)
            return eplb_state

        logical_to_physical_map = torch.full(
            (
                num_moe_layers,
                model.num_logical_experts,
                eplb_model_state.logical_to_physical_map.shape[2],
            ),
            -1,
            dtype=torch.int64,
        )
        logical_replica_count = torch.zeros(
            (num_moe_layers, model.num_logical_experts),
            dtype=torch.int64,
        )
        expanded_physical_to_logical_numpy = expanded_physical_to_logical.cpu().numpy()
        for layer_idx in range(num_moe_layers):
            for phys_idx in range(num_physical_experts):
                logical_idx = expanded_physical_to_logical_numpy[layer_idx, phys_idx]
                if logical_idx >= 0:
                    replica_idx = logical_replica_count[layer_idx, logical_idx]
                    logical_to_physical_map[layer_idx, logical_idx, replica_idx] = (
                        phys_idx
                    )
                    logical_replica_count[layer_idx, logical_idx] += 1

        logical_to_physical_map = logical_to_physical_map.to(device)
        logical_replica_count = logical_replica_count.to(device)
        eplb_model_state.logical_to_physical_map.copy_(logical_to_physical_map)
        eplb_model_state.logical_replica_count.copy_(logical_replica_count)
        return eplb_state


@dataclass
class EplbLayerState:
    """Runtime EPLB data stored in the MoE layer."""

    expert_load_view: torch.Tensor | None = None
    logical_to_physical_map: torch.Tensor | None = None
    logical_replica_count: torch.Tensor | None = None
    next_gate_weight: torch.Tensor | None = None
    expert_load_fgate_view: torch.Tensor | None = None


def _node_count_with_rank_mapping(
    pg: ProcessGroup | StatelessProcessGroup,
    rank_mapping: dict[int, int],
) -> int:
    if isinstance(pg, ProcessGroup):
        world_size = torch.distributed.get_world_size(group=pg)
    else:
        world_size = pg.world_size

    if world_size == 1:
        return 1

    # Build node assignment map
    node_assignment = [0] * world_size  # rank -> node_id
    next_node_id = 0

    for current_rank in range(world_size):
        if node_assignment[current_rank] != 0:
            continue  # Already assigned to a node

        assert current_rank in rank_mapping
        if rank_mapping[current_rank] == -1:
            continue  # Pending shutdown

        # Assign current rank to a new node
        next_node_id += 1
        node_assignment[current_rank] = next_node_id

        # Find all ranks on the same node as current_rank
        same_node_flags = in_the_same_node_as(pg, current_rank)
        for other_rank, is_same_node in enumerate(same_node_flags):
            if is_same_node and node_assignment[other_rank] == 0:
                node_assignment[other_rank] = next_node_id

    return next_node_id
