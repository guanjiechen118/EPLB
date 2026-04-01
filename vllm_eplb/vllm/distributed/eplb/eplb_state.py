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
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.distributed import ProcessGroup, ReduceOp, all_reduce

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
    move_to_buffer,
    move_to_buffer_local_only,
    rearrange_expert_weights_inplace,
)

logger = init_logger(__name__)

# Algorithms where each EP rank holds the same replicated global logical
# estimate (fgate window sums / peer-cache predictions). All-reduce AVG.
# fgate-hybrid-cache is excluded: per-rank logical EMA may differ; use SUM in
# rearrange(), and avoid per-step collectives in _step_hybrid_peer_cache (DP /
# scheduling can desynchronize ranks and deadlock NCCL).
_EPLB_ALGORITHMS_REPLICATED_LOGICAL_LOAD: frozenset[str] = frozenset(
    {"fgate", "fgate-v2", "fgate-peer-cache"}
)


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
    peer_cache_local_dynamic_bank_physical_ids: torch.Tensor | None = None
    """
    For fgate-hybrid-cache, the local physical expert ids reserved as the two
    double-buffered dynamic banks on this EP rank.
    Shape: (2, num_local_bank_slots)
    """
    peer_cache_local_dynamic_logical_ids: torch.Tensor | None = None
    """
    For fgate-hybrid-cache, the local logical assignments currently materialized
    in each dynamic shadow bank on this EP rank.
    Shape: (num_moe_layers, 2, num_local_bank_slots)
    """
    peer_cache_active_physical_to_logical_map: torch.Tensor | None = None
    """
    For fgate-hybrid-cache, the effective physical->logical map with the
    currently active local shadow bank patched in.
    Shape: (num_moe_layers, num_physical_experts)
    """
    peer_cache_periodic_synced_physical_to_logical_map: torch.Tensor | None = None
    """
    For fgate-hybrid-cache, a periodic-rearrange-only snapshot of the
    physical->logical map with all ranks' current local dynamic shadow
    assignments patched in. This is intentionally kept separate from the live
    runtime map to avoid mutating routing metadata mid-serve. Keep this on CPU
    so periodic static refresh never performs GPU metadata patching after the
    all_gather returns.
    Shape: (num_moe_layers, num_physical_experts)
    """
    fast_refresh_inflight: bool = False
    """Whether a decode-time async peer-cache refresh is in flight."""
    fast_refresh_thread: threading.Thread | None = None
    """Background thread staging the next peer-cache refresh into expert_buffer."""
    fast_refresh_ready_event: torch.cuda.Event | None = None
    """CUDA event recorded when the staged refresh buffer is ready to consume."""
    fast_refresh_consumed_event: torch.cuda.Event | None = None
    """CUDA event recorded after the staged refresh buffer is consumed."""
    fast_refresh_error: str | None = None
    """Background fast-refresh error propagated before the next forward."""
    fast_refresh_layer_idx: int = -1
    """Target layer index for the staged decode-time fast refresh."""
    fast_refresh_new_physical_to_logical_layer: torch.Tensor | None = None
    """Staged physical->logical mapping slice for decode-time fast refresh."""
    fast_refresh_new_logical_to_physical_layer: torch.Tensor | None = None
    """Staged logical->physical mapping slice for decode-time fast refresh."""
    fast_refresh_new_logical_replica_count_layer: torch.Tensor | None = None
    """Staged logical replica-count slice for decode-time fast refresh."""
    fast_refresh_new_active_dynamic_bank_idx: int | None = None
    """Staged active bank for hybrid-cache decode-time fast refresh."""
    fast_refresh_updates_global_mapping: bool = True
    """Whether the staged fast refresh should overwrite global EPLB maps."""
    fast_refresh_new_local_dynamic_logical_ids: torch.Tensor | None = None
    """Staged local shadow logical ids for the target hybrid-cache bank."""
    fast_refresh_is_unchanged: np.ndarray | None = None
    """Intermediate metadata for applying the staged fast refresh."""
    fast_refresh_is_received_locally: np.ndarray | None = None
    """Intermediate metadata for applying the staged fast refresh."""
    fast_refresh_recv_metadata: RecvMetadata | None = None
    """Intermediate metadata for applying the staged fast refresh."""


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
    def _assign_shadow_slots_stably(
        desired_logical_ids: torch.Tensor,
        current_logical_ids: torch.Tensor,
    ) -> torch.Tensor:
        desired = desired_logical_ids.long()
        current = current_logical_ids.long().to(desired.device)
        assigned = torch.full_like(current, -1)
        used = torch.zeros(
            (desired.shape[0],), dtype=torch.bool, device=desired.device
        )

        for slot_idx in range(current.shape[0]):
            matches = (~used) & (desired == current[slot_idx])
            if not torch.any(matches):
                continue
            match_idx = int(torch.nonzero(matches, as_tuple=False)[0].item())
            assigned[slot_idx] = current[slot_idx]
            used[match_idx] = True

        remaining = desired[~used]
        empty_slots = assigned < 0
        if torch.any(empty_slots):
            assigned[empty_slots] = remaining[: int(empty_slots.sum().item())]
        return assigned

    @staticmethod
    def _select_local_shadow_logical_ids(
        dynamic_weight: torch.Tensor,
        num_slots: int,
    ) -> torch.Tensor:
        if num_slots <= 0:
            return torch.empty(
                (0,), dtype=torch.long, device=dynamic_weight.device
            )

        positive_mask = dynamic_weight > 0
        if not torch.any(positive_mask):
            return torch.empty(
                (0,), dtype=torch.long, device=dynamic_weight.device
            )

        total_weight = dynamic_weight.sum()
        if float(total_weight.item()) <= 0.0:
            return torch.empty(
                (0,), dtype=torch.long, device=dynamic_weight.device
            )

        expected = dynamic_weight.float() * (
            float(num_slots) / float(total_weight.item())
        )
        slot_counts = torch.floor(expected).to(torch.long)
        remaining = num_slots - int(slot_counts.sum().item())
        if remaining > 0:
            residual = expected - slot_counts.float()
            residual = residual.masked_fill(~positive_mask, -1.0)
            extra_idx = torch.topk(residual, k=remaining, sorted=True).indices
            slot_counts[extra_idx] += 1

        candidate_ids = torch.nonzero(slot_counts > 0, as_tuple=False).flatten()
        if candidate_ids.numel() == 0:
            return torch.full(
                (num_slots,),
                int(torch.argmax(dynamic_weight).item()),
                dtype=torch.long,
                device=dynamic_weight.device,
            )

        candidate_load = dynamic_weight[candidate_ids]
        order = torch.argsort(candidate_load, descending=True)
        candidate_ids = candidate_ids[order]
        desired = torch.repeat_interleave(candidate_ids, slot_counts[candidate_ids])

        if desired.numel() < num_slots:
            pad = candidate_ids[0].expand(num_slots - desired.numel())
            desired = torch.cat((desired, pad), dim=0)
        elif desired.numel() > num_slots:
            desired = desired[:num_slots]
        return desired.long()

    def _sync_model_layer_shadow_views(self, model_state: EplbModelState) -> None:
        local_bank_ids = model_state.peer_cache_local_dynamic_bank_physical_ids
        local_shadow_logical_ids = model_state.peer_cache_local_dynamic_logical_ids
        active_bank_idx = model_state.peer_cache_active_dynamic_bank_idx
        for layer_idx, layer in enumerate(model_state.model.moe_layers):
            layer.eplb_state.physical_to_logical_map = model_state.physical_to_logical_map[
                layer_idx
            ]
            if (
                local_bank_ids is not None
                and local_shadow_logical_ids is not None
                and active_bank_idx is not None
            ):
                layer.eplb_state.local_dynamic_shadow_physical_ids = local_bank_ids
                layer.eplb_state.local_dynamic_shadow_logical_ids = (
                    local_shadow_logical_ids[layer_idx]
                )
                layer.eplb_state.local_dynamic_shadow_active_bank_idx = (
                    active_bank_idx[layer_idx]
                )
                self._sync_layer_active_shadow_view(model_state, layer_idx)
        self._refresh_active_shadow_physical_to_logical_map(model_state)

    def _refresh_active_shadow_physical_to_logical_map(
        self,
        model_state: EplbModelState,
        *,
        layer_idx: int | None = None,
    ) -> None:
        if (
            model_state.peer_cache_local_dynamic_bank_physical_ids is None
            or model_state.peer_cache_local_dynamic_logical_ids is None
            or model_state.peer_cache_active_dynamic_bank_idx is None
            or model_state.peer_cache_local_dynamic_bank_physical_ids.numel() == 0
        ):
            model_state.peer_cache_active_physical_to_logical_map = None
            return

        if (
            model_state.peer_cache_active_physical_to_logical_map is None
            or model_state.peer_cache_active_physical_to_logical_map.shape
            != model_state.physical_to_logical_map.shape
            or model_state.peer_cache_active_physical_to_logical_map.device
            != model_state.physical_to_logical_map.device
        ):
            model_state.peer_cache_active_physical_to_logical_map = (
                model_state.physical_to_logical_map.clone()
            )
            layer_idx = None

        cached_map = model_state.peer_cache_active_physical_to_logical_map
        assert cached_map is not None

        if layer_idx is None:
            for idx in range(model_state.model.num_moe_layers):
                cached_map[idx].copy_(
                    self._patch_local_shadow_into_layer_indices(
                        model_state=model_state,
                        layer_idx=idx,
                        active_bank_only=True,
                    )
                )
            return

        cached_map[layer_idx].copy_(
            self._patch_local_shadow_into_layer_indices(
                model_state=model_state,
                layer_idx=layer_idx,
                active_bank_only=True,
            )
        )

    def _build_hybrid_peer_cache_layer_fast_refresh(
        self,
        model_state: EplbModelState,
        layer_idx: int,
        local_predicted_load: torch.Tensor,
    ) -> tuple[torch.Tensor | None, int | None, int]:
        assert model_state.peer_cache_local_dynamic_bank_physical_ids is not None
        assert model_state.peer_cache_local_dynamic_logical_ids is not None
        assert model_state.peer_cache_active_dynamic_bank_idx is not None

        local_bank_ids = model_state.peer_cache_local_dynamic_bank_physical_ids.long()
        if local_bank_ids.numel() == 0 or not torch.count_nonzero(local_predicted_load):
            return None, None, 0

        num_logical_experts = model_state.model.num_logical_experts
        ep_rank = get_ep_group().device_group.rank()
        num_local_physical_experts = model_state.model.num_local_physical_experts
        local_begin = ep_rank * num_local_physical_experts
        local_end = local_begin + num_local_physical_experts
        local_bank_row_offsets = (local_bank_ids - local_begin).long()

        local_source_logicals = model_state.physical_to_logical_map[
            layer_idx, local_begin:local_end
        ].clone()
        for bank_idx in range(local_bank_row_offsets.shape[0]):
            local_source_logicals[local_bank_row_offsets[bank_idx]] = (
                model_state.peer_cache_local_dynamic_logical_ids[
                    layer_idx, bank_idx
                ].long()
            )

        sourceable_logicals = torch.zeros(
            (num_logical_experts,),
            dtype=torch.bool,
            device=self.device,
        )
        valid_source_logicals = local_source_logicals[local_source_logicals >= 0].long()
        if valid_source_logicals.numel() > 0:
            sourceable_logicals[valid_source_logicals] = True

        dynamic_weight = local_predicted_load.masked_fill(~sourceable_logicals, 0)
        if not torch.count_nonzero(dynamic_weight):
            return None, None, 0

        desired_dynamic_ids = self._select_local_shadow_logical_ids(
            dynamic_weight=dynamic_weight,
            num_slots=local_bank_ids.shape[1],
        )
        if desired_dynamic_ids.numel() == 0:
            return None, None, 0

        active_bank = int(model_state.peer_cache_active_dynamic_bank_idx[layer_idx].item())
        standby_bank = 1 - active_bank
        active_current = model_state.peer_cache_local_dynamic_logical_ids[
            layer_idx, active_bank
        ].long()
        active_assigned = self._assign_shadow_slots_stably(
            desired_logical_ids=desired_dynamic_ids,
            current_logical_ids=active_current,
        )
        if torch.equal(active_assigned, active_current):
            return None, None, 0

        standby_current = model_state.peer_cache_local_dynamic_logical_ids[
            layer_idx, standby_bank
        ].long()
        standby_assigned = self._assign_shadow_slots_stably(
            desired_logical_ids=desired_dynamic_ids,
            current_logical_ids=standby_current,
        )
        changed_slots = int(torch.count_nonzero(standby_assigned != standby_current).item())
        if changed_slots == 0:
            return None, None, 0

        return standby_assigned, standby_bank, changed_slots

    def _consume_hybrid_layer_refresh(
        self,
        model_state: EplbModelState,
        layer_idx: int,
    ) -> None:
        if not self._hybrid_enable_immediate_layer_refresh():
            return
        if torch.compiler.is_compiling():
            return
        if self.device.type != "cuda":
            return
        if torch.cuda.is_current_stream_capturing():
            return
        if (
            not model_state.fast_refresh_inflight
            or model_state.fast_refresh_layer_idx != layer_idx
        ):
            return

        ep_group = get_ep_group().device_group
        current_stream = torch.cuda.current_stream(self.device)
        self._consume_fast_refresh_for_model(model_state, current_stream, ep_group)

    def _maybe_schedule_hybrid_next_layer_refresh(
        self,
        model_state: EplbModelState,
        target_layer_idx: int,
        predicted_load: torch.Tensor,
    ) -> None:
        if not self._hybrid_enable_immediate_layer_refresh():
            return
        if torch.compiler.is_compiling():
            return
        if self.device.type != "cuda":
            return
        if torch.cuda.is_current_stream_capturing():
            return
        if model_state.fast_refresh_inflight:
            return
        if target_layer_idx < 0 or target_layer_idx >= model_state.model.num_moe_layers:
            return

        local_predicted_load = predicted_load.detach().to(
            device=self.device, dtype=torch.float32
        )
        local_predicted_load = local_predicted_load.reshape(-1)
        if local_predicted_load.numel() != model_state.model.num_logical_experts:
            return

        (
            new_local_dynamic_logical_ids,
            new_active_dynamic_bank_idx,
            changed_slots,
        ) = self._build_hybrid_peer_cache_layer_fast_refresh(
            model_state=model_state,
            layer_idx=target_layer_idx,
            local_predicted_load=local_predicted_load,
        )
        if (
            changed_slots <= 0
            or new_local_dynamic_logical_ids is None
            or new_active_dynamic_bank_idx is None
        ):
            return

        staged_layer_indices = self._patch_local_shadow_into_layer_indices(
            model_state=model_state,
            layer_idx=target_layer_idx,
            override_bank_idx=new_active_dynamic_bank_idx,
            override_bank_logical_ids=new_local_dynamic_logical_ids,
        )
        self._launch_async_fast_refresh(
            model_state=model_state,
            layer_idx=target_layer_idx,
            new_physical_to_logical_layer=staged_layer_indices,
            new_logical_to_physical_layer=model_state.logical_to_physical_map[
                target_layer_idx
            ],
            new_logical_replica_count_layer=model_state.logical_replica_count[
                target_layer_idx
            ],
            new_active_dynamic_bank_idx=new_active_dynamic_bank_idx,
            updates_global_mapping=False,
            new_local_dynamic_logical_ids=new_local_dynamic_logical_ids,
            local_only=True,
        )

    def _install_hybrid_layer_runtime_hooks(self, model_state: EplbModelState) -> None:
        if model_state.eplb_algorithm != "fgate-hybrid-cache":
            return

        num_layers = model_state.model.num_moe_layers
        for layer_idx, layer in enumerate(model_state.model.moe_layers):
            layer.eplb_state.consume_pending_layer_refresh = None
            layer.eplb_state.schedule_next_layer_shadow_refresh = None
            layer.eplb_state.next_gate_weight = None
            layer.eplb_state.expert_load_fgate_view = None
            layer.next_gate_weight = None
            layer.expert_load_fgate_view = None
            layer.fgate_skip_prefill = False

        if not self._hybrid_enable_immediate_layer_refresh():
            return

        for layer_idx, layer in enumerate(model_state.model.moe_layers):
            layer.eplb_state.consume_pending_layer_refresh = (
                lambda model_state=model_state, layer_idx=layer_idx: (
                    self._consume_hybrid_layer_refresh(model_state, layer_idx)
                )
            )
            if layer_idx + 1 < num_layers:
                layer.eplb_state.schedule_next_layer_shadow_refresh = (
                    lambda predicted_load,
                    model_state=model_state,
                    next_layer_idx=layer_idx + 1: (
                        self._maybe_schedule_hybrid_next_layer_refresh(
                            model_state, next_layer_idx, predicted_load
                        )
                    )
                )
            else:
                layer.eplb_state.schedule_next_layer_shadow_refresh = None

    def _sync_layer_active_shadow_view(
        self, model_state: EplbModelState, layer_idx: int
    ) -> None:
        layer = model_state.model.moe_layers[layer_idx]
        local_bank_ids = model_state.peer_cache_local_dynamic_bank_physical_ids
        local_shadow_logical_ids = model_state.peer_cache_local_dynamic_logical_ids
        active_bank_idx = model_state.peer_cache_active_dynamic_bank_idx

        if (
            local_bank_ids is None
            or local_shadow_logical_ids is None
            or active_bank_idx is None
            or local_bank_ids.numel() == 0
        ):
            layer.eplb_state.local_dynamic_shadow_active_physical_ids = None
            layer.eplb_state.local_dynamic_shadow_active_logical_ids = None
            return

        bank_idx = int(active_bank_idx[layer_idx].item())
        active_physical_ids = local_bank_ids[bank_idx].long()
        active_logical_ids = local_shadow_logical_ids[layer_idx, bank_idx].long()

        if (
            layer.eplb_state.local_dynamic_shadow_active_physical_ids is None
            or layer.eplb_state.local_dynamic_shadow_active_physical_ids.shape
            != active_physical_ids.shape
            or layer.eplb_state.local_dynamic_shadow_active_physical_ids.device
            != active_physical_ids.device
        ):
            layer.eplb_state.local_dynamic_shadow_active_physical_ids = (
                active_physical_ids.clone()
            )
        else:
            layer.eplb_state.local_dynamic_shadow_active_physical_ids.copy_(
                active_physical_ids
            )

        if (
            layer.eplb_state.local_dynamic_shadow_active_logical_ids is None
            or layer.eplb_state.local_dynamic_shadow_active_logical_ids.shape
            != active_logical_ids.shape
            or layer.eplb_state.local_dynamic_shadow_active_logical_ids.device
            != active_logical_ids.device
        ):
            layer.eplb_state.local_dynamic_shadow_active_logical_ids = (
                active_logical_ids.clone()
            )
        else:
            layer.eplb_state.local_dynamic_shadow_active_logical_ids.copy_(
                active_logical_ids
            )

    @staticmethod
    def _get_hybrid_periodic_source_map(model_state: EplbModelState) -> torch.Tensor:
        synced_map = model_state.peer_cache_periodic_synced_physical_to_logical_map
        if synced_map is not None:
            return synced_map
        return model_state.physical_to_logical_map

    def _should_barrier_after_hybrid_periodic_rearrange(self) -> bool:
        if self.is_async:
            return False
        if self.parallel_config.data_parallel_size <= 1:
            return False
        if not self.parallel_config.eplb_config.hybrid_periodic_rearrange_with_multi_dp:
            return False
        return any(
            model_state.eplb_algorithm == "fgate-hybrid-cache"
            for model_state in self.model_states.values()
        )

    def _hybrid_enable_immediate_layer_refresh(self) -> bool:
        if self.parallel_config.data_parallel_size <= 1:
            return True
        if not self.parallel_config.eplb_config.hybrid_periodic_rearrange_with_multi_dp:
            return True
        return False

    def _hybrid_periodic_rearrange_ready(self, local_ready: bool) -> bool:
        return local_ready

    def _patch_local_shadow_into_layer_indices(
        self,
        model_state: EplbModelState,
        layer_idx: int,
        *,
        active_bank_only: bool = False,
        override_bank_idx: int | None = None,
        override_bank_logical_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        layer_indices = model_state.physical_to_logical_map[layer_idx].clone()
        local_bank_ids = model_state.peer_cache_local_dynamic_bank_physical_ids
        local_shadow_logical_ids = model_state.peer_cache_local_dynamic_logical_ids
        active_bank_idx = model_state.peer_cache_active_dynamic_bank_idx
        if (
            local_bank_ids is None
            or local_shadow_logical_ids is None
            or active_bank_idx is None
            or local_bank_ids.numel() == 0
        ):
            return layer_indices

        if active_bank_only:
            bank_idx = int(active_bank_idx[layer_idx].item())
            layer_indices[local_bank_ids[bank_idx].long()] = local_shadow_logical_ids[
                layer_idx, bank_idx
            ].to(layer_indices.device)
            return layer_indices

        for bank_idx in range(local_bank_ids.shape[0]):
            bank_logical_ids = local_shadow_logical_ids[layer_idx, bank_idx]
            if override_bank_idx is not None and bank_idx == override_bank_idx:
                assert override_bank_logical_ids is not None
                bank_logical_ids = override_bank_logical_ids
            layer_indices[local_bank_ids[bank_idx].long()] = bank_logical_ids.to(
                layer_indices.device
            )
        return layer_indices

    def _sync_hybrid_local_shadow_metadata(self, model_state: EplbModelState) -> None:
        local_shadow_logical_ids = model_state.peer_cache_local_dynamic_logical_ids
        global_bank_ids = model_state.peer_cache_dynamic_bank_physical_ids
        if (
            local_shadow_logical_ids is None
            or model_state.peer_cache_local_dynamic_bank_physical_ids is None
            or global_bank_ids is None
            or local_shadow_logical_ids.numel() == 0
        ):
            return

        ep_group = get_ep_group().device_group
        ep_rank = ep_group.rank()
        ep_size = ep_group.size()
        num_local_physical_experts = model_state.model.num_local_physical_experts
        gathered = [
            torch.empty_like(local_shadow_logical_ids) for _ in range(ep_size)
        ]
        torch.distributed.all_gather(gathered, local_shadow_logical_ids, group=ep_group)
        live_map_cpu = model_state.physical_to_logical_map.cpu()
        global_bank_ids_cpu = global_bank_ids.cpu()
        synced_map = model_state.peer_cache_periodic_synced_physical_to_logical_map
        if (
            synced_map is None
            or synced_map.shape != live_map_cpu.shape
            or synced_map.device.type != "cpu"
        ):
            synced_map = live_map_cpu.clone()
            model_state.peer_cache_periodic_synced_physical_to_logical_map = synced_map
        else:
            synced_map.copy_(live_map_cpu)
        for rank_idx, rank_local_shadow in enumerate(gathered):
            rank_local_shadow_cpu = rank_local_shadow.cpu()
            rank_begin = rank_idx * num_local_physical_experts
            rank_end = rank_begin + num_local_physical_experts
            for bank_idx in range(global_bank_ids_cpu.shape[0]):
                rank_bank_ids = global_bank_ids_cpu[bank_idx]
                rank_bank_ids = rank_bank_ids[
                    (rank_bank_ids >= rank_begin) & (rank_bank_ids < rank_end)
                ].long()
                if rank_bank_ids.numel() == 0:
                    continue
                synced_map[:, rank_bank_ids] = rank_local_shadow_cpu[:, bank_idx]
        if ep_rank == 0:
            logger.debug(
                "Synchronized local hybrid shadow metadata for model %s",
                model_state.model_name,
            )

    def _build_effective_physical_to_logical_map(
        self,
        model_state: EplbModelState,
        *,
        active_bank_only: bool,
    ) -> torch.Tensor:
        if (
            active_bank_only
            and model_state.peer_cache_active_physical_to_logical_map is not None
        ):
            return model_state.peer_cache_active_physical_to_logical_map
        return torch.stack(
            [
                self._patch_local_shadow_into_layer_indices(
                    model_state=model_state,
                    layer_idx=layer_idx,
                    active_bank_only=active_bank_only,
                )
                for layer_idx in range(model_state.model.num_moe_layers)
            ],
            dim=0,
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
        del active_dynamic_bank_idx

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
            (
                layer_logical_to_physical,
                layer_logical_replica_count,
            ) = self.build_logical_mapping_from_physical_subset(
                physical_to_logical_map=physical_to_logical_map[layer_idx].long(),
                included_physical_ids=base_physical_ids,
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

        source_physical_to_logical_map = self._get_hybrid_periodic_source_map(
            model_state
        )
        new_physical_to_logical_map = source_physical_to_logical_map.clone()
        static_ids = model_state.peer_cache_static_physical_ids.long()
        static_ids_for_source_map = static_ids.to(source_physical_to_logical_map.device)
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
            source_physical_to_logical_map[:, static_ids_for_source_map]
            .cpu()
            .numpy()
            .astype(np.int64)
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
        local_predicted_load: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        assert model_state.peer_cache_local_dynamic_bank_physical_ids is not None
        assert model_state.peer_cache_local_dynamic_logical_ids is not None
        assert model_state.peer_cache_active_dynamic_bank_idx is not None

        new_local_dynamic_logical_ids = (
            model_state.peer_cache_local_dynamic_logical_ids.clone()
        )
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

        local_bank_ids = model_state.peer_cache_local_dynamic_bank_physical_ids.long()
        if local_bank_ids.numel() == 0 or not torch.count_nonzero(local_predicted_load):
            return (
                new_local_dynamic_logical_ids,
                new_active_dynamic_bank_idx,
                layer_swap_mask,
                layer_changed_slots,
            )

        num_logical_experts = model_state.model.num_logical_experts
        ep_rank = get_ep_group().device_group.rank()
        num_local_physical_experts = model_state.model.num_local_physical_experts
        local_begin = ep_rank * num_local_physical_experts
        local_end = local_begin + num_local_physical_experts
        local_bank_row_offsets = (local_bank_ids - local_begin).long()

        for layer_idx in range(model_state.model.num_moe_layers):
            local_source_logicals = model_state.physical_to_logical_map[
                layer_idx, local_begin:local_end
            ].clone()
            for bank_idx in range(local_bank_row_offsets.shape[0]):
                if local_bank_row_offsets[bank_idx].numel() == 0:
                    continue
                local_source_logicals[local_bank_row_offsets[bank_idx]] = (
                    model_state.peer_cache_local_dynamic_logical_ids[
                        layer_idx, bank_idx
                    ].long()
                )

            sourceable_logicals = torch.zeros(
                (num_logical_experts,),
                dtype=torch.bool,
                device=self.device,
            )
            valid_source_logicals = local_source_logicals[local_source_logicals >= 0].long()
            if valid_source_logicals.numel() > 0:
                sourceable_logicals[valid_source_logicals] = True

            dynamic_weight = local_predicted_load[layer_idx].masked_fill(
                ~sourceable_logicals, 0
            )
            if not torch.count_nonzero(dynamic_weight):
                continue

            desired_dynamic_ids = self._select_local_shadow_logical_ids(
                dynamic_weight=dynamic_weight,
                num_slots=local_bank_ids.shape[1],
            )
            if desired_dynamic_ids.numel() == 0:
                continue

            active_bank = int(model_state.peer_cache_active_dynamic_bank_idx[layer_idx].item())
            standby_bank = 1 - active_bank
            active_current = model_state.peer_cache_local_dynamic_logical_ids[
                layer_idx, active_bank
            ].long()
            active_assigned = self._assign_shadow_slots_stably(
                desired_logical_ids=desired_dynamic_ids,
                current_logical_ids=active_current,
            )
            if torch.equal(active_assigned, active_current):
                continue

            standby_current = model_state.peer_cache_local_dynamic_logical_ids[
                layer_idx, standby_bank
            ].long()
            standby_assigned = self._assign_shadow_slots_stably(
                desired_logical_ids=desired_dynamic_ids,
                current_logical_ids=standby_current,
            )
            changed_slots = int(
                torch.count_nonzero(standby_assigned != standby_current).item()
            )
            if changed_slots > 0:
                new_local_dynamic_logical_ids[layer_idx, standby_bank] = (
                    standby_assigned.to(new_local_dynamic_logical_ids.device)
                )
            layer_changed_slots[layer_idx] = changed_slots
            layer_swap_mask[layer_idx] = True
            new_active_dynamic_bank_idx[layer_idx] = standby_bank

        return (
            new_local_dynamic_logical_ids,
            new_active_dynamic_bank_idx,
            layer_swap_mask,
            layer_changed_slots,
        )

    def _launch_async_fast_refresh(
        self,
        model_state: EplbModelState,
        layer_idx: int,
        new_physical_to_logical_layer: torch.Tensor,
        new_logical_to_physical_layer: torch.Tensor,
        new_logical_replica_count_layer: torch.Tensor,
        new_active_dynamic_bank_idx: int | None = None,
        updates_global_mapping: bool = True,
        new_local_dynamic_logical_ids: torch.Tensor | None = None,
        local_only: bool = False,
    ) -> bool:
        if model_state.fast_refresh_inflight:
            if self.device.type == "cuda":
                ep_group = get_ep_group().device_group
                current_stream = torch.cuda.current_stream(self.device)
                self._consume_fast_refresh_for_model(
                    model_state, current_stream, ep_group
                )
            else:
                return False
            if model_state.fast_refresh_inflight:
                return False
        if self.device.type != "cuda":
            return False

        device_index = model_state.cuda_device_index or self.cuda_device_index
        if device_index is None:
            return False

        ready_event = torch.cuda.Event(blocking=False)
        model_state.fast_refresh_inflight = True
        model_state.fast_refresh_thread = None
        model_state.fast_refresh_ready_event = ready_event
        model_state.fast_refresh_error = None
        model_state.fast_refresh_layer_idx = layer_idx
        model_state.fast_refresh_new_physical_to_logical_layer = (
            new_physical_to_logical_layer.detach().clone()
        )
        model_state.fast_refresh_new_logical_to_physical_layer = (
            new_logical_to_physical_layer.detach().clone()
        )
        model_state.fast_refresh_new_logical_replica_count_layer = (
            new_logical_replica_count_layer.detach().clone()
        )
        model_state.fast_refresh_new_active_dynamic_bank_idx = (
            new_active_dynamic_bank_idx
        )
        model_state.fast_refresh_updates_global_mapping = updates_global_mapping
        model_state.fast_refresh_new_local_dynamic_logical_ids = (
            new_local_dynamic_logical_ids.detach().clone()
            if new_local_dynamic_logical_ids is not None
            else None
        )
        model_state.fast_refresh_is_unchanged = None
        model_state.fast_refresh_is_received_locally = None
        model_state.fast_refresh_recv_metadata = None

        if updates_global_mapping:
            old_layer_tensor = model_state.physical_to_logical_map[layer_idx]
        else:
            old_layer_tensor = self._patch_local_shadow_into_layer_indices(
                model_state=model_state,
                layer_idx=layer_idx,
            )
        old_layer_indices = old_layer_tensor.cpu().numpy().astype(np.int64)
        new_layer_indices = (
            model_state.fast_refresh_new_physical_to_logical_layer.cpu()
            .numpy()
            .astype(np.int64)
        )

        def _worker() -> None:
            torch.accelerator.set_device_index(device_index)
            cuda_stream = torch.cuda.Stream(device=device_index)
            try:
                with model_state.buffer_lock:
                    if model_state.fast_refresh_consumed_event is not None:
                        cuda_stream.wait_event(model_state.fast_refresh_consumed_event)
                    with torch.cuda.stream(cuda_stream):
                        if local_only:
                            (
                                model_state.fast_refresh_is_unchanged,
                                model_state.fast_refresh_is_received_locally,
                                model_state.fast_refresh_recv_metadata,
                            ) = move_to_buffer_local_only(
                                num_local_experts=(
                                    model_state.model.num_local_physical_experts
                                ),
                                old_indices=old_layer_indices,
                                new_indices=new_layer_indices,
                                expert_weights=(
                                    model_state.model.expert_weights[layer_idx]
                                ),
                                expert_weights_buffers=model_state.expert_buffer,
                            )
                        else:
                            (
                                model_state.fast_refresh_is_unchanged,
                                model_state.fast_refresh_is_received_locally,
                                model_state.fast_refresh_recv_metadata,
                            ) = move_to_buffer(
                                num_local_experts=(
                                    model_state.model.num_local_physical_experts
                                ),
                                old_indices=old_layer_indices,
                                new_indices=new_layer_indices,
                                expert_weights=(
                                    model_state.model.expert_weights[layer_idx]
                                ),
                                expert_weights_buffers=model_state.expert_buffer,
                                cuda_stream=cuda_stream,
                                ep_group=get_ep_group().device_group,
                            )
                    cuda_stream.record_event(ready_event)
            except Exception as exc:  # pragma: no cover - diagnostic path
                model_state.fast_refresh_error = (
                    "decode-time async fast refresh failed for "
                    f"{model_state.model_name} layer {layer_idx}: {exc}"
                )
                torch.cuda.current_stream(device=device_index).record_event(ready_event)

        thread = threading.Thread(target=_worker, daemon=True)
        model_state.fast_refresh_thread = thread
        thread.start()
        return True

    def _allreduce_expert_load_window(
        self, tensor: torch.Tensor, eplb_algorithm: str
    ) -> None:
        """In-place all-reduce for rearrange() load tensors."""
        ep_group = get_ep_group().device_group
        if ep_group.size() <= 1:
            return
        if eplb_algorithm in _EPLB_ALGORITHMS_REPLICATED_LOGICAL_LOAD:
            all_reduce(tensor, op=ReduceOp.AVG, group=ep_group)
        else:
            all_reduce(tensor, group=ep_group)

    def _allreduce_fgate_prediction_tensors(
        self, tensor_list: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """
        In-place average across EP ranks. Each rank holds the same fgate
        accumulation for replicated batches; SUM would inflate load by world size.
        """
        ep_group = get_ep_group().device_group
        if ep_group.size() <= 1:
            return tensor_list
        for t in tensor_list:
            all_reduce(t, op=ReduceOp.AVG, group=ep_group)
        return tensor_list

    def _consume_fast_refresh_for_model(
        self,
        model_state: EplbModelState,
        current_stream: torch.cuda.Stream,
        ep_group: ProcessGroup,
    ) -> None:
        if not model_state.fast_refresh_inflight:
            return

        assert model_state.fast_refresh_ready_event is not None
        current_stream.wait_event(model_state.fast_refresh_ready_event)
        if model_state.fast_refresh_thread is not None:
            model_state.fast_refresh_thread.join()
            model_state.fast_refresh_thread = None

        if model_state.fast_refresh_error is not None:
            error = model_state.fast_refresh_error
            model_state.fast_refresh_inflight = False
            model_state.fast_refresh_error = None
            raise RuntimeError(error)

        layer_idx = model_state.fast_refresh_layer_idx
        if layer_idx < 0:
            model_state.fast_refresh_inflight = False
            model_state.fast_refresh_ready_event = None
            model_state.fast_refresh_layer_idx = -1
            model_state.fast_refresh_new_physical_to_logical_layer = None
            model_state.fast_refresh_new_logical_to_physical_layer = None
            model_state.fast_refresh_new_logical_replica_count_layer = None
            model_state.fast_refresh_new_active_dynamic_bank_idx = None
            model_state.fast_refresh_updates_global_mapping = True
            model_state.fast_refresh_new_local_dynamic_logical_ids = None
            model_state.fast_refresh_is_unchanged = None
            model_state.fast_refresh_is_received_locally = None
            model_state.fast_refresh_recv_metadata = None
            return

        assert model_state.fast_refresh_is_unchanged is not None
        assert model_state.fast_refresh_is_received_locally is not None
        assert model_state.fast_refresh_recv_metadata is not None
        assert model_state.fast_refresh_new_physical_to_logical_layer is not None
        assert model_state.fast_refresh_new_logical_to_physical_layer is not None
        assert model_state.fast_refresh_new_logical_replica_count_layer is not None

        move_from_buffer(
            expert_weights=model_state.model.expert_weights[layer_idx],
            expert_weights_buffers=model_state.expert_buffer,
            is_unchanged=model_state.fast_refresh_is_unchanged,
            is_received_locally=model_state.fast_refresh_is_received_locally,
            recv_metadata=model_state.fast_refresh_recv_metadata,
            new_indices=(
                model_state.fast_refresh_new_physical_to_logical_layer.cpu().numpy()
            ),
            ep_rank=ep_group.rank(),
        )

        consumed_event = torch.cuda.Event(blocking=False)
        consumed_event.record(current_stream)
        model_state.fast_refresh_consumed_event = consumed_event
        if model_state.fast_refresh_updates_global_mapping:
            model_state.physical_to_logical_map[layer_idx].copy_(
                model_state.fast_refresh_new_physical_to_logical_layer
            )
            model_state.logical_to_physical_map[layer_idx].copy_(
                model_state.fast_refresh_new_logical_to_physical_layer
            )
            model_state.logical_replica_count[layer_idx].copy_(
                model_state.fast_refresh_new_logical_replica_count_layer
            )
        elif (
            model_state.peer_cache_local_dynamic_logical_ids is not None
            and model_state.fast_refresh_new_local_dynamic_logical_ids is not None
            and model_state.fast_refresh_new_active_dynamic_bank_idx is not None
        ):
            model_state.peer_cache_local_dynamic_logical_ids[
                layer_idx, model_state.fast_refresh_new_active_dynamic_bank_idx
            ] = model_state.fast_refresh_new_local_dynamic_logical_ids.to(
                model_state.peer_cache_local_dynamic_logical_ids.device
            )
        if (
            model_state.peer_cache_active_dynamic_bank_idx is not None
            and model_state.fast_refresh_new_active_dynamic_bank_idx is not None
        ):
            model_state.peer_cache_active_dynamic_bank_idx[layer_idx] = (
                model_state.fast_refresh_new_active_dynamic_bank_idx
            )
            self._sync_layer_active_shadow_view(model_state, layer_idx)
            self._refresh_active_shadow_physical_to_logical_map(
                model_state, layer_idx=layer_idx
            )

        model_state.fast_refresh_inflight = False
        model_state.fast_refresh_ready_event = None
        model_state.fast_refresh_layer_idx = -1
        model_state.fast_refresh_new_physical_to_logical_layer = None
        model_state.fast_refresh_new_logical_to_physical_layer = None
        model_state.fast_refresh_new_logical_replica_count_layer = None
        model_state.fast_refresh_new_active_dynamic_bank_idx = None
        model_state.fast_refresh_updates_global_mapping = True
        model_state.fast_refresh_new_local_dynamic_logical_ids = None
        model_state.fast_refresh_is_unchanged = None
        model_state.fast_refresh_is_received_locally = None
        model_state.fast_refresh_recv_metadata = None

    def prepare_for_forward(self) -> None:
        if self.device.type != "cuda":
            return
        if torch.cuda.is_current_stream_capturing():
            if any(
                model_state.fast_refresh_inflight
                for model_state in self.model_states.values()
            ):
                raise RuntimeError(
                    "decode-time peer-cache refresh is still pending during CUDA "
                    "graph capture; this indicates skip_eplb was bypassed."
                )
            return

        ep_group = get_ep_group().device_group
        current_stream = torch.cuda.current_stream(self.device)
        for model_state in self.model_states.values():
            self._consume_fast_refresh_for_model(
                model_state, current_stream, ep_group
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

        global_predicted_loads = self._allreduce_fgate_prediction_tensors(
            predicted_loads
        )
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
            if changed_layers.numel() == 0:
                continue
            selected_layer = int(changed_layers[0].item())
            self._launch_async_fast_refresh(
                model_state=model_state,
                layer_idx=selected_layer,
                new_physical_to_logical_layer=new_physical_to_logical_map[selected_layer],
                new_logical_to_physical_layer=new_logical_to_physical_map[selected_layer],
                new_logical_replica_count_layer=new_logical_replica_count[selected_layer],
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
            local_ready = self.expert_rearrangement_step >= (
                self.expert_rearrangement_step_interval
            )
            if self._hybrid_periodic_rearrange_ready(local_ready):
                if self.parallel_config.data_parallel_size > 1 and (
                    not self.parallel_config.eplb_config.hybrid_periodic_rearrange_with_multi_dp
                ):
                    self.expert_rearrangement_step = 0
                    return
                if any(
                    model_state.fast_refresh_inflight
                    for model_state in self.model_states.values()
                ):
                    return
                self.expert_rearrangement_step = 0
                self.rearrange()
            return

        for model_state in self.model_states.values():
            assert model_state.expert_load_logical_ema is not None
            effective_physical_to_logical_map = (
                model_state.peer_cache_active_physical_to_logical_map
            )
            if effective_physical_to_logical_map is None:
                effective_physical_to_logical_map = (
                    self._build_effective_physical_to_logical_map(
                        model_state=model_state,
                        active_bank_only=True,
                    )
                )
            logical_load = self._logical_expert_load_from_physical(
                physical_load=model_state.expert_load_pass,
                physical_to_logical_map=effective_physical_to_logical_map,
                num_logical_experts=model_state.model.num_logical_experts,
            )
            alpha = model_state.eplb_ema_alpha
            model_state.expert_load_logical_ema.mul_(1.0 - alpha).add_(
                logical_load, alpha=alpha
            )
            assert model_state.expert_load_fgate is not None
            model_state.expert_load_fgate.zero_()
            model_state.expert_load_pass.zero_()

        self.expert_rearrangement_step += 1
        local_ready = (
            self.expert_rearrangement_step
            >= self.expert_rearrangement_step_interval
        )
        if self._hybrid_periodic_rearrange_ready(local_ready):
            if self.parallel_config.data_parallel_size > 1 and (
                not self.parallel_config.eplb_config.hybrid_periodic_rearrange_with_multi_dp
            ):
                # Skip periodic rearrange under multi-DP unless explicitly enabled
                # (can stall; DP workers may be out of sync).
                self.expert_rearrangement_step = 0
                return
            if any(
                model_state.fast_refresh_inflight
                for model_state in self.model_states.values()
            ):
                return
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

        peer_cache_dynamic_physical_ids = None
        peer_cache_static_physical_to_logical_map = None
        peer_cache_home_rank = None
        peer_cache_primary_physical_to_logical_map = None
        peer_cache_static_physical_ids = None
        peer_cache_dynamic_bank_physical_ids = None
        peer_cache_active_dynamic_bank_idx = None
        peer_cache_local_dynamic_bank_physical_ids = None
        peer_cache_local_dynamic_logical_ids = None
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
                if (
                    self.parallel_config.eplb_config.hybrid_periodic_rearrange_with_multi_dp
                ):
                    logger.warning(
                        "fgate-hybrid-cache layer-level immediate refresh is disabled "
                        "for data_parallel_size=%d while periodic global static "
                        "refresh is enabled; decode runs in periodic-only mode so "
                        "all ranks share the same post-DP-synced rearrange counter.",
                        self.parallel_config.data_parallel_size,
                    )
                else:
                    logger.warning(
                        "fgate-hybrid-cache immediate local shadow refresh is enabled "
                        "for data_parallel_size=%d, but periodic global static "
                        "refresh is disabled to avoid long expert-migration stalls "
                        "during online serving.",
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
            local_begin = ep_group.rank() * num_local_physical_experts
            local_end = local_begin + num_local_physical_experts
            peer_cache_local_dynamic_bank_physical_ids = torch.stack(
                [
                    bank_ids[
                        (bank_ids >= local_begin) & (bank_ids < local_end)
                    ].long()
                    for bank_ids in peer_cache_dynamic_bank_physical_ids
                ]
            ).to(self.device)
            peer_cache_local_dynamic_logical_ids = torch.stack(
                [
                    physical_to_logical_map[:, bank_ids.long()]
                    for bank_ids in peer_cache_local_dynamic_bank_physical_ids
                ],
                dim=1,
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
                (
                    layer_logical_to_physical,
                    layer_logical_replica_count,
                ) = EplbState.build_logical_mapping_from_physical_subset(
                    physical_to_logical_map=physical_to_logical_map[layer_idx].long(),
                    included_physical_ids=hybrid_base_physical_ids,
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

        model_expert_load_fgate = expert_load_fgate
        if (
            eplb_algorithm == "fgate-hybrid-cache"
            and not self._hybrid_enable_immediate_layer_refresh()
        ):
            model_expert_load_fgate = None

        model.set_eplb_state(
            expert_load_pass,
            logical_to_physical_map,
            logical_replica_count,
            expert_load_fgate=model_expert_load_fgate,
            fgate_skip_prefill=(
                eplb_algorithm in (
                    "fgate-v2",
                    "fgate-peer-cache",
                )
            ),
        )
        if not model.expert_weights or not model.expert_weights[0]:
            raise RuntimeError(
                "EPLB expected MoE expert weights to be collected after "
                "set_eplb_state(), but model.expert_weights is empty."
            )
        expert_buffer = [torch.empty_like(w) for w in model.expert_weights[0]]

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
            peer_cache_local_dynamic_bank_physical_ids=peer_cache_local_dynamic_bank_physical_ids,
            peer_cache_local_dynamic_logical_ids=peer_cache_local_dynamic_logical_ids,
            peer_cache_active_physical_to_logical_map=None,
            peer_cache_periodic_synced_physical_to_logical_map=(
                physical_to_logical_map.cpu().clone()
                if eplb_algorithm == "fgate-hybrid-cache"
                else None
            ),
        )
        self.model_states[model_config.compute_hash()] = model_state
        self.num_valid_physical_experts = model.num_physical_experts
        self._sync_model_layer_shadow_views(model_state)
        self._refresh_active_shadow_physical_to_logical_map(model_state)
        self._install_hybrid_layer_runtime_hooks(model_state)

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
            self.model_states
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
            if eplb_model_state.eplb_algorithm == "fgate-hybrid-cache":
                self._sync_hybrid_local_shadow_metadata(eplb_model_state)
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
        # Perform all-reduce to get the expert load across all ranks for each model.
        # Replicated fgate / hybrid-EMA tensors use AVG; partial per-rank views use SUM.
        model_states_ordered = list(self.model_states.values())
        for eplb_model_state, gw in zip(
            model_states_ordered, global_expert_load_windows
        ):
            self._allreduce_expert_load_window(gw, eplb_model_state.eplb_algorithm)

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
                    current_rearrange_source_map = self._get_hybrid_periodic_source_map(
                        eplb_model_state
                    )
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
                        static_ids_for_source_map = static_ids.to(
                            current_rearrange_source_map.device
                        )
                        changed_slots = (
                            current_rearrange_source_map[:, static_ids_for_source_map]
                            != new_physical_to_logical_map[:, static_ids_for_source_map]
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
                    (
                        current_rearrange_source_map
                        if eplb_model_state.eplb_algorithm == "fgate-hybrid-cache"
                        else eplb_model_state.physical_to_logical_map
                    ),
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
                    if eplb_model_state.eplb_algorithm == "fgate-hybrid-cache":
                        synced_map = (
                            eplb_model_state.peer_cache_periodic_synced_physical_to_logical_map
                        )
                        if (
                            synced_map is None
                            or synced_map.shape != new_physical_to_logical_map.shape
                            or synced_map.device.type != "cpu"
                        ):
                            eplb_model_state.peer_cache_periodic_synced_physical_to_logical_map = (
                                new_physical_to_logical_map.cpu().clone()
                            )
                        else:
                            synced_map.copy_(new_physical_to_logical_map.cpu())
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
                    if eplb_model_state.eplb_algorithm == "fgate-hybrid-cache":
                        self._refresh_active_shadow_physical_to_logical_map(
                            eplb_model_state
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
        if self._should_barrier_after_hybrid_periodic_rearrange():
            get_ep_group().barrier()
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
        if model_state.eplb_algorithm == "fgate-hybrid-cache":
            synced_map = model_state.peer_cache_periodic_synced_physical_to_logical_map
            if (
                synced_map is None
                or synced_map.shape != model_state.physical_to_logical_map.shape
                or synced_map.device.type != "cpu"
            ):
                model_state.peer_cache_periodic_synced_physical_to_logical_map = (
                    model_state.physical_to_logical_map.cpu().clone()
                )
            else:
                synced_map[layer].copy_(new_physical[layer].cpu())

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
        if model_state.eplb_algorithm == "fgate-hybrid-cache":
            self._refresh_active_shadow_physical_to_logical_map(
                model_state, layer_idx=layer
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
        if (
            eplb_model_state.eplb_algorithm == "fgate-hybrid-cache"
            and eplb_model_state.peer_cache_periodic_synced_physical_to_logical_map
            is not None
        ):
            eplb_model_state.peer_cache_periodic_synced_physical_to_logical_map.copy_(
                expanded_physical_to_logical.cpu()
            )

        if eplb_model_state.eplb_algorithm == "fgate-hybrid-cache":
            assert eplb_model_state.peer_cache_active_dynamic_bank_idx is not None
            if (
                eplb_model_state.peer_cache_local_dynamic_bank_physical_ids is not None
                and eplb_model_state.peer_cache_local_dynamic_logical_ids is not None
            ):
                for bank_idx, bank_ids in enumerate(
                    eplb_model_state.peer_cache_local_dynamic_bank_physical_ids
                ):
                    if bank_ids.numel() == 0:
                        continue
                    eplb_model_state.peer_cache_local_dynamic_logical_ids[:, bank_idx] = (
                        expanded_physical_to_logical[:, bank_ids.long()].to(device)
                    )
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
            eplb_state._sync_model_layer_shadow_views(eplb_model_state)
            eplb_state._install_hybrid_layer_runtime_hooks(eplb_model_state)
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
    physical_to_logical_map: torch.Tensor | None = None
    logical_replica_count: torch.Tensor | None = None
    expert_map: torch.Tensor | None = None
    next_gate_weight: torch.Tensor | None = None
    expert_load_fgate_view: torch.Tensor | None = None
    local_dynamic_shadow_physical_ids: torch.Tensor | None = None
    local_dynamic_shadow_logical_ids: torch.Tensor | None = None
    local_dynamic_shadow_active_bank_idx: torch.Tensor | None = None
    local_dynamic_shadow_active_physical_ids: torch.Tensor | None = None
    local_dynamic_shadow_active_logical_ids: torch.Tensor | None = None
    consume_pending_layer_refresh: Callable[[], None] | None = None
    schedule_next_layer_shadow_refresh: Callable[[torch.Tensor], None] | None = None


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
