# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import abstractmethod
from collections.abc import Callable

import torch

from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.forward_context import is_forward_context_prefill_batch
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.platforms import current_platform

if current_platform.is_cuda_alike():

    @torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
    def eplb_map_to_physical(
        topk_ids: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> torch.Tensor:
        """
        Map the logical expert ids to physical expert ids
        and record the expert load metrics.

        This will select a pseudo-random replica for each logical expert.
        Only used for EPLB.

        Args:
            topk_ids: The logical expert ids.
            expert_load_view: The expert load view.
            logical_to_physical_map: The logical to physical map.
            logical_replica_count: The logical replica count.

        Returns:
            The physical expert ids.
        """

        # 1. Convert the logical expert ids to physical expert ids
        # Directly select a random replica for each logical expert

        # In case `indices_type` is not `torch.long` or `torch.int`,
        # e.g. `torch.uint32` as required by dispatch/combine kernels
        topk_ids_long = topk_ids.long()
        # Use (token position) modulo (replica count)
        # to deterministically choose a replica
        replica_count = logical_replica_count[topk_ids_long]
        # Flatten-position based index, reshaped back to `topk_ids` shape
        pos_indices = torch.arange(
            topk_ids.numel(), device=topk_ids.device, dtype=torch.long
        ).reshape_as(topk_ids)
        # Compute pseudo-random indices by modulo
        replica_indices = (pos_indices % replica_count).unsqueeze(-1)
        physical_ids = (
            logical_to_physical_map[topk_ids_long]
            .gather(-1, replica_indices)
            .squeeze(-1)
        )

        topk_ids = physical_ids

        return topk_ids

    @torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
    def eplb_record_physical_expert_load(
        topk_ids: torch.Tensor,
        expert_load_view: torch.Tensor,
    ) -> torch.Tensor:
        # Record expert load metrics after any local expert-id rewrites.

        # TODO(bowen): When using `FusedMoEModularKernel`, this
        # can be done in a more unified way, since
        # `FusedMoEPrepareAndFinalizeModular` will return the expert
        # token count, in some cases directly from the kernel.
        # However, now there are many code paths not using
        # the modular kernel, e.g. calling `fused_experts`,
        # so we decide to keep the logic here.
        #
        # If later refactor moved all the MoE kernel calls
        # to the modular kernel, we can move this logic there
        # to achieve better efficiency.

        # `expert_load_view`: (num_physical_experts,)

        # `torch.bincount` is not compilable, so use `scatter_add_` instead.
        topk_ids_flatten = topk_ids.flatten()
        expert_load_view.scatter_add_(
            dim=0,
            index=topk_ids_flatten.long(),
            src=torch.ones_like(topk_ids_flatten).to(expert_load_view),
        )
        return topk_ids

    @torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
    def eplb_apply_local_dynamic_shadow_mapping(
        topk_ids: torch.Tensor,
        active_shadow_ids: torch.Tensor,
        active_shadow_logicals: torch.Tensor,
        physical_to_logical_map: torch.Tensor,
        local_token_mask: torch.Tensor,
        local_shadow_mask: torch.Tensor,
    ) -> torch.Tensor:
        current_logical_ids = physical_to_logical_map[topk_ids.long()]
        shadow_match = (
            local_token_mask.unsqueeze(-1)
            & local_shadow_mask.view(1, 1, -1)
            & (current_logical_ids.unsqueeze(-1) == active_shadow_logicals.view(1, 1, -1))
        )
        shadow_count = shadow_match.sum(dim=-1)
        pos_indices = torch.arange(
            topk_ids.numel(), device=topk_ids.device, dtype=torch.long
        ).reshape_as(topk_ids)
        route_mod = torch.remainder(pos_indices, shadow_count.long() + 1)
        use_shadow = shadow_count > 0
        use_shadow &= local_token_mask
        use_shadow &= route_mod > 0

        match_rank = torch.cumsum(shadow_match.to(torch.int32), dim=-1) - 1
        selector = (route_mod - 1).clamp_min(0).unsqueeze(-1)
        selected_shadow_ids = torch.where(
            shadow_match & (match_rank == selector),
            active_shadow_ids.view(1, 1, -1),
            -1,
        ).amax(dim=-1)
        return torch.where(use_shadow, selected_shadow_ids, topk_ids.long())
else:

    def eplb_map_to_physical(
        topk_ids: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> torch.Tensor:
        # CPU fallback: no EPLB so just return as is
        return topk_ids

    def eplb_record_physical_expert_load(
        topk_ids: torch.Tensor,
        expert_load_view: torch.Tensor,
    ) -> torch.Tensor:
        return topk_ids

    def eplb_apply_local_dynamic_shadow_mapping(
        topk_ids: torch.Tensor,
        active_shadow_ids: torch.Tensor,
        active_shadow_logicals: torch.Tensor,
        physical_to_logical_map: torch.Tensor,
        local_token_mask: torch.Tensor,
        local_shadow_mask: torch.Tensor,
    ) -> torch.Tensor:
        return topk_ids


class BaseRouter(FusedMoERouter):
    """
    Base router class that provides common functionality for all router implementations.

    This class implements the template method pattern where select_experts() handles
    common pre-processing and post-processing, delegating the actual routing logic
    to the abstract _compute_routing() method.
    """

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        eplb_state: EplbLayerState,
        enable_eplb: bool = False,
        # TODO(bnell): Once the MK is constructed at layer init time, we
        # can make this a plain value instead of a callback.
        indices_type_getter: Callable[[], torch.dtype | None] | None = None,
    ):
        """
        Note: the indices dtype might not be available at router construction
        time, so we need to supply a callback to get it at runtime.  This is
        because the indices type is supplied by modular kernels which are
        created after MoE layer/router construction.
        """
        super().__init__()
        self.top_k = top_k
        self.global_num_experts = global_num_experts
        self.eplb_state = eplb_state
        self.enable_eplb = enable_eplb
        self.indices_type_getter = indices_type_getter
        self.capture_fn: Callable[[torch.Tensor], None] | None = None

    def set_capture_fn(self, capture_fn: Callable[[torch.Tensor], None] | None) -> None:
        """Set a capture callback for logical routed expert IDs."""
        self.capture_fn = capture_fn

    def _validate_eplb_state(self) -> None:
        """Validate that EPLB state is properly initialized if EPLB is enabled."""
        if self.enable_eplb:
            if self.eplb_state.expert_load_view is None:
                raise ValueError("enable_eplb=True requires expert_load_view != None")
            if self.eplb_state.logical_to_physical_map is None:
                raise ValueError(
                    "enable_eplb=True requires logical_to_physical_map != None"
                )
            if self.eplb_state.logical_replica_count is None:
                raise ValueError(
                    "enable_eplb=True requires logical_replica_count != None"
                )

    def _get_indices_type(self) -> torch.dtype | None:
        """Get the desired indices dtype from the getter function."""
        return (
            self.indices_type_getter() if self.indices_type_getter is not None else None
        )

    def _use_prefill_primary_only_mapping(self) -> bool:
        return bool(
            self.eplb_state.prefill_ignore_redundant
            and is_forward_context_prefill_batch()
        )

    def _apply_eplb_mapping(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """Apply EPLB mapping to convert logical expert IDs to physical expert IDs."""
        if self.enable_eplb:
            assert self.eplb_state.expert_load_view is not None
            assert self.eplb_state.logical_to_physical_map is not None
            assert self.eplb_state.logical_replica_count is not None
            use_prefill_primary_only = self._use_prefill_primary_only_mapping()
            logical_to_physical_map = self.eplb_state.logical_to_physical_map
            logical_replica_count = self.eplb_state.logical_replica_count
            if use_prefill_primary_only:
                logical_to_physical_map = logical_to_physical_map[..., :1]
                logical_replica_count = torch.ones_like(logical_replica_count)
            topk_ids = eplb_map_to_physical(
                topk_ids=topk_ids,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
            )
            if not use_prefill_primary_only:
                topk_ids = self._apply_local_dynamic_shadow_mapping(topk_ids)
                return eplb_record_physical_expert_load(
                    topk_ids=topk_ids,
                    expert_load_view=self.eplb_state.expert_load_view,
                )
            return topk_ids
        return topk_ids

    def _apply_local_dynamic_shadow_mapping(
        self, topk_ids: torch.Tensor
    ) -> torch.Tensor:
        active_shadow_ids = (
            self.eplb_state.local_dynamic_shadow_active_physical_ids
        )
        active_shadow_logicals = (
            self.eplb_state.local_dynamic_shadow_active_logical_ids
        )
        physical_to_logical_map = self.eplb_state.physical_to_logical_map
        if (
            active_shadow_ids is None
            or active_shadow_logicals is None
            or physical_to_logical_map is None
            or active_shadow_ids.numel() == 0
        ):
            return topk_ids

        if active_shadow_ids.numel() == 0:
            return topk_ids

        expert_map = self.eplb_state.expert_map
        if expert_map is None:
            local_token_mask = torch.ones_like(topk_ids, dtype=torch.bool)
            local_shadow_mask = torch.ones_like(
                active_shadow_ids, dtype=torch.bool, device=active_shadow_ids.device
            )
        else:
            local_token_mask = expert_map[topk_ids.long()] >= 0
            local_shadow_mask = expert_map[active_shadow_ids] >= 0

        return eplb_apply_local_dynamic_shadow_mapping(
            topk_ids=topk_ids,
            active_shadow_ids=active_shadow_ids,
            active_shadow_logicals=active_shadow_logicals,
            physical_to_logical_map=physical_to_logical_map,
            local_token_mask=local_token_mask,
            local_shadow_mask=local_shadow_mask,
        )

    def _convert_indices_dtype(
        self, topk_ids: torch.Tensor, indices_type: torch.dtype | None
    ) -> torch.Tensor:
        """Convert topk_ids to the desired dtype if needed."""
        if (indices_type is not None) and topk_ids.dtype != indices_type:
            topk_ids = topk_ids.to(dtype=indices_type)

        assert topk_ids.dtype == indices_type or indices_type is None
        return topk_ids

    @abstractmethod
    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the actual routing logic.

        This method must be implemented by subclasses to provide the specific
        routing algorithm (e.g., grouped_topk, fused_topk, custom routing, etc.).

        Args:
            hidden_states: Input hidden states
            router_logits: Router logits for expert selection
            indices_type: Desired dtype for expert indices (may be None)

        Returns:
            tuple of (topk_weights, topk_ids)
        """
        raise NotImplementedError

    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Route the input hidden states to the top-k experts based on the
        router logits.

        This method implements the template method pattern:
        1. Validates EPLB state
        2. Gets indices type
        3. Calls _compute_routing() to get topk_weights and topk_ids
        4. Applies EPLB mapping if enabled
        5. Converts indices dtype if needed

        Returns:
            (topk_weights, topk_ids)
            (tuple[torch.Tensor, torch.Tensor]):
            The weights and expert ids computation result.

            **Compatibility**: When EPLB is not enabled, the returned ids are
            equivalent to global logical ids, so should be compatible with
            plain MoE implementations without redundant experts.
        """
        # Step 1: Validate EPLB state
        self._validate_eplb_state()

        # Step 2: Get indices type.
        indices_type = self._get_indices_type()

        # Step 3: Compute routing (delegated to subclass)
        topk_weights, topk_ids = self._compute_routing(
            hidden_states, router_logits, indices_type
        )

        # Capture logical ids before EPLB mapping.
        if self.capture_fn is not None:
            self.capture_fn(topk_ids)

        # Step 4: Apply EPLB mapping
        topk_ids = self._apply_eplb_mapping(topk_ids)

        # Step 5: Convert indices dtype
        topk_ids = self._convert_indices_dtype(topk_ids, indices_type)

        return topk_weights, topk_ids
