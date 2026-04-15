import torch

from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.router import base_router as base_router_module
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter


class _DummyRouter(BaseRouter):
    @property
    def routing_method_type(self) -> RoutingMethodType:
        return RoutingMethodType.Default

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


def _make_router(prefill_ignore_redundant: bool) -> _DummyRouter:
    eplb_state = EplbLayerState(
        expert_load_view=torch.zeros(32, dtype=torch.int32),
        logical_to_physical_map=torch.tensor(
            [[0, 4], [1, 5], [2, 6], [3, 7]], dtype=torch.long
        ),
        logical_replica_count=torch.tensor([2, 2, 2, 2], dtype=torch.long),
        physical_to_logical_map=torch.arange(8, dtype=torch.long) % 4,
        local_dynamic_shadow_active_physical_ids=torch.tensor([8, 9], dtype=torch.long),
        local_dynamic_shadow_active_logical_ids=torch.tensor([0, 1], dtype=torch.long),
        prefill_ignore_redundant=prefill_ignore_redundant,
    )
    return _DummyRouter(
        top_k=2,
        global_num_experts=4,
        eplb_state=eplb_state,
        enable_eplb=True,
    )


def test_prefill_ignore_redundant_uses_primary_only_and_skips_shadow(
    monkeypatch,
):
    router = _make_router(prefill_ignore_redundant=True)
    logical_to_physical = router.eplb_state.logical_to_physical_map.clone()
    replica_count = router.eplb_state.logical_replica_count.clone()
    topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)

    def fake_map(
        topk_ids: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> torch.Tensor:
        assert logical_to_physical_map.shape == (4, 1)
        assert torch.equal(logical_to_physical_map[:, 0], logical_to_physical[:, 0])
        assert torch.equal(logical_replica_count, torch.ones_like(replica_count))
        return topk_ids + 10

    monkeypatch.setattr(
        base_router_module, "is_forward_context_prefill_batch", lambda default=False: True
    )
    monkeypatch.setattr(base_router_module, "eplb_map_to_physical", fake_map)
    monkeypatch.setattr(
        base_router_module,
        "eplb_apply_local_dynamic_shadow_mapping",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("prefill should skip local dynamic shadow mapping")
        ),
    )
    monkeypatch.setattr(
        base_router_module,
        "eplb_record_physical_expert_load",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("prefill should skip expert load bookkeeping")
        ),
    )

    out = router._apply_eplb_mapping(topk_ids)

    assert torch.equal(out, topk_ids + 10)


def test_decode_path_keeps_runtime_map_and_shadow_behavior(monkeypatch):
    router = _make_router(prefill_ignore_redundant=True)
    logical_to_physical = router.eplb_state.logical_to_physical_map
    replica_count = router.eplb_state.logical_replica_count
    topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    calls = {"shadow": 0, "record": 0}

    def fake_map(
        topk_ids: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> torch.Tensor:
        assert torch.equal(logical_to_physical_map, logical_to_physical)
        assert torch.equal(logical_replica_count, replica_count)
        return topk_ids + 10

    def fake_shadow(
        topk_ids: torch.Tensor,
        active_shadow_ids: torch.Tensor,
        active_shadow_logicals: torch.Tensor,
        physical_to_logical_map: torch.Tensor,
        local_token_mask: torch.Tensor,
        local_shadow_mask: torch.Tensor,
    ) -> torch.Tensor:
        calls["shadow"] += 1
        return topk_ids + 20

    def fake_record(topk_ids: torch.Tensor, expert_load_view: torch.Tensor) -> torch.Tensor:
        calls["record"] += 1
        return topk_ids

    monkeypatch.setattr(
        base_router_module, "is_forward_context_prefill_batch", lambda default=False: False
    )
    monkeypatch.setattr(base_router_module, "eplb_map_to_physical", fake_map)
    monkeypatch.setattr(
        base_router_module, "eplb_apply_local_dynamic_shadow_mapping", fake_shadow
    )
    monkeypatch.setattr(base_router_module, "eplb_record_physical_expert_load", fake_record)

    out = router._apply_eplb_mapping(topk_ids)

    assert calls == {"shadow": 1, "record": 1}
    assert torch.equal(out, topk_ids + 30)
