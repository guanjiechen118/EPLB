from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from vllm.config.parallel import EPLBConfig, ParallelConfig
from vllm.distributed.eplb import eplb_state as eplb_state_module
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.distributed.eplb.rebalance_execute import (
    move_from_buffer,
    move_to_buffer_local_only,
)


def test_peer_cache_initial_map_balances_redundant_slots():
    num_logical_experts = 128
    num_redundant_experts = 16
    ep_size = 8
    num_local_physical_experts = (num_logical_experts + num_redundant_experts) // ep_size

    physical_to_logical_map = EplbState.build_initial_global_physical_to_logical_map(
        num_logical_experts,
        num_redundant_experts,
        ep_size=ep_size,
        balance_redundant=True,
    )
    dynamic_ids = EplbState.get_peer_cache_dynamic_physical_ids(
        num_logical_experts,
        num_logical_experts + num_redundant_experts,
        ep_size,
    ).tolist()

    assert len(physical_to_logical_map) == num_logical_experts + num_redundant_experts
    assert len(dynamic_ids) == num_redundant_experts

    for rank in range(ep_size):
        rank_slice = physical_to_logical_map[
            rank * num_local_physical_experts : (rank + 1) * num_local_physical_experts
        ]
        assert len(rank_slice) == num_local_physical_experts
        assert len(dynamic_ids[rank * 2 : (rank + 1) * 2]) == 2


def test_assign_peer_cache_dynamic_slots_preserves_existing_assignments():
    desired = np.array([5, 7, 9, 11], dtype=np.int64)
    current = np.array([7, 3, 5, 1], dtype=np.int64)
    slot_ranks = np.array([0, 1, 2, 3], dtype=np.int64)
    home_ranks = np.array([0] * 12, dtype=np.int64)

    assigned = EplbState._assign_peer_cache_dynamic_slots(
        desired_logical_ids=desired,
        current_logical_ids=current,
        slot_ranks=slot_ranks,
        home_ranks=home_ranks,
    )

    assert assigned.tolist().count(5) == 1
    assert assigned.tolist().count(7) == 1
    assert set(assigned.tolist()) == {5, 7, 9, 11}
    assert assigned[0] == 7
    assert assigned[2] == 5


def test_hybrid_peer_cache_slot_layout_splits_static_and_double_buffered_banks():
    static_ids, dynamic_bank_ids = EplbState.get_hybrid_peer_cache_physical_ids(
        num_logical_experts=4,
        num_physical_experts=10,
        ep_size=2,
        num_static_redundant_experts=2,
    )

    assert static_ids.tolist() == [2, 7]
    assert dynamic_bank_ids.tolist() == [[3, 8], [4, 9]]


def test_hybrid_peer_cache_runtime_mapping_ignores_inactive_bank():
    eplb_state = EplbState(
        parallel_config=ParallelConfig(),
        device=torch.device("cpu"),
    )
    model_state = SimpleNamespace(
        peer_cache_primary_physical_to_logical_map=torch.tensor(
            [[0, 1, 0, 2, 3, 2, 3, 1, 0, 2]], dtype=torch.long
        ),
        peer_cache_static_physical_ids=torch.tensor([2, 7], dtype=torch.long),
        peer_cache_dynamic_bank_physical_ids=torch.tensor(
            [[3, 8], [4, 9]], dtype=torch.long
        ),
        logical_to_physical_map=torch.full((1, 4, 5), -1, dtype=torch.long),
        logical_replica_count=torch.zeros((1, 4), dtype=torch.long),
        model=SimpleNamespace(num_moe_layers=1, num_logical_experts=4),
    )
    physical_to_logical_map = torch.tensor(
        [[0, 1, 0, 2, 3, 2, 3, 1, 0, 2]], dtype=torch.long
    )

    logical_to_physical_map, logical_replica_count = (
        eplb_state._build_hybrid_peer_cache_runtime_logical_mapping(
            model_state=model_state,
            physical_to_logical_map=physical_to_logical_map,
            active_dynamic_bank_idx=torch.tensor([0], dtype=torch.long),
        )
    )

    assert 4 not in logical_to_physical_map[0].tolist()[3]
    assert 9 not in logical_to_physical_map[0].tolist()[2]
    assert set(logical_to_physical_map[0, 0, :2].tolist()) == {0, 2}
    assert logical_replica_count.tolist() == [[2, 2, 1, 1]]


def test_fgate_only_runtime_mapping_excludes_all_dynamic_banks_from_global_map():
    eplb_state = EplbState(
        parallel_config=ParallelConfig(),
        device=torch.device("cpu"),
    )
    model_state = SimpleNamespace(
        peer_cache_primary_physical_to_logical_map=torch.tensor(
            [[0, 1, 0, 2, 3, 2, 3, 1, 0, 2]], dtype=torch.long
        ),
        peer_cache_static_physical_ids=torch.empty((0,), dtype=torch.long),
        peer_cache_dynamic_bank_physical_ids=torch.tensor(
            [[3, 8], [4, 9]], dtype=torch.long
        ),
        logical_to_physical_map=torch.full((1, 4, 5), -1, dtype=torch.long),
        logical_replica_count=torch.zeros((1, 4), dtype=torch.long),
        model=SimpleNamespace(num_moe_layers=1, num_logical_experts=4),
    )
    physical_to_logical_map = torch.tensor(
        [[0, 1, 0, 2, 3, 2, 3, 1, 0, 2]], dtype=torch.long
    )

    logical_to_physical_map, logical_replica_count = (
        eplb_state._build_hybrid_peer_cache_runtime_logical_mapping(
            model_state=model_state,
            physical_to_logical_map=physical_to_logical_map,
            active_dynamic_bank_idx=torch.tensor([0], dtype=torch.long),
        )
    )

    flat_ids = set(logical_to_physical_map[0].reshape(-1).tolist())
    assert 3 not in flat_ids
    assert 4 not in flat_ids
    assert 8 not in flat_ids
    assert 9 not in flat_ids
    assert set(logical_to_physical_map[0, 0, :2].tolist()) == {0, 2}
    assert logical_replica_count.tolist() == [[2, 2, 1, 1]]


def test_fgate_only_config_keeps_no_static_slots():
    config = EPLBConfig(
        algorithm="fgate-only",
        num_redundant_experts=8,
    )

    assert config.resolved_static_redundant_experts() == 0
    assert config.fgate_skip_prefill is True
    assert config.fgate_prefill_ignore_redundant is False


@pytest.mark.parametrize("algorithm", ["fgate", "fgate-v2", "fgate-peer-cache"])
def test_legacy_fgate_algorithms_are_rejected(algorithm: str):
    with pytest.raises(ValidationError):
        EPLBConfig(algorithm=algorithm, num_redundant_experts=8)


def test_hybrid_peer_cache_static_slots_default_to_half():
    config = EPLBConfig(
        algorithm="fgate-hybrid-cache",
        num_redundant_experts=8,
    )

    assert config.resolved_static_redundant_experts() == 4


def test_hybrid_ablation_flags_default_to_legacy_behavior():
    config = EPLBConfig(algorithm="fgate-hybrid-cache", num_redundant_experts=8)
    assert config.hybrid_immediate_layer_refresh is None
    assert config.hybrid_barrier_after_periodic_rearrange is None
    assert config.hybrid_skip_fgate_on_decode is False
    assert config.hybrid_skip_fgate_on_prefill is False


class _FakeDeviceGroup:
    def rank(self) -> int:
        return 0

    def size(self) -> int:
        return 1


class _FakeLayer:
    def __init__(self):
        self.eplb_state = SimpleNamespace(
            local_dynamic_shadow_active_physical_ids=None,
            local_dynamic_shadow_active_logical_ids=None,
        )
        self.next_gate_weight = None
        self.expert_load_fgate_view = None
        self.fgate_skip_prefill = False


class _FakeMoEModel:
    def __init__(self):
        self.num_routed_experts = 4
        self.num_redundant_experts = 2
        self.num_physical_experts = 6
        self.num_logical_experts = 4
        self.num_expert_groups = 1
        self.num_moe_layers = 1
        self.num_local_physical_experts = 6
        self.moe_layers = [_FakeLayer()]
        self.expert_weights: list[list[torch.Tensor]] = []
        self.set_eplb_state_calls: list[dict[str, object]] = []

    def set_eplb_state(
        self,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
        expert_load_fgate: torch.Tensor | None = None,
        enable_next_gate_prediction: bool = False,
        fgate_skip_prefill: bool = False,
        fgate_prefill_ignore_redundant: bool = False,
    ) -> None:
        self.set_eplb_state_calls.append(
            {
                "expert_load_view": expert_load_view,
                "logical_to_physical_map": logical_to_physical_map,
                "logical_replica_count": logical_replica_count,
                "expert_load_fgate": expert_load_fgate,
                "enable_next_gate_prediction": enable_next_gate_prediction,
                "fgate_skip_prefill": fgate_skip_prefill,
                "fgate_prefill_ignore_redundant": fgate_prefill_ignore_redundant,
            }
        )
        self.expert_weights = [[torch.zeros((self.num_local_physical_experts, 1))]]


def test_fgate_only_add_model_skips_dead_fgate_tensor_but_keeps_predictor(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_ep_group = SimpleNamespace(device_group=_FakeDeviceGroup())
    monkeypatch.setattr(eplb_state_module, "get_ep_group", lambda: fake_ep_group)

    parallel_config = ParallelConfig()
    parallel_config.eplb_config = EPLBConfig(
        algorithm="fgate-only",
        num_redundant_experts=2,
    )
    eplb_state = EplbState(parallel_config=parallel_config, device=torch.device("cpu"))
    model = _FakeMoEModel()
    model_config = SimpleNamespace(
        model="fake-fgate-only",
        compute_hash=lambda: "fake-fgate-only-hash",
    )

    eplb_state.add_model(model=model, model_config=model_config)

    assert len(model.set_eplb_state_calls) == 1
    call = model.set_eplb_state_calls[0]
    assert call["expert_load_fgate"] is None
    assert call["enable_next_gate_prediction"] is True
    assert call["fgate_skip_prefill"] is True
    assert call["fgate_prefill_ignore_redundant"] is False

    model_state = eplb_state.model_states["fake-fgate-only-hash"]
    assert model_state.expert_load_fgate is None


def test_fgate_only_add_model_can_enable_prefill_fgate(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_ep_group = SimpleNamespace(device_group=_FakeDeviceGroup())
    monkeypatch.setattr(eplb_state_module, "get_ep_group", lambda: fake_ep_group)

    parallel_config = ParallelConfig()
    parallel_config.eplb_config = EPLBConfig(
        algorithm="fgate-only",
        num_redundant_experts=2,
        fgate_skip_prefill=False,
    )
    eplb_state = EplbState(parallel_config=parallel_config, device=torch.device("cpu"))
    model = _FakeMoEModel()
    model_config = SimpleNamespace(
        model="fake-fgate-only-prefill",
        compute_hash=lambda: "fake-fgate-only-prefill-hash",
    )

    eplb_state.add_model(model=model, model_config=model_config)

    assert len(model.set_eplb_state_calls) == 1
    call = model.set_eplb_state_calls[0]
    assert call["expert_load_fgate"] is None
    assert call["enable_next_gate_prediction"] is True
    assert call["fgate_skip_prefill"] is False
    assert call["fgate_prefill_ignore_redundant"] is False


def test_fgate_only_add_model_can_ignore_redundant_on_prefill(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_ep_group = SimpleNamespace(device_group=_FakeDeviceGroup())
    monkeypatch.setattr(eplb_state_module, "get_ep_group", lambda: fake_ep_group)

    parallel_config = ParallelConfig()
    parallel_config.eplb_config = EPLBConfig(
        algorithm="fgate-only",
        num_redundant_experts=2,
        fgate_prefill_ignore_redundant=True,
    )
    eplb_state = EplbState(parallel_config=parallel_config, device=torch.device("cpu"))
    model = _FakeMoEModel()
    model_config = SimpleNamespace(
        model="fake-fgate-only-ignore-prefill",
        compute_hash=lambda: "fake-fgate-only-ignore-prefill-hash",
    )

    eplb_state.add_model(model=model, model_config=model_config)

    call = model.set_eplb_state_calls[0]
    assert call["fgate_skip_prefill"] is True
    assert call["fgate_prefill_ignore_redundant"] is True


def test_move_to_buffer_local_only_and_move_from_buffer_batch_duplicate_rows(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_ep_group = SimpleNamespace(device_group=_FakeDeviceGroup())
    monkeypatch.setattr(
        "vllm.distributed.eplb.rebalance_execute.get_ep_group",
        lambda: fake_ep_group,
    )

    old_indices = np.array([0, 1, 2, 0], dtype=np.int64)
    new_indices = np.array([1, 0, 2, 1], dtype=np.int64)
    expert_weight = torch.tensor(
        [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]]
    )
    original_weight = expert_weight.clone()
    expert_buffer = [torch.empty_like(expert_weight)]

    is_unchanged, is_received_locally, recv_metadata = move_to_buffer_local_only(
        num_local_experts=4,
        old_indices=old_indices,
        new_indices=new_indices,
        expert_weights=[expert_weight],
        expert_weights_buffers=expert_buffer,
    )
    move_from_buffer(
        expert_weights=[expert_weight],
        expert_weights_buffers=expert_buffer,
        is_unchanged=is_unchanged,
        is_received_locally=is_received_locally,
        recv_metadata=recv_metadata,
        new_indices=new_indices,
        ep_rank=0,
    )

    expected = torch.stack(
        (
            original_weight[1],
            original_weight[0],
            original_weight[2],
            original_weight[1],
        )
    )
    assert torch.equal(expert_weight, expected)
