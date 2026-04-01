from types import SimpleNamespace

import numpy as np
import torch

from vllm.config.parallel import EPLBConfig, ParallelConfig
from vllm.distributed.eplb.eplb_state import EplbState


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


def test_hybrid_peer_cache_static_slots_default_to_half():
    config = EPLBConfig(
        algorithm="fgate-hybrid-cache",
        num_redundant_experts=8,
    )

    assert config.resolved_static_redundant_experts() == 4
