import numpy as np

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

