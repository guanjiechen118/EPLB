import torch

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.models.moe_fgate_utils import (
    maybe_fused_gate_and_next_logits,
    predicted_load_from_topk,
    project_with_linear_weight,
)


class _FakeGate:
    def __init__(
        self,
        weight: torch.Tensor,
        *,
        quant_method=None,
        bias=None,
        out_dtype: torch.dtype | None = None,
    ):
        self.weight = weight
        self.quant_method = quant_method
        self.bias = bias
        self.out_dtype = out_dtype

    def __call__(self, hidden_states: torch.Tensor):
        return project_with_linear_weight(hidden_states, self.weight), None


def test_maybe_fused_gate_and_next_logits_matches_two_separate_linears():
    hidden_states = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=torch.bfloat16,
    )
    gate_weight = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=torch.bfloat16,
    )
    next_gate_weight = torch.tensor(
        [[2.0, 1.0], [1.0, 2.0]],
        dtype=torch.bfloat16,
    )
    gate = _FakeGate(
        gate_weight,
        quant_method=UnquantizedLinearMethod(),
    )

    router_logits, pred_logits = maybe_fused_gate_and_next_logits(
        hidden_states,
        gate,
        next_gate_weight,
    )

    expected_router = project_with_linear_weight(hidden_states, gate_weight)
    expected_pred = project_with_linear_weight(hidden_states, next_gate_weight)

    assert torch.equal(router_logits, expected_router)
    assert torch.equal(pred_logits, expected_pred)


def test_maybe_fused_gate_and_next_logits_falls_back_for_non_unquantized_gate():
    hidden_states = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    gate_weight = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    next_gate_weight = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    gate = _FakeGate(gate_weight, quant_method=object())

    router_logits, pred_logits = maybe_fused_gate_and_next_logits(
        hidden_states,
        gate,
        next_gate_weight,
    )

    assert torch.equal(router_logits, torch.tensor([[1.0]], dtype=torch.float32))
    assert torch.equal(pred_logits, torch.tensor([[1.5]], dtype=torch.float32))


def test_predicted_load_from_topk_counts_hot_experts_without_softmax():
    pred_logits = torch.tensor(
        [[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]],
        dtype=torch.float32,
    )

    pred_load = predicted_load_from_topk(
        pred_logits,
        top_k=2,
        stride_scale=2.0,
    )

    assert torch.equal(pred_load, torch.tensor([2.0, 2.0, 4.0]))
