# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F
from torch import nn

from vllm.model_executor.layers.linear import UnquantizedLinearMethod


def project_with_linear_weight(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    if hidden_states.dtype != weight.dtype:
        hidden_states = hidden_states.to(weight.dtype)
    return F.linear(hidden_states, weight)


def maybe_fused_gate_and_next_logits(
    hidden_states: torch.Tensor,
    gate: nn.Module,
    next_gate_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    gate_weight = getattr(gate, "weight", None)
    gate_quant_method = getattr(gate, "quant_method", None)
    gate_bias = getattr(gate, "bias", None)

    if (
        gate_weight is None
        or gate_bias is not None
        or gate_weight.dim() != 2
        or next_gate_weight.dim() != 2
        or gate_weight.shape[1] != next_gate_weight.shape[1]
        or gate_weight.dtype != next_gate_weight.dtype
        or gate_weight.device != next_gate_weight.device
        or not isinstance(gate_quant_method, UnquantizedLinearMethod)
    ):
        router_logits, _ = gate(hidden_states)
        pred_logits = project_with_linear_weight(hidden_states, next_gate_weight)
        return router_logits, pred_logits

    combined_logits = project_with_linear_weight(
        hidden_states,
        torch.cat((gate_weight, next_gate_weight), dim=0),
    )
    gate_width = gate_weight.shape[0]
    router_logits = combined_logits[..., :gate_width]
    pred_logits = combined_logits[..., gate_width:]

    out_dtype = getattr(gate, "out_dtype", None)
    if out_dtype is not None and router_logits.dtype != out_dtype:
        router_logits = router_logits.to(out_dtype)

    return router_logits, pred_logits


def predicted_load_from_topk(
    pred_logits: torch.Tensor,
    top_k: int,
    stride_scale: float = 1.0,
) -> torch.Tensor:
    num_experts = pred_logits.shape[-1]
    if pred_logits.shape[0] == 0:
        pred_load = torch.zeros(
            num_experts,
            device=pred_logits.device,
            dtype=torch.float32,
        )
    else:
        pred_topk_ids = torch.topk(
            pred_logits,
            k=min(top_k, num_experts),
            dim=-1,
            sorted=False,
        ).indices.reshape(-1)
        pred_load = torch.bincount(
            pred_topk_ids,
            minlength=num_experts,
        ).to(device=pred_logits.device, dtype=torch.float32)

    if stride_scale != 1.0:
        pred_load = pred_load * stride_scale
    return pred_load
