#!/bin/bash
set -e

export PATH=/usr/local/bin:$PATH
export PATH=/usr/local/cuda-12.6/bin:$PATH
export PATH=/usr/local/nvidia/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

VLLM_BIN="/usr/local/bin/vllm"
# MODEL="/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--deepseek-ai--DeepSeek-V3-Base/snapshots/afb92e1fa402c2be2a9eb085312bb02e0384d6c7"
MODEL="/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507/snapshots/56e16a623ffb2855ca901a65166a9170e99df127"

export TORCHDYNAMO_VERBOSE=1
export VLLM_RPC_TIMEOUT=300
export VLLM_ENGINE_READY_TIMEOUT_S=1800


EPLB_CONFIG=$(
  ALGORITHM="fgate-hybrid-cache" \
  WINDOW_SIZE="1000" \
  STEP_INTERVAL="3000" \
  NUM_REDUNDANT_EXPERTS="24" \
  NUM_STATIC_REDUNDANT_EXPERTS="8" \
  LOG_BALANCEDNESS="false" \
  EMA_ALPHA="0.01" \
  HYBRID_PERIODIC_REARRANGE_MULTI_DP="1" \
  FGATE_DECODE_STRIDE="1" \
  HYBRID_IMMEDIATE_LAYER_REFRESH="true" \
  HYBRID_BARRIER_AFTER_PERIODIC_REARRANGE="0" \
  HYBRID_SKIP_FGATE_ON_DECODE="false" \
  HYBRID_SKIP_FGATE_ON_PREFILL="false" \
  python - <<'PY'
import json
import os

def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")

def _env_optional_bool(name: str):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return _env_bool(name)

cfg = {
    "algorithm": os.environ["ALGORITHM"],
    "ema_alpha": float(os.environ["EMA_ALPHA"]),
    "window_size": int(os.environ["WINDOW_SIZE"]),
    "step_interval": int(os.environ["STEP_INTERVAL"]),
    "num_redundant_experts": int(os.environ["NUM_REDUNDANT_EXPERTS"]),
    "num_static_redundant_experts": int(os.environ["NUM_STATIC_REDUNDANT_EXPERTS"]),
    "log_balancedness": os.environ["LOG_BALANCEDNESS"].lower() == "true",
    "hybrid_periodic_rearrange_with_multi_dp": _env_bool(
        "HYBRID_PERIODIC_REARRANGE_MULTI_DP"
    ),
    "fgate_decode_stride": int(os.environ["FGATE_DECODE_STRIDE"]),
}
imm = _env_optional_bool("HYBRID_IMMEDIATE_LAYER_REFRESH")
if imm is not None:
    cfg["hybrid_immediate_layer_refresh"] = imm
bar = _env_optional_bool("HYBRID_BARRIER_AFTER_PERIODIC_REARRANGE")
if bar is not None:
    cfg["hybrid_barrier_after_periodic_rearrange"] = bar
if _env_bool("HYBRID_SKIP_FGATE_ON_DECODE"):
    cfg["hybrid_skip_fgate_on_decode"] = True
if _env_bool("HYBRID_SKIP_FGATE_ON_PREFILL"):
    cfg["hybrid_skip_fgate_on_prefill"] = True

print(json.dumps(cfg, separators=(",", ":")))
PY
)

# Stability-first server config for benchmarking.
"$VLLM_BIN" serve "$MODEL"  \
    --tensor-parallel-size 1 \
    --data-parallel-size 8   \
    --enable-expert-parallel  \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --enable-eplb \
    --eplb-config "${EPLB_CONFIG}"
