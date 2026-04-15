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
  ALGORITHM="ema" \
  WINDOW_SIZE="1000" \
  STEP_INTERVAL="1000" \
  NUM_REDUNDANT_EXPERTS="16" \
  LOG_BALANCEDNESS="false" \
  EMA_ALPHA="0.01" \
  python - <<'PY'
import json
import os

print(json.dumps({
    "algorithm": os.environ["ALGORITHM"],
    "ema_alpha": float(os.environ["EMA_ALPHA"]),
    "window_size": int(os.environ["WINDOW_SIZE"]),
    "step_interval": int(os.environ["STEP_INTERVAL"]),
    "num_redundant_experts": int(os.environ["NUM_REDUNDANT_EXPERTS"]),
    "log_balancedness": os.environ["LOG_BALANCEDNESS"].lower() == "true",
}, separators=(",", ":")))
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
