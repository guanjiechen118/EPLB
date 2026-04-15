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

TP_SIZE=1
DP_SIZE=8
EP_SIZE=$((TP_SIZE * DP_SIZE))

# fgate-only 要求所有冗余槽都进入两组 dynamic shadow banks，
# 因此 NUM_REDUNDANT_EXPERTS 必须能被 2 * EP_SIZE 整除。
NUM_REDUNDANT_EXPERTS=16
LOG_BALANCEDNESS=false
FGATE_DECODE_STRIDE=1
FGATE_SKIP_PREFILL=true
FGATE_PREFILL_IGNORE_REDUNDANT=true

if (( NUM_REDUNDANT_EXPERTS <= 0 )); then
  echo "NUM_REDUNDANT_EXPERTS must be > 0" >&2
  exit 1
fi

if (( NUM_REDUNDANT_EXPERTS % (2 * EP_SIZE) != 0 )); then
  echo "NUM_REDUNDANT_EXPERTS must be divisible by 2 * EP_SIZE=$((2 * EP_SIZE))" >&2
  exit 1
fi

EPLB_CONFIG=$(
  ALGORITHM="fgate-only" \
  NUM_REDUNDANT_EXPERTS="${NUM_REDUNDANT_EXPERTS}" \
  LOG_BALANCEDNESS="${LOG_BALANCEDNESS}" \
  FGATE_DECODE_STRIDE="${FGATE_DECODE_STRIDE}" \
  FGATE_SKIP_PREFILL="${FGATE_SKIP_PREFILL}" \
  FGATE_PREFILL_IGNORE_REDUNDANT="${FGATE_PREFILL_IGNORE_REDUNDANT}" \
  python - <<'PY'
import json
import os

cfg = {
    "algorithm": os.environ["ALGORITHM"],
    "num_redundant_experts": int(os.environ["NUM_REDUNDANT_EXPERTS"]),
    "log_balancedness": os.environ["LOG_BALANCEDNESS"].lower() == "true",
    "fgate_decode_stride": int(os.environ["FGATE_DECODE_STRIDE"]),
    "fgate_skip_prefill": os.environ["FGATE_SKIP_PREFILL"].lower() == "true",
    "fgate_prefill_ignore_redundant": (
        os.environ["FGATE_PREFILL_IGNORE_REDUNDANT"].lower() == "true"
    ),
}

print(json.dumps(cfg, separators=(",", ":")))
PY
)

# Stability-first server config for benchmarking.
"$VLLM_BIN" serve "$MODEL"  \
    --tensor-parallel-size "${TP_SIZE}" \
    --data-parallel-size "${DP_SIZE}"   \
    --enable-expert-parallel  \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --enable-eplb \
    --eplb-config "${EPLB_CONFIG}"
