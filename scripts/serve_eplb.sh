#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
source "${SCRIPT_DIR}/common.sh"
resolve_model_profile

VLLM_BIN=vllm
HOST=0.0.0.0
PORT=8000
TP_SIZE=1
DP_SIZE=8
DTYPE=bfloat16
GPU_MEMORY_UTILIZATION=0.85
ENABLE_EPLB=1
EPLB_ALGO=fgate
WINDOW_SIZE=1000
STEP_INTERVAL=1000
NUM_REDUNDANT_EXPERTS=2
LOG_BALANCEDNESS=true
TRUST_REMOTE_CODE=0
RESULTS_DIR=${ROOT_DIR}/results
mkdir -p "${RESULTS_DIR}"
LOG_FILE=${RESULTS_DIR}/server_${MODEL_TAG}_${EPLB_ALGO}.log

EP_SIZE=$((TP_SIZE * DP_SIZE))
if [[ "${EPLB_ALGO}" == "fgate-peer-cache" ]] && (( NUM_REDUNDANT_EXPERTS % EP_SIZE != 0 )); then
  echo "fgate-peer-cache requires NUM_REDUNDANT_EXPERTS divisible by EP_SIZE=${EP_SIZE}" >&2
  exit 1
fi

EPLB_ARGS=()
if [[ "${ENABLE_EPLB}" == "1" ]]; then
  export EPLB_ALGO WINDOW_SIZE STEP_INTERVAL NUM_REDUNDANT_EXPERTS LOG_BALANCEDNESS
  EPLB_CONFIG=$(python - <<'PY'
import json
import os
print(json.dumps({
    "algorithm": os.environ["EPLB_ALGO"],
    "window_size": int(os.environ["WINDOW_SIZE"]),
    "step_interval": int(os.environ["STEP_INTERVAL"]),
    "num_redundant_experts": int(os.environ["NUM_REDUNDANT_EXPERTS"]),
    "log_balancedness": os.environ["LOG_BALANCEDNESS"].lower() == "true",
}, separators=(",", ":")))
PY
)
  EPLB_ARGS=(--enable-eplb --eplb-config "${EPLB_CONFIG}")
fi

TRUST_REMOTE_CODE_ARGS=()
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  TRUST_REMOTE_CODE_ARGS=(--trust-remote-code)
fi

echo "MODEL_KEY=${MODEL_KEY}"
echo "MODEL=${MODEL}"
echo "EPLB_ALGO=${EPLB_ALGO}"
echo "LOG_FILE=${LOG_FILE}"

"${VLLM_BIN}" serve "${MODEL}"   --tokenizer "${TOKENIZER}"   --host "${HOST}"   --port "${PORT}"   --tensor-parallel-size "${TP_SIZE}"   --data-parallel-size "${DP_SIZE}"   --dtype "${DTYPE}"   --max-model-len "${MAX_MODEL_LEN}"   --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"   --enable-expert-parallel   "${TRUST_REMOTE_CODE_ARGS[@]}"   "${EPLB_ARGS[@]}"   2>&1 | tee "${LOG_FILE}"
