#!/usr/bin/env bash
set -euo pipefail

MODEL_KEY=qwen3_30b_a3b
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
source "${SCRIPT_DIR}/common.sh"
resolve_model_profile

VLLM_BIN=${VLLM_BIN:-vllm}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8010}
TP_SIZE=${TP_SIZE:-1}
DP_SIZE=${DP_SIZE:-4}
DTYPE=${DTYPE:-bfloat16}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-0}
SERVER_SEED=${SERVER_SEED:-0}
VISIBLE_GPUS=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
WINDOW_SIZE=${WINDOW_SIZE:-1000}
STEP_INTERVAL=${STEP_INTERVAL:-1000}
EMA_ALPHA=${EMA_ALPHA:-0.01}
NUM_REDUNDANT_EXPERTS=${NUM_REDUNDANT_EXPERTS:-16}
LOG_BALANCEDNESS=${LOG_BALANCEDNESS:-false}
# 单机多实例 vLLM（DP>1）时需错开，默认留空使用 vLLM 内置端口。
DATA_PARALLEL_RPC_PORT=${DATA_PARALLEL_RPC_PORT:-}
RESULTS_DIR=${RESULTS_DIR:-${ROOT_DIR}/results}
mkdir -p "${RESULTS_DIR}"

LOG_FILE=${LOG_FILE:-${RESULTS_DIR}/server_${MODEL_TAG}_ema_tp${TP_SIZE}_dp${DP_SIZE}_r${NUM_REDUNDANT_EXPERTS}.log}

if (( NUM_REDUNDANT_EXPERTS <= 0 )); then
  echo "NUM_REDUNDANT_EXPERTS must be > 0" >&2
  exit 1
fi

EPLB_CONFIG=$(
  ALGORITHM="ema" \
  WINDOW_SIZE="${WINDOW_SIZE}" \
  STEP_INTERVAL="${STEP_INTERVAL}" \
  NUM_REDUNDANT_EXPERTS="${NUM_REDUNDANT_EXPERTS}" \
  LOG_BALANCEDNESS="${LOG_BALANCEDNESS}" \
  EMA_ALPHA="${EMA_ALPHA}" \
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

echo "MODEL_KEY=${MODEL_KEY}"
echo "MODEL=${MODEL}"
echo "VISIBLE_GPUS=${VISIBLE_GPUS}"
echo "PORT=${PORT}"
echo "TP_SIZE=${TP_SIZE}"
echo "DP_SIZE=${DP_SIZE}"
echo "SERVER_SEED=${SERVER_SEED}"
echo "LOG_FILE=${LOG_FILE}"
echo "EPLB_CONFIG=${EPLB_CONFIG}"
echo "DATA_PARALLEL_RPC_PORT=${DATA_PARALLEL_RPC_PORT:-<vllm default>}"

exec > >(tee "${LOG_FILE}") 2>&1

export CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}"

CMD=(
  "${VLLM_BIN}" serve "${MODEL}"
  --tokenizer "${TOKENIZER}"
  --host "${HOST}"
  --port "${PORT}"
  --seed "${SERVER_SEED}"
  --tensor-parallel-size "${TP_SIZE}"
  --data-parallel-size "${DP_SIZE}"
  --dtype "${DTYPE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --enable-expert-parallel
  --enable-eplb
  --eplb-config "${EPLB_CONFIG}"
)

if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  CMD+=(--trust-remote-code)
fi

if [[ -n "${DATA_PARALLEL_RPC_PORT}" ]]; then
  CMD+=(--data-parallel-rpc-port "${DATA_PARALLEL_RPC_PORT}")
fi

exec "${CMD[@]}"
