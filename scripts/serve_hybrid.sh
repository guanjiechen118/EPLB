#!/usr/bin/env bash
set -euo pipefail

MODEL_KEY=qwen3_30b_a3b
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
source "${SCRIPT_DIR}/common.sh"
resolve_model_profile

VLLM_BIN=${VLLM_BIN:-vllm}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8011}
TP_SIZE=${TP_SIZE:-1}
DP_SIZE=${DP_SIZE:-4}
DTYPE=${DTYPE:-bfloat16}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-0}
ENFORCE_EAGER=${ENFORCE_EAGER:-0}
SERVER_SEED=${SERVER_SEED:-0}
# Honor pre-exported CUDA_VISIBLE_DEVICES; default is 8 GPUs for this profile.
VISIBLE_GPUS=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
WINDOW_SIZE=${WINDOW_SIZE:-1000}
STEP_INTERVAL=${STEP_INTERVAL:-1000}
EMA_ALPHA=${EMA_ALPHA:-0.01}
NUM_REDUNDANT_EXPERTS=${NUM_REDUNDANT_EXPERTS:-16}
NUM_STATIC_REDUNDANT_EXPERTS=${NUM_STATIC_REDUNDANT_EXPERTS:-8}
LOG_BALANCEDNESS=${LOG_BALANCEDNESS:-false}
# Set to 1 to run periodic global rearrange when data_parallel_size>1.
# This is now enabled by default for hybrid so the EMA-managed static path is
# active again on normal decode steps.
HYBRID_PERIODIC_REARRANGE_MULTI_DP=${HYBRID_PERIODIC_REARRANGE_MULTI_DP:-1}
# Decode-only fgate subsampling; 1 = every step.
# For idea2-style layer-local hybrid refresh, keep this at 1 by default.
FGATE_DECODE_STRIDE=${FGATE_DECODE_STRIDE:-1}
# Optional fgate-hybrid-cache ablation overrides (omit from JSON when unset).
# Tri-state bools: set to 1/0 or true/false to send explicit JSON; unset = default behavior.
HYBRID_IMMEDIATE_LAYER_REFRESH=${HYBRID_IMMEDIATE_LAYER_REFRESH:-}
HYBRID_BARRIER_AFTER_PERIODIC_REARRANGE=${HYBRID_BARRIER_AFTER_PERIODIC_REARRANGE:-}
HYBRID_SKIP_FGATE_ON_DECODE=${HYBRID_SKIP_FGATE_ON_DECODE:-0}
# 1 = skip forward-gate on prefill (max_query_len>1); decode still uses fgate unless HYBRID_SKIP_FGATE_ON_DECODE.
HYBRID_SKIP_FGATE_ON_PREFILL=${HYBRID_SKIP_FGATE_ON_PREFILL:-0}
# 单机多实例 vLLM（如双路各 DP4）必须错开，否则会争用默认 29550。
DATA_PARALLEL_RPC_PORT=${DATA_PARALLEL_RPC_PORT:-}
RESULTS_DIR=${RESULTS_DIR:-${ROOT_DIR}/results}
mkdir -p "${RESULTS_DIR}"

LOG_FILE=${LOG_FILE:-${RESULTS_DIR}/server_${MODEL_TAG}_fgate-hybrid-cache_tp${TP_SIZE}_dp${DP_SIZE}_r${NUM_REDUNDANT_EXPERTS}_s${NUM_STATIC_REDUNDANT_EXPERTS}.log}

EP_SIZE=$((TP_SIZE * DP_SIZE))
if (( NUM_REDUNDANT_EXPERTS <= 0 )); then
  echo "NUM_REDUNDANT_EXPERTS must be > 0" >&2
  exit 1
fi

if (( NUM_STATIC_REDUNDANT_EXPERTS <= 0 )); then
  echo "NUM_STATIC_REDUNDANT_EXPERTS must be > 0" >&2
  exit 1
fi

if (( NUM_STATIC_REDUNDANT_EXPERTS >= NUM_REDUNDANT_EXPERTS )); then
  echo "NUM_STATIC_REDUNDANT_EXPERTS must be smaller than NUM_REDUNDANT_EXPERTS" >&2
  exit 1
fi

if (( NUM_STATIC_REDUNDANT_EXPERTS % EP_SIZE != 0 )); then
  echo "NUM_STATIC_REDUNDANT_EXPERTS must be divisible by EP_SIZE=${EP_SIZE}" >&2
  exit 1
fi

HYBRID_DYNAMIC_REDUNDANT=$((NUM_REDUNDANT_EXPERTS - NUM_STATIC_REDUNDANT_EXPERTS))
if (( HYBRID_DYNAMIC_REDUNDANT % (2 * EP_SIZE) != 0 )); then
  echo "dynamic redundant experts must be divisible by 2 * EP_SIZE=$((2 * EP_SIZE))" >&2
  exit 1
fi

EPLB_CONFIG=$(
  ALGORITHM="fgate-hybrid-cache" \
  WINDOW_SIZE="${WINDOW_SIZE}" \
  STEP_INTERVAL="${STEP_INTERVAL}" \
  NUM_REDUNDANT_EXPERTS="${NUM_REDUNDANT_EXPERTS}" \
  NUM_STATIC_REDUNDANT_EXPERTS="${NUM_STATIC_REDUNDANT_EXPERTS}" \
  LOG_BALANCEDNESS="${LOG_BALANCEDNESS}" \
  EMA_ALPHA="${EMA_ALPHA}" \
  HYBRID_PERIODIC_REARRANGE_MULTI_DP="${HYBRID_PERIODIC_REARRANGE_MULTI_DP}" \
  FGATE_DECODE_STRIDE="${FGATE_DECODE_STRIDE}" \
  HYBRID_IMMEDIATE_LAYER_REFRESH="${HYBRID_IMMEDIATE_LAYER_REFRESH}" \
  HYBRID_BARRIER_AFTER_PERIODIC_REARRANGE="${HYBRID_BARRIER_AFTER_PERIODIC_REARRANGE}" \
  HYBRID_SKIP_FGATE_ON_DECODE="${HYBRID_SKIP_FGATE_ON_DECODE}" \
  HYBRID_SKIP_FGATE_ON_PREFILL="${HYBRID_SKIP_FGATE_ON_PREFILL}" \
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

echo "MODEL_KEY=${MODEL_KEY}"
echo "MODEL=${MODEL}"
echo "VISIBLE_GPUS=${VISIBLE_GPUS}"
echo "PORT=${PORT}"
echo "TP_SIZE=${TP_SIZE}"
echo "DP_SIZE=${DP_SIZE}"
echo "SERVER_SEED=${SERVER_SEED}"
echo "LOG_FILE=${LOG_FILE}"
echo "EPLB_CONFIG=${EPLB_CONFIG}"
echo "ENFORCE_EAGER=${ENFORCE_EAGER}"
echo "HYBRID_PERIODIC_REARRANGE_MULTI_DP=${HYBRID_PERIODIC_REARRANGE_MULTI_DP}"
echo "FGATE_DECODE_STRIDE=${FGATE_DECODE_STRIDE}"
echo "HYBRID_IMMEDIATE_LAYER_REFRESH=${HYBRID_IMMEDIATE_LAYER_REFRESH:-<unset>}"
echo "HYBRID_BARRIER_AFTER_PERIODIC_REARRANGE=${HYBRID_BARRIER_AFTER_PERIODIC_REARRANGE:-<unset>}"
echo "HYBRID_SKIP_FGATE_ON_DECODE=${HYBRID_SKIP_FGATE_ON_DECODE}"
echo "HYBRID_SKIP_FGATE_ON_PREFILL=${HYBRID_SKIP_FGATE_ON_PREFILL}"
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

if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  CMD+=(--enforce-eager)
fi

if [[ -n "${DATA_PARALLEL_RPC_PORT}" ]]; then
  CMD+=(--data-parallel-rpc-port "${DATA_PARALLEL_RPC_PORT}")
fi

exec "${CMD[@]}"
