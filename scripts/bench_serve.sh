#!/usr/bin/env bash
set -euo pipefail
unset OPENAI_API_KEY CODEX_API_KEY
MODEL_KEY=qwen3_30b_a3b
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
source "${SCRIPT_DIR}/common.sh"
resolve_model_profile

VLLM_BIN=${VLLM_BIN:-vllm}
BENCH_HOST=${BENCH_HOST:-127.0.0.1}
EMA_PORT=${EMA_PORT:-8010}
HYBRID_PORT=${HYBRID_PORT:-8011}
BASELINE_PORT=${BASELINE_PORT:-8009}
TP_SIZE=${TP_SIZE:-1}
EMA_DP_SIZE=${EMA_DP_SIZE:-4}
HYBRID_DP_SIZE=${HYBRID_DP_SIZE:-4}
BASELINE_DP_SIZE=${BASELINE_DP_SIZE:-4}
READY_TIMEOUT_SEC=${READY_TIMEOUT_SEC:-30}
READY_POLL_INTERVAL_SEC=${READY_POLL_INTERVAL_SEC:-2}
RESULTS_DIR=${RESULTS_DIR:-${ROOT_DIR}/results}
mkdir -p "${RESULTS_DIR}"

DATASET_NAME=custom
DATASET_PATH=${DATASET_PATH:-${ROOT_DIR}/data/domain_shift_hf/scenarios/alternating_2.jsonl}
NUM_PROMPTS=${NUM_PROMPTS:-2048}
REQUEST_RATE=${REQUEST_RATE:-128}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-256}
RANDOM_INPUT_LEN=${RANDOM_INPUT_LEN:-128}
RANDOM_OUTPUT_LEN=${RANDOM_OUTPUT_LEN:-128}
DATASET_SEED=${DATASET_SEED:-0}
REQUEST_SEED=${REQUEST_SEED:-0}
TEMPERATURE=${TEMPERATURE:-0}
TOP_P=${TOP_P:-}
TOP_K=${TOP_K:-}
MIN_P=${MIN_P:-}
FREQUENCY_PENALTY=${FREQUENCY_PENALTY:-}
PRESENCE_PENALTY=${PRESENCE_PENALTY:-}
REPETITION_PENALTY=${REPETITION_PENALTY:-}

NUM_REDUNDANT_EXPERTS=${NUM_REDUNDANT_EXPERTS:-32}
NUM_STATIC_REDUNDANT_EXPERTS=${NUM_STATIC_REDUNDANT_EXPERTS:-16}
TARGETS=${TARGETS:-ema,hybrid}
# Optional suffix for result basenames, e.g. eval run id (empty = unchanged).
BENCH_SUFFIX=${BENCH_SUFFIX:-}

# Result filenames use *_DP_SIZE; they do not configure the server. Start the
# server with the same DP_SIZE as HYBRID_DP_SIZE / EMA_DP_SIZE when comparing runs.

if [[ "${DATASET_NAME}" == "custom" && ! -f "${DATASET_PATH}" ]]; then
  echo "DATASET_PATH does not exist: ${DATASET_PATH}" >&2
  exit 1
fi

if [[ "${DATASET_NAME}" == "custom" ]]; then
  DATASET_TAG=$(basename "${DATASET_PATH}")
  DATASET_TAG=${DATASET_TAG%.jsonl}
else
  DATASET_TAG=${DATASET_NAME}
fi

wait_for_server() {
  local name=$1
  local port=$2
  local deadline=$((SECONDS + READY_TIMEOUT_SEC))

  echo "waiting for ${name} server on http://${BENCH_HOST}:${port}/health ..."
  while (( SECONDS < deadline )); do
    if python - "${BENCH_HOST}" "${port}" <<'PY' >/dev/null 2>&1
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

s = socket.socket()
s.settimeout(2.0)
try:
    s.connect((host, port))
    s.sendall(
        b"GET /health HTTP/1.1\r\n"
        + f"Host: {host}\r\n".encode()
        + b"Connection: close\r\n\r\n"
    )
    data = s.recv(256)
    if b"200 OK" not in data:
        raise SystemExit(1)
finally:
    s.close()
PY
    then
      echo "${name} server is ready."
      return 0
    fi
    sleep "${READY_POLL_INTERVAL_SEC}"
  done

  echo "${name} server did not become ready within ${READY_TIMEOUT_SEC}s" >&2
  return 1
}

run_bench() {
  local name=$1
  local port=$2
  local result_basename=$3
  local log_file=$4
  local extra_body

  local -a cmd=(
    "${VLLM_BIN}" bench serve
    --backend vllm
    --host "${BENCH_HOST}"
    --port "${port}"
    --model "${MODEL}"
    --dataset-name "${DATASET_NAME}"
    --num-prompts "${NUM_PROMPTS}"
    --request-rate "${REQUEST_RATE}"
    --max-concurrency "${MAX_CONCURRENCY}"
    --seed "${DATASET_SEED}"
    --temperature "${TEMPERATURE}"
    --save-result
    --result-dir "${RESULTS_DIR}"
    --result-filename "${result_basename}.json"
  )

  extra_body=$(
    REQUEST_SEED="${REQUEST_SEED}" \
    TOP_P="${TOP_P}" \
    TOP_K="${TOP_K}" \
    MIN_P="${MIN_P}" \
    FREQUENCY_PENALTY="${FREQUENCY_PENALTY}" \
    PRESENCE_PENALTY="${PRESENCE_PENALTY}" \
    REPETITION_PENALTY="${REPETITION_PENALTY}" \
    python - <<'PY'
import json
import os

body = {"seed": int(os.environ["REQUEST_SEED"])}
optional_ints = ("TOP_K",)
optional_floats = (
    "TOP_P",
    "MIN_P",
    "FREQUENCY_PENALTY",
    "PRESENCE_PENALTY",
    "REPETITION_PENALTY",
)
for name in optional_ints:
    value = os.environ.get(name, "").strip()
    if value:
        body[name.lower()] = int(value)
for name in optional_floats:
    value = os.environ.get(name, "").strip()
    if value:
        body[name.lower()] = float(value)
print(json.dumps(body, separators=(",", ":")))
PY
  )
  cmd+=(--extra-body "${extra_body}")

  if [[ "${DATASET_NAME}" == "random" ]]; then
    cmd+=(
      --random-input-len "${RANDOM_INPUT_LEN}"
      --random-output-len "${RANDOM_OUTPUT_LEN}"
    )
  elif [[ "${DATASET_NAME}" == "custom" ]]; then
    cmd+=(
      --dataset-path "${DATASET_PATH}"
    )
  else
    echo "Unsupported DATASET_NAME: ${DATASET_NAME}" >&2
    exit 1
  fi

  echo "running ${name} benchmark -> ${RESULTS_DIR}/${result_basename}.json"
  NO_PROXY="${BENCH_HOST},127.0.0.1,localhost" \
  no_proxy="${BENCH_HOST},127.0.0.1,localhost" \
  "${cmd[@]}" 2>&1 | tee "${log_file}"
}

BASELINE_RESULT_BASENAME=bench_${MODEL_TAG}_baseline_tp${TP_SIZE}_dp${BASELINE_DP_SIZE}_${DATASET_TAG}
EMA_RESULT_BASENAME=bench_${MODEL_TAG}_ema_tp${TP_SIZE}_dp${EMA_DP_SIZE}_r${NUM_REDUNDANT_EXPERTS}_${DATASET_TAG}
HYBRID_RESULT_BASENAME=bench_${MODEL_TAG}_fgate-hybrid-cache_tp${TP_SIZE}_dp${HYBRID_DP_SIZE}_r${NUM_REDUNDANT_EXPERTS}_s${NUM_STATIC_REDUNDANT_EXPERTS}_${DATASET_TAG}
if [[ -n "${BENCH_SUFFIX}" ]]; then
  BASELINE_RESULT_BASENAME="${BASELINE_RESULT_BASENAME}__${BENCH_SUFFIX}"
  EMA_RESULT_BASENAME="${EMA_RESULT_BASENAME}__${BENCH_SUFFIX}"
  HYBRID_RESULT_BASENAME="${HYBRID_RESULT_BASENAME}__${BENCH_SUFFIX}"
fi
BASELINE_BENCH_LOG=${RESULTS_DIR}/${BASELINE_RESULT_BASENAME}.log
EMA_BENCH_LOG=${RESULTS_DIR}/${EMA_RESULT_BASENAME}.log
HYBRID_BENCH_LOG=${RESULTS_DIR}/${HYBRID_RESULT_BASENAME}.log

echo "MODEL_KEY=${MODEL_KEY}"
echo "MODEL=${MODEL}"
echo "DATASET_NAME=${DATASET_NAME}"
echo "DATASET_TAG=${DATASET_TAG}"
echo "NUM_PROMPTS=${NUM_PROMPTS}"
echo "REQUEST_RATE=${REQUEST_RATE}"
echo "MAX_CONCURRENCY=${MAX_CONCURRENCY}"
echo "DATASET_SEED=${DATASET_SEED}"
echo "REQUEST_SEED=${REQUEST_SEED}"
echo "TEMPERATURE=${TEMPERATURE}"
echo "TOP_P=${TOP_P}"
echo "TOP_K=${TOP_K}"
echo "TARGETS=${TARGETS}"
if [[ "${DATASET_NAME}" == "custom" ]]; then
  echo "DATASET_PATH=${DATASET_PATH}"
fi

IFS=',' read -r -a SELECTED_TARGETS <<< "${TARGETS}"
for target in "${SELECTED_TARGETS[@]}"; do
  case "${target}" in
    baseline)
      wait_for_server "baseline" "${BASELINE_PORT}"
      run_bench "baseline" "${BASELINE_PORT}" "${BASELINE_RESULT_BASENAME}" "${BASELINE_BENCH_LOG}"
      ;;
    ema)
      wait_for_server "ema" "${EMA_PORT}"
      run_bench "ema" "${EMA_PORT}" "${EMA_RESULT_BASENAME}" "${EMA_BENCH_LOG}"
      ;;
    hybrid)
      wait_for_server "hybrid" "${HYBRID_PORT}"
      run_bench "hybrid" "${HYBRID_PORT}" "${HYBRID_RESULT_BASENAME}" "${HYBRID_BENCH_LOG}"
      ;;
    *)
      echo "Unsupported target in TARGETS: ${target}" >&2
      exit 1
      ;;
  esac
done

echo "bench finished."
