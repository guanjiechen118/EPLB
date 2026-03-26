#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
source "${SCRIPT_DIR}/common.sh"
resolve_model_profile

VLLM_BIN=vllm
PORT=8000
DATASET_NAME=random
NUM_PROMPTS=10
RESULTS_DIR=${ROOT_DIR}/results
mkdir -p "${RESULTS_DIR}"
RESULT_BASENAME=bench_${MODEL_TAG}_${DATASET_NAME}
LOG_FILE=${RESULTS_DIR}/${RESULT_BASENAME}.log

CMD=("${VLLM_BIN}" bench serve
  --backend vllm
  --model "${MODEL}"
  --port "${PORT}"
  --dataset-name "${DATASET_NAME}"
  --num-prompts "${NUM_PROMPTS}"
  --save-result
  --result-dir "${RESULTS_DIR}"
  --result-filename "${RESULT_BASENAME}.json")

if [[ "${DATASET_NAME}" == "random" ]]; then
  CMD+=(
    --random-input-len "128"
    --random-output-len "128"
    --request-rate "10"
    --max-concurrency "10"
  )
elif [[ "${DATASET_NAME}" == "custom" ]]; then
  : "${DATASET_PATH:?DATASET_PATH is required when DATASET_NAME=custom}"
  CMD+=(
    --dataset-path "${DATASET_PATH}"
  )
else
  echo "Unsupported DATASET_NAME: ${DATASET_NAME}" >&2
  exit 1
fi

echo "MODEL_KEY=${MODEL_KEY}"
echo "DATASET_NAME=${DATASET_NAME}"
echo "LOG_FILE=${LOG_FILE}"

"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
