#!/bin/bash
set -euo pipefail

EXPERIMENT_NAME_PREFIX=${EXPERIMENT_NAME_PREFIX:-random-decode-heavy-ema}

export PATH=/usr/local/bin:$PATH
export PATH=/usr/local/cuda-12.6/bin:$PATH
export PATH=/usr/local/nvidia/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

MODEL=${MODEL:-/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507/snapshots/56e16a623ffb2855ca901a65166a9170e99df127}
RESULTS_DIR=${RESULTS_DIR:-./results-qwen3-200B-random}
PORT=${PORT:-8000}
NUM_PROMPTS=${NUM_PROMPTS:-2048}
REQUEST_RATE=${REQUEST_RATE:-inf}
RANDOM_INPUT_LEN=${RANDOM_INPUT_LEN:-32}
RANDOM_OUTPUT_LEN=${RANDOM_OUTPUT_LEN:-512}
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-0.0}
SEED=${SEED:-0}
TEMPERATURE=${TEMPERATURE:-0}

# 默认扫两档并发；如果要更激进，可以改成 "128 256 512"。
MAX_CONCURRENCY_LIST=(64 128 256 512)

mkdir -p "$RESULTS_DIR"

request_rate_tag=${REQUEST_RATE//./p}
range_ratio_tag=${RANDOM_RANGE_RATIO//./p}

for max_concurrency in "${MAX_CONCURRENCY_LIST[@]}"; do
    result_filename="${EXPERIMENT_NAME_PREFIX}_rr${request_rate_tag}_mc${max_concurrency}_in${RANDOM_INPUT_LEN}_out${RANDOM_OUTPUT_LEN}_range${range_ratio_tag}.json"

    vllm bench serve \
        --backend vllm \
        --model "$MODEL" \
        --port "$PORT" \
        --dataset-name random \
        --num-prompts "$NUM_PROMPTS" \
        --request-rate "$REQUEST_RATE" \
        --max-concurrency "$max_concurrency" \
        --random-input-len "$RANDOM_INPUT_LEN" \
        --random-output-len "$RANDOM_OUTPUT_LEN" \
        --random-range-ratio "$RANDOM_RANGE_RATIO" \
        --seed "$SEED" \
        --temperature "$TEMPERATURE" \
        --save-result \
        --result-dir "$RESULTS_DIR" \
        --result-filename "$result_filename"
done
