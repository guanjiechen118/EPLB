#!/bin/bash
set -euo pipefail

EXPERIMENT_NAME="baseline-EP"

export PATH=/usr/local/bin:$PATH
export PATH=/usr/local/cuda-12.6/bin:$PATH
export PATH=/usr/local/nvidia/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

MODEL=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--deepseek-ai--DeepSeek-V3-Base/snapshots/afb92e1fa402c2be2a9eb085312bb02e0384d6c7
RESULTS_DIR=./results
mkdir -p "$RESULTS_DIR"

vllm bench serve \
    --backend vllm \
    --model "$MODEL" \
    --port 8000 \
    --dataset-name random \
    --random-input-len 128 \
    --request-rate 1 \
    --max-concurrency 1 \
    --random-output-len 128 \
    --num-prompts 10 \
    --save-result \
    --result-dir "$RESULTS_DIR" \
    --result-filename "${EXPERIMENT_NAME}.json"
