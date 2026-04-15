#!/bin/bash
set -euo pipefail

EXPERIMENT_NAME="baseline-EP"

export PATH=/usr/local/bin:$PATH
export PATH=/usr/local/cuda-12.6/bin:$PATH
export PATH=/usr/local/nvidia/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

MODEL="/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507/snapshots/56e16a623ffb2855ca901a65166a9170e99df127"
RESULTS_DIR=./results-qwen3-200B
mkdir -p "$RESULTS_DIR"


vllm bench serve \
    --backend vllm \
    --model "$MODEL" \
    --port 8000 \
    --dataset-name custom \
    --dataset-path /mnt/shared-storage-user/chenguanjie/huawei_eplb/data/domain_shift_hf/scenarios/alternating_2.jsonl \
    --num-prompts 2048 \
    --request-rate inf \
    --max-concurrency 512 \
    --save-result \
    --seed 0 \
    --temperature 0 \
    --result-dir "$RESULTS_DIR" \
    --result-filename "${EXPERIMENT_NAME}.json"

vllm bench serve \
    --backend vllm \
    --model "$MODEL" \
    --port 8000 \
    --dataset-name custom \
    --dataset-path /mnt/shared-storage-user/chenguanjie/huawei_eplb/data/domain_shift_hf/scenarios/alternating_2.jsonl \
    --num-prompts 2048 \
    --request-rate inf \
    --max-concurrency 512 \
    --save-result \
    --seed 0 \
    --temperature 0 \
    --result-dir "$RESULTS_DIR" \
    --result-filename "${EXPERIMENT_NAME}.json"


vllm bench serve \
    --backend vllm \
    --model "$MODEL" \
    --port 8000 \
    --dataset-name custom \
    --dataset-path /mnt/shared-storage-user/chenguanjie/huawei_eplb/data/domain_shift_hf/scenarios/alternating_2.jsonl \
    --num-prompts 2048 \
    --request-rate inf \
    --max-concurrency 512 \
    --save-result \
    --seed 0 \
    --temperature 0 \
    --result-dir "$RESULTS_DIR" \
    --result-filename "${EXPERIMENT_NAME}.json"