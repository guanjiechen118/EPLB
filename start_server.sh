#!/bin/bash
set -e

export PATH=/usr/local/bin:$PATH
export PATH=/usr/local/cuda-12.6/bin:$PATH
export PATH=/usr/local/nvidia/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

VLLM_BIN="/usr/local/bin/vllm"
MODEL="/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--deepseek-ai--DeepSeek-V3-Base/snapshots/afb92e1fa402c2be2a9eb085312bb02e0384d6c7"
TOKENIZER="/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--deepseek-ai--DeepSeek-V3-Base/snapshots/afb92e1fa402c2be2a9eb085312bb02e0384d6c7"

export TORCHDYNAMO_VERBOSE=1
export VLLM_RPC_TIMEOUT=300
export VLLM_ENGINE_READY_TIMEOUT_S=1800

# Stability-first server config for benchmarking.
"$VLLM_BIN" serve "$MODEL"  \
    --tokenizer "$TOKENIZER" \
    --tensor-parallel-size 1 \
    --data-parallel-size 8   \
    --enable-expert-parallel  \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096
