#!/usr/bin/env bash
set -euo pipefail

export PATH=/usr/local/bin:${PATH}
export PATH=/usr/local/cuda-12.6/bin:${PATH}
export PATH=/usr/local/nvidia/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH}

resolve_model_profile() {
  MODEL_KEY=${MODEL_KEY:-deepseek_v2_lite}
  case "${MODEL_KEY}" in
    deepseek_v2_lite)
      MODEL=/mnt/shared-storage-user/moegroup/share_models/DeepSeek-V2-Lite
      MAX_MODEL_LEN_DEFAULT=163840
      ;;
    deepseek_v3)
      MODEL=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--deepseek-ai--DeepSeek-V3-Base/snapshots/afb92e1fa402c2be2a9eb085312bb02e0384d6c7
      MAX_MODEL_LEN_DEFAULT=163840
      ;;
    qwen3_30b_a3b)
      MODEL=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/4c446470ba0aec43e22ac1128f9ffd915f338ba3
      MAX_MODEL_LEN_DEFAULT=40960
      ;;
    qwen3_5_122b_a10b)
      MODEL=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-122B-A10B/snapshots/b000b2eb18a7f4cdf3153c4215842da339e09d99
      MAX_MODEL_LEN_DEFAULT=262144
      ;;
    qwen3_5_397b_a17b)
      MODEL=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-397B-A17B/snapshots/285b7b5d3792e7357b31101b858806a0eddd3e3c
      MAX_MODEL_LEN_DEFAULT=262144
      ;;
    *)
      echo "Unknown MODEL_KEY: ${MODEL_KEY}" >&2
      return 1
      ;;
  esac

  TOKENIZER=${MODEL}
  MAX_MODEL_LEN=${MAX_MODEL_LEN:-${MAX_MODEL_LEN_DEFAULT}}
  MODEL_TAG=${MODEL_KEY}

  export MODEL_KEY MODEL MODEL_TAG TOKENIZER MAX_MODEL_LEN
}
