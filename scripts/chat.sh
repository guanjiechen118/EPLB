#!/usr/bin/env bash
set -euo pipefail

MODEL_KEY=${MODEL_KEY:-qwen3_30b_a3b}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"
resolve_model_profile

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8010}
PROMPT=${PROMPT:-人工智能的未来发展趋势是}
MAX_TOKENS=${MAX_TOKENS:-1024}
TEMPERATURE=${TEMPERATURE:-0}
REQUEST_SEED=${REQUEST_SEED:-0}
CURL_MAX_TIME=${CURL_MAX_TIME:-60}

PAYLOAD=$(
  MODEL="${MODEL}" \
  PROMPT="${PROMPT}" \
  MAX_TOKENS="${MAX_TOKENS}" \
  TEMPERATURE="${TEMPERATURE}" \
  REQUEST_SEED="${REQUEST_SEED}" \
  python - <<'PY'
import json
import os

print(json.dumps({
    "model": os.environ["MODEL"],
    "prompt": os.environ["PROMPT"],
    "max_tokens": int(os.environ["MAX_TOKENS"]),
    "temperature": float(os.environ["TEMPERATURE"]),
    "seed": int(os.environ["REQUEST_SEED"]),
}, ensure_ascii=False))
PY
)

echo "MODEL_KEY=${MODEL_KEY}"
echo "MODEL=${MODEL}"
echo "URL=http://${HOST}:${PORT}/v1/completions"
echo "MAX_TOKENS=${MAX_TOKENS}"
echo "TEMPERATURE=${TEMPERATURE}"
echo "REQUEST_SEED=${REQUEST_SEED}"

curl -sS --fail-with-body --max-time "${CURL_MAX_TIME}" \
  "http://${HOST}:${PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "${PAYLOAD}"
echo
