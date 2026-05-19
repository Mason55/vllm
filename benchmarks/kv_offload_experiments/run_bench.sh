#!/usr/bin/env bash
# Run vllm bench serve workload against a running server.
# Usage: ./run_bench.sh <w1|w2|w3> [port] [output_json]
set -euo pipefail

WORKLOAD="${1:?w1|w2|w3}"
PORT="${2:-8000}"
OUT="${3:-}"

MODEL="${MODEL:-/data1/models/Qwen/Qwen3-4B-Instruct-2507}"

BASE=(
  --backend openai-chat
  --endpoint /v1/chat/completions
  --model "$MODEL"
  --host 127.0.0.1
  --port "$PORT"
  --dataset-name random
  --save-result
)

case "$WORKLOAD" in
  w1)
    EXTRA=(--random-prefix-len 8192 --input-len 512 --output-len 128
           --num-prompts 200 --request-rate 4)
    ;;
  w2)
    EXTRA=(--random-prefix-len 0 --input-len 4096 --output-len 64
           --num-prompts 150 --request-rate 2)
    ;;
  w3)
    EXTRA=(--random-prefix-len 4096 --input-len 2048 --output-len 256
           --num-prompts 100 --request-rate inf --burstiness 0.5)
    ;;
  *)
    echo "Unknown workload: $WORKLOAD" >&2
    exit 1
    ;;
esac

if [[ -n "$OUT" ]]; then
  mkdir -p "$(dirname "$OUT")"
  BASE+=(--result-dir "$(dirname "$OUT")" --result-filename "$(basename "$OUT")")
fi

echo "[bench] workload=$WORKLOAD model=$MODEL port=$PORT"
exec vllm bench serve "${BASE[@]}" "${EXTRA[@]}"
