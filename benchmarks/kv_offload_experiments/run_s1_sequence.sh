#!/usr/bin/env bash
# Layer 0: S1 sequence T0 → T1 → T2 (store / evict / load).
# Usage: ./run_s1_sequence.sh [port] [output_dir] [model] [prefix_tokens]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${1:-8000}"
OUT_DIR="${2:-results/s1}"
MODEL="${3:-/data1/models/Qwen/Qwen3-4B-Instruct-2507}"
PREFIX_TOKENS="${4:-65536}"
PYTHON="${PYTHON:-/data1/lmy/vllm/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python3
fi

PROMPT_DIR="$OUT_DIR/prompts"
mkdir -p "$OUT_DIR"

echo "[s1] Generating bit-exact prompts (${PREFIX_TOKENS} tokens)..."
"$PYTHON" gen_coding_agent_trace.py \
  --model "$MODEL" \
  --prefix-tokens "$PREFIX_TOKENS" \
  --seed 42 \
  --s1-prompts "$PROMPT_DIR"

PROMPT_A="$PROMPT_DIR/s1_prompt_a.txt"
PROMPT_B="$PROMPT_DIR/s1_prompt_b.txt"

run_step() {
  local label="$1"
  local prompt_file="$2"
  local out_json="$OUT_DIR/${label}.json"
  echo "[s1] ${label} ..."
  "$PYTHON" - "$PORT" "$MODEL" "$prompt_file" "$out_json" <<'PY'
import json, sys, time, urllib.request

port, model, prompt_path, out_path = sys.argv[1:5]
text = open(prompt_path, encoding="utf-8").read()
body = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": text}],
    "max_tokens": 8,
    "temperature": 0.0,
    "stream": True,
}).encode()
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=body,
    headers={"Content-Type": "application/json"},
    method="POST",
)
t0 = time.perf_counter()
ttft = None
with urllib.request.urlopen(req, timeout=3600) as resp:
    for raw in resp:
        line = raw.decode().strip()
        if line.startswith("data:") and line[5:].strip() != "[DONE]":
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            break
total = (time.perf_counter() - t0) * 1000
result = {"mean_ttft_ms": ttft or total, "total_ms": total, "label": out_path}
open(out_path, "w").write(json.dumps(result, indent=2))
print(f"  ttft_ms={result['mean_ttft_ms']:.1f} total_ms={total:.1f}")
PY
}

echo "[s1] T0 prompt_A (Eager store observation window)"
run_step "T0" "$PROMPT_A"

echo "[s1] T1 prompt_B (evict A from GPU)"
run_step "T1" "$PROMPT_B"

echo "[s1] T2 prompt_A again (CPU load if T1 evicted GPU copy)"
run_step "T2" "$PROMPT_A"

echo "[s1] Done. Compare T0 vs T2 TTFT in $OUT_DIR/*.json"
echo "[s1] grep 'Store event' in server.log for E1; Load signals on T2 for E2."
