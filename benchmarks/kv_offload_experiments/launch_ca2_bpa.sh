#!/usr/bin/env bash
# Launch CA2 B-PA: server (ca profile) + CA2 replayer + offload witness check.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="/data1/anaconda3/envs/py312/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-1800000}"
unset PYTORCH_CUDA_ALLOC_CONF

MODEL="${MODEL:-/data1/models/Qwen/Qwen3-4B-Instruct-2507}"
PORT="${PORT:-8000}"
PROFILE="${PROFILE:-ca}"
KV_GIB="${KV_GIB:-32}"
OUTDIR="${OUTDIR:-results/ca2_bpa}"
TRACE_A="${TRACE_A:-traces/ca2_task_a.jsonl}"
TRACE_B="${TRACE_B:-traces/ca2_task_b.jsonl}"
SERVER_WAIT_S="${SERVER_WAIT_S:-1800}"
PYTHON="${PYTHON:-/data1/lmy/vllm/.venv/bin/python}"
REQUIRE_OFFLOAD_WITNESS="${REQUIRE_OFFLOAD_WITNESS:-1}"

mkdir -p "$OUTDIR"
ulimit -l unlimited 2>/dev/null || true

pkill -9 -f "vllm serve.*${PORT}" 2>/dev/null || true
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
pkill -9 -f "run_coding_agent_replayer.py ca2" 2>/dev/null || true
sleep 2

: > "$OUTDIR/server.log"
nohup ./run_server.sh simple "$MODEL" "$PORT" "$KV_GIB" "$PROFILE" >>"$OUTDIR/server.log" 2>&1 &
echo $! >"$OUTDIR/server.pid"
echo "[launch] server pid=$(cat "$OUTDIR/server.pid")"

for i in $(seq 1 $((SERVER_WAIT_S / 3))); do
  if curl -sf --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null; then
    echo "[launch] server ready after ${i}x3s"
    break
  fi
  if grep -qE "ValueError|ImportError|ModuleNotFoundError|EngineCore failed|RuntimeError: Engine core|cannot open shared object file|Traceback \\(most recent call last\\)" "$OUTDIR/server.log" 2>/dev/null; then
    echo "[launch] server failed:" >&2
    tail -20 "$OUTDIR/server.log" >&2
    exit 1
  fi
  sleep 3
done

curl -sf --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null || {
  echo "[launch] server health timeout" >&2
  tail -20 "$OUTDIR/server.log" >&2
  exit 1
}

: > "$OUTDIR/run.log"
"$PYTHON" run_coding_agent_replayer.py ca2 \
  --trace-a "$TRACE_A" \
  --trace-b "$TRACE_B" \
  --port "$PORT" \
  --output "$OUTDIR/ca2.json" >>"$OUTDIR/run.log" 2>&1

if [[ "$REQUIRE_OFFLOAD_WITNESS" == "1" ]]; then
  "$PYTHON" - <<'PY' "$OUTDIR/ca2.json"
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    data = json.load(f)
witnessed = data.get("offload_witness", {}).get("witnessed")
if not witnessed:
    raise SystemExit(f"offload witness missing: {path}")
print(f"offload witness ok: {path}")
PY
fi

echo "[launch] done -> $OUTDIR/ca2.json"
