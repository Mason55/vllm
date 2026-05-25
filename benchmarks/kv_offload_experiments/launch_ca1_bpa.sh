#!/usr/bin/env bash
# Launch CA1 B-PA: server (ca profile) + replayer. Safe to re-run.
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
BLOCK_SIZE="${BLOCK_SIZE:-}"
KV_GIB="${KV_GIB:-32}"
OUTDIR="${OUTDIR:-results/ca1_bpa}"
CONCURRENCY="${CONCURRENCY:-16}"
SERVER_WAIT_S="${SERVER_WAIT_S:-1800}"
PYTHON="${PYTHON:-/data1/anaconda3/envs/py312/bin/python}"
SAMPLE_GPU="${SAMPLE_GPU:-0}"
SAMPLE_METRICS="${SAMPLE_METRICS:-0}"
FANOUT_MODE="${FANOUT_MODE:-cold}"
WAIT_REPLAYER="${WAIT_REPLAYER:-0}"

mkdir -p "$OUTDIR"
ulimit -l unlimited 2>/dev/null || true

pkill -9 -f "vllm serve.*${PORT}" 2>/dev/null || true
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
pkill -9 -f "run_coding_agent_replayer.py ca1" 2>/dev/null || true
sleep 2

: > "$OUTDIR/server.log"
SERVER_CMD=(./run_server.sh simple "$MODEL" "$PORT" "$KV_GIB" "$PROFILE")
if [[ -n "$BLOCK_SIZE" ]]; then
  SERVER_CMD+=("$BLOCK_SIZE")
fi
nohup "${SERVER_CMD[@]}" >>"$OUTDIR/server.log" 2>&1 &
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
CMD=("$PYTHON" run_coding_agent_replayer.py ca1 \
  --trace traces/ca_100k_16r.jsonl \
  --port "$PORT" \
  --concurrency "$CONCURRENCY" \
  --fanout-mode "$FANOUT_MODE" \
  --output "$OUTDIR/ca1.json")
if [[ "$SAMPLE_METRICS" == "1" ]]; then
  CMD+=(--sample-metrics)
elif [[ "$SAMPLE_GPU" == "1" ]]; then
  CMD+=(--sample-gpu)
fi
nohup "${CMD[@]}" >>"$OUTDIR/run.log" 2>&1 &
echo $! >"$OUTDIR/replayer.pid"
echo "[launch] replayer pid=$(cat "$OUTDIR/replayer.pid") concurrency=$CONCURRENCY mode=$FANOUT_MODE profile=$PROFILE block_size=${BLOCK_SIZE:-default}"
echo "[launch] tail -f $OUTDIR/run.log"

if [[ "$WAIT_REPLAYER" == "1" ]]; then
  REPLAYER_PID="$(cat "$OUTDIR/replayer.pid")"
  while kill -0 "$REPLAYER_PID" 2>/dev/null; do
    sleep 5
  done
  wait "$REPLAYER_PID"
fi
