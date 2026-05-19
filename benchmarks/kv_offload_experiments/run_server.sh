#!/usr/bin/env bash
# Start vLLM OpenAI server for KV offload experiments.
# Usage: ./run_server.sh <baseline|simple|native> <model_path> [port] [kv_gib] [profile] [block_size]
# profile: short (16K, default) | l1 (64K Layer 0) | ca (100K+ Layer 1 Coding Agent / B-PA)
#        | ca-l3 (100K+ Layer 1, large PA block — B-L3 control group for the RFC)
# block_size: overrides PA --block-size; must be % 16 == 0 (vLLM constraint).
#             ca-l3 default 1024 (PA + large block, isolates block-size contribution).
set -euo pipefail

MODE="${1:?mode: baseline|simple|native}"
MODEL="${2:?model path}"
PORT="${3:-8000}"
KV_GIB="${4:-16}"
PROFILE="${5:-short}"
BLOCK_SIZE="${6:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Always use py312 editable vLLM (0.21.0rc3); system /usr vllm 0.17 breaks SimpleCPUOffload.
export PATH="/data1/anaconda3/envs/py312/bin:${PATH}"
VLLM_BIN="${VLLM_BIN:-/data1/anaconda3/envs/py312/bin/vllm}"
# Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True with SimpleCPUOffload
# (invalidates pinned KV memory; vLLM 0.21 rejects at startup).

case "$PROFILE" in
  short)
    MAX_LEN=16384
    MAX_SEQS=64
    GPU_UTIL=0.88
    ;;
  l1)
    MAX_LEN=65536
    MAX_SEQS=2
    GPU_UTIL=0.92
    KV_GIB="${KV_GIB:-24}"
    ;;
  ca)
    # B-PA: vLLM default block_size (Layer 1 Coding Agent baseline).
    # 131072 fails startup on 24GB 3090: vLLM requires GPU KV for max_model_len
    # even with CPU offload (~13 GiB avail vs ~14 GiB for 100K+).
    # max-model-len must exceed 100K prefix + tool deltas (~102700+212).
    # On v0.21.0rc3 + cudagraph memory profiling, 0.96 can miss by ~0.3 GiB;
    # 0.965 restores enough KV headroom for 104448 on 3090.
    MAX_LEN=104448
    MAX_SEQS=16
    GPU_UTIL=0.965
    MAX_BATCHED=4096
    KV_GIB="${KV_GIB:-32}"
    ;;
  ca-l3)
    # B-L3: PA + large block (RFC §4.1.3 control group; isolates block-size win).
    MAX_LEN=104448
    MAX_SEQS=16
    GPU_UTIL=0.965
    MAX_BATCHED=4096
    KV_GIB="${KV_GIB:-32}"
    BLOCK_SIZE="${BLOCK_SIZE:-1024}"
    ;;
  *)
    echo "Unknown profile: $PROFILE (short|l1|ca|ca-l3)" >&2
    exit 1
    ;;
esac

COMMON=(
  --host 0.0.0.0
  --port "$PORT"
  --enable-prefix-caching
  --max-model-len "$MAX_LEN"
  --gpu-memory-utilization "$GPU_UTIL"
  --max-num-seqs "$MAX_SEQS"
  --max-num-batched-tokens "${MAX_BATCHED:-8192}"
)

if [[ -n "$BLOCK_SIZE" ]]; then
  if (( BLOCK_SIZE % 16 != 0 )); then
    echo "[server] block_size=$BLOCK_SIZE must be a multiple of 16" >&2
    exit 1
  fi
  COMMON+=(--block-size "$BLOCK_SIZE")
fi

echo "[server] vllm=$("$VLLM_BIN" --version 2>/dev/null | head -1 || echo unknown)"
echo "[server] profile=$PROFILE max-model-len=$MAX_LEN max-num-seqs=$MAX_SEQS kv_gib=$KV_GIB block_size=${BLOCK_SIZE:-default(16)} batched=${MAX_BATCHED:-8192}"

case "$MODE" in
  baseline)
    echo "[server] baseline (GPU KV only), model=$MODEL port=$PORT"
    exec "$VLLM_BIN" serve "$MODEL" "${COMMON[@]}"
    ;;
  simple)
    echo "[server] SimpleCPUOffload, kv-offloading-size=${KV_GIB}GiB"
    export VLLM_USE_SIMPLE_KV_OFFLOAD=1
    exec "$VLLM_BIN" serve "$MODEL" "${COMMON[@]}" --kv-offloading-size "$KV_GIB"
    ;;
  native)
    echo "[server] native OffloadingConnector, kv-offloading-size=${KV_GIB}GiB"
    unset VLLM_USE_SIMPLE_KV_OFFLOAD
    exec "$VLLM_BIN" serve "$MODEL" "${COMMON[@]}" --kv-offloading-size "$KV_GIB" --disable-hybrid-kv-cache-manager
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 1
    ;;
esac
