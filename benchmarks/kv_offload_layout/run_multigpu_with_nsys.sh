#!/usr/bin/env bash
set -euo pipefail

OUTDIR="${1:-benchmarks/kv_offload_layout/results/nsys}"
shift || true

mkdir -p "$OUTDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
NSYS_BASENAME="$OUTDIR/multigpu_${STAMP}"
BENCH_OUT="${NSYS_BASENAME}.txt"

NSYS_BIN="${NSYS_BIN:-nsys}"
TRACE_SET="${TRACE_SET:-cuda,nvtx,osrt}"
CUDA_GRAPH_TRACE="${CUDA_GRAPH_TRACE:-node}"
FORCE_OVERWRITE="${FORCE_OVERWRITE:-true}"

echo "nsys_rep=${NSYS_BASENAME}.nsys-rep"
echo "bench_out=${BENCH_OUT}"

"$NSYS_BIN" profile \
  --trace="$TRACE_SET" \
  --cuda-graph-trace="$CUDA_GRAPH_TRACE" \
  --force-overwrite="$FORCE_OVERWRITE" \
  --output="$NSYS_BASENAME" \
  .venv/bin/python benchmarks/kv_offload_layout/benchmark_multigpu_batch_lock.py "$@" \
  | tee "$BENCH_OUT"

echo "nsys_stats_hint: nsys stats ${NSYS_BASENAME}.nsys-rep"
