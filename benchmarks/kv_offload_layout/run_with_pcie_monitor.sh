#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
OUTDIR="${1:-benchmarks/kv_offload_layout/results}"
shift || true

mkdir -p "$OUTDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
BENCH_OUT="$OUTDIR/bench_${STAMP}.txt"
PCIE_CSV="$OUTDIR/pcie_${STAMP}.csv"

cleanup() {
  if [[ -n "${DMON_PID:-}" ]]; then
    kill "$DMON_PID" >/dev/null 2>&1 || true
    wait "$DMON_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT

echo "pcie_csv=$PCIE_CSV"
echo "bench_out=$BENCH_OUT"

nvidia-smi dmon -i "$GPU" -s t -d 1 --format csv,nounit -o DT -f "$PCIE_CSV" &
DMON_PID=$!
sleep 1

.venv/bin/python benchmarks/kv_offload_layout/benchmark_a_vs_c.py "$@" | tee "$BENCH_OUT"

cleanup
unset DMON_PID

.venv/bin/python benchmarks/kv_offload_layout/summarize_pcie_csv.py "$PCIE_CSV"
