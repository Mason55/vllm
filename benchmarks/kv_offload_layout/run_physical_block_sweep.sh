#!/usr/bin/env bash
set -euo pipefail

OUTROOT="${1:-benchmarks/kv_offload_layout/results/physical_block_sweep}"
mkdir -p "$OUTROOT"

NUM_LAYERS="${NUM_LAYERS:-36}"
TOTAL_MIB="${TOTAL_MIB:-576}"
PATTERN="${PATTERN:-random}"
REPEAT="${REPEAT:-20}"
WARMUP="${WARMUP:-5}"

TOTAL_BYTES=$((TOTAL_MIB * 1024 * 1024))

# Blog-inspired effective physical block sizes.
# Keep total bytes fixed. Sweep bytes_per_block, solve num_blocks.
BLOCK_SIZES=(
  "65536|64_kib_baseline"
  "524288|512_kib"
  "1048576|1_mib"
  "2097152|2_mib"
)

echo "num_layers=$NUM_LAYERS total_mib=$TOTAL_MIB pattern=$PATTERN repeat=$REPEAT warmup=$WARMUP"

for entry in "${BLOCK_SIZES[@]}"; do
  IFS="|" read -r bytes_per_block slug <<<"$entry"
  denom=$((NUM_LAYERS * bytes_per_block))
  if (( TOTAL_BYTES % denom != 0 )); then
    echo "skip $slug: total_bytes=$TOTAL_BYTES not divisible by num_layers*bytes_per_block=$denom"
    continue
  fi
  num_blocks=$((TOTAL_BYTES / denom))
  outdir="$OUTROOT/$slug"
  mkdir -p "$outdir"
  echo "== $slug =="
  echo "bytes_per_block=$bytes_per_block num_blocks=$num_blocks total_bytes=$TOTAL_BYTES"
  bash benchmarks/kv_offload_layout/run_with_pcie_monitor.sh \
    "$outdir" \
    --num-layers "$NUM_LAYERS" \
    --num-blocks "$num_blocks" \
    --bytes-per-block "$bytes_per_block" \
    --pattern "$PATTERN" \
    --warmup "$WARMUP" \
    --repeat "$REPEAT"
done
