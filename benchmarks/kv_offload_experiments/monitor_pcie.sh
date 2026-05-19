#!/usr/bin/env bash
# Record GPU PCIe and utilization during benchmark.
# Usage: ./monitor_pcie.sh [gpu_id] [output_csv]
GPU="${1:-0}"
OUT="${2:-pcie_$(date +%Y%m%d_%H%M%S).csv}"
echo "Logging GPU $GPU to $OUT (Ctrl+C to stop)"
exec nvidia-smi dmon -i "$GPU" -s pucvmet -d 1 -f "$OUT"
