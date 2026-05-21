#!/usr/bin/env bash
set -euo pipefail

OUTROOT="${1:-benchmarks/kv_offload_layout/results/common_models}"
mkdir -p "$OUTROOT"

# Derived from official Hugging Face config.json files.
# bytes_per_block = 2 (K,V) * num_key_value_heads * head_dim * block_size(16) * bf16(2B)
MODELS=(
  "qwen3_4b|Qwen/Qwen3-4B|36|65536"
  "qwen3_8b|Qwen/Qwen3-8B|36|65536"
  "mistral_7b_v03|mistralai/Mistral-7B-v0.3|32|65536"
  "qwen2_5_7b_instruct|Qwen/Qwen2.5-7B-Instruct|28|32768"
  "deepseek_r1_distill_qwen_7b|deepseek-ai/DeepSeek-R1-Distill-Qwen-7B|28|32768"
  "qwen2_5_14b_instruct|Qwen/Qwen2.5-14B-Instruct|48|65536"
)

for entry in "${MODELS[@]}"; do
  IFS="|" read -r slug model_id num_layers bytes_per_block <<<"$entry"
  outdir="$OUTROOT/$slug"
  mkdir -p "$outdir"
  echo "== $slug =="
  echo "model_id=$model_id num_layers=$num_layers bytes_per_block=$bytes_per_block"
  bash benchmarks/kv_offload_layout/run_with_pcie_monitor.sh \
    "$outdir" \
    --num-layers "$num_layers" \
    --num-blocks 256 \
    --bytes-per-block "$bytes_per_block" \
    --repeat 20
done
