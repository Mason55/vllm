#!/usr/bin/env bash
# Install vLLM into conda py312 with settings tuned for slow/large wheel downloads.
set -euo pipefail

PY312=/data1/anaconda3/envs/py312/bin/python
REPO=/data1/lmy/vllm
LOG="${REPO}/benchmarks/kv_offload_experiments/results/install_py312.log"

# uv defaults: connect 10s, read 30s — too short for 500MB+ nvidia wheels.
export UV_HTTP_TIMEOUT=600
export UV_HTTP_CONNECT_TIMEOUT=60
export UV_HTTP_RETRIES=10
export UV_CONCURRENT_DOWNLOADS=2
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

cd "${REPO}"

echo "=== $(date -Is) stage 1: torch 2.11 ===" | tee -a "${LOG}"
uv pip install "torch==2.11.0" --python "${PY312}" --torch-backend=auto -v 2>&1 | tee -a "${LOG}"

echo "=== $(date -Is) stage 2: vllm editable (precompiled) ===" | tee -a "${LOG}"
# Skip slow/fragile git version fetch (ambiguous v0.21.0rc3 tag on this repo).
export SETUPTOOLS_SCM_PRETEND_VERSION=0.21.0rc3
uv pip install setuptools-scm cmake ninja packaging 'setuptools>=77,<81' wheel jinja2 vcs-versioning \
  --python "${PY312}" 2>&1 | tee -a "${LOG}"
VLLM_USE_PRECOMPILED=1 uv pip install -e . --no-build-isolation \
  --python "${PY312}" --torch-backend=auto -v 2>&1 | tee -a "${LOG}"

echo "=== $(date -Is) verify ===" | tee -a "${LOG}"
"${PY312}" -c "import torch, vllm; print('torch', torch.__version__); print('vllm', vllm.__version__)" 2>&1 | tee -a "${LOG}"
