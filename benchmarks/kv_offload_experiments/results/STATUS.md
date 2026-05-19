# KV Offload 实验状态（压缩 handoff）

> 计划：[kv_cache_offload_experiments.md](../../../docs/features/kv_cache_offload_experiments.md) · RFC：[streaming_kv_management_rfc.md](../../../docs/features/streaming_kv_management_rfc.md)

## 环境

| 项 | 值 |
|----|-----|
| Python | `/data1/anaconda3/envs/py312/bin/python` |
| vLLM | 0.21.0rc3 editable `/data1/lmy/vllm` |
| 模型 | `/data1/models/Qwen/Qwen3-4B-Instruct-2507` |
| GPU | RTX 3090 ×7，`CUDA_VISIBLE_DEVICES=0` |
| 驱动 | **580.126.09**（原 550→Simple segfault，已修） |
| Server 前置 | `ulimit -l unlimited`（pin 24G CPU KV 必需） |
| 安装脚本 | `install_py312.sh`（清华镜像+长 timeout+本地 wheel） |

## Layer 0 进度（S1 @ 65500 tok, profile `l1`）

| ID | 目录 | Backend | T0 | T1 | T2 | 门禁 |
|----|------|---------|----|----|-----|------|
| E1+E2 rerun | `s1_l1_rerun_20260520_102558_logged/` | A1 simple, CPU 24GiB | 27.6s | 28.1s | **1.07s** | ✅ E1/E2（`server.log` + `metrics_after.prom`；T0/T2 25.75x） |
| E1+E2 | `s1_l1/` | A1 simple, CPU 24GiB | 27.8s | 28.2s | **1.05s** | ✅ E2（T2≪T0 ~26×） |
| E3 | `s1_l1_e3/` | A0 baseline, 无 offload | 27.8s | 28.0s | 20.8s | ⚠️ 无 preempt/OOM（顺序请求，KV peak ~60%） |
| smoke | `s1/` | simple, profile short 14K | 2.9s | 2.5s | 0.10s | 驱动升级验证 |
| 废弃 | `s1_64k_segfault/` | 旧驱动 550 | T0 only | — | — | segfault |

**当前 Layer 0 结论以 `s1_l1_rerun_20260520_102558_logged/` + `s1_l1_e3/` 为准；旧 `s1_l1/` 保留作历史对照。**

rerun 关键证据：`external_prefix_cache_hits_total=42336`，`external_kv_transfer_tokens=42336`，`prompt_tokens_cached_total=65504`。

E3 未触发：`num_preemptions_total=0`，无 OOM。若要补信号→调紧 GPU 池或并发 T0+T1（§4.2）。

## Layer 1

未开始。无 `traces/`，无 CA JSON。B-L1 PoC 未建。

## 常用命令

```bash
export PATH=/data1/anaconda3/envs/py312/bin:$PATH
export PYTHON=/data1/anaconda3/envs/py312/bin/python
ulimit -l unlimited

# E1/E2
./run_server.sh simple $MODEL 8000 24 l1
./run_s1_sequence.sh 8000 results/s1_l1_rerun_20260520_102558_logged $MODEL 65500

# E3
./run_server.sh baseline $MODEL 8000 0 l1
# 同 prompts（已 cp 到 s1_l1_e3/prompts）
```

## 代码改动（本会话）

- `gen_coding_agent_trace.py`：`_pad_text_to_tokens` O(N²)→批量 tokenize（64K prompt 6s）
- `install_py312.sh`：镜像/timeout/no-build-isolation/本地 wheel

## 下一步

1. （可选）E3 重跑：压 GPU 池或并发
2. Layer 1：gen trace 100K×16 → CA1 B-PA / B-L3
3. SE1 block-size scan
4. B-L1 mini-runtime（未实现）
