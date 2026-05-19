# KV Cache Offload 实验脚本

配合文档：[docs/features/kv_cache_offload_experiments.md](../../docs/features/kv_cache_offload_experiments.md) · [CONTEXT.md](../../docs/features/CONTEXT.md) · [Streaming KV RFC](../../docs/features/streaming_kv_management_rfc.md)

## 实验结构

| 类别 | 场景 | 脚本 |
|----|------|------|
| **Prereq / Gate** | S1：64K T0→T1→T2，验证 store/load | `run_s1_sequence.sh` |
| **Mainline** | **CA2**：100K task swap | `gen_coding_agent_trace.py` + `run_coding_agent_replayer.py ca2` |
| **Appendix** | CA1 / CA3 / CA4 | 同上 |

`Prereq` 通过后再跑 `Mainline`。结果不混表。

## 前置

```bash
cd /data1/lmy/vllm
source .venv/bin/activate
```

## 文件

| 文件 | 作用 |
|------|------|
| `run_server.sh` | 启动 server；profile `short`（16K）/ `l1`（64K）/ `ca`（100K+ B-PA）/ `ca-l3`（100K+ B-L3 大 block 对照） |
| `run_s1_sequence.sh` | **Layer 0**：S1 顺序 T0→T1→T2 |
| `gen_coding_agent_trace.py` | **Mainline / Appendix**：生成 CA trace JSONL（真实 tokenizer） |
| `run_coding_agent_replayer.py` | **Mainline / Appendix**：CA 场景 driver |
| `gen_shared_prefix_dataset.py` | S2 多租户 64K system（遗留，非 CA 主路径） |
| `run_bench.sh` | 短上下文 w1/w2/w3（遗留，非主路径） |
| `monitor_pcie.sh` | 后台 `nvidia-smi dmon` |
| `summarize_results.py` | 汇总 bench / replayer JSON |
| `launch_ca2_bpa.sh` | **Mainline / CA2**：task swap baseline，要求结果满足主表 Offload evidence 口径 |

### Mainline 三组对照（B-PA / sat-BS large-block baseline / B-L1）

当前主线只保留 `CA2`，但仍需三组对照：B-PA（vLLM 默认）、**sat-BS large-block baseline**（PA + 大 block，block size 来自 SE1 饱和点）、B-L1（RFC PoC，独立 runtime）。详见实验指南 §11。

## 快速开始

### Layer 0（64K S1）

```bash
# 终端 A
./run_server.sh simple /data1/models/Qwen/Qwen3-4B-Instruct-2507 8000 24 l1

# 终端 B
./run_s1_sequence.sh 8000 results/s1
```

### Mainline（100K CA2 task swap）

```bash
# 生成 traces
.venv/bin/python gen_coding_agent_trace.py \
  --model /data1/models/Qwen/Qwen3-4B-Instruct-2507 \
  --task-id ca2-task-a --prefix-tokens 102400 --num-rollouts 1 \
  -o traces/ca2_task_a.jsonl

.venv/bin/python gen_coding_agent_trace.py \
  --model /data1/models/Qwen/Qwen3-4B-Instruct-2507 \
  --task-id ca2-task-b --prefix-tokens 102400 --num-rollouts 1 \
  -o traces/ca2_task_b.jsonl

# B-PA
TRACE_A=traces/ca2_task_a.jsonl TRACE_B=traces/ca2_task_b.jsonl \
  OUTDIR=results/ca2_bpa SERVER_WAIT_S=1800 ./launch_ca2_bpa.sh
```

要进入主表，`CA2` 结果必须同时给出：

- `gpu_to_cpu` **Exact bytes**
- `cpu_to_gpu` **Exact bytes**
- `Exact tokens` 可选，作为辅证
