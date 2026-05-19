# KV Cache Offload 实验指南

> 当前文档按 **Prereq / Mainline / Appendix** 组织：`Prereq` 先验证机制与 telemetry，`Mainline` 只保留 RFC 主表最小闭环，`Appendix` 收纳探索性场景。
> 配套脚本：[`benchmarks/kv_offload_experiments/`](../../benchmarks/kv_offload_experiments/README.md)
> 术语：[CONTEXT.md](./CONTEXT.md) · 原理：[KV Cache Offload 使用与实现指南](kv_cache_offload_guide.md)

**范围声明**：Layer 0 聚焦超长单文档（64K）；Layer 1 聚焦 agentic-RL（100K 共享 prefix + rollout）。短上下文（≤16K）不在范围内。

---

## 0. 实验结构（必读）

| 类别 | 回答的问题 | 场景 | 脚本 | 角色 |
|----|-----------|------|------|-----------|
| **Prereq / Gate** | backend + telemetry 是否可用？offload 机制是否发生？ | S1：T0→T1→T2（64K） | `run_s1_sequence.sh` | 进入 Mainline 前置 |
| **Mainline** | task swap / offload cycle 的成本是否被量化，且 B-L1 是否优于对照组？ | **CA2**：100K task swap | `gen_coding_agent_trace.py` + `run_coding_agent_replayer.py ca2` | RFC 主表最小闭环 |
| **Appendix** | fan-out、delta、multi-task 等探索性或后续场景 | CA1 / CA3 / CA4 | trace replayer | 不阻塞 Mainline |

```text
Prereq S1 (mechanism + telemetry)  ──▶  Mainline CA2 (task swap, bidirectional exact bytes)
                                └──▶  Appendix (CA1 / CA3 / CA4, later)
```

`Prereq`、`Mainline`、`Appendix` **不混表**。

**环境**：单机 RTX 3090（24GB）、251GB 主机内存、模型目录 `/data1/models/Qwen/`。

### 门槛表述（统一）

| 类别 | 性质 | 门槛 | 未通过时 |
|----|------|------|----------|
| **Prereq / Gate** | 机制验证 + telemetry 验收 | **E1/E2 必须有 store/load/reuse 信号**；若要进入 Mainline，还必须确认所选 backend 能给 **Bidirectional exact bytes** | 无信号则暂停 Mainline；只剩 token / estimated bytes 的 backend 只能进 Appendix |
| **Mainline** | RFC 主表最小闭环 | **CA2** 必须同时提交 `gpu_to_cpu` 与 `cpu_to_gpu` 的 **Exact bytes**，并对比 `B-PA`、`sat-BS large-block baseline`、`B-L1` | 不满足 exact-bytes 或 task swap 量化条件，则不进入主表 |
| **Appendix** | 探索性 / 后续阶段 | 不设 kill 阈值；用于保留 CA1 / CA3 / CA4 与历史对照 | 不阻塞 Mainline |

**v0 store 模式**：**Eager**（默认，不加 `lazy_offload`）。Lazy 仅 v1 T3 对照。

---

## 1. 「超长序列」在本实验中的定义

| 量级 | token 数 | Qwen3-4B 单条 KV | 单卡 3090 能否装下完整 KV | 本实验角色 |
|------|----------|------------------|---------------------------|------------|
| **L1** | 64K | ~9 GiB | 能（GPU KV 池 ~13 GiB） | **Layer 0 主测** |
| **CA-L1** | 100K | ~14 GiB | **不能**（必须 offload） | **Layer 1 主测（固定 100K，不做 64K 过渡）** |
| **L2** | 128K | ~18 GiB | 不能（需 TP=2 或更小模型） | 扩展目标 |
| L3 | 256K+ | ~36 GiB+ | 不能 | 暂不覆盖 |

主模型：`Qwen3-4B-Instruct-2507`（`max_position_embeddings=262144`，64K/128K 均在原生上限内，**不必开 YaRN**）。

单 token KV（bf16，36 层 × 8 heads × 128 dim）：

```text
bytes_per_token ≈ 144 KiB
block_size=16  → 每 block（全层）≈ 2.25 MiB
64K  → 4096 blocks → ~9 GiB / 序列
128K → 8192 blocks → ~18 GiB / 序列
```

---

## 2. 超长序列下 offload 能测什么、不能测什么

Simple / Native offload 依赖 **prefix caching + block hash**。Store 时机取决于 **Eager / Lazy**（v0 用 Eager）：

| 路径 | Eager（v0 默认） | Lazy（v1 对照 B1） |
|------|------------------|-------------------|
| **Store（GPU→CPU）** | prefill **进行中**，已确认满块逐步 copy 到 CPU（GPU 仍保留副本） | 块 `ref_cnt=0` 进 `free_queue`、临近 evict 时才 store |
| **Load（CPU→GPU）** | 新请求 prefix hash 命中 CPU 池，且该 hash **不在 GPU** | 同左 |

**Load 前提**：T0 完成后 A 的块可能仍在 GPU → 直接 T2 可能纯 GPU prefix hit、不走 PCIe。**T1 必须**：用 prompt_B 挤掉 A 在 GPU 上的块，T2 才测真 Load。

v0（Eager）S1 时序：

```text
T0: prompt_A 64K prefill  →  Eager store 进行中（E1 盯这步）；测冷 TTFT
T1: prompt_B 64K          →  挤 GPU，清掉 A 的 GPU 驻留（非 v0 store 主观测点）
T2: prompt_A 再次         →  CPU load（E2）；TTFT 应显著低于 T0
```

> 实现参考：`manager.py` `_prepare_eager_store_specs`（Eager）；`_prepare_lazy_store_specs`（Lazy，走 `free_queue`）。Eager 有 TODO：请求最后一步结束的尾块可能漏 store。`cuda_mem_ops.copy_blocks` 用 `cuMemcpyBatchAsync` batch 提交 `num_blocks × num_layers` 条 memcpy。

---

## 3. 超长序列场景（只保留 3 类）

### S1 — 同一超长文档重复问答（主场景）

**用户故事**：一份 64K 长文档先 ingest 一次，之后多次基于同一文档提问。

| 步骤 | 请求 | 测什么（v0 Eager） |
|------|------|-------------------|
| T0 | prompt_A（64K 固定内容）+ 短 output | 冷 prefill TTFT；**Eager store**（E1 主观测） |
| T1 | prompt_B（64K 不同内容）+ 短 output | **挤 GPU**，清 A 的 GPU 驻留（为 T2 load 铺路） |
| T2 | prompt_A 再次 | **CPU load**（E2）→ TTFT vs T0 |

**参数要点**：

- `input_len=65536`，`output_len=8`（短 decode，让 prefill/offload 占主导）
- `max-num-seqs=2`（两条 64K 并发 ≈ 18 GiB KV 需求 > GPU 池 ~13 GiB）
- prompt 必须 **bit-exact 可复现**（固定 seed 或 JSONL）

### S2 — 多租户共享超长 system prompt

**用户故事**：K 个租户各有一份 64K system prompt，N 条短 user 问句。

- 用 `gen_shared_prefix_dataset.py` 生成 JSONL：`--dataset-name custom`
- 控制命中率：同一 system prompt 下多条 user 请求 → 测 **load 并发**
- 仍属超长序列（prefix 64K），不是短 prompt 压测

### S3 — 多轮对话累积到超长（扩展）

**用户故事**：同 session 多轮，上下文从 32K 累积到 64K+。

- turn1: 32K → turn2: 续接到 64K
- 每 turn 结束 store，下 turn 部分 load
- v0 先不做；S1 通过后再加

**v0 只做 S1**；S2/S3 在 E1/E2 通过后按需扩展。

---

## 4. 环境与 server 配置（64K / L1）

### 4.1 环境

```bash
cd /data1/lmy/vllm
uv venv --python 3.12 && source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

单卡：`CUDA_VISIBLE_DEVICES=0`（v0 不用 TP）。

### 4.2 Server（64K 超长序列专用）

固定：`--enable-prefix-caching`（Simple 未开会禁用 offload）。**v0 不加** `lazy_offload`（保持 Eager 默认）。

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /data1/models/Qwen/Qwen3-4B-Instruct-2507 \
  --enable-prefix-caching \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 8192 \
  --kv-offloading-size 24
  # v0: 不加 --kv-transfer-config lazy_offload
```

| 参数 | 值 | 为何与超长序列相关 |
|------|-----|-------------------|
| `max-model-len` | 65536 | 允许单条 64K prefill |
| `max-num-seqs` | 2 | 两条 64K 叠满 GPU KV 池，触发 store/evict |
| `kv-offloading-size` | 24 GiB | CPU 池可缓存 ~2.6 条完整 64K KV |
| `max-num-batched-tokens` | 8192 | chunked prefill，避免单 step OOM |

显存预算：

```text
24 GiB − 8 GiB(权重) − 2 GiB(workspace) ≈ 13 GiB GPU KV 池
64K 单条 ≈ 9 GiB  →  池内约 1.4 条  →  第二条 64K 必挤第一条
```

### 4.3 128K 扩展（L2，可选）

128K + 4B 单卡装不下（~18 GiB KV > ~13 GiB 池）。选项：

| 方案 | 配置 | 代价 |
|------|------|------|
| TP=2 | `tensor_parallel_size=2`，两卡各 ~9 GiB KV | store 需 rank 对齐，噪声大 |
| 换 1.7B | KV/token ~56 KiB，128K ~7 GiB | 非 4B 真实负载 |

128K 实验单独开一节记录，不与 64K 结果混表。

---

## 5. v0 门禁：证明超长序列下 offload 真的发生

**E1/E2 不过，不进入任何 Layer 1 或带宽对比。E3 是诊断项：用于说明 GPU-only baseline 是否真的被压紧；E3 无信号不单独阻断 Layer 1。**

Workload：**S1 固定 T0→T1→T2**（见 §3.1）。

### E1 — Store（Eager：盯 T0）

```text
T0: prompt_A (64K) prefill 进行中  →  Eager 逐步 store 到 CPU
```

| Backend | 判据 |
|---------|------|
| A1 Simple | **T0/T2 联合判定**：Simple connector 启动；T2 出现 external prefix hit；TTFT(T2) ≪ TTFT(T0) |
| A2 Native | **T0 期间** `/metrics`: `vllm:kv_offload_total_bytes{transfer_type="gpu_to_cpu"}` ↑ |

> Simple 的 `Store event ... completed` 多为 debug 日志，默认 info 日志可能看不到。E1 对 Simple 不只依赖 grep store log；T2 external hit + 大幅 TTFT 下降可反证 T0/T1 期间已有 CPU cache 可用。T1 不是 v0 E1 主观测点（T1 作用是挤 GPU，见 E2 前置条件）。

### E2 — Load（需 T1 前置）

```text
T1: prompt_B (64K)  →  挤掉 A 在 GPU 上的块
T2: prompt_A 再次   →  CPU prefix hit，load 回 GPU
```

| 信号 | 判据 |
|------|------|
| TTFT(T2) | ≪ TTFT(T0) |
| A1 | `External prefix cache hit rate` ↑；TTFT(T2) 明显低于 baseline T2 |
| A2 | `cpu_to_gpu` bytes ↑；`external_prefix_cache_hits` ↑ |

> 2026-05-20 latest rerun（`results/s1_l1_rerun_20260520_102558_logged/`）：T0=`27.61s`，T1=`28.06s`，T2=`1.07s`，T0/T2=`25.75x`；`external_prefix_cache_hits_total=42336`，`prompt_tokens_cached_total=65504`。结论：SimpleCPUOffload 在 64K S1 上通过 E1/E2，满足进入 Layer 1 的软门禁。

### E3 — GPU 池真紧（A0）

A0（无 `--kv-offloading-size`），同 S1 序列。E3 用于确认 baseline 压力，不是 Layer 1 硬门槛。

判据（其一）：`num_preemptions_total` ↑；或 `gpu_cache_usage_perc` ≈ 1.0；或 T1 OOM。

无信号 → 记录为「GPU-only baseline 未触发压力」；若必须证明抢占，再回 §4.2 调紧 `max-num-seqs` / `gpu-memory-utilization` 或改并发 T0/T1。

### v0 矩阵

| ID | Backend | 序列 | 主观测步 |
|----|---------|------|----------|
| E1 | A1 + pool 24G（Eager） | T0 | Store 于 T0 prefill |
| E2 | A1 + pool 24G（Eager） | T0→T1→T2 | Load 于 T2 |
| E3 | A0 | T0→T1→T2 | 抢占/OOM 于 T1（诊断项） |

每组 ≥3 次，取中位数；保存 bench JSON + `server.log` + metrics 快照。

---

## 6. v1 扩展（S1 有信号后，仍限超长序列）

所有 workload **固定 S1 或 S2**，只扫单轴：

| 表 | 扫什么 | 固定 |
|----|--------|------|
| **T1 On/Off** | A0 vs A1 | S1；64K；pool=24G |
| **T2 CPU 池** | `kv-offloading-size` 16/24/32/48 GiB | S1；A1 |
| **T3 Backend / Store** | A1 Eager / A2 Native / **B1 Lazy** | S1；pool=24G；Lazy 时 E1 观测点改 T1 |
| **T4 长度** | 32K / 64K /（可选 128K TP=2） | S1 序列；看 TTFT/load 字节随长度变化 |
| **T5 共享 prefix（S2）** | K=2/4 租户 × 64K system | A1；测 load 并发 |

Lazy store（**仅 v1 T3**，v0 不用）：

```bash
--kv-transfer-config '{"kv_connector_extra_config":{"lazy_offload":true}}'
```

---

## 7. 观测指标（超长序列语境）

### 7.1 必采信号

| 信号 | 超长序列含义 | A1 Simple | A2 Native |
|------|-------------|-----------|-----------|
| store bytes/blocks | T0 prefill 期间 Eager 搬了多少 KV | **T0 期间** log grep `Store event` | **T0 期间** `kv_offload_total_bytes{gpu_to_cpu}` |
| load bytes/blocks | 同一 64K 再请求时拉回多少 | log grep Load | `kv_offload_total_bytes{cpu_to_gpu}` |
| TTFT(T0) vs TTFT(T2) | 冷 64K prefill vs CPU hit 后 load | bench JSON | 同 |
| prefix hit | 64K hash 是否对齐 | log / metrics | `prefix_cache_hits`, `external_prefix_cache_hits` |
| preemption | GPU 装不下第二条 64K | `num_preemptions_total` | 同 |
| PCIe TX/RX | 64K store/load 有效带宽 | `monitor_pcie.sh` | 同 |

> A1 Simple **无** Prom offload counter（`manager.py` TODO）；必须 `tee server.log`。

### 7.2 metrics 采样

```bash
while true; do
  date -Is
  curl -s localhost:8000/metrics | grep -E 'kv_offload_total_bytes|prefix_cache|num_preemptions|gpu_cache_usage'
  sleep 1
done | tee metrics_snap.log
```

### 7.3 带宽估算（64K 量级）

```text
一次完整 64K store/load ≈ 4096 blocks × 2.25 MiB ≈ 9 GiB
PCIe 4.0 x16 峰值 ~25 GB/s；3090 实测 12–20 GB/s 已不错
有效带宽 ≈ 9 GiB / transfer_time
```

batch 内 memcpy 条数 = `4096 × 36 layers`。打不满原因见主指南 [§10.5](kv_cache_offload_guide.md#105-dma-带宽与局部性)。

### 7.4 探索性解读（超长序列）

| 观察 | 含义 |
|------|------|
| A0 T1 抢占/OOM，A1 完成 | offload 让「两条 64K 顺序服务」成为可能 |
| TTFT(T2) ≪ TTFT(T0) | 64K prefix 在 CPU，load 比全量 prefill 快 |
| TTFT(T2) 仍很高 | load 带宽或 block 粒度是瓶颈 |
| A1 全程无 store/load 计数 | 实验设计错（未走 S1 序列或未开 prefix caching） |

---

## 8. 执行流程

```bash
cd /data1/lmy/vllm/benchmarks/kv_offload_experiments
export PATH=/data1/anaconda3/envs/py312/bin:$PATH
export PYTHON=/data1/anaconda3/envs/py312/bin/python

# 终端 A：server（64K 参数见 §4.2）
CUDA_VISIBLE_DEVICES=0 VLLM_USE_SIMPLE_KV_OFFLOAD=1 \
  vllm serve /data1/models/Qwen/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 --port 8000 \
  --enable-prefix-caching \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 8192 \
  --kv-offloading-size 24 \
  2>&1 | tee server.log

# 终端 B：metrics + 可选 PCIe
# ./monitor_pcie.sh 0 pcie_s1.csv

# 终端 C：S1 序列 T0→T1→T2
./run_s1_sequence.sh 8000 results/s1
# （不要用 run_bench.sh w1/w2/w3 — ≤8K 短上下文，非 Layer 0 主路径）
```

建议顺序：

```text
Day1  E1/E2/E3（S1，64K，A1）— 确认超长序列下 store/load/抢占有信号
Day2  T1（A0 vs A1）+ T4（32K vs 64K 长度曲线）
Day3  T2/T3 或 T5（S2 共享 64K system prompt）+ PCIe/nsys
```

---

## 9. 常见坑（超长序列专用）

1. **只跑一条 64K** → Eager 下 T0 能测 store，但测不到 load；必须 T0→T1→T2。
2. **跳过 T1 直接 T2** → T0 块可能还在 GPU，T2 纯 GPU hit，E2 假阴性。
3. **把 T1 当 E1 观测点** → v0 Eager 下 store 主要在 T0，T1 只负责挤 GPU。
4. **prompt 不可复现** → T2 hash miss，表现为「同一文档却不 load」。
5. **block 差 1 token** → 整 block（16 token）miss；64K 下影响被放大。
6. **未开 prefix caching** → Simple offload 禁用。
7. **v0 误开 lazy_offload** → store 观测点变 T1，与 E1 设计不一致。
8. **混用 `--cpu-offload-gb`（权重）与 `--kv-offloading-size`（KV 池）**。
9. **用 run_bench.sh w1/w2/w3** → ≤8K 短上下文，非超长序列。
10. **128K + 4B 单卡** → 第一次 prefill 装不下；勿与 64K 混比。
11. **A1 只看 /metrics** → offload 计数在 log。

---

## 10. 对照组速查

| 编号 | 名称 | 关键参数 |
|------|------|----------|
| **A0** | 基线（仅 GPU KV） | 不设 `--kv-offloading-size` |
| **A1** | Simple offload | `VLLM_USE_SIMPLE_KV_OFFLOAD=1` + `--kv-offloading-size N` |
| **A2** | Native offload | 仅 `--kv-offloading-size N` |
| **B1** | Lazy store | A1 + `lazy_offload=true` |

---

## 11. Mainline — CA2 Task Swap

> 对齐 [Streaming KV RFC §6](./streaming_kv_management_rfc.md) 的 **Phase 1**：先验证 task swap / offload cycle 假设，不假装一次性完成 RFC 全部四维验收。
> **前置**：Prereq S1 已通过；且所选 backend 已确认能提供 **Bidirectional exact bytes**。

### 11.1 Mainline 只保留 CA2

当前阶段：

- **CA2** 进入 `Mainline`
- **CA1**、**CA3**、**CA4** 降到 `Appendix`
- `Mainline` 不再被 fan-out / delta / multi-task 容量实验阻塞

### 11.2 三组对照（B-PA / sat-BS large-block baseline / B-L1）

`CA2` 固定跑三组配置，对齐 [RFC §4.1.3 / §6](./streaming_kv_management_rfc.md)：

| 配置 | 实现 | block size | 启动方式 | 角色 |
|------|------|-----------|----------|------|
| **B-PA** | vLLM PA + offload backend | 16（默认） | `run_server.sh <mode> <model> 8000 32 ca` | 当前 baseline |
| **sat-BS large-block baseline** | 同上，仅 `--block-size` 增大 | **SE1 扫描得到的饱和点** | `run_server.sh <mode> <model> 8000 32 ca-l3 <sat_bs>` | **隔离 block 大小贡献** |
| **B-L1** | RFC PoC mini-runtime | arena 整段 | （PoC 阶段独立 runtime） | 本 RFC 提案 |

说明：

- `sat-BS large-block baseline` 是 `B-L3` 的规范叫法
- 不再把 `1024` 写成定义；`1024` 只能是候选值
- 若 backend 没有 **Bidirectional exact bytes**，该 backend 只能进 `Appendix`

### 11.3 Block-size 扫描子实验（SE1）

`sat-BS large-block baseline` 先由 SE1 决定，再进入 `Mainline`：

对每个 `BS ∈ {16,128,256,512,1024,2048}`，重复下面模板：

```bash
export PATH=/data1/anaconda3/envs/py312/bin:$PATH
export PYTHON=/data1/anaconda3/envs/py312/bin/python

# 终端 A：启动 server
./run_server.sh simple /data1/models/Qwen/Qwen3-4B-Instruct-2507 8000 32 ca-l3 <BS>

# 终端 B：运行 replayer
mkdir -p results/se1_bs<BS>
$PYTHON run_coding_agent_replayer.py ca2 \
  --trace-a traces/ca2_task_a.jsonl \
  --trace-b traces/ca2_task_b.jsonl \
  --port 8000 \
  --output results/se1_bs<BS>/ca2.json
```

全部 block size 跑完后再汇总：

```bash
$PYTHON summarize_results.py results/se1_bs*/ca2.json
```

测量：

- `gpu_to_cpu` exact bytes
- `cpu_to_gpu` exact bytes
- task swap wall-clock
- PCIe 利用率

找到收益**饱和点**后，才把该 block size 升格为 `sat-BS large-block baseline`。

### Trace 格式

由 `gen_coding_agent_trace.py` 生成（**真实 tokenizer 计数**，bit-exact 可复现）：

```jsonc
{
  "task_id": "swe-bench-django-12345",
  "prefix_tokens": 102400,
  "prefix_messages": [
    {"role": "system", "content": "... 仓库上下文 ..."},
    {"role": "user", "content": "... 任务描述 ..."}
  ],
  "rollouts": [
    {
      "rollout_id": 0,
      "steps": [
        {"type": "decode", "tokens": 312},
        {"type": "tool_call", "name": "grep", "input_tokens": 18, "output_tokens": 2048},
        {"type": "decode", "tokens": 156},
        {"type": "tool_call", "name": "read_file", "input_tokens": 22, "output_tokens": 4096}
      ]
    }
  ]
}
```

- `prefix_messages`：所有 rollout **完全相同**；对应 RFC **Prefix Arena**
- `steps`：rollout 私有增量；对应 RFC **Delta Buffer**
- `tool_call`：`input_tokens` = 模型发出 tool 参数；`output_tokens` = tool observation 写入 context

### CA2 场景定义（RFC Regime A / 任务切换成本）

**用户故事**：Task A（100K prefix + rollout）→ Task B（不同 100K prefix，挤 GPU）→ 回到 Task A 继续 rollout。

| 步骤 | 动作 |
|------|------|
| T0 | Task A：prefill prefix + rollout 0 decode |
| T1 | Task B：prefill 不同 prefix（挤 GPU / 触发 evict） |
| T2 | Task A：同一 prefix 再次 prefill/续 rollout |

**测什么**：

- T2 TTFT vs T0（load 是否生效）
- Task 切换 wall-clock（T1 结束 → T2 首 token）
- PCIe 带宽（`monitor_pcie.sh`）
- **Offload evidence required**
  - `gpu_to_cpu` **Exact bytes**
  - `cpu_to_gpu` **Exact bytes**
  - `Exact tokens` 可作 load-side 辅证，但不能代替 bytes
- **RFC 关切**：Phase 1 的 bulk swap / offload-cycle 成本 baseline

当前脚本若仍输出 `offload_witness` 字段名，主表语义按 **Offload evidence** 解读。

```bash
./run_coding_agent_replayer.py ca2 --trace-a traces/task_a.jsonl --trace-b traces/task_b.jsonl
```

### Appendix（当前不实现）

以下场景保留在文档中，但当前阶段**不阻塞 Mainline**：

| 场景 | 当前定位 | 未来作用 |
|------|----------|----------|
| **CA1** | Appendix only；当前不实现 | shared-prefix / fan-out / cold-warm sharing |
| **CA3** | Appendix only；当前不实现 | delta / decode regression |
| **CA4** | Appendix only；保留、后续实现 | GPU 内存有效利用 / active task capacity |

### Server 配置（CA-L1，100K prefix）

与 Layer 0 的 64K 配置不同；100K + 4B 单条约 14 GiB KV，需更紧的 GPU 池：

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_SIMPLE_KV_OFFLOAD=1 \
VLLM_ENGINE_READY_TIMEOUT_S=1800 VLLM_RPC_TIMEOUT=1800000 \
  vllm serve /data1/models/Qwen/Qwen3-4B-Instruct-2507 \
  --enable-prefix-caching \
  --max-model-len 104448 \
  --gpu-memory-utilization 0.96 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 4096 \
  --kv-offloading-size 32 \
  --port 8000 2>&1 | tee server_ca.log
```

或使用 `./run_server.sh simple <model> 8000 32 ca`（见 benchmark README）。

> 3090 上 `max-model-len=131072` 可能在 server 启动阶段失败；当前 CA profile 用 104448 覆盖 100K prefix + 短 decode/tool delta。EngineCore ready 可能超过默认 600s，Layer 1 建议显式设置 `VLLM_ENGINE_READY_TIMEOUT_S=1800`。

显存预算（100K / 4B）：

```text
24 GiB − 8 GiB(权重) − 2 GiB(workspace) ≈ 13 GiB GPU KV 池
100K 单条 ≈ 14 GiB KV  →  单条装不下完整 KV  →  必须 offload
16 rollout 并发 fan-out → max-num-seqs=16（与 --concurrency 对齐）
```

### Mainline 观测指标

`Mainline` 只围绕 `CA2` 采集：

| 指标 | 含义 | 来源 |
|------|------|------|
| `gpu_to_cpu` **Exact bytes** | Task A 被挤出时写出了多少数据 | `/metrics` |
| `cpu_to_gpu` **Exact bytes** | Task A 回来时读回了多少数据 | `/metrics` |
| `external_kv_transfer` **Exact tokens** | load-side 被外部 KV transfer 满足的 token 数 | `/metrics` |
| T0 / T2 prefix TTFT | cold vs reload 的体感差异 | replayer JSON |
| task swap wall-clock | T1 结束到 T2 首 token 的切换成本 | replayer JSON |
| PCIe TX/RX | 搬运是否接近链路上限 | `monitor_pcie.sh` |

进入主表前，至少要满足：

- `gpu_to_cpu` **Exact bytes** 已知
- `cpu_to_gpu` **Exact bytes** 已知
- `Exact tokens` 若存在，只作为辅证
- 没有 exact bytes 的 backend / run，一律降到 `Appendix`

### Mainline 执行流程

```bash
cd /data1/lmy/vllm/benchmarks/kv_offload_experiments
export PATH=/data1/anaconda3/envs/py312/bin:$PATH
export PYTHON=/data1/anaconda3/envs/py312/bin/python

# 1. 生成 CA2 traces（100K prefix）
$PYTHON gen_coding_agent_trace.py \
  --model /data1/models/Qwen/Qwen3-4B-Instruct-2507 \
  --task-id ca2-task-a --prefix-tokens 102400 --num-rollouts 1 \
  -o traces/ca2_task_a.jsonl

$PYTHON gen_coding_agent_trace.py \
  --model /data1/models/Qwen/Qwen3-4B-Instruct-2507 \
  --task-id ca2-task-b --prefix-tokens 102400 --num-rollouts 1 \
  -o traces/ca2_task_b.jsonl

# 2. CA2 @ B-PA
TRACE_A=traces/ca2_task_a.jsonl TRACE_B=traces/ca2_task_b.jsonl \
  OUTDIR=results/ca2_bpa SERVER_WAIT_S=1800 ./launch_ca2_bpa.sh

# 3. CA2 @ sat-BS large-block baseline（sat_bs 由 SE1 给出）
# 终端 A
./run_server.sh simple /data1/models/Qwen/Qwen3-4B-Instruct-2507 8000 32 ca-l3 <sat_bs>

# 终端 B
mkdir -p results/ca2_bl3_satbs
$PYTHON run_coding_agent_replayer.py ca2 \
  --trace-a traces/ca2_task_a.jsonl \
  --trace-b traces/ca2_task_b.jsonl \
  --port 8000 \
  --output results/ca2_bl3_satbs/ca2.json

# 4. 汇总
$PYTHON summarize_results.py results/*.json
```

建议顺序：

```text
Day1  Prereq S1（64K）— 确认机制 + telemetry
Day2  SE1（定 sat-BS）
Day3  CA2 @ B-PA
Day4  CA2 @ sat-BS large-block baseline
Day5  B-L1 PoC 对照 + 汇总
```

### Mainline 常见坑

1. **用字符数估 token** → 100K prefix 实际不足，hash 对不齐；必须用 `gen_coding_agent_trace.py`。
2. **CA2 跳过 Task B** → Task A 块仍在 GPU，T2 纯 GPU hit。
3. **只记录发生，不记录 amount** → 不满足主表要求。
4. **只有 token，没有 bytes** → 只能进 `Appendix`，不能进主表。
5. **只有单侧 bytes** → 不能回答完整 swap cycle；仍不能进主表。
6. **把 `1024` 写死成 B-L3 定义** → 必须先做 SE1，得到 `sat-BS large-block baseline`。
7. **Layer 0 与 Mainline 混表** → 64K S1 与 100K CA2 长度不同，不可直接比 TTFT。
