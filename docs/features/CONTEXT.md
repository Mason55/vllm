# KV Cache Offload / Agentic-RL 实验域

探索性测量：超长序列与 **Coding Agent agentic-RL** 场景下 KV offload 行为、带宽与延迟；并为 [Streaming KV RFC](./streaming_kv_management_rfc.md) PoC 准备 baseline workload。

## 实验分层

| 层 | 目的 | 场景 | 文档 |
|----|------|------|------|
| **Layer 0** | 证明 vLLM PA + offload **机制有效**（store/load/抢占） | S1：64K 单文档 T0→T1→T2 | [§5–§8 实验指南](./kv_cache_offload_experiments.md) |
| **Layer 1** | 模拟 **Coding Agent RL** 工作负载，反映 RFC 关切 | CA1–CA4：rollout fan-out、task swap、tool delta、多 task | [§11 实验指南](./kv_cache_offload_experiments.md) |

Layer 0 是 Layer 1 的前置：**E1–E3 无信号则不做 CA 实验**。

---

## Language — 通用 offload

**超长序列（L1）**:
单条 prompt 约 64K token；Qwen3-4B 单条 KV ≈ 9 GiB。
_Avoid_: 长上下文（未量化）、大 prompt

**超长序列（L2）**:
单条 prompt 约 128K token；单卡 3090 + 4B 装不下完整 KV。
_Avoid_: 128K（裸数字无模型/硬件语境）

**超长序列（CA-L1）**:
Coding Agent 共享 prefix 约 100K token（系统提示 + 仓库代码 + 任务）；Qwen3-4B 单条 KV ≈ 14 GiB。
_Avoid_: 100K（无 agent 语境）

**S1 场景**:
同一超长文档先 ingest（T0），换文档挤 GPU（T1），再问同一文档（T2）。
_Avoid_: 重复问答、长文档实验

**Store（GPU→CPU）**:
已确认 KV 块从 GPU 复制到 CPU 池。
_Avoid_: offload、下刷

**Load（CPU→GPU）**:
新请求 prefix hash 命中 CPU 池，块从 CPU 拉回 GPU。
_Avoid_: 预热、cache hit（未区分 GPU/CPU 层）

**Eager store**:
prefill 进行中，已确认满块逐步 store；默认模式。
_Avoid_: 即时 store

**Lazy store**:
仅扫描 GPU free_queue 中临近 evict 的块 store。
_Avoid_: 延迟 offload

---

## Language — Coding Agent / agentic-RL

**Task**:
一次 RL 采样单元：固定 `task_id`、一份共享 prefix（仓库 + 任务描述）、N 条 rollout。
_Avoid_: 请求、session（未区分 rollout）

**Rollout**:
同一 task 下 fan-out 的一条 trajectory；共享 prefix，delta（tool call + decode）各自不同。
_Avoid_: 并发请求（未说明 prefix 共享）

**Prefix Arena（RFC L1）**:
task 粒度、不可变、所有 rollout **物理共享**的 KV 段（系统 + 仓库 + 任务）。
_Avoid_: system prompt（未说明 KV 共享语义）

**Delta Buffer（RFC L1）**:
rollout 粒度、append-only 的增量 KV（tool 输出 + 模型回复）。
_Avoid_: 续写、增量

**Tool call step**:
trace 中一步：`input_tokens`（模型发出 tool 参数）+ `output_tokens`（tool 返回 observation）。
_Avoid_: function call（未区分 KV 增量）

**Rollout fan-out**:
同一 task **并发** in-flight N 条 rollout（典型 4~16；CA1 用 `--concurrency N`）；测 prefix 共享与 GPU 内存压力。
_Avoid_: batch size（未说明并发 prefix 共享）

**物理共享效率（physical_sharing_ratio）**:
`ΔGPU_KV / (N × KV_theory_single_prefix)`。≈1.0 无物理共享；≈1/N 理想 arena。CA1 必采（`--sample-gpu`）。
_Avoid_: GPU 利用率（未区分 prefix 共享）

**Task swap**:
Task A 在 GPU/CPU 驻留后，Task B 挤占；再切回 Task A。对应 RFC Regime A **bulk arena swap**。
_Avoid_: 冷热切换（未区分 task 粒度）

**B-PA / B-L3 / B-L1**:
RFC PoC 三组对照——**B-PA** vLLM 默认（block=16）；**B-L3** PA + 大 block（256~2048，隔离 block 大小贡献）；**B-L1** RFC mini-runtime（layout 设计 + 物理共享 + 流水）。RFC 命题被证实的判据：B-L1 必须同时赢 B-PA 与 B-L3。
_Avoid_: baseline（未指明 B-PA 还是 B-L3）

**sat-BS large-block baseline**:
`B-L3` 的规范命名。表示使用 SE1 扫描得到的 block-size 收益饱和点作为 large-block control baseline，而不是把 `1024` 之类单个数值当成定义本身。
_Avoid_: 1024 baseline、fixed-1024 baseline

**Offload evidence**:
进入 RFC 主表前，Layer 1 baseline 必须提交的定量 offload 证据对象。至少包含：`happened`（是否发生）、`amount`（发生多少）、`unit`（优先 bytes，其次 tokens）、`quality`（`exact` / `estimated`）。
_Avoid_: offload witness（只表达布尔门禁）、offload happened（未量化）

**Exact bytes**:
运行时直接观测到的 KV 物理搬运字节量。用于回答“实际搬了多少数据”，是 RFC 主表准入的硬门槛。
_Avoid_: estimated bytes、token-equivalent bytes

**Exact tokens**:
运行时直接观测到的、由外部 KV transfer 满足的 token 数。可作为 load-side 精确辅证，但不能替代 `Exact bytes` 回答物理搬运量。
_Avoid_: bytes、offload amount（单独使用时）

**Bidirectional exact bytes**:
同一 baseline 需要同时提交 `gpu_to_cpu` 与 `cpu_to_gpu` 两侧的 `Exact bytes`。用于完整刻画 task swap / offload cycle 的写出与读回成本。
_Avoid_: single-sided exact bytes

**Mainline**:
为 RFC 主表服务的最小实验闭环。只保留回答 RFC 核心命题所必需、且证据口径满足主表硬门槛的实验。
_Avoid_: 全量实验、探索性大全

**Prereq / Gate**:
进入 `Mainline` 前的前置机制验证。用于证明 backend / telemetry / workload setup 可用，但不进入 RFC 主表。
_Avoid_: mainline experiment、主表结果

**Appendix**:
不进入 RFC 主表的探索性实验、诊断实验、历史兼容实验、实现细节验证。
_Avoid_: main experiment

**SE1 block-size scan**:
绑定 CA1 的子实验：扫 `--block-size {16,128,256,512,1024,2048}` 找 B-L3 的收益饱和点。
_Avoid_: 调大 block（无饱和曲线语境）

---

## Relationships

### Layer 0（机制验证）

- **S1** 由 T0、T1、T2 三步组成
- **E1** 验证 **Store** 发生；**E2** 验证 **Load** 发生
- **Eager store** 下 Store 主要发生在 **T0**；**Lazy store** 下 Store 主要发生在 **T1**
- **T1** 的作用：挤掉 T0 在 GPU 上的块，使 **T2** 必须走 CPU **Load**（非纯 GPU hit）

### Layer 1（RFC 关切映射）

| RFC §6 关切 | CA 场景 | 测什么 |
|-------------|---------|--------|
| 长共享 prefix + rollout fan-out | **CA1** | Appendix only；当前阶段不实现 |
| Task swap / Regime A | **CA2** | cold→hot 任务切换 wall-clock、**Offload evidence**、PCIe 利用率 |
| Delta 增量 + decode 不退化 | **CA3** | Appendix only；当前阶段不实现 |
| GPU 内存有效利用 | **CA4** | Appendix only；保留、后续实现 |
| PoC 4 维成功标准 | 分阶段汇总 | 见 [RFC §6.3](./streaming_kv_management_rfc.md) |

---

## Example dialogue

> **Dev:** "T0 跑完 64K，T2 再发同一 prompt，能测 Load 吗？"
> **Domain expert:** "不一定。T0 块可能还在 GPU → T2 纯 GPU prefix hit，不走 PCIe。必须先 T1 挤 GPU，T2 才测 Load。"

> **Dev:** "16 rollout 各发 100K prefix，能测 RFC 的物理共享吗？"
> **Domain expert:** "vLLM PA 下是 block ref_cnt 逻辑共享，不是 RFC 的 Prefix Arena。CA1 先量 **实际 KV 占用与 prefill 次数**，PoC mini-runtime 再比 L1 layout。"

> **Dev:** "B-L1 比 B-PA 赢了 2×，能写进 PoC 通过吗？"
> **Domain expert:** "不能。还要看是否赢 B-L3。如果 B-L3 已经从 B-PA 拿到了 1.8× 收益，说明 80% 的赢面来自 block 大小，layout 设计的独立贡献不显著，RFC 命题没立住。"

---

## Flagged ambiguities

- "offload 发生" 曾混指 Store 与 Load — 已拆：**E1=Store，E2=Load**
- 文档曾写 Store 只在 `ref_cnt=0` 后 — 仅 **Lazy store**；**Eager（默认）** 在 prefill 进行中 store
- **v0 决策**：Layer 0 锁定 **Eager**；E1 观测 T0，T1 仅挤 GPU，Lazy 留 v1 T3
- **Layer 1** 与 Layer 0 **不混表**：CA 场景用 trace replayer；S1 用 `run_s1_sequence.sh`
- CA trace 的 `prefix_tokens` 必须用 **真实 tokenizer 计数**，不能用字符估算
- **Layer 1 必须跑 B-PA + B-L3 + B-L1 三组**：只比 B-L1 vs B-PA 不能证伪 PA 替代命题（block 大小变量未隔离）
- B-L3 的 block size 来自 SE1 扫描的**饱和点**，不是任意值；规范称呼用 **sat-BS large-block baseline**
- **Layer 0 = 软门禁**（E1–E3 无 signal 暂停 Layer 1）；**Layer 1 B-L1 PoC = 硬门槛**（RFC §6.3）
- CA1 **必须并发 fan-out**（`--concurrency`）；Layer 1 trace **固定 100K**，不用 64K 过渡
- **offload-first 原则**：Layer 1 任何想进入 RFC 主表的 baseline，必须带 **Offload evidence**。只有 prefix sharing / local prefix hit，不算 offload baseline
- **旧词冲突已解决**：`offload witness` 仅能表示布尔门禁；RFC 主表改用 **Offload evidence**
- **主表硬门槛**：每个 Layer 1 baseline 必须至少提供一项 **Exact bytes**；`Exact tokens` 只能作辅证，不能单独替代物理搬运量
- **主表更强硬门槛**：对 task-swap / offload-cycle baseline，必须提供 **Bidirectional exact bytes**（`gpu_to_cpu` + `cpu_to_gpu`）；单侧 bytes 不足以进入主表
- 文档整理优先级：先定义 **Mainline**，其余内容再降到 **Appendix**，避免把探索性实验和 RFC 主表闭环混在一起
- **Layer 0 定位已收敛**：`Layer 0` 属于 **Prereq / Gate**，用于机制验证，不进入 RFC 主表，也不与 `Mainline` 并列
- **当前裁剪决定**：`CA1` 与 `CA3` 先降到 **Appendix**，且当前阶段不实现；`Mainline` 不再依赖它们
- **当前主线收敛**：`Mainline = CA2 only`
- **CA4 当前定位**：保留在 **Appendix**，不删除，但当前阶段不阻塞 `Mainline`
