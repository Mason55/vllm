# KV Offload Layout Microbench

目标：最小验证 `A` vs `C`。

- `A`: `cudaMemcpyAsync` 一次连续大块搬运
- `C`: `cuMemcpyBatchAsync` 搬 `layer x block` 离散小块

范围故意很小：

- 不接 scheduler
- 不接真实 model
- 不改主代码路径
- 只回答「连续大块 copy 上限」vs「当前 Simple 路径 copy 形态」
- CPU 侧只要求 pinned host memory；不复刻 `cudaHostRegister` 分配细节

## 文件

- `benchmark_a_vs_c.py`: 独立 bench
- `benchmark_multigpu_batch_lock.py`: 多 GPU 并发 `contiguous` vs `batched`，看 `cuMemcpyBatchAsync` 提交/完成是否有锁冲突
- `run_with_pcie_monitor.sh`: bench + `nvidia-smi dmon` 封装
- `summarize_pcie_csv.py`: 汇总 `dmon` 的 PCIe Rx/Tx
- `run_common_models.sh`: 跑一组常见开源模型 preset
- `run_physical_block_sweep.sh`: 固定总字节，扫 `physical block` 大小

## 前置

```bash
cd /data1/lmy/vllm
source .venv/bin/activate
```

要求：

- CUDA 可用
- `.venv` 内已有 `torch`、`cuda.bindings`

## 跑法

默认同时测 `H2D` 和 `D2H`：

```bash
.venv/bin/python benchmarks/kv_offload_layout/benchmark_a_vs_c.py
```

多 GPU 并发 lock-contention bench：

```bash
.venv/bin/python benchmarks/kv_offload_layout/benchmark_multigpu_batch_lock.py \
  --gpus 0,1 \
  --direction h2d \
  --mode both \
  --execution-model thread \
  --run-mode both \
  --num-layers 36 \
  --num-blocks 256 \
  --bytes-per-block 65536 \
  --repeat 20 \
  --output benchmarks/kv_offload_layout/results/multigpu_h2d_2gpu.json
```

同一口径切到多进程：

```bash
.venv/bin/python benchmarks/kv_offload_layout/benchmark_multigpu_batch_lock.py \
  --gpus 0,1 \
  --direction h2d \
  --mode both \
  --execution-model process \
  --run-mode concurrent-only \
  --num-layers 36 \
  --num-blocks 256 \
  --bytes-per-block 65536 \
  --repeat 20 \
  --output benchmarks/kv_offload_layout/results/multigpu_h2d_2gpu_process.json
```

用 Nsight Systems 观测当前多 GPU microbench：

```bash
bash benchmarks/kv_offload_layout/run_multigpu_with_nsys.sh \
  benchmarks/kv_offload_layout/results/nsys \
  --gpus 0,1,2,3 \
  --direction both \
  --mode both \
  --execution-model thread \
  --run-mode concurrent-only \
  --num-layers 1 \
  --num-blocks 256 \
  --bytes-per-block 65536 \
  --repeat 20 \
  --output benchmarks/kv_offload_layout/results/multigpu_4gpu_thread_concurrent.json
```

切到多进程只改：

```bash
  --execution-model process
```

带 PCIe 监控一起跑：

```bash
bash benchmarks/kv_offload_layout/run_with_pcie_monitor.sh \
  benchmarks/kv_offload_layout/results \
  --num-layers 36 \
  --num-blocks 256 \
  --bytes-per-block 65536 \
  --repeat 20
```

按常见模型 preset 批量跑：

```bash
bash benchmarks/kv_offload_layout/run_common_models.sh
```

固定总字节，扫大 `physical block`：

```bash
bash benchmarks/kv_offload_layout/run_physical_block_sweep.sh
```

默认：

- `NUM_LAYERS=36`
- `TOTAL_MIB=576`
- `PATTERN=random`
- sweep: `64 KiB`, `512 KiB`, `1 MiB`, `2 MiB`

默认 `--pattern all`，会顺序跑：

- `contiguous`: block id 连续
- `runs`: 连续短 run 打散
- `random`: block id 全随机

缩小规模先做 smoke：

```bash
.venv/bin/python benchmarks/kv_offload_layout/benchmark_a_vs_c.py \
  --num-layers 8 \
  --num-blocks 32 \
  --bytes-per-block 4096 \
  --warmup 2 \
  --repeat 5
```

更接近 KV block 量级：

```bash
.venv/bin/python benchmarks/kv_offload_layout/benchmark_a_vs_c.py \
  --num-layers 36 \
  --num-blocks 256 \
  --bytes-per-block 65536 \
  --repeat 20
```

只跑更离散场景：

```bash
.venv/bin/python benchmarks/kv_offload_layout/benchmark_a_vs_c.py \
  --direction h2d \
  --num-layers 36 \
  --num-blocks 256 \
  --bytes-per-block 65536 \
  --pattern random \
  --repeat 20
```

跑短 run 场景：

```bash
.venv/bin/python benchmarks/kv_offload_layout/benchmark_a_vs_c.py \
  --direction h2d \
  --num-layers 36 \
  --num-blocks 256 \
  --bytes-per-block 65536 \
  --pattern runs \
  --run-length 8 \
  --repeat 20
```

## 输出

每个方向打印三行：

- `A contiguous`: 连续大块 copy
- `C batched`: 当前离散 batch copy
- `ratio C/A`: `C` 相对 `A` 倍数

字段：

- `stream_us`: stream 完成耗时中位数
- `gbps`: 有效带宽

`pattern` 只影响 `C`。`A` 始终是单次连续大块 copy，上限基线不变。

`run_with_pcie_monitor.sh` 额外打印：

- `avg_rx_mib_s` / `avg_tx_mib_s`
- `peak_rx_mib_s` / `peak_tx_mib_s`

## 多 GPU lock-contention bench 输出

`benchmark_multigpu_batch_lock.py` 每个组合会先跑：

- `baseline_1gpu`: 只用 `--gpus` 里的第一张卡
- `concurrent_ngpu`: 所有指定 GPU 同时起跑

字段：

- `submit_us_p50/p95`: host 侧提交一次 copy 的耗时
- `complete_us_p50/p95`: 从提交开始到 stream 完成的总耗时
- `gbps_p50/p95`: 有效带宽
- `speedup_vs_1gpu`: `concurrent_ngpu.aggregate.gbps_p50 / baseline_1gpu.aggregate.gbps_p50`
- `scaling_efficiency`: `speedup_vs_1gpu / num_gpus`
- `execution_model`: `thread` 或 `process`
- `run_mode`: `baseline-only` / `concurrent-only` / `both`

设计口径：

- 单进程
- 每 GPU 一个线程
- 每轮用 barrier 同步起跑
- 第一版只测 `H2D` / `D2H`
- 第一版不测 `G2G`

现在已支持两种并发模型：

- `thread`: `1 process / N threads / N GPUs`
- `process`: `N processes / 1 GPU each`

建议口径：

- `baseline-only`: 只补一次 1GPU baseline
- `concurrent-only`: 只跑 `N GPU` 同时起跑，适合 `thread vs process` 对照
- `both`: 一次命令同时产出 baseline + concurrent

## 用 `nsys` 看什么

对这个 microbench，`nsys` 主要回答两类问题：

1. **submit 路径是否被串行化**
   - 看 host 侧 CUDA API 时间线
   - 对比 `thread` vs `process`
   - 若 `thread` 下多 GPU 的 API 调用明显排队，而 `process` 下缓解，说明 submission-side contention 存在

2. **copy 执行本体是否仍被共享链路限制**
   - 看各 GPU 的 memcpy 时间线是否高度重叠
   - 看整体 CUDA memcpy 持续时间是否与 aggregate `gbps` 结论一致
   - 若 `process` 只改善 submit，不明显改善 memcpy 完成跨度，说明主瓶颈仍是平台级 copy ceiling

建议先抓：

- `run-mode=concurrent-only`
- 同一组参数分别跑：
  - `execution-model=thread`
  - `execution-model=process`

这样 trace 最干净，最容易对照。

## 解释

看两件事：

- `gbps`: `C` 离 `A` 上限差多少
- `peak_rx/tx`: PCIe 链路峰值流量大概到哪

若 `C` 已接近 `A`：

- 大改 KV layout 收益可能有限

若 `C` 明显落后 `A`：

- `block-major` / `cross-layer` / `run-contiguous` 值得继续做

## 当前结论

基于 2026-05-21 本机实测，结果目录：

- `benchmarks/kv_offload_layout/results/`
- `benchmarks/kv_offload_layout/results/common_models/`

### 1. 总体结论

- `A` 总是快于 `C`
- 只看 memcpy 本体，`C` 相对 `A` 差距主要受 `bytes_per_block` 影响
- `pattern=contiguous/runs/random` 影响通常不大
- PCIe 峰值稳定在 `~13 GiB/s Rx`、`~14 GiB/s Tx`

### 2. 典型结果

`64 KiB/block` 家族：

- `Qwen3-4B` / `Qwen3-8B` / `Mistral-7B-v0.3`
  - `C ~= 0.89x A`
  - `H2D`: `10.20~10.21 GB/s` vs `A 11.49~11.50 GB/s`
  - `D2H`: `10.87~10.88 GB/s` vs `A 12.28 GB/s`

- `Qwen2.5-14B-Instruct`
  - `C ~= 0.94x A`
  - `H2D`: `10.85 GB/s` vs `A 11.49 GB/s`
  - `D2H`: `11.58 GB/s` vs `A 12.28 GB/s`

`32 KiB/block` 家族：

- `Qwen2.5-7B-Instruct`
  - `C ~= 0.76x~0.80x A`
  - `H2D`: `8.76~9.16 GB/s` vs `A 11.49 GB/s`
  - `D2H`: `9.30~9.31 GB/s` vs `A 12.27 GB/s`

- `DeepSeek-R1-Distill-Qwen-7B`
  - `C ~= 0.80x A`
  - `H2D`: `9.15~9.17 GB/s` vs `A 11.49~11.50 GB/s`
  - `D2H`: `9.78~9.79 GB/s` vs `A 12.27 GB/s`

### 3. 解读

- `32 KiB/block` 比 `64 KiB/block` 更吃亏，说明小块 batched memcpy 惩罚更明显
- block id 顺序通常不是主矛盾；主要问题更像 `layers x blocks` 这种 copy 组织方式
- 下一步最值钱实验不是继续调 `pattern`，而是补 `full-block copy` 中间态：
  - `copy count = layers x blocks` -> 当前 `C`
  - `copy count = blocks` -> `block-major` / `cross-layer` 候选
  - `copy count = runs / 1` -> 连续布局上限

### 4. 大 physical block sweep

脚本：

- `benchmarks/kv_offload_layout/run_physical_block_sweep.sh`

口径：

- 固定 `total_bytes = 576 MiB`
- 固定 `num_layers = 36`
- 固定 `pattern = random`
- 只改 `bytes_per_block`
- 自动反推 `num_blocks`

结果（2026-05-21）：

| bytes_per_block | num_blocks | H2D `C/A` | D2H `C/A` | 结论 |
|---|---:|---:|---:|---|
| `64 KiB` | 256 | `0.89x` | `0.89x` | 当前小块基线 |
| `512 KiB` | 32 | `0.98x` | `0.98x` | 已非常接近 `A` |
| `1 MiB` | 16 | `0.99x` | `0.99x` | 几乎贴住 `A` |
| `2 MiB` | 8 | `1.00x` | `0.99x` | 与 `A` 基本重合 |

这组实验说明：

- 把 **physical block** 从 `64 KiB` 拉大到 `0.5~2 MiB`，`C` 的 memcpy 本体性能会迅速逼近 `A`
- 这和 blog 里结论一致：DMA 更喜欢 **大且连续** 的 copy
- 因此，若能把当前 `layer x block` 碎片化搬运重组为 `0.5~2 MiB` 级别 physical block，单看 memcpy 本体，收益非常实在

### 5. Multi-GPU concurrent copy probe (`cuMemcpyBatchAsync` lock question)

新增多 GPU microbench：

- `benchmark_multigpu_batch_lock.py`
- 口径：**单进程 + 每 GPU 一个线程 + barrier 同步起跑**
- 第一阶段只测：
  - `H2D` / `D2H`
  - `contiguous` (`cudaMemcpyAsync`)
  - `batched` (`cuMemcpyBatchAsync`)

#### 5.1 关注问题

要回答的不是单卡 layout 收益，而是：

- 多 GPU 并发时，`cuMemcpyBatchAsync` 是否存在明显的 driver-side lock / serialization
- 若多卡不扩展，主因更像：
  - `cuMemcpyBatchAsync` 提交路径锁
  - 还是共享的 host<->device copy 通道 / PCIe / root complex ceiling

#### 5.2 当前观测（GPU 组：`0,1,2,3`）

已测 3 档总量，`bytes_per_block` 固定 `64 KiB`：

| 形态 | 参数 | 每卡总量 | 结论 |
|---|---|---:|---|
| CA-like large | `36 layers × 256 blocks × 64 KiB` | `576 MiB` | 4 卡 aggregate 几乎等于单卡 aggregate |
| reduced bytes | `36 layers × 28 blocks × 64 KiB` | `~63 MiB` | 4 卡 aggregate 仍几乎等于单卡 aggregate |
| tiny bytes | `1 layer × 256 blocks × 64 KiB` | `16 MiB` | 4 卡 aggregate 仍几乎等于单卡 aggregate |

典型现象：

- `contiguous` 单卡：
  - `H2D ~= 11.4~11.5 GB/s`
  - `D2H ~= 12.2~12.3 GB/s`
- `contiguous` 4 卡 aggregate：
  - 仍约 `11.5 GB/s` / `12.2 GB/s`
- 因此 4 卡每卡只分到约 `1/4`：
  - `H2D ~= 2.88~2.90 GB/s / GPU`
  - `D2H ~= 3.06~3.08 GB/s / GPU`

`batched` 也类似：

- 单卡 aggregate 低于 `contiguous`
- 4 卡 aggregate 也仍被钉在接近单卡 aggregate 的水平
- `submit_us` 随并发增加而上升，但 `complete_us` 仍占主导

#### 5.3 当前解读

这几组结果更支持：

- **多 GPU 不扩展是真现象**
- 但主因**不像** `cuMemcpyBatchAsync` 独有 driver lock
- 更像 **共享 host<->device copy 通道总预算固定**
  - PCIe / root complex / host memory DMA path ceiling

原因：

1. `contiguous` 和 `batched` 都不扩展  
   若主因是 `cuMemcpyBatchAsync` 锁，`contiguous` 不该同样塌到单卡 aggregate。

2. 每卡带宽近似严格 `1/N` 平分  
   这更像共享总池子被均匀切分，不像某个线程/某张卡被锁异常拖慢。

3. `batched` 的 `submit_us` 虽然更高，但不是总完成时间主导项  
   说明 submission contention 存在痕迹，但不是当前 scaling collapse 的第一矛盾。

#### 5.4 当前结论

截至这轮实验，可以先下一个保守结论：

- **当前证据不支持“多卡不扩展主要由 `cuMemcpyBatchAsync` driver lock 导致”**
- 更强的解释是：
  - **平台级 host<->device aggregate copy ceiling** 先出现
- `cuMemcpyBatchAsync` 主要额外损失体现在：
  - 单卡 `A vs C` 差距
  - 更高的 `submit_us`
  - 但它不是 4 卡 aggregate 不扩展的主因

#### 5.5 thread vs process 对照

新增 `execution_model` 对照后，`0,1,2,3` 这组在 `64 KiB/block`、每卡 `16 MiB` 下可观察到：

- `thread -> process` 后，`submit_us` 明显下降
  - `contiguous` 最明显，常从 `~30-40 us` 回到 `~5-15 us`
  - `batched` 也下降，常从 `~300-500 us` 回到 `~230-290 us`
- 但 `complete_us` 与 aggregate `gbps` 变化很小
  - `contiguous` aggregate 几乎不变
  - `batched` aggregate 只小幅改善

这说明：

- **单进程多线程路径确实存在 submission-side contention**
- 但 **它不是 4 卡 aggregate 不扩展的主因**
- 主导瓶颈仍更像 **平台级 host<->device aggregate copy ceiling**

因此当前最稳的判断是：

- 有锁痕迹，但不是主犯
- `process` 模式适合作为证据对照
- 若要继续深挖根因，优先做 `nsys` 观测，再看是否需要 GPU 拓扑对照

#### 5.6 nsys 观察

对同一组 `concurrent-only` 参数分别抓 `thread` / `process` trace 后，`nsys stats` 给出的定量信息与上面的 microbench 结论一致：

- 两个 trace 的 workload 基本一致
  - `H2D`: `200` 次 copy，总量约 `3355 MB`
  - `D2H`: `216` 次 copy，总量约 `3624 MB`
  - 单次 copy 大小基本固定在 `16.777 MB`

- `process` 模式下，host 侧提交成本确实下降
  - `cuMemcpyBatchAsync`
    - `thread`: total `70.4 ms`，avg `352 us`
    - `process`: total `49.9 ms`，avg `250 us`
  - `cudaMemcpyAsync`
    - `thread`: total `195.3 ms`，avg `872 us`
    - `process`: total `182.1 ms`，avg `813 us`

- 但 GPU 上真实 memcpy 执行时间只小幅改善
  - `H2D` GPU mem time
    - `thread`: `989 ms`
    - `process`: `917 ms`
  - `D2H` GPU mem time
    - `thread`: `1146 ms`
    - `process`: `1049 ms`

这说明：

- **单进程多线程路径存在 submission-side contention**
- 但 **真实 memcpy 执行跨度没有随之大幅缩短**
- 所以当前主瓶颈仍更像 **平台级 host<->device aggregate copy ceiling**

另外，`process` trace 里 `cudaHostAlloc`、`cudaMalloc`、`cudaStreamCreateWithPriority`、`cuLibraryLoadData` 更重，主要是因为 `4` 个进程各自初始化一套 CUDA runtime / stream / alloc 路径，不应把这部分误判成 steady-state copy 退化。

#### 5.7 下一步

若还要继续追 `driver lock`，比继续降 payload 更值钱的是：

1. **换 GPU 拓扑组合**
   - 如 `0,1` / `0,4` / `0,1,2,3` / `4,5,6,7`
   - 看 aggregate ceiling 是否随 GPU 组合变化

2. **对比 thread vs process**
   - `thread`: `1 process / N threads / N GPUs`
   - `process`: `N processes / 1 GPU each`
   - 若 `process` 显著更好，才更像进程内 driver/runtime lock
   - 若两者都一样差，更像平台共享通道问题


你这个观察和总结非常精准，直接点出了底层高性能存储和内存拷贝的核心痛点：**平摊开销（Amortized Overhead）**。

在现代系统架构中，“单看 memcpy 本体，收益非常实在”这个结论背后，其实有几个非常硬核的底层硬件和软件驱动机制在支撑。我们可以把这个现象拆开来看看：

### 1. 为什么 0.5~2 MiB 是个黄金区间？

当 Block 从 64 KiB 扩大到 0.5~2 MiB 时，性能逼近 A（极值）的主要原因在于**硬件流水线被彻底喂饱了**，同时**软件/协议栈的开销被稀释到了接近于零**。

* **页表与 TLB 命中率（Huge Pages 的隐形福利）：**
在 Linux 等系统中，标准的物理内存页（Page）是 4 KiB。64 KiB 只需要 16 个页表项，而 2 MiB 正好是一个大页（Huge Page）的尺寸。如果你的 Physical Block 达到了 2 MiB 且地址是对齐的，系统可以直接利用硬件大页，大大减少了 TLB（Translation Lookaside Buffer，页表缓存）的 miss 损耗。
* **DMA 与 CPU Cache 的预取（Prefetching）：**
现代 CPU 的硬件预取器（Hardware Prefetcher）非常聪明。当你开始拷贝一段连续内存时，预取器会预测你接下来的访问路径，提前把数据从内存拉到 L1/L2/L3 Cache 中。64 KiB 太短了，CPU 刚把流水线和预取跑满，拷贝就结束了；而 0.5~2 MiB 的尺寸足够让 CPU 的 L3 缓存流水线或 DMA 控制器在中途进入**满载的“巡航模式”**。

### 2. 为什么 DMA 更喜欢“大且连续”？

正如你提到的 Blog 结论，DMA（Direct Memory Access）在处理大块连续内存时效率极高。这是因为 DMA 传输的核心瓶颈往往不在于“传输带宽”，而在于“启动和握手开销”。

如果执行碎片化搬运（比如大量的 64 KiB）：

* **Descriptor 链表变长：** DMA 需要维护一个庞大的 Scatter-Gather List（离散-聚集列表）。每处理完一个小块，DMA 控制器就要去读下一个描述符（Descriptor），甚至触发一次中断或轮询，这带来了极大的内核态/硬件级上下文切换开销。
* **总线利用率低：** 每次 DMA 传输启动都有一个 Setup Time（建立时间）。块越小，Setup Time 占总时间的比例就越高。

当合并为 0.5~2 MiB 的大块时，DMA 只需要一次 Setup 信号，就能连续不断地拉干总线带宽。

---

### 💡 架构设计上的“终极一问”：天下没有免费的午餐

你提到的构想非常完美：**“把当前 layer x block 碎片化搬运重组为 0.5~2 MiB 级别的 physical block”**。这在学术和工业界有一个专门的技术叫 **Compaction（内存/存储紧凑化）** 或 **Coalescing（合并）**。

在实施这个重组时，你需要权衡一个核心的 trade-off（权衡）：

> **重组（Aggregation）本身是有成本的。**
> 如果你要把碎片“搬运并重组”成 2 MiB 的大块，你在重组的阶段**已经发生了一次额外的内存拷贝或 CPU 调度**。
> * **不划算的情况：** 如果你为了拼凑这 2 MiB，在内存里用 CPU 做了好几次小规模的 memcpy 或者是复杂的链表遍历，那么重组带来的 DMA 收益，可能会被前面“拼凑碎片”的 CPU 算力损耗给直接对冲掉。
> * **划算的情况（零拷贝/硬件合并）：**
> 1. **Scatter-Gather 硬件合并：** 利用支持 Scatter-Gather 的高级 DMA 直接在传输时合并（不需要 CPU 搬运）。
> 2. **预分配与空间换时间：** 在 Layer 写入的源头，就通过内存池（Memory Pool）直接申请连续的 2 MiB 空间，即使没写满也占着，从而在根源上避免碎片化。
> 
> 
> 
> 

**总结来说：** 你的直觉和数据完全正确。在现代高性能计算、分布式存储（如 SPDK, RDMA 架构）或大模型推理（KV Cache 管理）中，**尽一切可能将 I/O 和内存操作对齐到 2 MiB 边界**，已经是公认的黄金法则。
