# vLLM KV Cache 内存布局分析

本文档详细分析 vLLM KV Offloading Connector 的内存布局设计，包括逐层布局（per-layer）和跨层布局（cross-layer）两种模式。

---

## 一、两种内存布局模式

vLLM 中 KV cache 有两条初始化路径，在 [gpu_model_runner.py:L6855-L6867](../v1/worker/gpu_model_runner.py#L6855-L6867) 中分流：

```python
if self.use_uniform_kv_cache(self.attn_groups, cache_dtype):
    # → 跨层布局 (cross-layer)
    kv_caches, cross_layers_kv_cache, attn_backend = (
        self.allocate_uniform_kv_caches(...)
    )
else:
    # → 逐层布局 (per-layer, 默认)
    kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)
```

随后注册到 offload connector 时再次分流（[gpu_model_runner.py:L6972-L6978](../v1/worker/gpu_model_runner.py#L6972-L6978)）：

```python
if self.cross_layers_kv_cache is not None:
    kv_transfer_group.register_cross_layers_kv_cache(
        self.cross_layers_kv_cache, self.cross_layers_attn_backend
    )
else:
    kv_transfer_group.register_kv_caches(kv_caches)
```

---

## 二、逐层布局（Per-layer，默认模式）

这是大多数场景下的默认布局。

### 2.1 内存结构

每个层（或 K/V 各半）拥有独立的 tensor，形状为 `(num_blocks, page_size_bytes)`。不同层之间通过 `KVCacheTensor.shared_by` 来共享底层 tensor，例如通用场景下第 `i` 层的 full attention 和 sliding window 的对应层共享同一个 tensor。

核心代码在 [kv_cache_utils.py:L1285-L1314](../v1/core/kv_cache_utils.py#L1285-L1314)：

```python
# General case: group_size 个内存池，每个被每个 group 的一层共享
# 3 groups (full.0, full.1), (sw.0, sw.2), (sw.1, padding) 时:
# full.0, sw.0, sw.1 → 共享 Tensor A
# full.1, sw.2 → 共享 Tensor B
group_size = max(len(group.layer_names) for group in kv_cache_groups)
page_size = get_uniform_page_size([group.kv_cache_spec for group in kv_cache_groups])
num_blocks = get_num_blocks(vllm_config, group_size, available_memory, page_size)
kv_cache_tensors = []
for i in range(group_size):
    shared_by = []
    for j in range(len(kv_cache_groups)):
        if i < len(kv_cache_groups[j].layer_names):
            shared_by.append(kv_cache_groups[j].layer_names[i])
    kv_cache_tensors.append(
        KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)
    )
```

### 2.2 page_size_bytes 计算

```
page_size_bytes = 2 × num_kv_heads × head_size × block_size
```

对应 K 和 V 拼接在一起的一层数据大小（KV cache dtype 对齐后的字节数）。

以 Llama-3.1-8B-Instruct 为例（`num_kv_heads=8, head_size=128, block_size=16`）：

```
page_size_bytes = 2 × 8 × 128 × 16 = 32,768 bytes ≈ 32KB
```

### 2.3 FlashAttention 的特殊处理

FlashAttention 的物理布局是 `(2, num_blocks, ...)`，K 和 V 分开存储。在 [offloading/worker.py:L140-L160](../../distributed/kv_transfer/kv_connector/v1/offloading/worker.py#L140-L160) 中会将其拆分为两个 `CanonicalKVCacheTensor`（每个 `half_page_size`），每个形状为 `(num_blocks, half_page_size)`：

```python
# Flash Attention case: (2, num_blocks, ...)
half_page_size = layer_kv_cache_spec.page_size_bytes // 2
raw = storage.view(2, num_blocks, half_page_size)
tensors_per_block[layer_name] = tuple(raw.unbind(0))
```

### 2.4 布局示意

```
逐层布局（Flash Attention NHD 后端）

Tensor 0 (K, layer_0):  ┌──────────────────────────────────┐
                         │ blk_0  │ blk_1  │ ... │ blk_N    │
                         │ K data │ K data │     │ K data   │
                         │ ~16KB  │ ~16KB  │     │ ~16KB    │
                         └──────────────────────────────────┘
Tensor 1 (V, layer_0):  ┌──────────────────────────────────┐
                         │ blk_0  │ blk_1  │ ... │ blk_N    │
                         │ V data │ V data │     │ V data   │
                         └──────────────────────────────────┘
...
Tensor 62 (K, layer_31):┌──────────────────────────────────┐
Tensor 63 (V, layer_31):┌──────────────────────────────────┘
```

此时单个逻辑块大小仅 ~16-32KB，远小于 DMA 最优的 2MB 阈值。

### 2.5 CanonicalKVCaches 构建

逐层模式通过 [offloading/worker.py:L59-L235](../../distributed/kv_transfer/kv_connector/v1/offloading/worker.py#L59-L235) 中的 `register_kv_caches()` 构建 `CanonicalKVCaches`：

```python
# 每个 KVCacheTensor (shared_by 组) 对应一组 CanonicalKVCacheTensor
for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
    for tensor in tensors_per_block[first_layer_name]:
        block_tensors.append(
            CanonicalKVCacheTensor(
                tensor=tensor,           # shape: (num_blocks, page_size_bytes)
                page_size_bytes=page_size_bytes,
            )
        )
```

每个层的 page_size_bytes 保持为单层的大小（~32KB），CPU offload handler 会为每个 layer data ref 分别计算指针、分批搬运。

---

## 三、跨层布局（Cross-layer / Uniform，优化模式）

这是博客中描述的 `2 × num_layers × block_size` 新布局。

### 3.1 启用条件

在 [kv_connector_model_runner_mixin.py:L156-L188](../v1/worker/kv_connector_model_runner_mixin.py#L156-L188) 的 `use_uniform_kv_cache()` 中判断：

1. **配置了 KV connector** 且 `prefer_cross_layer_blocks == True`
2. **只有单一 KV cache group**（所有层 `page_size` 相同）
3. **Attention backend 支持 `include_num_layers_dimension`**（stride_order 长度 = shape 长度 + 1）
4. **`stride_order[0] != 0`**（num_layers 不在物理第 0 维，即不是 identity permutation）

```python
# stride_order[0] == 0 意味着 num_layers 保持在物理布局的第一维
# （identity permutation），此时跨层布局不支持
return kv_cache_stride_order[0] != 0
```

### 3.2 分配过程

在 [kv_connector_model_runner_mixin.py:L192-L282](../v1/worker/kv_connector_model_runner_mixin.py#L192-L282) 的 `allocate_uniform_kv_caches()` 中：

```python
num_layers = len(kv_cache_config.kv_cache_tensors)
total_size = tensor_size * num_layers  # = page_size × num_blocks × num_layers

# 根据后端的 stride_order 确定物理布局
kv_cache_shape = attn_backend.get_kv_cache_shape(...)
kv_cache_shape = (num_layers,) + kv_cache_shape  # 在逻辑形状前加 num_layers 维
kv_cache_stride_order = attn_backend.get_kv_cache_stride_order(
    include_num_layers_dimension=True
)
kv_cache_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)

# 分配一个超大连续 buffer 容纳所有层
cross_layers_kv_cache = (
    torch.zeros(total_size, dtype=torch.int8, device=device)
    .view(kv_cache_spec.dtype)
    .view(kv_cache_shape)
)

# 还原为逻辑形状的 permute 视图，供模型前向使用
inv_order = [kv_cache_stride_order.index(i) for i in range(len(kv_cache_stride_order))]
permuted_kv_cache = cross_layers_kv_cache.permute(*inv_order)

kv_caches = {}
for i, kv_cache_tensor in enumerate(kv_cache_config.kv_cache_tensors):
    tensor = permuted_kv_cache[i]  # 第 i 层是 permuted 后的第 i 个 slice
    for layer_name in kv_cache_tensor.shared_by:
        kv_caches[layer_name] = tensor
```

### 3.3 不同后端物理布局对比

| Backend | 物理布局 | num_blocks 在物理 dim 0? |
|---------|---------|--------------------------|
| **Flash Attention NHD** | `(num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)` | **✅ 是** |
| FlashInfer NHD | `(2, num_layers, num_blocks, block_size, num_kv_heads, head_size)` | ❌ 否（在 dim 2） |
| FlashInfer HND | `(2, num_kv_heads, num_layers, num_blocks, block_size, head_size)` | ❌ 否（在 dim 3） |
| Triton NHD | `(2, num_layers, num_blocks, block_size, num_kv_heads, head_size)` | ❌ 否（在 dim 2） |
| Triton HND | `(2, num_kv_heads, num_layers, num_blocks, block_size, head_size)` | ❌ 否（在 dim 3） |

stride_order 定义在各后端文件中：

- [flash_attn.py:L156-L170](../v1/attention/backends/flash_attn.py#L156-L170) — Flash Attention
- [flashinfer.py:L374-L390](../v1/attention/backends/flashinfer.py#L374-L390) — FlashInfer
- [triton_attn.py:L333-L349](../v1/attention/backends/triton_attn.py#L333-L349) — Triton

### 3.4 "真正连续"的跨层块——`register_cross_layers_kv_cache()`

这条路径在 [offloading/worker.py:L239-L280](../../distributed/kv_transfer/kv_connector/v1/offloading/worker.py#L239-L280) 中有一个关键断言：

```python
# 断言：num_blocks 必须在物理维度 0（即 blocks 在内存中连续）
test_shape = attn_backend.get_kv_cache_shape(
    num_blocks=1234, block_size=16, num_kv_heads=1, head_size=256
)
num_blocks_logical_dim = test_shape.index(1234) + 1  # +1 补上 num_layers 维
physical_to_logical = attn_backend.get_kv_cache_stride_order(
    include_num_layers_dimension=True
)
num_blocks_physical_dim = physical_to_logical.index(num_blocks_logical_dim)
assert num_blocks_physical_dim == 0  # ← 关键断言

# 创建一个统一的 CanonicalKVCacheTensor：
page_size_bytes = kv_cache_spec.page_size_bytes * num_layers
tensor = storage.view(num_blocks, page_size_bytes)
#                       ↑          ↑
#                   num_blocks   单层page × num_layers
canonical_kv_caches = CanonicalKVCaches(
    tensors=[CanonicalKVCacheTensor(tensor=tensor, page_size_bytes=page_size_bytes)],
    group_data_refs=[[CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=page_size_bytes)]]
)
```

**只有 Flash Attention NHD 能通过这个断言！**

### 3.5 跨层布局示意（Flash Attention NHD）

```
Flash Attention NHD 跨层连续布局

┌─────────────────────────────────────────────────────────┐
│ block_0: all 32 layers, K+V together                   │
│  = 32 × 32KB = 1MB  ← 一次 DMA 即可搬运                │
├─────────────────────────────────────────────────────────┤
│ block_1: all 32 layers, K+V together                   │
│  = 1MB                                                 │
├─────────────────────────────────────────────────────────┤
│ block_2: ...                                           │
│ ...                                                    │
└─────────────────────────────────────────────────────────┘

关键特性：给定一个 block_id，所有层的 KV 数据在物理上连续！
```

### 3.6 跨层块大小公式

```
跨层块大小 = page_size_bytes × num_layers
           = (2 × num_kv_heads × head_size × block_size) × num_layers
```

以 Llama-3.1-8B-Instruct 为例（`num_layers=32, num_kv_heads=8, head_size=128, block_size=16`）：

```
跨层块大小 = 2 × 8 × 128 × 16 × 32 = 1,048,576 bytes = 1MB
```

正好落入博客中描述的 0.5-2MB 范围，此时 DMA (`cuMemcpyBatchAsync`) 达到最优吞吐 ~50GB/s。

**这就是博客里描述的 `2 × num_layers × block_size` 布局。** 更准确地说，完整的因子是 `2 × num_kv_heads × head_size × block_size × num_layers`。

对于 FlashInfer 和 Triton，虽然 `allocate_uniform_kv_caches()` 也分配了一个统一的跨层 tensor，但由于物理布局中 `num_blocks` 不在第 0 维，无法通过 `register_cross_layers_kv_cache()` 的断言。这些后端会回退到逐层的 `register_kv_caches()` 路径，只是叠加上 `block_size_factor` 来提高传输效率。

---

## 四、block_size_factor——GPU→CPU 子块合并

这是与层合并**正交**的另一个概念。在 CPU offload handler ([cpu/gpu_worker.py:L385-L445](../v1/kv_offload/cpu/gpu_worker.py#L385-L445)) 中：

```python
gpu_page_size_bytes = kv_cache_tensor.page_size_bytes  # 例如 32KB
cpu_page_size_bytes = gpu_page_size_bytes * block_size_factor  # 例如 32KB × 32 = 1MB

cpu_tensor = torch.zeros(
    (num_cpu_blocks, cpu_page_size_bytes),
    dtype=torch.int8,
    device="cpu",
)
```

`block_size_factor` 是将 N 个 GPU 逻辑块（子块）合并为 1 个 CPU 物理块。在 [cpu/spec.py:L44](../v1/kv_offload/cpu/spec.py#L44) 中：

```python
kv_bytes_per_block = total_gpu_kv_bytes // kv_cache_config.num_blocks
kv_bytes_per_offloaded_block = kv_bytes_per_block * block_size_factor
```

### 工作方式

子块指针计算在 [cpu/gpu_worker.py:L38-L82](../v1/kv_offload/cpu/gpu_worker.py#L38-L82) 的 `compute_sub_block_ptrs()` 中：

```python
# 每个 block 包含 block_size_factor 个子块
# 子块 j 的指针 = base_ptr + block_id × row_stride + j × sub_block_size
sub_block_size = tensor.shape[1] // block_size_factor
sub_offsets = np.arange(block_size_factor, dtype=np.int64) * sub_block_size
all_ptrs = (
    base_ptr + block_ids.astype(np.int64)[:, np.newaxis] * row_stride
) + sub_offsets[np.newaxis, :]
```

### 示意图

```
block_size_factor = 4 示意：

GPU blocks (32KB each):   [b0] [b1] [b2] [b3]   [b4] [b5] [b6] [b7] ...
                              ↘________________↙      ↘________________↙
CPU blocks (128KB each):  [      cpu_b0        ]   [      cpu_b1        ] ...

G2H 时：expand b0→b3 为 4 个子块指针 → batch memcpy → 写入 cpu_b0
H2G 时：expand b0→b3 为 4 个子块指针 → batch memcpy → 从 cpu_b0 读取
```

由于 CPU 块比 GPU 块大 N 倍，首尾块可能存在对齐偏移（partial block），通过 `skip_count` 处理：

```python
compute_sub_block_ptrs(
    group_src,
    self.src_block_size_factor,
    all_src[op_idx:end_idx],
    self.src_tensors[t_idx],
    skip_count=src_logical_blocks_to_skip,  # 跳过首块的前几个子块
)
```

---

## 五、GPU-CPU 传输路径总览

### 5.1 CanonicalKVCaches 两层含义

| 路径 | CanonicalKVCaches 结构 | page_size_bytes 含义 |
|------|-----------------------|---------------------|
| 逐层 | N 个 CanonicalKVCacheTensor，每个对应一个 shared_by 组 | 单层 page_size（~32KB） |
| 跨层 | 1 个 CanonicalKVCacheTensor | 单层 page_size × num_layers（~1MB） |

### 5.2 传输的批量处理

在 [cpu/gpu_worker.py:L156-L316](../v1/kv_offload/cpu/gpu_worker.py#L156-L316) 的 `SingleDirectionOffloadingHandler.transfer_async()` 中：

```python
# 遍历每个 group 的每个 layer data ref，展开 block_id 为子块指针
for group_size, block_idx, group_data_refs in zip(
    group_sizes, block_indices, kv_cache_groups_data_refs
):
    for data_ref in group_data_refs:
        t_idx = data_ref.tensor_idx
        compute_sub_block_ptrs(group_src, self.src_block_size_factor, all_src[...],
                               self.src_tensors[t_idx], skip_count=...)
        compute_sub_block_ptrs(group_dst, self.dst_block_size_factor, all_dst[...],
                               self.dst_tensors[t_idx], skip_count=...)
        all_sizes[op_idx:end_idx] = data_ref.page_size_bytes  # 每个子块的大小

# 一次 batch 调用完成所有子块搬运
ops.swap_blocks_batch(batch_src, batch_dst, batch_sizes, is_src_access_order_any=...)
```

- **逐层 + block_size_factor > 1**：每个 block 展开为 N 个子块指针，数据大小 = `page_size_bytes / block_size_factor`（每个子块）
- **跨层 + block_size_factor = 1**：不需要展开，单个 CanonicalKVCacheTensor 的 page_size 已经是 `num_layers × page_size_bytes`，一次搬运 1MB

---

## 六、端到端对比总结

| 维度 | 逐层布局（默认） | 跨层布局（优化） |
|------|-----------------|-----------------|
| 启用条件 | 默认 | KV connector + Flash Attn NHD + 单一 group |
| 单 block 大小 | `2 × n_kv_heads × head_size × block_size` (~32KB) | `× num_layers` (~1MB) |
| CanonicalKVCaches | N 个 tensor，每层一个 | 1 个 tensor，全层一体 |
| Flash Attention | ✅ K/V 各半，两个 tensor | ✅ 全层 K+V 一体，物理连续 |
| FlashInfer | ✅ 每层一个 tensor | ⚠️ 有 uniform tensor 但 block 不连续，回退逐层 |
| Triton | ✅ 每层一个 tensor | ⚠️ 同上，回退逐层 |
| `block_size_factor` | ✅ 合并 GPU→CPU 块 | ✅ 也可叠加使用 |
| DMA 效率 | 块太小（~32KB），效率低 | 块 ~1MB，DMA 吞吐 ~50GB/s |
| 对应博客描述 | 旧布局（高度碎片化） | 新布局（`2×num_layers×block_size`） |

---

## 七、关键文件索引

| 用途 | 文件路径 |
|------|---------|
| 布局模式选择 | [v1/worker/gpu_model_runner.py:L6855-L6867](../v1/worker/gpu_model_runner.py#L6855-L6867) |
| uniform 条件判断 | [v1/worker/kv_connector_model_runner_mixin.py:L121-L188](../v1/worker/kv_connector_model_runner_mixin.py#L121-L188) |
| 跨层 tensor 分配 | [v1/worker/kv_connector_model_runner_mixin.py:L192-L282](../v1/worker/kv_connector_model_runner_mixin.py#L192-L282) |
| Flash Attn stride_order | [v1/attention/backends/flash_attn.py:L156-L171](../v1/attention/backends/flash_attn.py#L156-L171) |
| FlashInfer stride_order | [v1/attention/backends/flashinfer.py:L374-L390](../v1/attention/backends/flashinfer.py#L374-L390) |
| Triton stride_order | [v1/attention/backends/triton_attn.py:L333-L349](../v1/attention/backends/triton_attn.py#L333-L349) |
| CanonicalKVCaches 定义 | [v1/kv_offload/base.py:L275-L324](../v1/kv_offload/base.py#L275-L324) |
| 逐层注册 | [distributed/.../offloading/worker.py:L59-L235](../../distributed/kv_transfer/kv_connector/v1/offloading/worker.py#L59-L235) |
| 跨层注册 | [distributed/.../offloading/worker.py:L239-L280](../../distributed/kv_transfer/kv_connector/v1/offloading/worker.py#L239-L280) |
| KV cache config 构建 | [v1/core/kv_cache_utils.py:L815-L1316](../v1/core/kv_cache_utils.py#L815-L1316) |
| CPU offload handler | [v1/kv_offload/cpu/gpu_worker.py:L385-L445](../v1/kv_offload/cpu/gpu_worker.py#L385-L445) |
| block_size_factor 计算 | [v1/kv_offload/cpu/spec.py:L37-L55](../v1/kv_offload/cpu/spec.py#L37-L55) |
| 子块指针计算 | [v1/kv_offload/cpu/gpu_worker.py:L38-L82](../v1/kv_offload/cpu/gpu_worker.py#L38-L82) |
| 批量传输调用 | [v1/kv_offload/cpu/gpu_worker.py:L278-L300](../v1/kv_offload/cpu/gpu_worker.py#L278-L300) |