# vLLM 架构梳理

> 基于 `vllm-project/vllm` main 分支，V1 引擎架构。

## 1. 整体分层

```
┌──────────────────────────────────────────────────────────────────┐
│                       入口层 (entrypoints)                        │
│  OpenAI Server │ Anthropic │ LLM (offline) │ Pooling │ MCP       │
│  Realtime │ Speech-to-text │ Generative Scoring │ SageMaker      │
│  gRPC │ CLI │ Serve (render/sleep/tokenize/lora/rpc/profile/     │
│         cache/disagg/elastic_ep/rlhf/instrumentator)             │
├──────────────────────────────────────────────────────────────────┤
│                       引擎层 (engine)                             │
│  LLMEngine ──→ EngineCoreClient ──→ EngineCore                   │
│  AsyncLLM    (通信代理: ZMQ/InProc)  (核心调度循环)               │
│  InputProcessor │ OutputProcessor │ Detokenizer │ Logprobs       │
├──────────────────────────────────────────────────────────────────┤
│                       调度层 (core)                               │
│  Scheduler │ KVCacheManager │ BlockPool │ KVCacheCoordinator     │
│  EncoderCache │ SingleTypeKVCacheManager │ KV Cache Metrics      │
├──────────────────────────────────────────────────────────────────┤
│                       采样层 (sample)                             │
│  Sampler │ LogitsProcessor │ TopK/TopP Ops │ Penalties          │
├──────────────────────────────────────────────────────────────────┤
│                       执行层 (executor)                           │
│  UniProc │ Multiproc │ Ray │ RayV2 │ ExternalLauncher            │
├──────────────────────────────────────────────────────────────────┤
│                       计算层 (worker)                             │
│  ModelRunner │ Attention Backends │ MoE │ Quantization            │
│  PoolingRunner │ EncoderRunner │ Structured Output               │
├──────────────────────────────────────────────────────────────────┤
│                       KV Offload 层                               │
│  CPU Offload (ARC/LRU) │ Simple KV Offload │ Reuse Manager       │
├──────────────────────────────────────────────────────────────────┤
│                       硬件层                                      │
│  CUDA Kernels │ Triton Kernels │ CPU Kernels │ ROCm │ TPU │ XPU  │
└──────────────────────────────────────────────────────────────────┘
```

## 2. 核心流程

```
用户请求
  │
  ▼
LLMEngine.add_request() / AsyncLLM.generate()
  │ InputProcessor: EngineInput → EngineCoreRequest
  ▼
EngineCoreClient (ZMQ/InProc)
  │
  ▼
EngineCore (核心循环)
  │
  ├─► Scheduler.schedule()
  │     ├─ 请求队列管理 (FCFS / Priority / Hybrid)
  │     ├─ KV Cache 分配 (prefix caching, KVCacheCoordinator)
  │     ├─ 推测解码调度 (Eagle/Medusa/Ngram/MTP)
  │     └─ 输出 SchedulerOutput
  │
  ├─► Executor.execute_model(scheduler_output)
  │     ├─ Worker.prepare_input()
  │     ├─ ModelRunner.execute_model()
  │     │    ├─ Attention (FlashAttn / FlashInfer / MLA / Triton)
  │     │    ├─ MoE (FusedMoE / DeepEP)
  │     │    └─ Linear (量化 GEMM)
  │     └─ 输出 ModelRunnerOutput
  │
  ├─► Sampler.sample()
  │     ├─ LogitsProcessor (temperature, top_k, top_p, penalties)
  │     ├─ Structured Output (xgrammar / outlines / guidance)
  │     └─ 输出采样后的 token_ids
  │
  └─► Scheduler.update_from_output()
        ├─ 更新请求状态, 释放 KV Cache
        └─ KV Offload (CPU offload / Simple KV offload)
  │
  ▼
OutputProcessor: EngineCoreOutput → RequestOutput
  │  Detokenizer: token_ids → text
  │  Logprobs: 计算 logprobs
  ▼
返回给用户 (streaming / non-streaming)
```

## 3. 关键模块

### 3.1 入口层 (`vllm/entrypoints/`)

| 入口 | 路径 | 说明 |
|------|------|------|
| OpenAI Server | `openai/` | Chat Completions, Completions, Responses, Models |
| Anthropic Server | `anthropic/` | Anthropic Messages API 兼容 |
| LLM (offline) | `llm.py` | 离线批量推理 |
| Pooling | `pooling/` | Embedding, Classification, Scoring, Pooling |
| MCP | `mcp/` | Model Context Protocol 工具服务器 |
| Realtime | `openai/realtime/` | WebSocket 实时语音/对话 |
| Speech-to-text | `openai/speech_to_text/` | 语音转文字 |
| Generative Scoring | `openai/generative_scoring/` | 生成式评分 |
| SageMaker | `sagemaker/` | AWS SageMaker 集成 |
| gRPC | `grpc_server.py` | gRPC 推理服务 |
| CLI | `cli/` | `vllm serve/benchmark/run_batch` 命令行 |

**Serve 子模块** (`entrypoints/serve/`):

| 模块 | 说明 |
|------|------|
| `render/` | 渲染服务 (图片/视频生成) |
| `sleep/` | 睡眠模式 (权重卸载到 CPU) |
| `tokenize/` | 独立 tokenize/detokenize 服务 |
| `lora/` | LoRA 适配器管理 API |
| `rpc/` | RPC 接口 (分布式通信) |
| `profile/` | Profiling 接口 |
| `cache/` | KV Cache 管理 API |
| `disagg/` | 分离式推理 (prefill/decode 分离) |
| `elastic_ep/` | 弹性 Expert Parallel |
| `rlhf/` | RLHF 训练推理接口 |
| `instrumentator/` | Prometheus 指标、健康检查 |

### 3.2 配置系统 (`vllm/config/`)

```
VllmConfig (vllm/config/vllm.py)
├── ModelConfig          # 模型架构、dtype、max_model_len
├── CacheConfig          # KV Cache 类型、block_size、prefix caching
├── ParallelConfig       # TP/PP/DP/EP/CP 并行配置
├── SchedulerConfig      # max_num_seqs、max_num_batched_tokens
├── CompilationConfig    # torch.compile、CUDA Graph 配置
├── KernelConfig         # 算子优先级 (flashinfer vs triton vs native)
├── LoRAConfig           # LoRA 适配器配置
├── SpeculativeConfig    # 推测解码 (Eagle/MTP/Ngram)
├── KVTransferConfig     # KV Cache 传输 (Mooncake/LMCache/NIXL)
├── AttentionConfig      # 注意力后端选择
├── LoadConfig           # 模型加载方式
├── DeviceConfig         # 设备配置
└── ObservabilityConfig  # 监控/追踪
```

优化等级 (O0-O3) 控制编译和融合策略:
- **O0**: 无优化，快速启动
- **O1**: Dynamo+Inductor 编译 + Piecewise CUDA Graph
- **O2** (默认): O1 + Full CUDA Graph + allreduce-rms 融合
- **O3**: O2 + flashinfer autotune

### 3.3 引擎层 (`vllm/v1/engine/`)

| 文件 | 职责 |
|------|------|
| `llm_engine.py` | 同步入口，管理 InputProcessor + OutputProcessor |
| `async_llm.py` | **AsyncLLM** — 异步入口，支持 `async for` 流式生成 |
| `core.py` | **EngineCore** — 核心调度循环，step() 驱动 |
| `core_client.py` | 通信代理 (InProc / ZMQ Multiproc) |
| `input_processor.py` | 输入预处理 (tokenize, multimodal) |
| `output_processor.py` | 输出后处理 (streaming, RequestOutput 构造) |
| `detokenizer.py` | token_ids → text 解码 |
| `logprobs.py` | Logprobs 计算 |
| `parallel_sampling.py` | 并行采样 (n > 1) |
| `tensor_ipc.py` | 跨进程 Tensor 传输 (共享内存) |
| `coordinator.py` | DP 协调器 (MoE wave 协调, LB 统计) |
| `exceptions.py` | 引擎异常定义 |
| `utils.py` | 引擎工具函数 |

### 3.4 调度层 (`vllm/v1/core/`)

| 文件 | 职责 |
|------|------|
| `sched/scheduler.py` | **Scheduler** — 请求调度、KV Cache 分配、前缀缓存 |
| `sched/async_scheduler.py` | 异步调度 (推测解码) |
| `sched/request_queue.py` | 请求队列 (FCFS / Priority / Hybrid) |
| `sched/output.py` | SchedulerOutput 数据结构 |
| `sched/utils.py` | 调度工具函数 |
| `kv_cache_manager.py` | Block 级 KV Cache 分配/回收 |
| `single_type_kv_cache_manager.py` | 单一类型 KV Cache 管理器 |
| `kv_cache_coordinator.py` | **KVCacheCoordinator** — 跨 DP 协调 KV Cache |
| `kv_cache_utils.py` | KV Cache 配置生成、block hash |
| `kv_cache_metrics.py` | KV Cache 使用率/命中率指标 |
| `block_pool.py` | Block 内存池 |
| `encoder_cache_manager.py` | 多模态 Encoder 缓存 |

### 3.5 采样层 (`vllm/v1/sample/`)

| 文件 | 职责 |
|------|------|
| `sampler.py` | **Sampler** — 采样主逻辑，调度 logits 处理 |
| `metadata.py` | 采样元数据 (temperature, top_k, top_p, seed) |
| `rejection_sampler.py` | 推测解码拒绝采样 |
| `thinking_budget_state.py` | 思考预算状态管理 |
| `logits_processor/` | Logits 处理器 (bad_words, penalties, state) |
| `ops/` | 采样算子 (topk_topp_sampler, logprobs, penalties) |

### 3.6 结构化输出 (`vllm/v1/structured_output/`)

| 文件 | 职责 |
|------|------|
| `backend_xgrammar.py` | XGrammar 后端 (默认) |
| `backend_outlines.py` | Outlines 后端 |
| `backend_guidance.py` | Guidance 后端 |
| `backend_lm_format_enforcer.py` | LM Format Enforcer 后端 |
| `backend_types.py` | 后端类型定义 |
| `utils.py` | 结构化输出工具函数 |
| `request.py` | 结构化输出请求处理 |

### 3.7 执行层 (`vllm/v1/executor/`)

| 执行器 | 适用场景 |
|--------|---------|
| `UniProcExecutor` | 单进程 (开发/调试) |
| `MultiprocExecutor` | 多进程 (生产) |
| `RayExecutor` | Ray 集群 (V1) |
| `RayExecutorV2` | Ray 集群 (V2, 优化) |
| `ExternalLauncherExecutor` | 外部启动器 (torchrun) |

### 3.8 注意力后端 (`vllm/v1/attention/backends/`)

```
Attention Backends
├── FlashAttn        # 通用 GPU (CUDA/ROCm)
├── FlashInfer       # Hopper+ GPU 优化
├── TritonAttn       # Triton 实现 (跨平台)
├── FlexAttention    # PyTorch FlexAttention
├── MLA 系列          # DeepSeek MLA 专用
│   ├── FlashMLA     # Hopper GPU
│   ├── CutlassMLA   # Blackwell GPU
│   ├── FlashInferMLA
│   ├── TritonMLA
│   └── AITER_MLA    # ROCm
├── Mamba/SDM        # Mamba SSM 注意力
├── TreeAttn         # 推测解码树注意力
├── LinearAttn       # 线性注意力
└── GDN              # Gated DeltaNet
```

### 3.9 模型层 (`vllm/model_executor/`)

```
model_executor/
├── models/          # 各模型实现 (Llama, Qwen, DeepSeek, ...)
├── layers/          # 基础层
│   ├── attention/   # Attention 层
│   ├── fused_moe/   # MoE 层 (router, experts, runner)
│   ├── fla/         # Flash Linear Attention
│   └── quantization/# 量化层
├── kernels/         # 线性层 kernel (FP8, NVFP4, MXFP8, ScaledMM)
└── model_loader/    # 模型权重加载
```

### 3.10 分布式通信 (`vllm/distributed/`)

```
distributed/
├── kv_transfer/     # KV Cache 跨节点传输
│   └── kv_connector/v1/
│       ├── mooncake/    # Mooncake 传输
│       ├── lmcache/     # LMCache
│       ├── nixl/        # NIXL (NVLink)
│       ├── p2p/         # P2P NCCL
│       ├── offloading/  # CPU Offload
│       └── hf3fs/       # 3FS 文件系统
├── ec_transfer/     # Encoder Cache 传输
├── eplb/            # Expert Parallel Load Balancing
├── elastic_ep/      # 弹性 Expert Parallel
└── weight_transfer/ # 权重传输 (RL 训练)
```

### 3.11 KV Offload (`vllm/v1/kv_offload/`, `vllm/v1/simple_kv_offload/`)

| 模块 | 路径 | 说明 |
|------|------|------|
| CPU Offload | `kv_offload/cpu/` | 将 KV Cache 卸载到 CPU 内存 |
| ARC Policy | `kv_offload/cpu/policies/arc.py` | Adaptive Replacement Cache 策略 |
| LRU Policy | `kv_offload/cpu/policies/lru.py` | LRU 淘汰策略 |
| Simple KV Offload | `simple_kv_offload/` | 轻量级 KV offload 实现 |
| Reuse Manager | `kv_offload/reuse_manager.py` | KV Cache 复用管理 |
| Factory | `kv_offload/factory.py` | Offload 后端工厂 |

### 3.12 Pool 层 (`vllm/v1/pool/`)

| 文件 | 职责 |
|------|------|
| `metadata.py` | Pooling 元数据 (embed/classify/score) |
| `late_interaction.py` | 延迟交互 (ColBERT 等) |

### 3.13 推测解码 (`vllm/v1/spec_decode/`)

| 文件 | 职责 |
|------|------|
| `eagle.py` | EAGLE 推测解码 |
| `medusa.py` | Medusa 多头推测 |
| `ngram_proposer.py` | N-gram 推测 (CPU) |
| `ngram_proposer_gpu.py` | N-gram 推测 (GPU) |
| `draft_model.py` | 草稿模型管理 |
| `dflash.py` | DFlash 推测 |
| `gemma4.py` | Gemma 4 推测 |
| `suffix_decoding.py` | 后缀解码 |
| `llm_base_proposer.py` | LLM 基础提议器 |
| `metadata.py` | 推测解码元数据 |
| `metrics.py` | 推测解码指标 (接受率等) |
| `utils.py` | 推测解码工具函数 |

### 3.14 指标与监控 (`vllm/v1/metrics/`)

| 文件 | 职责 |
|------|------|
| `prometheus.py` | Prometheus 指标暴露 |
| `stats.py` | 统计信息收集 |
| `loggers.py` | 日志记录 (吞吐、延迟) |
| `reader.py` | 指标读取接口 |
| `perf.py` | 性能分析 |
| `ray_wrappers.py` | Ray 环境指标包装 |
| `utils.py` | 指标工具函数 |

## 4. 数据流

```
EngineInput (用户输入)
  │  prompt, sampling_params, multimodal_data
  ▼
EngineCoreRequest (内部请求)
  │  request_id, prompt_token_ids, sampling_params
  ▼
Request (调度器内部)
  │  status, token_ids, kv_cache_blocks, ...
  ▼
SchedulerOutput (调度结果)
  │  scheduled_new_reqs, scheduled_cached_reqs, finished_req_ids
  ▼
ModelRunnerOutput (模型输出)
  │  hidden_states / logits, kv_connector_output
  ▼
SamplerOutput (采样结果)
  │  sampled_token_ids, logprobs (经 structured output 约束)
  ▼
EngineCoreOutput (引擎输出)
  │  request_id, new_token_ids, finish_reason
  ▼
RequestOutput (用户输出)
  │  prompt, outputs[].text, metrics, finish_reason
```

## 5. 关键设计决策

1. **V1 是唯一引擎**: `vllm/engine/llm_engine.py` 直接 `from vllm.v1.engine.llm_engine import LLMEngine`
2. **EngineCore 与 LLMEngine 解耦**: 通过 `EngineCoreClient` 通信，支持 InProc 和 ZMQ Multiproc 两种模式
3. **Scheduler 可插拔**: 通过 `SchedulerInterface` 抽象，支持同步/异步调度
4. **注意力后端自动选择**: 根据 GPU 能力 (compute capability) 和模型类型自动选择最优后端
5. **Block 级 KV Cache 管理**: 支持 prefix caching、chunked prefill、KV offloading (CPU/simple)
6. **torch.compile + CUDA Graph**: 通过编译优化和 graph capture 减少 kernel launch overhead
7. **结构化输出可插拔**: 支持 xgrammar、outlines、guidance、lm-format-enforcer 多种后端
8. **推测解码多策略**: EAGLE、Medusa、N-gram、DFlash、Gemma4 等多种推测方法
9. **多入口协议**: 同时支持 OpenAI、Anthropic、SageMaker、gRPC 等多种 API 协议
10. **分离式推理**: 通过 `disagg` 模块支持 prefill/decode 分离部署
