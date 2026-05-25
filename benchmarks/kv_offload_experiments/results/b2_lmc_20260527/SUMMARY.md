# B2-LMC S1 Results (2026-05-27)

Model: `/data1/models/Qwen/Qwen3-4B-Instruct-2507`

Workload: S1, `T0 prompt_A -> T1 prompt_B -> T2 prompt_A`, `prefix_tokens=65500`, `max_tokens=8`, `max_model_len=65536`, single RTX 3090.

| Variant | Config | T0 TTFT ms | T1 TTFT ms | T2 TTFT ms | T0/T2 | Notes |
|---------|--------|------------|------------|------------|-------|-------|
| Simple vLLM | `SimpleCPUOffloadConnector`, `kv-offloading-size=24` | 27631.9 | 28241.6 | 944.9 | 29.24x | Current vLLM baseline |
| B2-LMC-base | `local_cpu=true`, `chunk_size=256` | 28484.6 | 29669.0 | 989.7 | 28.78x | Retrieved 34048 tokens, 4.6758 GB, 411.7 ms, 11.36 GB/s |
| B2-LMC-async | `enable_async_loading=true` | 28511.6 | 29614.2 | 979.3 | 29.11x | Retrieved 34048 tokens, 4.6758 GB, 409.8 ms, 11.41 GB/s |
| B2-LMC-layerwise | `enable_async_loading=true`, `use_layerwise=true` | 28486.4 | 28986.9 | 21877.8 | 1.30x | T2 log shows `LMCache hit tokens: 0`; layerwise path did not reuse in this run |
| B2-LMC-threshold1024 | `enable_async_loading=true`, `min_retrieve_tokens=1024` | 28556.7 | 29525.1 | 979.1 | 29.17x | Same as async for a 65K hit; threshold only matters for small hits |

Conclusion: on this S1 single-request reload workload, LMCache base/async/threshold does not improve TTFT over Simple vLLM. It is roughly equal within run noise. Layerwise regresses because this run did not hit/retrieve on T2.

Why TTFT: S1 isolates long-prefill reload. The visible user win is first-token latency after a repeated long prefix. Throughput/req-s is dominated by concurrency and decode scheduling, and with `max_tokens=8` it does not isolate CPU->GPU reload cost.
