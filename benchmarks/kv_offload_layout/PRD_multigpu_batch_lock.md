# PRD: Multi-GPU `cuMemcpyBatchAsync` Lock Contention Microbench

## Problem Statement

Current KV offload layout microbench coverage is single-process, single-GPU, and focused on the memcpy shape difference between one contiguous copy and the current SimpleCPUOffloadConnector batched `layer x block` copy shape. That is enough to answer whether small batched copies leave bandwidth on the table on one GPU, but it does not answer a separate question that matters for KV cache offload on multi-GPU serving systems:

- Does `cuMemcpyBatchAsync` show driver-side lock contention when multiple GPUs submit batched copies concurrently?
- If multi-GPU scaling is poor, is the regression visible in submission latency, completion latency, or both?
- Is the scaling behavior materially worse for batched copies than for the contiguous-copy baseline?

From the user’s perspective, the missing capability is a focused microbench that isolates multi-GPU concurrent H2D and D2H copy behavior without pulling in scheduler, model execution, or vLLM serving noise.

## Solution

Add a dedicated multi-GPU KV offload layout microbench that keeps the current benchmark philosophy: measure copy-shape behavior in isolation, with minimum code and no serving integration. The new benchmark will run one process with one worker thread per GPU, synchronize all workers with a barrier, then measure concurrent copy submission and completion for both:

- contiguous `cudaMemcpyAsync`
- batched `cuMemcpyBatchAsync`

The benchmark will support explicit GPU lists, H2D and D2H directions, and JSON output for scripted sweeps. The primary goal is to detect whether multi-GPU concurrent batched copies suffer from submission-side contention that is not explained by plain link saturation.

## User Stories

1. As a KV offload developer, I want a multi-GPU microbench for concurrent batched copies, so that I can tell whether `cuMemcpyBatchAsync` has scaling issues beyond single-GPU bandwidth loss.
2. As a KV offload developer, I want the benchmark to stay separate from serving and scheduler code, so that I can attribute regressions to copy behavior instead of runtime noise.
3. As a KV offload developer, I want to compare contiguous and batched copy modes under the same multi-GPU harness, so that I can isolate whether contention is API-specific or platform-wide.
4. As a KV offload developer, I want explicit GPU-list selection, so that I can compare same-switch, cross-switch, and full-machine GPU sets deterministically.
5. As a KV offload developer, I want all GPU workers to start each measured round together, so that the benchmark maximizes contention and does not hide locking behind staggered starts.
6. As a KV offload developer, I want separate submit-time and complete-time metrics, so that I can distinguish driver-side submission contention from transport or DMA execution bottlenecks.
7. As a KV offload developer, I want aggregate throughput numbers, so that I can compare 1-GPU, 2-GPU, 4-GPU, and 8-GPU scaling directly.
8. As a KV offload developer, I want per-GPU statistics, so that I can see whether contention is symmetric or whether one GPU becomes a straggler.
9. As a KV offload developer, I want JSON output, so that I can run scripted sweeps and summarize results later without scraping stdout.
10. As a KV offload developer, I want stdout summaries, so that I can quickly inspect whether a run is healthy before doing deeper analysis.
11. As a KV offload developer, I want H2D and D2H to both be supported, so that I can test store-side and load-side offload paths independently.
12. As a KV offload developer, I want to reuse the same bytes-per-block and layer/block vocabulary as the current layout microbench, so that single-GPU and multi-GPU results stay comparable.
13. As a KV offload developer, I want the batched mode to preserve the current SimpleCPUOffloadConnector copy shape semantics, so that the benchmark remains relevant to real KV offload behavior.
14. As a performance investigator, I want the benchmark to report scaling against the 1-GPU baseline, so that I can spot under-scaling immediately.
15. As a performance investigator, I want percentiles such as p50 and p95 for submit and complete time, so that I can detect lock-driven tail latency and not just median drift.
16. As a performance investigator, I want the first version to avoid G2G and P2P complexity, so that the initial signal stays tied to KV cache offload and host-device transport.
17. As a code maintainer, I want the existing single-GPU benchmark to remain untouched in behavior, so that historical layout results remain comparable.
18. As a code maintainer, I want the new benchmark to be a separate module, so that multi-GPU coordination logic does not make the single-GPU benchmark harder to read or maintain.
19. As a code maintainer, I want the result schema to be stable and explicit, so that follow-up sweep wrappers can build on it without rewriting the benchmark core.
20. As a future implementer, I want the design to leave room for a second-phase multi-process variant, so that we can later distinguish process-local locking from broader driver or platform contention without redesigning the first benchmark.

## Implementation Decisions

- The work will extend the `kv_offload_layout` microbench suite rather than the serving or offload experiment suite. This keeps the benchmark scoped to copy-shape behavior instead of scheduler or request-path behavior.
- A new dedicated benchmark module will be added for multi-GPU lock-contention experiments. The existing single-GPU contiguous-vs-batched benchmark remains the reference baseline and should not be folded into a more complex code path.
- The first implementation target is one process with one worker thread per GPU. This is the minimum design that can expose process-local runtime or driver locking without introducing multi-process orchestration noise.
- GPU selection will be explicit rather than count-based. The caller will provide an ordered GPU list so topology-sensitive runs can be reproduced exactly.
- Each measured round will use a synchronization barrier so all worker threads submit work at the same time. This is required to maximize contention and make lock symptoms observable.
- The benchmark will support two copy modes:
  - contiguous copy as the baseline
  - batched `cuMemcpyBatchAsync` copy matching the current SimpleCPUOffloadConnector copy organization
- The benchmark will support two directions only in the first phase:
  - H2D
  - D2H
- G2G and P2P paths are explicitly excluded from the first phase because the question being answered is KV cache offload behavior, not peer transport behavior.
- The benchmark will record two timing surfaces per GPU:
  - submit time: the host-side interval around enqueueing the copy work
  - complete time: the end-to-end interval until the submitted work is complete on the stream
- Aggregate throughput will be reported alongside per-GPU throughput. The benchmark must make it easy to compare `N`-GPU aggregate scaling against the 1-GPU baseline.
- The result model will include both per-GPU and aggregate statistics. This is the deep module in the design: a stable result schema that wrappers and future summaries can consume without knowing thread or stream internals.
- The result model will include the experiment configuration needed to reproduce the run: selected GPUs, direction, mode, layer count, block count, bytes per block, warmup count, repeat count, and synchronization strategy.
- JSON output is the primary machine-readable artifact. Stdout remains a human-readable summary only.
- The design intentionally does not add PCIe monitor integration in the first implementation decision set. Existing wrappers may be adapted later, but the first benchmark should prove lock-contention signal with benchmark-native timing first.
- A second-phase extension for `N` processes on `N` GPUs is acknowledged but deferred. The first benchmark should produce a clean answer on whether single-process multi-threaded concurrent submission already shows contention.

## Testing Decisions

- Good tests will validate external behavior and invariants, not implementation details such as internal thread sequencing.
- Tests should verify that argument parsing, GPU-list handling, result schema generation, and mode/direction validation behave correctly.
- Tests should verify summary and JSON-shape behavior for deterministic synthetic data where possible.
- Tests should not try to prove a particular GPU performance number. Performance values are environment-dependent and are not stable unit-test targets.
- The deepest testable module should be the result aggregation and statistics layer. It should accept timing samples and emit deterministic per-GPU and aggregate summaries.
- The next testable module should be configuration normalization: explicit GPU list parsing, direction/mode selection, and run-shape validation.
- Any integration-style smoke test should focus on “run completes and writes well-formed output” rather than asserting throughput thresholds.
- Prior art in the codebase:
  - the current single-GPU `kv_offload_layout` benchmark shows the intended benchmark scope and vocabulary
  - existing `simple_kv_offload` tests show the project preference for testing behavior rather than internals when dealing with offload components

## Out of Scope

- vLLM multi-GPU serving benchmarks
- tensor parallel, pipeline parallel, or data parallel request-path experiments
- scheduler integration
- model execution integration
- G2G or P2P copy modes
- multi-process lock-contention benchmarking in the first phase
- plotting or dashboard generation in the first phase
- automatic topology discovery or topology-aware scheduling
- changing the current SimpleCPUOffloadConnector implementation
- redesigning the existing single-GPU `A vs C` benchmark

## Further Notes

- The benchmark should continue using KV offload layout vocabulary: layers, blocks, bytes per block, H2D, D2H, contiguous copy, and batched copy.
- The benchmark should preserve relevance to the current SimpleCPUOffloadConnector path by keeping the batched copy semantics aligned with the current `layer x block` organization.
- The main decision already made in design review is that this work is about detecting driver-side contention symptoms, not about proving end-to-end serving wins.
- If the first-phase benchmark shows poor multi-GPU scaling for both contiguous and batched modes, the next investigation branch is platform or transport contention. If only batched mode degrades materially, the next branch is API- or submission-path-specific contention.
