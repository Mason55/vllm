# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-GPU lock-contention microbench for contiguous vs batched copies.

Focus:
    Measure whether concurrent multi-GPU ``cuMemcpyBatchAsync`` submission shows
    lock-like scaling loss compared with a contiguous ``cudaMemcpyAsync``
    baseline.

Scope:
    - One worker per GPU
    - Thread or process execution model
    - Explicit GPU list
    - Barrier-synchronized measured rounds
    - H2D / D2H only
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from cuda.bindings import runtime as cudart

from vllm.platforms import current_platform
from vllm.v1.simple_kv_offload.cuda_mem_ops import build_params, copy_blocks

Direction = Literal["h2d", "d2h"]
Mode = Literal["contiguous", "batched"]
Pattern = Literal["contiguous", "random", "runs"]
ExecutionModel = Literal["thread", "process"]
RunMode = Literal["baseline-only", "concurrent-only", "both"]


@dataclass
class Sample:
    round_idx: int
    gpu: int
    submit_start_ns: int
    submit_end_ns: int
    complete_end_ns: int


@dataclass
class WorkerStats:
    gpu: int
    submit_us_p50: float
    submit_us_p95: float
    complete_us_p50: float
    complete_us_p95: float
    gbps_p50: float
    gbps_p95: float


@dataclass
class AggregateStats:
    submit_us_p50: float
    submit_us_p95: float
    complete_us_p50: float
    complete_us_p95: float
    gbps_p50: float
    gbps_p95: float


@dataclass
class RunStats:
    gpus: list[int]
    num_gpus: int
    bytes_per_gpu: int
    total_bytes: int
    per_gpu: list[WorkerStats]
    aggregate: AggregateStats


@dataclass
class ExperimentResult:
    execution_model: ExecutionModel
    run_mode: RunMode
    direction: Direction
    mode: Mode
    pattern: Pattern
    baseline_1gpu: RunStats | None
    concurrent_ngpu: RunStats | None
    speedup_vs_1gpu: float | None
    scaling_efficiency: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure multi-GPU concurrent contiguous vs batched copy scaling."
    )
    parser.add_argument(
        "--gpus",
        required=True,
        help="Comma-separated GPU list, e.g. 0,1 or 0,1,2,3",
    )
    parser.add_argument(
        "--direction",
        choices=("h2d", "d2h", "both"),
        default="both",
        help="Copy direction to benchmark.",
    )
    parser.add_argument(
        "--mode",
        choices=("contiguous", "batched", "both"),
        default="both",
        help="Copy mode to benchmark.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=36,
        help="Layer count for batched path.",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=256,
        help="Block count copied in one iteration.",
    )
    parser.add_argument(
        "--bytes-per-block",
        type=int,
        default=64 * 1024,
        help="Bytes per layer block for batched path.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations per GPU before measured rounds.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=20,
        help="Measured rounds per experiment.",
    )
    parser.add_argument(
        "--pattern",
        choices=("contiguous", "random", "runs", "all"),
        default="random",
        help="Block-id layout used by batched path.",
    )
    parser.add_argument(
        "--run-length",
        type=int,
        default=8,
        help="Run length when --pattern=runs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed for block-id generation.",
    )
    parser.add_argument(
        "--execution-model",
        choices=("thread", "process"),
        default="thread",
        help="Run one worker per GPU as threads or spawn'ed processes.",
    )
    parser.add_argument(
        "--run-mode",
        choices=("baseline-only", "concurrent-only", "both"),
        default="both",
        help="Run just the 1-GPU baseline, just the N-GPU concurrent phase, or both.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def parse_gpu_list(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError("GPU list must not be empty")
    gpus = [int(part) for part in parts]
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"GPU list contains duplicates: {value}")
    if min(gpus) < 0:
        raise ValueError(f"GPU ids must be >= 0: {value}")
    return gpus


def alloc_host(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.empty(shape, dtype=torch.uint8, device="cpu", pin_memory=True)


def make_layer_tensors(
    num_layers: int,
    num_blocks: int,
    bytes_per_block: int,
    device: str,
) -> dict[str, torch.Tensor]:
    return {
        f"layer_{idx}": torch.empty(
            (num_blocks, bytes_per_block),
            dtype=torch.uint8,
            device=device,
            pin_memory=(device == "cpu"),
        )
        for idx in range(num_layers)
    }


def init_pattern(tensor: torch.Tensor, seed: int) -> None:
    values = (
        torch.arange(tensor.numel(), dtype=torch.int64, device=tensor.device) + seed
    ) % 251
    tensor.copy_(values.to(torch.uint8).view_as(tensor))


def init_layer_patterns(tensors: dict[str, torch.Tensor], seed: int) -> None:
    for idx, tensor in enumerate(tensors.values()):
        init_pattern(tensor, seed + idx * 17)


def validate_contiguous(dst: torch.Tensor, src: torch.Tensor) -> None:
    if not torch.equal(dst, src):
        raise AssertionError("contiguous path validation failed")


def validate_layer_tensors(
    dst: dict[str, torch.Tensor],
    src: dict[str, torch.Tensor],
) -> None:
    for name in src:
        if not torch.equal(dst[name], src[name]):
            raise AssertionError(f"batched path validation failed at {name}")


def make_block_ids(
    num_blocks: int,
    pattern: Pattern,
    run_length: int,
    seed: int,
) -> list[int]:
    if pattern == "contiguous":
        return list(range(num_blocks))

    rng = random.Random(seed)
    if pattern == "random":
        block_ids = list(range(num_blocks))
        rng.shuffle(block_ids)
        return block_ids

    if run_length <= 0:
        raise ValueError("run_length must be > 0")
    if num_blocks % run_length != 0:
        raise ValueError("num_blocks must be divisible by run_length for runs pattern")

    run_starts = list(range(0, num_blocks, run_length))
    rng.shuffle(run_starts)
    block_ids: list[int] = []
    for start in run_starts:
        block_ids.extend(range(start, start + run_length))
    return block_ids


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires non-empty values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def gb_per_s(num_bytes: int, duration_us: float) -> float:
    seconds = duration_us / 1e6
    return num_bytes / seconds / (1024**3)


def cuda_memcpy_async(
    dst_ptr: int,
    src_ptr: int,
    num_bytes: int,
    direction: Direction,
    stream: torch.cuda.Stream,
) -> None:
    kind = (
        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        if direction == "h2d"
        else cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    )
    (err,) = cudart.cudaMemcpyAsync(
        dst_ptr,
        src_ptr,
        num_bytes,
        kind,
        stream.cuda_stream,
    )
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaMemcpyAsync failed: {err}")


class WorkerState:
    def __init__(
        self,
        gpu: int,
        direction: Direction,
        mode: Mode,
        num_layers: int,
        num_blocks: int,
        bytes_per_block: int,
        block_ids: list[int],
    ) -> None:
        device = torch.device(f"cuda:{gpu}")
        current_platform.set_device(device)
        torch.cuda.set_device(device)
        self.gpu = gpu
        self.direction = direction
        self.mode = mode
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.bytes_per_block = bytes_per_block
        self.total_bytes = num_layers * num_blocks * bytes_per_block
        self.stream = torch.cuda.Stream(device=device)
        self.src_contig: torch.Tensor | None = None
        self.dst_contig: torch.Tensor | None = None
        self.src_layers: dict[str, torch.Tensor] | None = None
        self.dst_layers: dict[str, torch.Tensor] | None = None
        self.batched_params = None

        self.block_ids = block_ids
        if mode == "contiguous":
            if direction == "h2d":
                self.src_contig = alloc_host((self.total_bytes,))
                self.dst_contig = torch.empty(
                    (self.total_bytes,), dtype=torch.uint8, device=device
                )
            else:
                self.src_contig = torch.empty(
                    (self.total_bytes,), dtype=torch.uint8, device=device
                )
                self.dst_contig = alloc_host((self.total_bytes,))
            init_pattern(self.src_contig, seed=11 + gpu * 101)
            self.dst_contig.zero_()
            return

        if direction == "h2d":
            self.src_layers = make_layer_tensors(
                num_layers, num_blocks, bytes_per_block, "cpu"
            )
            self.dst_layers = make_layer_tensors(
                num_layers, num_blocks, bytes_per_block, f"cuda:{gpu}"
            )
        else:
            self.src_layers = make_layer_tensors(
                num_layers, num_blocks, bytes_per_block, f"cuda:{gpu}"
            )
            self.dst_layers = make_layer_tensors(
                num_layers, num_blocks, bytes_per_block, "cpu"
            )
        init_layer_patterns(self.src_layers, seed=29 + gpu * 101)
        for tensor in self.dst_layers.values():
            tensor.zero_()
        self.batched_params = build_params(self.src_layers, self.dst_layers, self.stream)

    def run_op(self) -> None:
        if self.mode == "contiguous":
            assert self.src_contig is not None and self.dst_contig is not None
            cuda_memcpy_async(
                self.dst_contig.data_ptr(),
                self.src_contig.data_ptr(),
                self.total_bytes,
                self.direction,
                self.stream,
            )
            return
        assert self.batched_params is not None
        copy_blocks(self.block_ids, self.block_ids, self.batched_params)

    def synchronize(self) -> None:
        self.stream.synchronize()

    def validate(self) -> None:
        if self.mode == "contiguous":
            assert self.src_contig is not None and self.dst_contig is not None
            validate_contiguous(self.dst_contig.cpu(), self.src_contig.cpu())
            return
        assert self.src_layers is not None and self.dst_layers is not None
        validate_layer_tensors(
            {k: v.cpu() for k, v in self.dst_layers.items()},
            {k: v.cpu() for k, v in self.src_layers.items()},
        )


def run_group_thread(
    gpus: list[int],
    *,
    direction: Direction,
    mode: Mode,
    num_layers: int,
    num_blocks: int,
    bytes_per_block: int,
    warmup: int,
    repeat: int,
    pattern: Pattern,
    run_length: int,
    seed: int,
) -> RunStats:
    block_ids = make_block_ids(num_blocks, pattern, run_length, seed)
    ready_barrier = threading.Barrier(len(gpus))
    round_barrier = threading.Barrier(len(gpus))
    samples: dict[int, list[Sample]] = {gpu: [] for gpu in gpus}
    errors: list[BaseException] = []
    bytes_per_gpu = num_layers * num_blocks * bytes_per_block

    def worker(gpu: int) -> None:
        try:
            state = WorkerState(
                gpu=gpu,
                direction=direction,
                mode=mode,
                num_layers=num_layers,
                num_blocks=num_blocks,
                bytes_per_block=bytes_per_block,
                block_ids=block_ids,
            )
            for _ in range(warmup):
                state.run_op()
                state.synchronize()
            ready_barrier.wait(timeout=300.0)
            for round_idx in range(repeat):
                round_barrier.wait(timeout=300.0)
                submit_start_ns = time.perf_counter_ns()
                state.run_op()
                submit_end_ns = time.perf_counter_ns()
                state.synchronize()
                complete_end_ns = time.perf_counter_ns()
                samples[gpu].append(
                    Sample(
                        round_idx=round_idx,
                        gpu=gpu,
                        submit_start_ns=submit_start_ns,
                        submit_end_ns=submit_end_ns,
                        complete_end_ns=complete_end_ns,
                    )
                )
            state.validate()
        except BaseException as exc:  # propagate after threads join
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpus
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if errors:
        raise errors[0]

    per_gpu_stats = [summarize_worker(samples[gpu], bytes_per_gpu) for gpu in gpus]
    aggregate = summarize_aggregate(samples, bytes_per_gpu)
    return RunStats(
        gpus=gpus,
        num_gpus=len(gpus),
        bytes_per_gpu=bytes_per_gpu,
        total_bytes=bytes_per_gpu * len(gpus),
        per_gpu=per_gpu_stats,
        aggregate=aggregate,
    )


def _worker_process_entry(
    gpu: int,
    result_queue,
    error_queue,
    barrier,
    *,
    direction: Direction,
    mode: Mode,
    num_layers: int,
    num_blocks: int,
    bytes_per_block: int,
    warmup: int,
    repeat: int,
    pattern: Pattern,
    run_length: int,
    seed: int,
) -> None:
    try:
        block_ids = make_block_ids(num_blocks, pattern, run_length, seed)
        state = WorkerState(
            gpu=gpu,
            direction=direction,
            mode=mode,
            num_layers=num_layers,
            num_blocks=num_blocks,
            bytes_per_block=bytes_per_block,
            block_ids=block_ids,
        )
        for _ in range(warmup):
            state.run_op()
            state.synchronize()
        barrier.wait(timeout=300.0)
        worker_samples: list[Sample] = []
        for round_idx in range(repeat):
            barrier.wait(timeout=300.0)
            submit_start_ns = time.perf_counter_ns()
            state.run_op()
            submit_end_ns = time.perf_counter_ns()
            state.synchronize()
            complete_end_ns = time.perf_counter_ns()
            worker_samples.append(
                Sample(
                    round_idx=round_idx,
                    gpu=gpu,
                    submit_start_ns=submit_start_ns,
                    submit_end_ns=submit_end_ns,
                    complete_end_ns=complete_end_ns,
                )
            )
        state.validate()
        result_queue.put((gpu, [asdict(sample) for sample in worker_samples]))
    except BaseException:
        error_queue.put((gpu, traceback.format_exc()))


def run_group_process(
    gpus: list[int],
    *,
    direction: Direction,
    mode: Mode,
    num_layers: int,
    num_blocks: int,
    bytes_per_block: int,
    warmup: int,
    repeat: int,
    pattern: Pattern,
    run_length: int,
    seed: int,
) -> RunStats:
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(len(gpus))
    result_queue = ctx.Queue()
    error_queue = ctx.Queue()
    bytes_per_gpu = num_layers * num_blocks * bytes_per_block
    processes: list[mp.Process] = []

    for gpu in gpus:
        proc = ctx.Process(
            target=_worker_process_entry,
            args=(gpu, result_queue, error_queue, barrier),
            kwargs={
                "direction": direction,
                "mode": mode,
                "num_layers": num_layers,
                "num_blocks": num_blocks,
                "bytes_per_block": bytes_per_block,
                "warmup": warmup,
                "repeat": repeat,
                "pattern": pattern,
                "run_length": run_length,
                "seed": seed,
            },
        )
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()

    errors: list[str] = []
    while not error_queue.empty():
        gpu, error = error_queue.get()
        errors.append(f"gpu={gpu}\n{error}")
    if errors:
        raise RuntimeError("\n".join(errors))

    for proc in processes:
        if proc.exitcode not in (0, None):
            raise RuntimeError(
                f"worker process pid={proc.pid} exited with code {proc.exitcode}"
            )

    samples: dict[int, list[Sample]] = {}
    for _ in gpus:
        gpu, raw_samples = result_queue.get()
        samples[gpu] = [Sample(**item) for item in raw_samples]

    per_gpu_stats = [summarize_worker(samples[gpu], bytes_per_gpu) for gpu in gpus]
    aggregate = summarize_aggregate(samples, bytes_per_gpu)
    return RunStats(
        gpus=gpus,
        num_gpus=len(gpus),
        bytes_per_gpu=bytes_per_gpu,
        total_bytes=bytes_per_gpu * len(gpus),
        per_gpu=per_gpu_stats,
        aggregate=aggregate,
    )


def run_group(
    gpus: list[int],
    *,
    execution_model: ExecutionModel,
    direction: Direction,
    mode: Mode,
    num_layers: int,
    num_blocks: int,
    bytes_per_block: int,
    warmup: int,
    repeat: int,
    pattern: Pattern,
    run_length: int,
    seed: int,
) -> RunStats:
    if execution_model == "thread":
        return run_group_thread(
            gpus,
            direction=direction,
            mode=mode,
            num_layers=num_layers,
            num_blocks=num_blocks,
            bytes_per_block=bytes_per_block,
            warmup=warmup,
            repeat=repeat,
            pattern=pattern,
            run_length=run_length,
            seed=seed,
        )
    return run_group_process(
        gpus,
        direction=direction,
        mode=mode,
        num_layers=num_layers,
        num_blocks=num_blocks,
        bytes_per_block=bytes_per_block,
        warmup=warmup,
        repeat=repeat,
        pattern=pattern,
        run_length=run_length,
        seed=seed,
    )


def summarize_worker(worker_samples: list[Sample], num_bytes: int) -> WorkerStats:
    submit_us = [
        (sample.submit_end_ns - sample.submit_start_ns) / 1e3
        for sample in worker_samples
    ]
    complete_us = [
        (sample.complete_end_ns - sample.submit_start_ns) / 1e3
        for sample in worker_samples
    ]
    gbps = [gb_per_s(num_bytes, duration_us) for duration_us in complete_us]
    return WorkerStats(
        gpu=worker_samples[0].gpu,
        submit_us_p50=percentile(submit_us, 0.50),
        submit_us_p95=percentile(submit_us, 0.95),
        complete_us_p50=percentile(complete_us, 0.50),
        complete_us_p95=percentile(complete_us, 0.95),
        gbps_p50=percentile(gbps, 0.50),
        gbps_p95=percentile(gbps, 0.95),
    )


def summarize_aggregate(
    samples: dict[int, list[Sample]],
    bytes_per_gpu: int,
) -> AggregateStats:
    submit_us: list[float] = []
    complete_us: list[float] = []
    gbps: list[float] = []
    num_rounds = len(next(iter(samples.values())))
    total_bytes = bytes_per_gpu * len(samples)

    for round_idx in range(num_rounds):
        round_samples = [gpu_samples[round_idx] for gpu_samples in samples.values()]
        round_submit_start_ns = min(sample.submit_start_ns for sample in round_samples)
        round_submit_end_ns = max(sample.submit_end_ns for sample in round_samples)
        round_complete_end_ns = max(sample.complete_end_ns for sample in round_samples)
        round_submit_us = (round_submit_end_ns - round_submit_start_ns) / 1e3
        round_complete_us = (round_complete_end_ns - round_submit_start_ns) / 1e3
        submit_us.append(round_submit_us)
        complete_us.append(round_complete_us)
        gbps.append(gb_per_s(total_bytes, round_complete_us))

    return AggregateStats(
        submit_us_p50=percentile(submit_us, 0.50),
        submit_us_p95=percentile(submit_us, 0.95),
        complete_us_p50=percentile(complete_us, 0.50),
        complete_us_p95=percentile(complete_us, 0.95),
        gbps_p50=percentile(gbps, 0.50),
        gbps_p95=percentile(gbps, 0.95),
    )


def directions_for(arg: str) -> list[Direction]:
    return ["h2d", "d2h"] if arg == "both" else [arg]


def modes_for(arg: str) -> list[Mode]:
    return ["contiguous", "batched"] if arg == "both" else [arg]


def patterns_for(arg: str) -> list[Pattern]:
    return ["contiguous", "runs", "random"] if arg == "all" else [arg]


def print_run(label: str, stats: RunStats) -> None:
    print(
        f"{label}: gpus={stats.gpus} total_bytes={stats.total_bytes} "
        f"aggregate_complete_us_p50={stats.aggregate.complete_us_p50:.2f} "
        f"aggregate_gbps_p50={stats.aggregate.gbps_p50:.2f}"
    )
    for worker in stats.per_gpu:
        print(
            f"  gpu={worker.gpu} "
            f"submit_us_p50={worker.submit_us_p50:.2f} "
            f"submit_us_p95={worker.submit_us_p95:.2f} "
            f"complete_us_p50={worker.complete_us_p50:.2f} "
            f"gbps_p50={worker.gbps_p50:.2f}"
        )


def run_experiment(
    gpus: list[int],
    *,
    execution_model: ExecutionModel,
    run_mode: RunMode,
    direction: Direction,
    mode: Mode,
    num_layers: int,
    num_blocks: int,
    bytes_per_block: int,
    warmup: int,
    repeat: int,
    pattern: Pattern,
    run_length: int,
    seed: int,
) -> ExperimentResult:
    baseline: RunStats | None = None
    concurrent: RunStats | None = None

    if run_mode in ("baseline-only", "both"):
        baseline = run_group(
            [gpus[0]],
            execution_model="thread",
            direction=direction,
            mode=mode,
            num_layers=num_layers,
            num_blocks=num_blocks,
            bytes_per_block=bytes_per_block,
            warmup=warmup,
            repeat=repeat,
            pattern=pattern,
            run_length=run_length,
            seed=seed,
        )

    if run_mode in ("concurrent-only", "both"):
        concurrent = run_group(
            gpus,
            execution_model=execution_model,
            direction=direction,
            mode=mode,
            num_layers=num_layers,
            num_blocks=num_blocks,
            bytes_per_block=bytes_per_block,
            warmup=warmup,
            repeat=repeat,
            pattern=pattern,
            run_length=run_length,
            seed=seed,
        )

    speedup: float | None = None
    efficiency: float | None = None
    if baseline is not None and concurrent is not None:
        speedup = concurrent.aggregate.gbps_p50 / baseline.aggregate.gbps_p50
        efficiency = speedup / len(gpus)

    return ExperimentResult(
        execution_model=execution_model,
        run_mode=run_mode,
        direction=direction,
        mode=mode,
        pattern=pattern,
        baseline_1gpu=baseline,
        concurrent_ngpu=concurrent,
        speedup_vs_1gpu=speedup,
        scaling_efficiency=efficiency,
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    args = parse_args()
    gpus = parse_gpu_list(args.gpus)
    available = torch.cuda.device_count()
    if max(gpus) >= available:
        raise ValueError(
            f"Requested GPUs {gpus}, but only {available} device(s) visible"
        )

    results: list[ExperimentResult] = []
    for direction in directions_for(args.direction):
        for mode in modes_for(args.mode):
            mode_patterns = ["contiguous"] if mode == "contiguous" else patterns_for(
                args.pattern
            )
            for pattern_idx, pattern in enumerate(mode_patterns):
                result = run_experiment(
                    gpus,
                    execution_model=args.execution_model,
                    run_mode=args.run_mode,
                    direction=direction,
                    mode=mode,
                    num_layers=args.num_layers,
                    num_blocks=args.num_blocks,
                    bytes_per_block=args.bytes_per_block,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    pattern=pattern,
                    run_length=args.run_length,
                    seed=args.seed + pattern_idx,
                )
                results.append(result)
                active_stats = result.concurrent_ngpu or result.baseline_1gpu
                assert active_stats is not None
                total_mib = active_stats.total_bytes / (1024**2)
                print(
                    f"[{direction}][{mode}] exec={args.execution_model} "
                    f"run={args.run_mode} pattern={pattern} gpus={gpus} "
                    f"total_bytes={total_mib:.2f} MiB"
                )
                if result.baseline_1gpu is not None:
                    print_run("baseline_1gpu", result.baseline_1gpu)
                if result.concurrent_ngpu is not None:
                    print_run("concurrent_ngpu", result.concurrent_ngpu)
                if result.speedup_vs_1gpu is not None and result.scaling_efficiency is not None:
                    print(
                        f"  speedup_vs_1gpu={result.speedup_vs_1gpu:.2f} "
                        f"scaling_efficiency={result.scaling_efficiency:.2f}"
                    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "gpus": gpus,
                "execution_model": args.execution_model,
                "run_mode": args.run_mode,
                "direction": args.direction,
                "mode": args.mode,
                "num_layers": args.num_layers,
                "num_blocks": args.num_blocks,
                "bytes_per_block": args.bytes_per_block,
                "warmup": args.warmup,
                "repeat": args.repeat,
                "pattern": args.pattern,
                "run_length": args.run_length,
                "seed": args.seed,
            },
            "results": [asdict(result) for result in results],
        }
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"json={args.output}")


if __name__ == "__main__":
    main()
