# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal benchmark for contiguous memcpy (A) vs batched block memcpy (C).

A:
    One contiguous cudaMemcpyAsync over all KV bytes.

C:
    cuMemcpyBatchAsync over layer-major [num_blocks, bytes_per_block] tensors,
    matching the current SimpleCPUOffloadConnector copy shape.
"""

from __future__ import annotations

import argparse
import random
import statistics
from dataclasses import dataclass
from typing import Literal

import torch
from cuda.bindings import runtime as cudart

from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    build_params,
    copy_blocks,
)


Direction = Literal["h2d", "d2h"]
Pattern = Literal["contiguous", "random", "runs"]


@dataclass
class Sample:
    stream_us: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare contiguous cudaMemcpyAsync vs cuMemcpyBatchAsync."
    )
    parser.add_argument(
        "--direction",
        choices=("h2d", "d2h", "both"),
        default="both",
        help="Copy direction to benchmark.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=36,
        help="Layer count for C path.",
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
        help="Bytes per layer block for C path.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations per path.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=20,
        help="Measured iterations per path.",
    )
    parser.add_argument(
        "--pattern",
        choices=("contiguous", "random", "runs", "all"),
        default="all",
        help="Block-id layout used by C path.",
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
        help="Seed for randomized block-id patterns.",
    )
    return parser.parse_args()


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
        raise AssertionError("A path validation failed")


def validate_layer_tensors(
    dst: dict[str, torch.Tensor],
    src: dict[str, torch.Tensor],
) -> None:
    for name in src:
        if not torch.equal(dst[name], src[name]):
            raise AssertionError(f"C path validation failed at {name}")


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


def time_contiguous(
    src: torch.Tensor,
    dst: torch.Tensor,
    direction: Direction,
    stream: torch.cuda.Stream,
) -> Sample:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record(stream)
    cuda_memcpy_async(dst.data_ptr(), src.data_ptr(), src.nbytes, direction, stream)
    end.record(stream)

    end.synchronize()
    stream_us = start.elapsed_time(end) * 1e3
    return Sample(stream_us=stream_us)


def time_batched(
    params,
    block_ids: list[int],
    stream: torch.cuda.Stream,
) -> Sample:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record(stream)
    copy_blocks(block_ids, block_ids, params)
    end.record(stream)

    end.synchronize()
    stream_us = start.elapsed_time(end) * 1e3
    return Sample(stream_us=stream_us)


def summarize(samples: list[Sample]) -> Sample:
    return Sample(stream_us=statistics.median(sample.stream_us for sample in samples))


def gb_per_s(num_bytes: int, stream_us: float) -> float:
    seconds = stream_us / 1e6
    return num_bytes / seconds / (1024**3)


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


def run_once(
    direction: Direction,
    num_layers: int,
    num_blocks: int,
    bytes_per_block: int,
    warmup: int,
    repeat: int,
    pattern: Pattern,
    run_length: int,
    seed: int,
) -> None:
    total_bytes = num_layers * num_blocks * bytes_per_block
    stream = torch.cuda.Stream()
    block_ids = make_block_ids(num_blocks, pattern, run_length, seed)

    if direction == "h2d":
        src_contig = alloc_host((total_bytes,))
        dst_contig = torch.empty((total_bytes,), dtype=torch.uint8, device="cuda")
        src_layers = make_layer_tensors(num_layers, num_blocks, bytes_per_block, "cpu")
        dst_layers = make_layer_tensors(num_layers, num_blocks, bytes_per_block, "cuda")
    else:
        src_contig = torch.empty((total_bytes,), dtype=torch.uint8, device="cuda")
        dst_contig = alloc_host((total_bytes,))
        src_layers = make_layer_tensors(num_layers, num_blocks, bytes_per_block, "cuda")
        dst_layers = make_layer_tensors(num_layers, num_blocks, bytes_per_block, "cpu")

    init_pattern(src_contig, seed=11)
    init_layer_patterns(src_layers, seed=29)
    dst_contig.zero_()
    for tensor in dst_layers.values():
        tensor.zero_()
    batched_params = build_params(src_layers, dst_layers, stream)

    for _ in range(warmup):
        time_contiguous(src_contig, dst_contig, direction, stream)
        time_batched(batched_params, block_ids, stream)
    torch.cuda.synchronize()

    a_samples = [
        time_contiguous(src_contig, dst_contig, direction, stream) for _ in range(repeat)
    ]
    c_samples = [time_batched(batched_params, block_ids, stream) for _ in range(repeat)]

    torch.cuda.synchronize()
    validate_contiguous(dst_contig.cpu(), src_contig.cpu())
    validate_layer_tensors(
        {k: v.cpu() for k, v in dst_layers.items()},
        {k: v.cpu() for k, v in src_layers.items()},
    )

    a = summarize(a_samples)
    c = summarize(c_samples)

    print(
        f"[{direction}] total_bytes={total_bytes} "
        f"({total_bytes / (1024**2):.2f} MiB) "
        f"layers={num_layers} blocks={num_blocks} bpb={bytes_per_block} "
        f"pattern={pattern}"
    )
    print(
        "A contiguous  "
        f"stream_us={a.stream_us:.2f} "
        f"gbps={gb_per_s(total_bytes, a.stream_us):.2f}"
    )
    print(
        "C batched     "
        f"stream_us={c.stream_us:.2f} "
        f"gbps={gb_per_s(total_bytes, c.stream_us):.2f}"
    )
    print(
        "ratio C/A     "
        f"stream={c.stream_us / a.stream_us:.2f}x "
        f"gbps={gb_per_s(total_bytes, c.stream_us) / gb_per_s(total_bytes, a.stream_us):.2f}x"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    args = parse_args()
    directions: list[Direction]
    patterns: list[Pattern]
    if args.direction == "both":
        directions = ["h2d", "d2h"]
    else:
        directions = [args.direction]

    if args.pattern == "all":
        patterns = ["contiguous", "runs", "random"]
    else:
        patterns = [args.pattern]

    for direction in directions:
        for pattern_idx, pattern in enumerate(patterns):
            run_once(
                direction=direction,
                num_layers=args.num_layers,
                num_blocks=args.num_blocks,
                bytes_per_block=args.bytes_per_block,
                warmup=args.warmup,
                repeat=args.repeat,
                pattern=pattern,
                run_length=args.run_length,
                seed=args.seed + pattern_idx,
            )


if __name__ == "__main__":
    main()
