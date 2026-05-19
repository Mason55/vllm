#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay Coding Agent traces against a running vLLM OpenAI server (Layer 1).

Scenarios: ca1 (fan-out), ca2 (task swap), ca3 (delta heavy), ca4 (multi-task).
See docs/features/kv_cache_offload_experiments.md §11.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse token counting / filler padding from trace generator.
from gen_coding_agent_trace import (
    _load_tokenizer,
    _messages_token_count,
    _pad_text_to_tokens,
)


@dataclass
class RequestMetric:
    scenario: str
    task_id: str
    rollout_id: int | None
    step_index: int | None
    label: str
    ttft_ms: float
    total_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class RunResult:
    scenario: str
    metrics: list[RequestMetric] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""
    gpu_samples: dict[str, Any] = field(default_factory=dict)
    offload_witness: dict[str, Any] = field(default_factory=dict)


def _fetch_metrics_text(host: str, port: int) -> str:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/metrics", timeout=10) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError):
        return ""


def _parse_metric_value(metrics_text: str, metric_name: str) -> float | None:
    for line in metrics_text.splitlines():
        if line.startswith("#"):
            continue
        if not line.startswith(metric_name):
            continue
        try:
            return float(line.rsplit(" ", 1)[-1])
        except ValueError:
            return None
    return None


def _parse_metric_value_with_labels(
    metrics_text: str,
    metric_name: str,
    required_labels: dict[str, str],
) -> float | None:
    for line in metrics_text.splitlines():
        if line.startswith("#") or not line.startswith(metric_name):
            continue
        if any(f'{key}="{value}"' not in line for key, value in required_labels.items()):
            continue
        try:
            return float(line.rsplit(" ", 1)[-1])
        except ValueError:
            return None
    return None


def _parse_info_labels(metrics_text: str, metric_name: str) -> dict[str, str]:
    for line in metrics_text.splitlines():
        if line.startswith("#") or not line.startswith(metric_name + "{"):
            continue
        match = re.search(r"\{(.*)\}\s+[0-9.eE+-]+$", line)
        if not match:
            continue
        labels_text = match.group(1)
        return {
            key: value
            for key, value in re.findall(r'([A-Za-z0-9_]+)="([^"]*)"', labels_text)
        }
    return {}


def _sample_kv_cache_usage(
    host: str,
    port: int,
    stop_event: threading.Event,
    samples: dict[str, Any],
    poll_interval_s: float,
) -> None:
    peak_usage = 0.0
    sample_count = 0
    last_usage = None
    while not stop_event.is_set():
        metrics_text = _fetch_metrics_text(host, port)
        usage = _parse_metric_value(metrics_text, "vllm:kv_cache_usage_perc")
        if usage is not None:
            last_usage = usage
            peak_usage = max(peak_usage, usage)
            sample_count += 1
        stop_event.wait(poll_interval_s)
    samples["peak_kv_cache_usage_perc"] = peak_usage
    samples["last_kv_cache_usage_perc"] = last_usage
    samples["num_metric_samples"] = sample_count


def _metric_delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return after - before


def _capture_offload_metrics(host: str, port: int) -> dict[str, float | None]:
    metrics_text = _fetch_metrics_text(host, port)
    return {
        "prefix_cache_hits_total": _parse_metric_value(
            metrics_text, "vllm:prefix_cache_hits_total"
        ),
        "external_prefix_cache_hits_total": _parse_metric_value(
            metrics_text, "vllm:external_prefix_cache_hits_total"
        ),
        "prompt_tokens_cached_total": _parse_metric_value(
            metrics_text, "vllm:prompt_tokens_cached_total"
        ),
        "external_kv_transfer_tokens_total": _parse_metric_value_with_labels(
            metrics_text,
            "vllm:prompt_tokens_by_source_total",
            {"source": "external_kv_transfer"},
        ),
        "kv_offload_gpu_to_cpu_bytes_total": _parse_metric_value_with_labels(
            metrics_text,
            "vllm:kv_offload_total_bytes_total",
            {"transfer_type": "gpu_to_cpu"},
        ),
        "kv_offload_cpu_to_gpu_bytes_total": _parse_metric_value_with_labels(
            metrics_text,
            "vllm:kv_offload_total_bytes_total",
            {"transfer_type": "cpu_to_gpu"},
        ),
    }


def _capture_offload_metrics_delta(
    before: dict[str, float | None],
    after: dict[str, float | None],
) -> dict[str, float | None]:
    return {key: _metric_delta(before.get(key), after.get(key)) for key in after}


def _load_kv_geometry(model: str) -> dict[str, Any]:
    try:
        from transformers import AutoConfig
    except ImportError:
        return {}

    try:
        cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
    except Exception:
        return {}

    num_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    num_attention_heads = getattr(cfg, "num_attention_heads", None) or getattr(
        cfg, "n_head", None
    )
    hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    num_kv_heads = (
        getattr(cfg, "num_key_value_heads", None)
        or getattr(cfg, "multi_query_group_num", None)
        or num_attention_heads
    )
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None and hidden_size and num_attention_heads:
        head_dim = hidden_size // num_attention_heads

    dtype_name = str(getattr(cfg, "torch_dtype", "float16")).replace("torch.", "")
    dtype_bytes = {
        "float16": 2,
        "half": 2,
        "bfloat16": 2,
        "float32": 4,
        "float": 4,
        "int8": 1,
        "uint8": 1,
    }.get(dtype_name, 2)

    if not all((num_layers, num_kv_heads, head_dim)):
        return {
            "dtype": dtype_name,
            "dtype_bytes": dtype_bytes,
        }

    bytes_per_token = 2 * int(num_layers) * int(num_kv_heads) * int(head_dim) * dtype_bytes
    return {
        "num_layers": int(num_layers),
        "num_kv_heads": int(num_kv_heads),
        "head_dim": int(head_dim),
        "dtype": dtype_name,
        "dtype_bytes": dtype_bytes,
        "bytes_per_token": bytes_per_token,
    }


def _enrich_phase_offload_amounts(
    phase: dict[str, float | None],
    kv_geometry: dict[str, Any],
) -> dict[str, Any]:
    enriched: dict[str, Any] = dict(phase)
    bytes_per_token = kv_geometry.get("bytes_per_token")
    ext_tokens = enriched.get("external_kv_transfer_tokens_total")
    if bytes_per_token is not None and ext_tokens is not None:
        enriched["external_kv_transfer_estimated_bytes"] = ext_tokens * bytes_per_token
    return enriched


def _build_ca1_sharing_stats(
    *,
    trace: dict[str, Any],
    metrics_before: str,
    metrics_after: str,
    sample_state: dict[str, Any],
    total_rollouts: int,
    measured_rollouts: int,
    fanout_mode: str,
) -> dict[str, Any]:
    labels = _parse_info_labels(metrics_before or metrics_after, "vllm:cache_config_info")
    block_size = int(labels["block_size"]) if labels.get("block_size") else None
    num_gpu_blocks = int(labels["num_gpu_blocks"]) if labels.get("num_gpu_blocks") else None
    prefix_tokens = int(trace.get("prefix_tokens", 0))
    prefix_blocks = (
        math.ceil(prefix_tokens / block_size)
        if prefix_tokens > 0 and block_size and block_size > 0
        else None
    )
    peak_usage = sample_state.get("peak_kv_cache_usage_perc")
    peak_allocated_blocks = (
        peak_usage * num_gpu_blocks
        if peak_usage is not None and num_gpu_blocks is not None
        else None
    )
    theoretical_total_blocks = (
        total_rollouts * prefix_blocks if prefix_blocks is not None else None
    )
    sharing_ratio = (
        peak_allocated_blocks / theoretical_total_blocks
        if peak_allocated_blocks is not None and theoretical_total_blocks
        else None
    )

    prefix_hits_before = _parse_metric_value(metrics_before, "vllm:prefix_cache_hits_total")
    prefix_hits_after = _parse_metric_value(metrics_after, "vllm:prefix_cache_hits_total")
    ext_hits_before = _parse_metric_value(
        metrics_before, "vllm:external_prefix_cache_hits_total"
    )
    ext_hits_after = _parse_metric_value(
        metrics_after, "vllm:external_prefix_cache_hits_total"
    )

    return {
        "measurement": "metrics_peak_allocated_blocks",
        "fanout_mode": fanout_mode,
        "rollouts_total": total_rollouts,
        "rollouts_measured": measured_rollouts,
        "prefix_tokens": prefix_tokens,
        "block_size": block_size,
        "prefix_blocks": prefix_blocks,
        "num_gpu_blocks": num_gpu_blocks,
        "peak_kv_cache_usage_perc": peak_usage,
        "peak_allocated_blocks": peak_allocated_blocks,
        "theoretical_total_prefix_blocks": theoretical_total_blocks,
        "physical_sharing_ratio": sharing_ratio,
        "prefix_cache_hits_delta": _metric_delta(prefix_hits_before, prefix_hits_after),
        "external_prefix_cache_hits_delta": _metric_delta(ext_hits_before, ext_hits_after),
    }


def _replay_rollout_prefix_only(
    trace: dict[str, Any],
    rollout: dict[str, Any],
    *,
    host: str,
    port: int,
    model: str,
    tokenizer,
    max_model_len: int,
    scenario: str,
    label_prefix: str,
) -> RequestMetric:
    """Concurrent fan-out phase: prefix prefill + short decode only."""
    messages = list(trace["prefix_messages"])
    task_id = trace["task_id"]
    rid = rollout["rollout_id"]
    max_tokens = _cap_max_tokens(tokenizer, messages, 8, max_model_len)
    if max_tokens <= 0:
        raise ValueError("prefix prompt already fills max_model_len")
    ttft, total, pt, ct = _chat_stream(
        host, port, model, messages, max_tokens=max_tokens
    )
    return RequestMetric(
        scenario=scenario,
        task_id=task_id,
        rollout_id=rid,
        step_index=-1,
        label=f"{label_prefix}_rollout{rid}_prefix_decode",
        ttft_ms=ttft,
        total_ms=total,
        prompt_tokens=pt,
        completion_tokens=ct,
    )


def _replay_rollout_steps_after_prefix(
    trace: dict[str, Any],
    rollout: dict[str, Any],
    *,
    host: str,
    port: int,
    model: str,
    tokenizer,
    max_model_len: int,
    scenario: str,
    label_prefix: str,
) -> list[RequestMetric]:
    """Continue rollout after prefix phase (tool calls + decode steps)."""
    messages = list(trace["prefix_messages"])
    task_id = trace["task_id"]
    rid = rollout["rollout_id"]
    rng = random.Random(rid)
    metrics: list[RequestMetric] = []

    # Mirror prefix decode context growth (approximate).
    messages.append(
        {
            "role": "assistant",
            "content": _pad_text_to_tokens(tokenizer, "Ack.\n", 8, rng),
        }
    )

    for i, step in enumerate(rollout["steps"]):
        if step["type"] == "tool_call":
            messages.extend(_synth_tool_messages(tokenizer, step, rng))
            continue
        if step["type"] == "decode":
            max_t = _cap_max_tokens(
                tokenizer, messages, int(step["tokens"]), max_model_len
            )
            if max_t <= 0:
                break
            ttft, total, pt, ct = _chat_stream(
                host, port, model, messages, max_tokens=max_t
            )
            metrics.append(
                RequestMetric(
                    scenario=scenario,
                    task_id=task_id,
                    rollout_id=rid,
                    step_index=i,
                    label=f"{label_prefix}_rollout{rid}_step{i}",
                    ttft_ms=ttft,
                    total_ms=total,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                )
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": _pad_text_to_tokens(
                        tokenizer, "Ack.\n", min(max_t, 32), rng
                    ),
                }
            )
    return metrics


def _chat_stream(
    host: str,
    port: int,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> tuple[float, float, int, int]:
    """Returns (ttft_ms, total_ms, prompt_tokens, completion_tokens)."""
    url = f"http://{host}:{port}/v1/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    ttft_ms = 0.0
    completion_tokens = 0
    prompt_tokens = 0
    first = True
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw_line in resp:
            line = raw_line.decode().strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("error"):
                raise RuntimeError(chunk["error"])
            choices = chunk.get("choices")
            if first and choices:
                ttft_ms = (time.perf_counter() - t0) * 1000
                first = False
            usage = chunk.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            if delta.get("content"):
                completion_tokens += 1
    total_ms = (time.perf_counter() - t0) * 1000
    return ttft_ms, total_ms, prompt_tokens, completion_tokens


def _cap_max_tokens(
    tokenizer,
    messages: list[dict[str, str]],
    requested_max_tokens: int,
    max_model_len: int,
) -> int:
    prompt_tokens = _messages_token_count(tokenizer, messages)
    available = max_model_len - prompt_tokens
    return max(0, min(requested_max_tokens, available))


def _load_traces(path: Path) -> list[dict[str, Any]]:
    traces = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces


def _synth_tool_messages(
    tokenizer,
    step: dict[str, Any],
    rng: random.Random,
) -> list[dict[str, str]]:
    name = step.get("name", "tool")
    in_tok = int(step["input_tokens"])
    out_tok = int(step["output_tokens"])
    assistant = _pad_text_to_tokens(
        tokenizer,
        f'{{"tool": "{name}", "args": "',
        in_tok,
        rng,
    )
    observation = _pad_text_to_tokens(
        tokenizer,
        f"Tool {name} output:\n",
        out_tok,
        rng,
    )
    return [
        {"role": "assistant", "content": assistant},
        {"role": "user", "content": observation},
    ]


def _replay_rollout(
    trace: dict[str, Any],
    rollout: dict[str, Any],
    *,
    host: str,
    port: int,
    model: str,
    tokenizer,
    max_model_len: int,
    scenario: str,
    label_prefix: str,
    metrics: list[RequestMetric],
) -> None:
    messages = list(trace["prefix_messages"])
    task_id = trace["task_id"]
    rid = rollout["rollout_id"]
    rng = random.Random(rid)

    # Initial prefill + first decode from prefix-only context.
    max_tokens = _cap_max_tokens(tokenizer, messages, 8, max_model_len)
    if max_tokens <= 0:
        raise ValueError("prefix prompt already fills max_model_len")
    ttft, total, pt, ct = _chat_stream(
        host, port, model, messages, max_tokens=max_tokens
    )
    metrics.append(
        RequestMetric(
            scenario=scenario,
            task_id=task_id,
            rollout_id=rid,
            step_index=-1,
            label=f"{label_prefix}_rollout{rid}_prefix_decode",
            ttft_ms=ttft,
            total_ms=total,
            prompt_tokens=pt,
            completion_tokens=ct,
        )
    )

    for i, step in enumerate(rollout["steps"]):
        if step["type"] == "tool_call":
            messages.extend(_synth_tool_messages(tokenizer, step, rng))
            continue
        if step["type"] == "decode":
            max_t = _cap_max_tokens(
                tokenizer, messages, int(step["tokens"]), max_model_len
            )
            if max_t <= 0:
                break
            ttft, total, pt, ct = _chat_stream(
                host, port, model, messages, max_tokens=max_t
            )
            metrics.append(
                RequestMetric(
                    scenario=scenario,
                    task_id=task_id,
                    rollout_id=rid,
                    step_index=i,
                    label=f"{label_prefix}_rollout{rid}_step{i}",
                    ttft_ms=ttft,
                    total_ms=total,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                )
            )
            # Append synthetic assistant reply for context growth (approx length).
            messages.append(
                {
                    "role": "assistant",
                    "content": _pad_text_to_tokens(
                        tokenizer, "Ack.\n", min(max_t, 32), rng
                    ),
                }
            )


def run_ca1(
    trace: dict[str, Any],
    *,
    host: str,
    port: int,
    model: str,
    tokenizer,
    max_rollouts: int | None,
    concurrency: int,
    sample_metrics: bool,
    fanout_mode: str,
    metrics_poll_interval_s: float,
    max_model_len: int,
) -> RunResult:
    result = RunResult(scenario="ca1", started_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    rollouts = trace["rollouts"]
    if max_rollouts is not None:
        rollouts = rollouts[:max_rollouts]
    n_rollouts = len(rollouts)
    if n_rollouts == 0:
        result.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        return result
    if fanout_mode == "warm" and n_rollouts < 2:
        raise ValueError("ca1 warm mode needs at least 2 rollouts")

    if fanout_mode == "warm":
        measured_rollouts = rollouts[1:]
        warmup_rollout = rollouts[0]
    else:
        measured_rollouts = rollouts
        warmup_rollout = None

    workers = min(concurrency, len(measured_rollouts))
    metrics_before = ""
    metrics_after = ""
    sample_state: dict[str, Any] = {}

    if warmup_rollout is not None:
        # Materialize prefix once before measuring warm fan-out sharing.
        _replay_rollout_prefix_only(
            trace,
            warmup_rollout,
            host=host,
            port=port,
            model=model,
            tokenizer=tokenizer,
            max_model_len=max_model_len,
            scenario="ca1",
            label_prefix="ca1_warmup",
        )

    if sample_metrics:
        metrics_before = _fetch_metrics_text(host, port)
        stop_event = threading.Event()
        sampler = threading.Thread(
            target=_sample_kv_cache_usage,
            args=(host, port, stop_event, sample_state, metrics_poll_interval_s),
            daemon=True,
        )
        sampler.start()
    else:
        stop_event = None
        sampler = None

    # Phase 1: concurrent prefix fan-out (RL rollout launch).
    prefix_metrics: list[RequestMetric] = []
    if workers > 0:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [
                pool.submit(
                    _replay_rollout_prefix_only,
                    trace,
                    rollout,
                    host=host,
                    port=port,
                    model=model,
                    tokenizer=tokenizer,
                    max_model_len=max_model_len,
                    scenario="ca1",
                    label_prefix="ca1",
                )
                for rollout in measured_rollouts
            ]
            for fut in as_completed(futs):
                prefix_metrics.append(fut.result())

    if sample_metrics and stop_event is not None and sampler is not None:
        stop_event.set()
        sampler.join(timeout=max(metrics_poll_interval_s * 4, 1.0))
        metrics_after = _fetch_metrics_text(host, port)
        result.gpu_samples = _build_ca1_sharing_stats(
            trace=trace,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            sample_state=sample_state,
            total_rollouts=n_rollouts,
            measured_rollouts=len(measured_rollouts),
            fanout_mode=fanout_mode,
        )

    result.metrics.extend(sorted(prefix_metrics, key=lambda m: m.rollout_id or 0))

    # Phase 2: remaining steps per rollout (sequential; prefix phase already done).
    for rollout in rollouts:
        result.metrics.extend(
            _replay_rollout_steps_after_prefix(
                trace,
                rollout,
                host=host,
                port=port,
                model=model,
                tokenizer=tokenizer,
                max_model_len=max_model_len,
                scenario="ca1",
                label_prefix="ca1",
            )
        )

    result.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    return result


def run_ca2(
    trace_a: dict[str, Any],
    trace_b: dict[str, Any],
    *,
    host: str,
    port: int,
    model: str,
    tokenizer,
    max_model_len: int,
) -> RunResult:
    result = RunResult(scenario="ca2", started_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    kv_geometry = _load_kv_geometry(model)
    metrics_before = _capture_offload_metrics(host, port)
    # T0: Task A prefix
    _replay_rollout(
        trace_a,
        trace_a["rollouts"][0],
        host=host,
        port=port,
        model=model,
        tokenizer=tokenizer,
        max_model_len=max_model_len,
        scenario="ca2",
        label_prefix="ca2_T0",
        metrics=result.metrics,
    )
    metrics_after_t0 = _capture_offload_metrics(host, port)
    # T1: Task B evict
    _replay_rollout(
        trace_b,
        trace_b["rollouts"][0],
        host=host,
        port=port,
        model=model,
        tokenizer=tokenizer,
        max_model_len=max_model_len,
        scenario="ca2",
        label_prefix="ca2_T1",
        metrics=result.metrics,
    )
    metrics_after_t1 = _capture_offload_metrics(host, port)
    # T2: Task A again
    _replay_rollout(
        trace_a,
        trace_a["rollouts"][0],
        host=host,
        port=port,
        model=model,
        tokenizer=tokenizer,
        max_model_len=max_model_len,
        scenario="ca2",
        label_prefix="ca2_T2",
        metrics=result.metrics,
    )
    metrics_after_t2 = _capture_offload_metrics(host, port)
    t0_prefix = next(m for m in result.metrics if m.label == "ca2_T0_rollout0_prefix_decode")
    t2_prefix = next(m for m in result.metrics if m.label == "ca2_T2_rollout0_prefix_decode")
    phase_t0 = _enrich_phase_offload_amounts(
        _capture_offload_metrics_delta(metrics_before, metrics_after_t0),
        kv_geometry,
    )
    phase_t1 = _enrich_phase_offload_amounts(
        _capture_offload_metrics_delta(metrics_after_t0, metrics_after_t1),
        kv_geometry,
    )
    phase_t2 = _enrich_phase_offload_amounts(
        _capture_offload_metrics_delta(metrics_after_t1, metrics_after_t2),
        kv_geometry,
    )
    witnessed = (
        (phase_t1.get("kv_offload_gpu_to_cpu_bytes_total") or 0) > 0
        or (phase_t2.get("kv_offload_cpu_to_gpu_bytes_total") or 0) > 0
        or (phase_t2.get("external_prefix_cache_hits_total") or 0) > 0
    )
    amount_known = (
        (phase_t1.get("kv_offload_gpu_to_cpu_bytes_total") or 0) > 0
        or (phase_t2.get("kv_offload_cpu_to_gpu_bytes_total") or 0) > 0
        or (phase_t2.get("external_kv_transfer_tokens_total") or 0) > 0
    )
    result.offload_witness = {
        "required": True,
        "witnessed": witnessed,
        "amount_known": amount_known,
        "criteria": [
            "phase_t1.kv_offload_gpu_to_cpu_bytes_total > 0",
            "phase_t2.kv_offload_cpu_to_gpu_bytes_total > 0",
            "phase_t2.external_prefix_cache_hits_total > 0",
        ],
        "phase_deltas": {
            "t0": phase_t0,
            "t1": phase_t1,
            "t2": phase_t2,
        },
        "behavioral": {
            "t0_prefix_ttft_ms": t0_prefix.ttft_ms,
            "t2_prefix_ttft_ms": t2_prefix.ttft_ms,
            "t0_over_t2_ratio": (
                t0_prefix.ttft_ms / t2_prefix.ttft_ms if t2_prefix.ttft_ms > 0 else None
            ),
        },
        "amount": {
            "kv_geometry": kv_geometry,
            "phase_t1_gpu_to_cpu_bytes": phase_t1.get("kv_offload_gpu_to_cpu_bytes_total"),
            "phase_t2_cpu_to_gpu_bytes": phase_t2.get("kv_offload_cpu_to_gpu_bytes_total"),
            "phase_t2_external_kv_transfer_tokens": phase_t2.get(
                "external_kv_transfer_tokens_total"
            ),
            "phase_t2_external_kv_transfer_estimated_bytes": phase_t2.get(
                "external_kv_transfer_estimated_bytes"
            ),
        },
    }
    result.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    return result


def run_ca3(
    trace: dict[str, Any],
    rollout_id: int,
    *,
    host: str,
    port: int,
    model: str,
    tokenizer,
    max_model_len: int,
) -> RunResult:
    result = RunResult(scenario="ca3", started_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    rollout = next(r for r in trace["rollouts"] if r["rollout_id"] == rollout_id)
    _replay_rollout(
        trace,
        rollout,
        host=host,
        port=port,
        model=model,
        tokenizer=tokenizer,
        max_model_len=max_model_len,
        scenario="ca3",
        label_prefix="ca3",
        metrics=result.metrics,
    )
    result.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    return result


def run_ca4(
    traces: list[dict[str, Any]],
    *,
    host: str,
    port: int,
    model: str,
    tokenizer,
    interleave: bool,
    max_model_len: int,
) -> RunResult:
    result = RunResult(scenario="ca4", started_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    if interleave:
        max_r = max(len(t["rollouts"]) for t in traces)
        for ridx in range(max_r):
            for trace in traces:
                if ridx < len(trace["rollouts"]):
                    _replay_rollout(
                        trace,
                        trace["rollouts"][ridx],
                        host=host,
                        port=port,
                        model=model,
                        tokenizer=tokenizer,
                        max_model_len=max_model_len,
                        scenario="ca4",
                        label_prefix=f"ca4_{trace['task_id']}",
                        metrics=result.metrics,
                    )
    else:
        for trace in traces:
            for rollout in trace["rollouts"][:1]:
                _replay_rollout(
                    trace,
                    rollout,
                    host=host,
                    port=port,
                    model=model,
                    tokenizer=tokenizer,
                    max_model_len=max_model_len,
                    scenario="ca4",
                    label_prefix=f"ca4_{trace['task_id']}",
                    metrics=result.metrics,
                )
    result.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    return result


def _write_result(result: RunResult, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenario": result.scenario,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "metrics": [asdict(m) for m in result.metrics],
        "gpu_samples": result.gpu_samples,
        "offload_witness": result.offload_witness,
        "summary": {
            "count": len(result.metrics),
            "mean_ttft_ms": (
                sum(m.ttft_ms for m in result.metrics) / len(result.metrics)
                if result.metrics
                else 0.0
            ),
            "physical_sharing_ratio": result.gpu_samples.get("physical_sharing_ratio"),
            "offload_witnessed": result.offload_witness.get("witnessed"),
        },
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output} ({len(result.metrics)} requests)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scenario", choices=("ca1", "ca2", "ca3", "ca4"))
    p.add_argument("--trace", type=Path, help="Trace JSONL (ca1/ca3)")
    p.add_argument("--trace-a", type=Path, help="Task A trace (ca2)")
    p.add_argument("--trace-b", type=Path, help="Task B trace (ca2)")
    p.add_argument("--traces", nargs="+", type=Path, help="Multiple traces (ca4)")
    p.add_argument("--model", default="/data1/models/Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-model-len", type=int, default=104448)
    p.add_argument("--max-rollouts", type=int, default=None, help="ca1: limit fan-out")
    p.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="ca1: concurrent in-flight rollouts for prefix fan-out (default: all)",
    )
    p.add_argument(
        "--sample-metrics",
        action="store_true",
        help="ca1: sample /metrics peak kv_cache_usage_perc to estimate sharing ratio",
    )
    p.add_argument(
        "--sample-gpu",
        action="store_true",
        help="Deprecated alias for --sample-metrics",
    )
    p.add_argument(
        "--fanout-mode",
        choices=("cold", "warm"),
        default="cold",
        help="ca1: cold=all rollouts concurrent; warm=1 rollout materializes prefix first",
    )
    p.add_argument(
        "--metrics-poll-interval-s",
        type=float,
        default=0.2,
        help="ca1: /metrics polling interval during concurrent prefix phase",
    )
    p.add_argument("--rollout-id", type=int, default=0, help="ca3")
    p.add_argument("--interleave", action="store_true", help="ca4: round-robin tasks")
    args = p.parse_args()

    tokenizer = _load_tokenizer(args.model)

    try:
        if args.scenario == "ca1":
            if not args.trace:
                sys.exit("ca1 requires --trace")
            trace = _load_traces(args.trace)[0]
            n_rollouts = (
                args.max_rollouts
                if args.max_rollouts is not None
                else len(trace["rollouts"])
            )
            concurrency = args.concurrency if args.concurrency is not None else n_rollouts
            result = run_ca1(
                trace,
                host=args.host,
                port=args.port,
                model=args.model,
                tokenizer=tokenizer,
                max_rollouts=args.max_rollouts,
                concurrency=concurrency,
                sample_metrics=args.sample_metrics or args.sample_gpu,
                fanout_mode=args.fanout_mode,
                metrics_poll_interval_s=args.metrics_poll_interval_s,
                max_model_len=args.max_model_len,
            )
        elif args.scenario == "ca2":
            if not args.trace_a or not args.trace_b:
                sys.exit("ca2 requires --trace-a and --trace-b")
            ta = _load_traces(args.trace_a)[0]
            tb = _load_traces(args.trace_b)[0]
            result = run_ca2(
                ta,
                tb,
                host=args.host,
                port=args.port,
                model=args.model,
                tokenizer=tokenizer,
                max_model_len=args.max_model_len,
            )
        elif args.scenario == "ca3":
            if not args.trace:
                sys.exit("ca3 requires --trace")
            trace = _load_traces(args.trace)[0]
            result = run_ca3(
                trace,
                args.rollout_id,
                host=args.host,
                port=args.port,
                model=args.model,
                tokenizer=tokenizer,
                max_model_len=args.max_model_len,
            )
        else:
            if not args.traces:
                sys.exit("ca4 requires --traces")
            traces = []
            for tp in args.traces:
                traces.extend(_load_traces(tp))
            result = run_ca4(
                traces,
                host=args.host,
                port=args.port,
                model=args.model,
                tokenizer=tokenizer,
                interleave=args.interleave,
                max_model_len=args.max_model_len,
            )
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(
            f"HTTP error (is server running on :{args.port}?): {e}\n{body}",
            file=sys.stderr,
        )
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"HTTP error (is server running on :{args.port}?): {e}", file=sys.stderr)
        sys.exit(1)

    _write_result(result, args.output)
    if result.gpu_samples:
        ratio = result.gpu_samples.get("physical_sharing_ratio")
        print(
            "  gpu_samples: "
            f"mode={result.gpu_samples.get('fanout_mode')} "
            f"peak_blocks={result.gpu_samples.get('peak_allocated_blocks')} "
            f"physical_sharing_ratio={ratio}"
        )
    if result.offload_witness:
        print(
            "  offload_witness: "
            f"witnessed={result.offload_witness.get('witnessed')} "
            f"t0/t2={result.offload_witness.get('behavioral', {}).get('t0_over_t2_ratio')}"
        )
    for m in result.metrics:
        print(f"  {m.label}: ttft={m.ttft_ms:.1f}ms total={m.total_ms:.1f}ms")


if __name__ == "__main__":
    main()
