#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate Coding Agent trace JSONL for Layer 1 KV offload experiments.

Uses the model tokenizer for exact token counts (bit-exact prefix for cache hits).
See docs/features/kv_cache_offload_experiments.md §11.2 and streaming_kv_management_rfc.md §6.1.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

FILLER_WORDS = (
    "repository function class module import def return async await "
    "grep read_file edit patch test benchmark latency throughput cache "
    "block scheduler worker dma tensor parallel prefix rollout task"
).split()

TOOL_NAMES = ("grep", "read_file", "edit", "run_terminal_cmd", "list_dir")


def _load_tokenizer(model: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise SystemExit(
            "transformers required: uv pip install transformers"
        ) from e
    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def _count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _pad_text_to_tokens(tokenizer, seed_text: str, target: int, rng: random.Random) -> str:
    # Build text by appending filler words in batches, tokenizing only the
    # filler chunk each round (O(target)), then truncate to `target` tokens.
    seed_ids = tokenizer.encode(seed_text, add_special_tokens=False)
    if len(seed_ids) >= target:
        ids = seed_ids[:target]
        text = tokenizer.decode(ids, skip_special_tokens=True)
        return text

    ids: list[int] = list(seed_ids)
    while len(ids) < target:
        # Generate a chunk of filler words sized to remaining budget.
        remaining = target - len(ids)
        chunk_words = max(64, min(remaining, 1024))
        filler = " " + " ".join(rng.choice(FILLER_WORDS) for _ in range(chunk_words))
        chunk_ids = tokenizer.encode(filler, add_special_tokens=False)
        ids.extend(chunk_ids)

    ids = ids[:target]
    text = tokenizer.decode(ids, skip_special_tokens=True)
    measured = _count_tokens(tokenizer, text)
    if measured != target:
        # Some tokenizers reshuffle bytes on decode→encode. Fix by trimming
        # the encoded text to target tokens once.
        ids2 = tokenizer.encode(text, add_special_tokens=False)[:target]
        text = tokenizer.decode(ids2, skip_special_tokens=True)
    return text


def _build_prefix_messages(
    tokenizer,
    task_id: str,
    prefix_tokens: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    """Split prefix budget: ~85% system (repo), ~15% user (task)."""
    system_budget = int(prefix_tokens * 0.85)
    user_budget = prefix_tokens - system_budget
    system_seed = (
        f"You are a coding agent working on {task_id}. "
        f"Repository context follows.\n"
    )
    user_seed = f"Task for {task_id}: fix the failing test and submit a patch.\n"
    system_content = _pad_text_to_tokens(tokenizer, system_seed, system_budget, rng)
    user_content = _pad_text_to_tokens(tokenizer, user_seed, user_budget, rng)
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def _messages_token_count(tokenizer, messages: list[dict[str, str]]) -> int:
    # Chat template if available; else sum roles.
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            encoded = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False
            )
            if hasattr(encoded, "input_ids"):
                input_ids = encoded["input_ids"]
                return len(input_ids[0] if input_ids and isinstance(input_ids[0], list) else input_ids)
            return len(encoded)
        except Exception:
            pass
    total = 0
    for m in messages:
        total += _count_tokens(tokenizer, m["content"])
    return total


def _default_steps(rng: random.Random, num_tool_calls: int) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for _ in range(num_tool_calls):
        steps.append({"type": "decode", "tokens": rng.randint(64, 256)})
        steps.append(
            {
                "type": "tool_call",
                "name": rng.choice(TOOL_NAMES),
                "input_tokens": rng.randint(12, 32),
                "output_tokens": rng.randint(512, 4096),
            }
        )
    steps.append({"type": "decode", "tokens": rng.randint(32, 128)})
    return steps


def build_task_trace(
    tokenizer,
    task_id: str,
    prefix_tokens: int,
    num_rollouts: int,
    num_tool_calls: int,
    rng: random.Random,
) -> dict[str, Any]:
    prefix_messages = _build_prefix_messages(tokenizer, task_id, prefix_tokens, rng)
    measured = _messages_token_count(tokenizer, prefix_messages)
    rollouts = []
    for rid in range(num_rollouts):
        r_rng = random.Random(rng.randint(0, 2**31 - 1))
        rollouts.append(
            {
                "rollout_id": rid,
                "steps": _default_steps(r_rng, num_tool_calls),
            }
        )
    return {
        "task_id": task_id,
        "prefix_tokens": measured,
        "prefix_messages": prefix_messages,
        "rollouts": rollouts,
    }


def write_s1_prompts(
    tokenizer,
    output_dir: Path,
    prefix_tokens: int,
    seed: int,
) -> tuple[Path, Path]:
    """Layer 0: bit-exact prompt_A / prompt_B for S1 T0→T1→T2."""
    rng_a = random.Random(seed)
    rng_b = random.Random(seed + 1)
    text_a = _pad_text_to_tokens(
        tokenizer,
        "S1 prompt A fixed document for KV offload store/load experiment.\n",
        prefix_tokens,
        rng_a,
    )
    text_b = _pad_text_to_tokens(
        tokenizer,
        "S1 prompt B different document to evict A from GPU KV pool.\n",
        prefix_tokens,
        rng_b,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path_a = output_dir / "s1_prompt_a.txt"
    path_b = output_dir / "s1_prompt_b.txt"
    path_a.write_text(text_a, encoding="utf-8")
    path_b.write_text(text_b, encoding="utf-8")
    print(f"Wrote {path_a} ({_count_tokens(tokenizer, text_a)} tokens)")
    print(f"Wrote {path_b} ({_count_tokens(tokenizer, text_b)} tokens)")
    return path_a, path_b


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Tokenizer / model path")
    p.add_argument("-o", "--output", type=Path, help="Output JSONL (one task per line)")
    p.add_argument("--task-id", default="swe-bench-synthetic-001")
    p.add_argument("--prefix-tokens", type=int, default=102400)
    p.add_argument("--num-rollouts", type=int, default=16)
    p.add_argument("--num-tool-calls", type=int, default=3)
    p.add_argument("--num-tasks", type=int, default=1, help="Lines in JSONL (multi-task CA4)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--s1-prompts",
        type=Path,
        metavar="DIR",
        help="Layer 0 only: write s1_prompt_a/b.txt at --prefix-tokens",
    )
    args = p.parse_args()

    tokenizer = _load_tokenizer(args.model)
    rng = random.Random(args.seed)

    if args.s1_prompts:
        write_s1_prompts(tokenizer, args.s1_prompts, args.prefix_tokens, args.seed)
        return

    if not args.output:
        print("error: -o/--output required unless --s1-prompts", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for i in range(args.num_tasks):
            tid = args.task_id if args.num_tasks == 1 else f"{args.task_id}-{i}"
            trace = build_task_trace(
                tokenizer,
                tid,
                args.prefix_tokens,
                args.num_rollouts,
                args.num_tool_calls,
                random.Random(args.seed + i),
            )
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
            print(
                f"task {tid}: prefix_tokens={trace['prefix_tokens']} "
                f"rollouts={len(trace['rollouts'])}",
                file=sys.stderr,
            )
    print(f"Wrote {args.num_tasks} task(s) to {args.output}")


if __name__ == "__main__":
    main()
