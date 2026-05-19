#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate a custom JSONL for vllm bench serve (--dataset-name custom).

Legacy S2 workload (multi-tenant shared system prompt). For Coding Agent
agentic-RL traces see gen_coding_agent_trace.py (Layer 1, RFC-aligned).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _lorem_tokens(n: int, rng: random.Random) -> str:
    words = (
        "analysis benchmark throughput latency offload prefix cache block "
        "scheduler worker dma memcpy batch async tensor parallel"
    ).split()
    return " ".join(rng.choice(words) for _ in range(max(1, n // 4)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--num-prompts", type=int, default=200)
    p.add_argument("--num-system-prompts", type=int, default=8)
    p.add_argument("--system-chars", type=int, default=12000,
                   help="Approx length of each shared system prompt")
    p.add_argument("--user-chars", type=int, default=400)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    systems = [
        _lorem_tokens(args.system_chars, rng) for _ in range(args.num_system_prompts)
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for i in range(args.num_prompts):
            sys_prompt = systems[i % args.num_system_prompts]
            user = _lorem_tokens(args.user_chars, rng)
            row = {
                "prompt": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user},
                ],
                "max_tokens": args.max_tokens,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {args.num_prompts} lines to {args.output}")


if __name__ == "__main__":
    main()
