#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize vllm bench serve --save-result JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", type=Path)
    args = p.parse_args()

    rows: list[tuple[str, float, float, float, float]] = []
    for path in args.files:
        if not path.is_file():
            print(f"skip missing: {path}", file=sys.stderr)
            continue
        d = _load(path)
        name = path.name
        # Replayer output (Layer 1): use summary.mean_ttft_ms
        if "metrics" in d and "summary" in d:
            ttft = float(d["summary"].get("mean_ttft_ms", 0.0))
            tpot = 0.0
            rps = float(d["summary"].get("count", 0))
            tok = float(d["summary"].get("physical_sharing_ratio") or 0.0)
        else:
            ttft = d.get("mean_ttft_ms") or d.get("ttft_mean") or 0.0
            tpot = d.get("mean_tpot_ms") or d.get("tpot_mean") or 0.0
            rps = d.get("request_throughput") or d.get("req_per_s") or 0.0
            tok = d.get("total_token_throughput") or d.get("total_token_throughput_tok_s") or 0.0
        rows.append((name, float(ttft), float(tpot), float(rps), float(tok)))

    if not rows:
        print("No valid result files.", file=sys.stderr)
        sys.exit(1)

    print(f"{'file':40} {'TTFT_ms':>10} {'TPOT_ms':>10} {'req/s':>10} {'share_ratio':>12}")
    print("-" * 86)
    for name, ttft, tpot, rps, tok in sorted(rows):
        print(f"{name:40} {ttft:10.1f} {tpot:10.2f} {rps:10.2f} {tok:12.1f}")


if __name__ == "__main__":
    main()
