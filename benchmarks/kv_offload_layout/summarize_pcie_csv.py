# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize nvidia-smi dmon PCIe throughput CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize PCIe Rx/Tx CSV.")
    parser.add_argument("csv_path", type=Path, help="CSV from nvidia-smi dmon -s t")
    return parser.parse_args()


def mbps_to_gbps(value: float) -> float:
    return value / 1024.0


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str]] = []

    with args.csv_path.open(newline="") as f:
        first_line = f.readline()
        if not first_line:
            raise RuntimeError("Empty CSV")
        if not first_line.startswith("#"):
            raise RuntimeError("Expected dmon CSV header")
        fieldnames = [item.strip() for item in first_line[1:].split(",")]
        reader = csv.DictReader(f, fieldnames=fieldnames, skipinitialspace=True)
        for row in reader:
            if not row:
                continue
            rows.append(row)

    if not rows:
        raise RuntimeError("No PCIe samples found")

    rx_values = [float(row["rxpci"]) for row in rows]
    tx_values = [float(row["txpci"]) for row in rows]

    avg_rx = sum(rx_values) / len(rx_values)
    avg_tx = sum(tx_values) / len(tx_values)
    peak_rx = max(rx_values)
    peak_tx = max(tx_values)

    print(
        f"pcie_samples={len(rows)} "
        f"avg_rx_mib_s={avg_rx:.2f} avg_tx_mib_s={avg_tx:.2f} "
        f"peak_rx_mib_s={peak_rx:.2f} peak_tx_mib_s={peak_tx:.2f}"
    )
    print(
        f"pcie_samples={len(rows)} "
        f"avg_rx_gib_s={mbps_to_gbps(avg_rx):.2f} "
        f"avg_tx_gib_s={mbps_to_gbps(avg_tx):.2f} "
        f"peak_rx_gib_s={mbps_to_gbps(peak_rx):.2f} "
        f"peak_tx_gib_s={mbps_to_gbps(peak_tx):.2f}"
    )


if __name__ == "__main__":
    main()
