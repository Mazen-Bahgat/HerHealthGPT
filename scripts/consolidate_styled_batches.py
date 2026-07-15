"""Concatenate per-batch styled CSVs (from generate_style_variants_batch.py)
into one file covering the full source dataset, in original row order.

Batches are non-overlapping row ranges of the same source CSV, each already
expanded to 6 style rows per seed and written independently so concurrent
batch work doesn't collide on one growing file (see
generate_style_variants_batch.py). This just concatenates them in the order
given -- callers are responsible for passing batches in source row order.

Usage:
    python scripts/consolidate_styled_batches.py \
        --batches path/to/batch1.csv path/to/batch2_3.csv \
        --output path/to/consolidated.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    fieldnames = None
    all_rows = []
    for batch_path in args.batches:
        with batch_path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise ValueError(f"{batch_path} header {reader.fieldnames} != {fieldnames}")
            rows = list(reader)
        all_rows.extend(rows)
        print(f"{batch_path.name}: {len(rows)} rows")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    seeds = len(all_rows) // 6
    print(f"\n{len(all_rows)} total rows ({seeds} seeds x 6 styles) -> {args.output}")


if __name__ == "__main__":
    main()
