"""Add row_id/split columns to the EN styled train/val files.

Brings train_canonical_styled.csv / validation_canonical_styled.csv to the
same row_id scheme already used by translation_handoff_v2/{fr,ar}.csv --
0-indexed position in the file, "train-0000"/"val-0000" etc. -- so an EN row
can be joined to its FR/AR counterparts by row_id directly instead of
recomputing index math. Matches exactly the enumeration
build_translation_handoff_v2.build_handoff_rows() already does over these
same two files, so no existing row_id in the handoff files is invalidated.

Purely additive (two new leading columns); prepare_ft_data_v2.py reads
Question/Answer/Topic by name only, so this doesn't affect the FT pipeline
already running against these files.

Usage:
    python scripts/add_row_id_split.py \
        --input Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/Train/train_canonical_styled.csv \
        --split train
    python scripts/add_row_id_split.py \
        --input Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/validate/validation_canonical_styled.csv \
        --split val
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def add_row_id_split(rows: list[dict], split: str) -> list[dict]:
    out = []
    for i, r in enumerate(rows):
        out.append({"row_id": f"{split}-{i:04d}", "split": split, **r})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["train", "val"])
    args = parser.parse_args()

    with args.input.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        orig_fields = reader.fieldnames

    if "row_id" in orig_fields:
        print(f"{args.input.name}: already has row_id, skipping")
        return

    out_rows = add_row_id_split(rows, args.split)
    fieldnames = ["row_id", "split"] + orig_fields

    with args.input.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(f"{args.input.name}: added row_id/split to {len(out_rows)} rows")


if __name__ == "__main__":
    main()
