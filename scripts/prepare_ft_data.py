"""Format the leakage-cleaned FT corpus for Qwen QLoRA training.

Reads only ``HerHealthGPT-LU_seed/ft_corpus_v1.jsonl`` and emits Qwen
chat-message records with a deterministic, balanced train/validation split.
"""

import argparse
import json
import pathlib
import random
from collections import defaultdict


CORPUS = pathlib.Path("HerHealthGPT-LU_seed/ft_corpus_v1.jsonl")
OUT_DIR = pathlib.Path("data/ft/en")


def to_chat_record(row: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": row["instruction"]},
            {"role": "user", "content": row["input"]},
            {"role": "assistant", "content": row["output"]},
        ],
        "category": row["category"],
    }


def split_train_val(
    rows: list[dict], val_frac: float, seed: int
) -> tuple[list[dict], list[dict]]:
    by_category = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)

    rng = random.Random(seed)
    train = []
    val = []
    for category in sorted(by_category):
        items = by_category[category][:]
        rng.shuffle(items)
        val_count = round(len(items) * val_frac)
        val.extend(items[:val_count])
        train.extend(items[val_count:])
    return train, val


def _write(path: pathlib.Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(to_chat_record(record), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with CORPUS.open(encoding="utf-8") as corpus:
        rows = [json.loads(line) for line in corpus]
    train, val = split_train_val(rows, args.val_frac, args.seed)
    _write(OUT_DIR / "train.jsonl", train)
    _write(OUT_DIR / "val.jsonl", val)
    print(f"corpus={len(rows)} train={len(train)} val={len(val)} -> {OUT_DIR}")


if __name__ == "__main__":
    main()
