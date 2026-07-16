"""Merge per-language v2 FT corpora into one shuffled joint corpus.

Reads data/ft/{lang}_v2_json/{split}.jsonl for each requested language,
concatenates the records verbatim (no re-serialization, so the training
records are byte-identical to the per-language files), shuffles with a
fixed seed for a reproducible language interleave, and writes
data/ft/{langs-joined}_v2_json/{split}.jsonl.

Run:
    python scripts/merge_ft_langs.py --langs en,fr,ar --split train --seed 3407
    python scripts/merge_ft_langs.py --langs en,fr,ar --split val   --seed 3407
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

DEFAULT_DATA_DIR = Path("data/ft")


def merge_lines(langs: list[str], split: str, seed: int,
                data_dir: Path = DEFAULT_DATA_DIR) -> list[str]:
    """Concatenated JSONL lines for `split` across `langs`, seed-shuffled."""
    lines: list[str] = []
    for lang in langs:
        path = data_dir / f"{lang}_v2_json" / f"{split}.jsonl"
        if not path.exists():
            raise SystemExit(f"missing per-language corpus: {path} "
                             f"(run prepare_ft_data_v2.py --lang {lang} first)")
        with path.open(encoding="utf-8") as f:
            file_lines = [ln.rstrip("\n") for ln in f if ln.strip()]
        if not file_lines:
            raise SystemExit(f"empty per-language corpus: {path}")
        lines.extend(file_lines)
    random.Random(seed).shuffle(lines)
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", required=True,
                    help="comma-separated language codes, e.g. en,fr,ar")
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = ap.parse_args()

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    lines = merge_lines(langs, args.split, args.seed, args.data_dir)

    out_dir = args.data_dir / f"{''.join(langs)}_v2_json"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"langs={','.join(langs)} split={args.split} seed={args.seed} "
          f"rows={len(lines)} -> {out_path}")


if __name__ == "__main__":
    main()
