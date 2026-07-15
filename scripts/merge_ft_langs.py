"""Merge per-language FT chat/JSON corpora into one shuffled multilingual corpus
for the Stage-5 joint EN+FR+AR fine-tune (run 2).

Each language must already be prepared with the SAME recipe as the winning EN
run (JSON format + oversample-clarify), i.e. via:

    python scripts/prepare_ft_data_v2.py --lang en --format json --oversample-clarify 4
    python scripts/prepare_ft_data_v2.py --lang fr --format json --oversample-clarify 4 \
        --train <fr styled csv> --val <fr styled csv>
    python scripts/prepare_ft_data_v2.py --lang ar --format json --oversample-clarify 4 \
        --train <ar styled csv> --val <ar styled csv>

That leaves data/ft/{en,fr,ar}_v2_json/{train,val}.jsonl. This script then
concatenates the three per-split, shuffles deterministically, and writes the
joint corpus that train_qlora.py consumes.

Run:
    python scripts/merge_ft_langs.py --langs en,fr,ar --split train --seed 3407
    python scripts/merge_ft_langs.py --langs en,fr,ar --split val   --seed 3407
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def merge_shuffle(paths: list[Path], seed: int) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        rows.extend(json.loads(l) for l in p.open(encoding="utf-8") if l.strip())
    random.Random(seed).shuffle(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="en,fr,ar")
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--suffix", default="_v2_json",
                    help="per-language dir suffix, e.g. _v2_json (JSON recipe)")
    ap.add_argument("--out-dir", type=Path, default=Path("data/ft/enfrar_v2_json"))
    args = ap.parse_args()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    paths = [Path(f"data/ft/{lang}{args.suffix}/{args.split}.jsonl") for lang in langs]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit("missing per-language inputs:\n  " + "\n  ".join(missing))

    rows = merge_shuffle(paths, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.split}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    per_lang = {lang: sum(1 for _ in Path(f"data/ft/{lang}{args.suffix}/{args.split}.jsonl")
                          .open(encoding="utf-8")) for lang in langs}
    print(f"{args.split}: {per_lang} -> {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
