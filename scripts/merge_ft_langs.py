<<<<<<< HEAD
"""Merge per-language v2 FT corpora into one shuffled joint corpus.

Reads data/ft/{lang}_v2_json/{split}.jsonl for each requested language,
concatenates the records verbatim (no re-serialization, so the training
records are byte-identical to the per-language files), shuffles with a
fixed seed for a reproducible language interleave, and writes
data/ft/{langs-joined}_v2_json/{split}.jsonl.
=======
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
>>>>>>> b640ef475ca2efa0696587c7a6fbf5d413d74217

Run:
    python scripts/merge_ft_langs.py --langs en,fr,ar --split train --seed 3407
    python scripts/merge_ft_langs.py --langs en,fr,ar --split val   --seed 3407
"""
from __future__ import annotations

import argparse
<<<<<<< HEAD
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
=======
import json
import random
from pathlib import Path


def merge_shuffle(paths: list[Path], seed: int) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        rows.extend(json.loads(l) for l in p.open(encoding="utf-8") if l.strip())
    random.Random(seed).shuffle(rows)
    return rows
>>>>>>> b640ef475ca2efa0696587c7a6fbf5d413d74217


def main() -> None:
    ap = argparse.ArgumentParser()
<<<<<<< HEAD
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
=======
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
>>>>>>> b640ef475ca2efa0696587c7a6fbf5d413d74217


if __name__ == "__main__":
    main()
