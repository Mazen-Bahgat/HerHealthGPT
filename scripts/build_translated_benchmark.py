"""Build a per-language benchmark JSONL from translated questions + English gold.

Takes the canonical English benchmark JSONL (all gold labels) and a filled
translation handoff (Question_translated keyed by item_key = seed_id__style),
and emits a benchmark JSONL identical to the English one except that
`style_text` is the translated question. Gold labels are carried over unchanged
-- they are language-invariant. Fails loudly on any missing/empty translation or
unknown item_key so a partial benchmark can never be scored.

Run:
    python scripts/build_translated_benchmark.py --lang fr \
        --translated .../Test/translation_handoff_benchmark/fr.csv
    # -> gold_seeds_styled_fr.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DATA = Path("Used_Datasets/Consolidated_Datasets/200_Seed_Dataset")
DEFAULT_BENCH = DATA / "gold_seeds_styled.jsonl"


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))


def translate_benchmark(bench: list[dict], handoff: list[dict]) -> list[dict]:
    by_key = {}
    for h in handoff:
        q = (h.get("Question_translated") or "").strip()
        if not q:
            continue
        by_key[h["item_key"]] = q

    out = []
    missing = []
    for b in bench:
        key = f"{b['seed_id']}__{b['style']}"
        q = by_key.get(key)
        if not q:
            missing.append(key)
            continue
        row = dict(b)
        row["style_text"] = q
        out.append(row)
    if missing:
        raise ValueError(f"{len(missing)} benchmark items have no translation, "
                         f"e.g. {missing[:5]}")
    extra = set(by_key) - {f"{b['seed_id']}__{b['style']}" for b in bench}
    if extra:
        raise ValueError(f"{len(extra)} handoff item_keys not in benchmark, "
                         f"e.g. {sorted(extra)[:5]}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["fr", "ar"])
    ap.add_argument("--benchmark", type=Path, default=DEFAULT_BENCH)
    ap.add_argument("--translated", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or DATA / f"gold_seeds_styled_{args.lang}.jsonl"

    bench = [json.loads(l) for l in args.benchmark.open(encoding="utf-8") if l.strip()]
    handoff = read_csv(args.translated)
    rows = translate_benchmark(bench, handoff)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} translated benchmark items ({args.lang}) -> {out}")


if __name__ == "__main__":
    main()
