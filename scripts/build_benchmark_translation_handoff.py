"""Build FR/AR translation handoff CSVs for the 540-item English benchmark.

The benchmark (Test set) is English-only. To evaluate FR/AR, the 540 benchmark
*questions* must be translated and native-speaker validated. This emits one CSV
per language for the translators, keyed by (seed_id, style) so the translations
join back to the benchmark deterministically. Gold labels are NOT included --
they are language-invariant and stay canonical in the English Test file; only
the question text is translated.

Input : Test benchmark JSONL (default: gold_seeds_styled.jsonl)
Output: Test/translation_handoff_benchmark/{fr,ar}.csv + README.md

Run: python scripts/build_benchmark_translation_handoff.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DATA = Path("Used_Datasets/Consolidated_Datasets/200_Seed_Dataset")
DEFAULT_BENCH = DATA / "gold_seeds_styled.jsonl"
OUT_DIR = DATA / "Test" / "translation_handoff_benchmark"
LANGS = ["fr", "ar"]
FIELDS = ["item_key", "seed_id", "style", "Topic", "Question", "Question_translated"]

README = """# Benchmark translation handoff (FR / AR)

These are the 540 English benchmark questions to translate for cross-lingual
evaluation.

- Fill ONLY `Question_translated`. Leave every other column untouched.
- `item_key` (seed_id + style) is the join key -- it MUST come back unchanged.
- Translate meaning, register, and naturalness. Keep the communication style:
  a `layperson` question stays lay, a `clinical` one stays clinical, an
  `ambiguous` one stays vague. Do NOT clarify or expand ambiguous questions.
- Do NOT translate gold labels -- there are none here, by design. The gold
  interpretation is language-invariant and stays in the English Test file.
- Keep the file CSV, UTF-8. Return one completed file per language.
- Native-speaker validation for meaning preservation and register is REQUIRED
  before these are used for evaluation.
"""


def build_rows(bench: list[dict]) -> list[dict]:
    rows = []
    for b in bench:
        rows.append({
            "item_key": f"{b['seed_id']}__{b['style']}",
            "seed_id": b["seed_id"],
            "style": b["style"],
            "Topic": b.get("topic_raw", ""),
            "Question": b["style_text"],
            "Question_translated": "",
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=Path, default=DEFAULT_BENCH)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    bench = [json.loads(l) for l in args.benchmark.open(encoding="utf-8") if l.strip()]
    rows = build_rows(bench)
    keys = {r["item_key"] for r in rows}
    if len(keys) != len(rows):
        raise SystemExit(f"item_key collision: {len(rows)} rows, {len(keys)} unique keys")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for lang in LANGS:
        path = args.out_dir / f"{lang}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"{len(rows)} rows -> {path}")
    (args.out_dir / "README.md").write_text(README, encoding="utf-8")


if __name__ == "__main__":
    main()
