"""Convert Used_Datasets/Consolidated_Datasets/Train_Val_Dataset/gold_seeds_styled.csv
into the benchmark JSONL shape run_local_inference.py / run_inference.py expect.

gold_seeds_styled.csv columns: Question, Answer, Topic, Keywords, Style. It has
NO gold safety labels (risk level, clarification-needed, condition) yet, so this
conversion leaves those fields absent -- only parse_ok_rate and category
comparisons are meaningful until gold labels are added. Topic is a 13-way
taxonomy broader than the model's 4-way prompt (menstrual/pcos/fertility/other),
so category accuracy against raw Topic will read low; category is normalized via
run_inference.normalize_category for a rough mapping, with topic_raw kept for
inspection.

Seed grouping: rows sharing the same Answer are style-siblings of one seed
(confirmed 90 groups x 6 styles = 540 rows).

Run: python scripts/convert_gold_seeds_styled.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_inference as inf  # noqa: E402

SRC = Path("Used_Datasets/Consolidated_Datasets/Train_Val_Dataset/gold_seeds_styled.csv")
OUT = Path("Used_Datasets/Consolidated_Datasets/Train_Val_Dataset/gold_seeds_styled.jsonl")


def main() -> None:
    rows = list(csv.DictReader(SRC.open(encoding="utf-8")))
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["Answer"]].append(r)

    out = []
    for i, (answer, group) in enumerate(sorted(groups.items())):
        seed_id = f"gss-{i:03d}"
        for r in group:
            out.append({
                "seed_id": seed_id,
                "style": r["Style"].strip().lower(),
                "category": inf.normalize_category(r["Topic"]),
                "topic_raw": r["Topic"],
                "style_text": r["Question"].strip(),
                "gold_answer": r["Answer"].strip(),
                "keywords": r["Keywords"],
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(out)} rows -> {OUT} ({len(groups)} seed groups)")


if __name__ == "__main__":
    main()
