"""Convert gold_seeds_styled_labeled.csv into the benchmark JSONL shape
run_local_inference.py / run_inference.py expect.

Reads the gold-labeled file (see complete_gold_labels_gss.py), which adds
gold_risk_level / requires_clarification / gold_condition on top of the raw
Question/Answer/Topic/Keywords/Style columns. Topic is a 13-way taxonomy
broader than the model's 4-way prompt (menstrual/pcos/fertility/other), so
category accuracy against raw Topic will read low; category is normalized via
run_inference.normalize_category for a rough mapping, with topic_raw kept for
inspection.

Seed grouping: rows sharing the same Answer are style-siblings of one seed
(confirmed 90 groups x 6 styles = 540 rows).

Run: python scripts/complete_gold_labels_gss.py   (once, or after source changes)
     python scripts/convert_gold_seeds_styled.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_inference as inf  # noqa: E402

SRC = Path("Used_Datasets/Consolidated_Datasets/Train_Val_Dataset/gold_seeds_styled_labeled.csv")
OUT = Path("Used_Datasets/Consolidated_Datasets/Train_Val_Dataset/gold_seeds_styled.jsonl")


def convert(src: Path, out: Path) -> int:
    rows = list(csv.DictReader(src.open(encoding="utf-8-sig")))
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["Answer"]].append(r)

    records = []
    for i, (answer, group) in enumerate(sorted(groups.items())):
        seed_id = f"gss-{i:03d}"
        for r in group:
            records.append({
                "seed_id": seed_id,
                "style": r["Style"].strip().lower(),
                "category": inf.normalize_category(r["Topic"]),
                "topic_raw": r["Topic"],
                "style_text": r["Question"].strip(),
                "gold_answer": r["Answer"].strip(),
                "keywords": r["Keywords"],
                "gold_risk_level": r["gold_risk_level"],
                "requires_clarification": r["requires_clarification"],
                "gold_condition": r["gold_condition"],
                "gold_action": r.get("gold_action", ""),
            })

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(records)} rows -> {out} ({len(groups)} seed groups)")
    return len(records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    convert(args.src, args.out)


if __name__ == "__main__":
    main()
