"""Patch gold-label fields of an existing inference JSONL from a benchmark JSONL.

Zero-shot predictions are independent of gold labels, so when only the gold
labels of a benchmark change (same 540 questions), the already-paid-for
generations can be re-scored by overwriting the gold fields in each record --
no GPU re-run. Join is by item_id, with input_text verified against the
benchmark style_text so a silent misjoin is impossible.

Usage:
    python scripts/patch_predictions_gold.py \
        --benchmark Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/gold_seeds_styled.jsonl \
        --predictions Used_Datasets/Consolidated_Datasets/Train_Val_Dataset/M2_gss_en.jsonl \
        --output Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/M2_gss_en.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_inference as inf  # noqa: E402

GOLD_FIELDS = ["gold_category", "gold_risk_level", "gold_action",
               "gold_condition", "requires_clarification"]


def patch_records(preds: list[dict], bench_rows: list[dict],
                  language: str = "en") -> list[dict]:
    bench_by_id = {f"{b['seed_id']}_{b['style']}_{language}": b for b in bench_rows}
    pred_ids = {p["item_id"] for p in preds}
    unknown = sorted(pred_ids - set(bench_by_id))
    if unknown:
        raise ValueError(f"prediction item_id not in benchmark: {unknown[:5]} "
                         f"({len(unknown)} total)")
    missing_preds = sorted(set(bench_by_id) - pred_ids)
    if missing_preds:
        raise ValueError(f"benchmark items with no prediction: {missing_preds[:5]} "
                         f"({len(missing_preds)} total)")
    out = []
    for p in preds:
        b = bench_by_id.get(p["item_id"])
        if b is None:
            raise ValueError(f"prediction item_id not in benchmark: {p['item_id']}")
        if p.get("input_text", "").strip() != b["style_text"].strip():
            raise ValueError(f"input_text mismatch for {p['item_id']}: "
                             f"{p.get('input_text')!r} != {b['style_text']!r}")
        patched = dict(p)
        patched["gold_category"] = inf.normalize_category(b.get("category"))
        patched["gold_risk_level"] = inf.normalize_risk(b.get("gold_risk_level"))
        patched["gold_action"] = b.get("gold_action", "")
        patched["gold_condition"] = b.get("gold_condition", "")
        patched["requires_clarification"] = (b.get("requires_clarification") or "").strip().lower()
        out.append(patched)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, type=Path)
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--language", default="en")
    args = ap.parse_args()

    bench = [json.loads(l) for l in args.benchmark.open(encoding="utf-8") if l.strip()]
    preds = [json.loads(l) for l in args.predictions.open(encoding="utf-8") if l.strip()]
    patched = patch_records(preds, bench, args.language)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in patched:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(patched)} records patched -> {args.output}")


if __name__ == "__main__":
    main()
