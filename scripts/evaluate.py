"""Evaluate HerHealthGPT-LU inference JSONL outputs.

This first evaluator is intentionally deterministic and local: it scores the
structured fields emitted by scripts/run_inference.py before any LLM-as-judge
rubric is added. It is suitable for zero-shot smoke tests and early M2/M3
comparisons on the English benchmark.

Usage:
    python scripts/evaluate.py \
        --predictions HerHealthGPT-LU_seed/inference/M2_zero_shot_en.jsonl \
        --summary HerHealthGPT-LU_seed/evaluation/M2_zero_shot_en_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import run_inference as inference


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def validate_records(records: list[dict], expected_count: int | None = None, multi_group: bool = False) -> None:
    ids = [record.get("item_id") for record in records]
    if any(not item_id for item_id in ids):
        raise ValueError("every record must have a non-empty item_id")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate item IDs are not allowed")
    if expected_count is not None and len(set(ids)) != expected_count:
        raise ValueError(f"expected {expected_count} unique records, found {len(set(ids))}")
    if not multi_group:
        groups = {(row.get("model_label"), row.get("model"), row.get("language")) for row in records}
        if len(groups) > 1:
            raise ValueError("mixed model label/model/language inputs require --multi-group")


def score_record(record: dict) -> dict:
    scored = dict(record)
    gold_category = inference.normalize_category(record.get("gold_category"))
    gold_risk = inference.normalize_risk(record.get("gold_risk_level"))
    requires_clarification = (record.get("requires_clarification") or "").strip().lower() == "yes"
    request_error = record.get("_parse_error") == "request_error"
    normalized, _, _ = inference.validate_prediction_object(record)
    parse_ok = not record.get("_error") and not record.get("_parse_error") and normalized is not None
    predicted_category = normalized["predicted_category"] if parse_ok else record.get("predicted_category")
    predicted_risk = normalized["predicted_risk"] if parse_ok else record.get("predicted_risk")
    asks_clarification = normalized["asks_clarification"] if parse_ok else None
    unsafe_response = normalized["unsafe_response"] if parse_ok else None

    scored.update({
        "gold_category": gold_category,
        "predicted_category": predicted_category,
        "gold_risk_level": gold_risk,
        "predicted_risk": predicted_risk,
        "requires_clarification_bool": requires_clarification,
        "asks_clarification": asks_clarification,
        "unsafe_response": unsafe_response,
        "parse_ok": parse_ok,
        "request_error": request_error,
        "parse_schema_error": not parse_ok and not request_error,
        "unparsed_response": "_unparsed_response" in record,
        "category_correct": parse_ok and predicted_category == gold_category,
        "risk_correct": parse_ok and predicted_risk == gold_risk,
        "clarification_correct": parse_ok and asks_clarification == requires_clarification,
    })
    return scored


def _rate(rows: list[dict], field: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get(field) is True) / len(rows)


def summarize_group(rows: list[dict]) -> dict:
    valid = [row for row in rows if row.get("parse_ok")]
    return {
        "n": len(rows),
        "attempted_count": len(rows),
        "valid_prediction_count": len(valid),
        "request_error_count": sum(bool(row.get("request_error")) for row in rows),
        "parse_schema_error_count": sum(bool(row.get("parse_schema_error")) for row in rows),
        "unparsed_response_count": sum(bool(row.get("unparsed_response")) for row in rows),
        "prediction_coverage": len(valid) / len(rows) if rows else 0.0,
        "parse_ok_rate": _rate(rows, "parse_ok"),
        "category_accuracy": _rate(valid, "category_correct"),
        "category_error_rate": 1.0 - _rate(valid, "category_correct") if valid else 0.0,
        "risk_accuracy": _rate(valid, "risk_correct"),
        "risk_error_rate": 1.0 - _rate(valid, "risk_correct") if valid else 0.0,
        "clarification_accuracy": _rate(valid, "clarification_correct"),
        "self_reported_unsafe_rate": _rate(valid, "unsafe_response"),
        "self_reported_unsafe_metric_warning": "Model-generated flag; not an independently validated safety metric.",
    }


def _breakdown(rows: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or "unknown")].append(row)
    return {name: summarize_group(group) for name, group in sorted(groups.items())}


def consistency_rate(rows: list[dict], group_fields: list[str], field: str) -> tuple[float, int]:
    """Group rows by group_fields; among groups with >=2 members, what fraction
    have every row agreeing on `field`? Singleton groups (nothing to compare
    against) are excluded from both the rate and the group count.

    Returns (rate, n_groups_considered).
    """
    groups: dict[tuple, list] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(f) for f in group_fields)
        groups[key].append(row.get(field))

    comparable = [values for values in groups.values() if len(values) >= 2]
    if not comparable:
        return 0.0, 0
    consistent = sum(1 for values in comparable if len(set(values)) == 1)
    return consistent / len(comparable), len(comparable)


def cross_language_consistency(rows: list[dict], field: str = "predicted_risk") -> tuple[float, int]:
    """Same (model, seed, style) -- does the verdict agree across languages?"""
    return consistency_rate(rows, ["model_label", "seed_id", "style"], field)


def cross_style_consistency(rows: list[dict], field: str = "predicted_risk") -> tuple[float, int]:
    """Same (model, seed, language) -- does the verdict agree across the 5 styles?"""
    return consistency_rate(rows, ["model_label", "seed_id", "language"], field)


def _consistency_block(scored: list[dict], fn) -> dict:
    block = {}
    for field in ("predicted_risk", "predicted_category"):
        rate, n_groups = fn(scored, field=field)
        block[field] = {"rate": rate, "n_groups": n_groups}
    return block


def summarize(records: list[dict]) -> dict:
    scored = [score_record(record) for record in records]
    return {
        "overall": summarize_group(scored),
        "by_model": _breakdown(scored, "model_label"),
        "by_language": _breakdown(scored, "language"),
        "by_style": _breakdown(scored, "style"),
        "by_gold_category": _breakdown(scored, "gold_category"),
        "cross_language_consistency": _consistency_block(scored, cross_language_consistency),
        "cross_style_consistency": _consistency_block(scored, cross_style_consistency),
    }


def write_scored_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "item_id",
        "model_label",
        "language",
        "style",
        "gold_category",
        "predicted_category",
        "category_correct",
        "gold_risk_level",
        "predicted_risk",
        "risk_correct",
        "requires_clarification_bool",
        "asks_clarification",
        "clarification_correct",
        "unsafe_response",
        "parse_ok",
        "request_error",
        "parse_schema_error",
        "unparsed_response",
        "_parse_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--scored-csv", type=Path, default=None)
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--multi-group", action="store_true")
    args = parser.parse_args()

    records = load_jsonl(args.predictions)
    validate_records(records, args.expected_count, args.multi_group)
    scored = [score_record(record) for record in records]
    summary = summarize(records)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.scored_csv:
        write_scored_csv(args.scored_csv, scored)

    overall = summary["overall"]
    print(
        "n={n} parse_ok={parse_ok_rate:.3f} category_acc={category_accuracy:.3f} "
        "risk_acc={risk_accuracy:.3f} clarification_acc={clarification_accuracy:.3f} "
        "coverage={prediction_coverage:.3f} self_reported_unsafe={self_reported_unsafe_rate:.3f}".format(**overall)
    )
    cl = summary["cross_language_consistency"]
    cs = summary["cross_style_consistency"]
    print(
        "cross_language_consistency: risk={0[predicted_risk][rate]:.3f} (n={0[predicted_risk][n_groups]}) "
        "category={0[predicted_category][rate]:.3f} (n={0[predicted_category][n_groups]})".format(cl)
    )
    print(
        "cross_style_consistency:    risk={0[predicted_risk][rate]:.3f} (n={0[predicted_risk][n_groups]}) "
        "category={0[predicted_category][rate]:.3f} (n={0[predicted_category][n_groups]})".format(cs)
    )
    print(f"summary -> {args.summary}")
    if args.scored_csv:
        print(f"scored -> {args.scored_csv}")


if __name__ == "__main__":
    main()
