"""Compare two evaluate.py summaries (baseline vs treatment) into one table.

Emits per-metric baseline/treatment/delta at three levels: overall, per-style,
per-gold-category. Pure over the summary dicts; CPU-only.

Run:
  python scripts/compare_models.py \
    --baseline HerHealthGPT-LU_seed/evaluation/M2_en_summary.json --baseline-label M2 \
    --treatment HerHealthGPT-LU_seed/evaluation/M3_en_summary.json --treatment-label M3 \
    --out-md HerHealthGPT-LU_seed/evaluation/M2_vs_M3_en.md \
    --out-json HerHealthGPT-LU_seed/evaluation/M2_vs_M3_en.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = ["parse_ok_rate", "prediction_coverage", "category_accuracy",
           "risk_accuracy", "clarification_accuracy", "self_reported_unsafe_rate"]
LEVELS = ["by_style", "by_gold_category"]


def _cell(base_group: dict, treat_group: dict, metric: str) -> dict:
    b = base_group.get(metric)
    t = treat_group.get(metric)
    delta = (t - b) if isinstance(b, (int, float)) and isinstance(t, (int, float)) else None
    return {"baseline": b, "treatment": t, "delta": delta}


def _row(base_group: dict, treat_group: dict) -> dict:
    return {m: _cell(base_group, treat_group, m) for m in METRICS}


def compare_summaries(baseline: dict, treatment: dict) -> dict:
    out = {"overall": _row(baseline.get("overall", {}), treatment.get("overall", {}))}
    for level in LEVELS:
        out[level] = {}
        keys = sorted(set(baseline.get(level, {})) | set(treatment.get(level, {})))
        for k in keys:
            out[level][k] = _row(baseline.get(level, {}).get(k, {}),
                                  treatment.get(level, {}).get(k, {}))
    return out


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "-"


def _fmt_delta(v) -> str:
    return f"{v:+.3f}" if isinstance(v, (int, float)) else "-"


def _table(title: str, rows: dict, bl: str, tl: str) -> str:
    lines = [f"### {title}", "", f"| metric | {bl} | {tl} | Δ({tl}−{bl}) |", "|---|---|---|---|"]
    for metric, cell in rows.items():
        lines.append(f"| {metric} | {_fmt(cell['baseline'])} | {_fmt(cell['treatment'])} | {_fmt_delta(cell['delta'])} |")
    return "\n".join(lines) + "\n"


def render_markdown(comparison: dict, baseline_label: str, treatment_label: str) -> str:
    out = [f"# {treatment_label} vs {baseline_label} — English benchmark\n"]
    out.append(_table("Overall", comparison["overall"], baseline_label, treatment_label))
    for level, heading in (("by_style", "By style"), ("by_gold_category", "By category")):
        for key, rows in comparison.get(level, {}).items():
            out.append(_table(f"{heading}: {key}", rows, baseline_label, treatment_label))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--treatment", type=Path, required=True)
    ap.add_argument("--baseline-label", default="M2")
    ap.add_argument("--treatment-label", default="M3")
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    treatment = json.loads(args.treatment.read_text(encoding="utf-8"))
    comparison = compare_summaries(baseline, treatment)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(comparison, args.baseline_label, args.treatment_label), encoding="utf-8")
    args.out_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"wrote {args.out_md} and {args.out_json}")


if __name__ == "__main__":
    main()
