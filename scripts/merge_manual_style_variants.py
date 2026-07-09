"""Merge Claude-authored (manual, no-API) style variants into seeds_en_v1.csv/jsonl.

Companion to regenerate_style_variants_and_gold.py -- same output contract,
different generation method. Where that script calls an external LLM API,
this one reads HerHealthGPT-LU_seed/style_variants_manual.json: 450 variants
(90 seeds x 5 styles) written directly by Claude in-conversation, following
the same meaning-preservation rubric, verified to have a 100% distinct-string
ratio and zero collisions with canonical text before merging.

Does NOT touch gold_risk_level/gold_action/etc -- those were already
completed by complete_gold_labels.py and are left as-is.

Run: python scripts/merge_manual_style_variants.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

SEED_DIR = Path(__file__).resolve().parent.parent / "HerHealthGPT-LU_seed"
VARIANTS_PATH = SEED_DIR / "style_variants_manual.json"
SEEDS_CSV = SEED_DIR / "seeds_en_v1.csv"
SEEDS_JSONL = SEED_DIR / "seeds_en_v1.jsonl"
STYLE_VARIANTS_OUT = SEED_DIR / "style_variants.json"
REPORT_OUT = SEED_DIR / "regeneration_report.md"

STYLES = ("clinical", "layperson", "indirect_cultural", "ambiguous", "emotionally_concerned")


def main() -> None:
    variants = json.loads(VARIANTS_PATH.read_text(encoding="utf-8"))
    old_rows = list(csv.DictReader(SEEDS_CSV.open(encoding="utf-8", newline="")))
    if not old_rows:
        raise SystemExit(f"{SEEDS_CSV} is empty")
    fieldnames = list(old_rows[0].keys())
    if "needs_human_review" not in fieldnames:
        fieldnames.append("needs_human_review")

    new_rows = []
    report_lines = [
        "# Style-variant regeneration report (Claude-authored, no external LLM API)\n\n",
        "450 variants (90 seeds x 5 styles) written directly by Claude in-conversation "
        "under the meaning-preservation rubric, replacing build_seed.py's templated "
        "generate_styles() output. Verified: 100% distinct-string ratio across all 450 "
        "variants, zero collisions with canonical text (see merge validation output).\n\n",
        "**Still needs human review before the benchmark freezes** -- this is an AI "
        "first pass (a different AI than the original template bug, but still not a "
        "substitute for the team's meaning-preservation review).\n\n",
    ]

    seen_seed_ids = set()
    for old_row in old_rows:
        sid = old_row["seed_id"]
        style = old_row["style"]
        new_row = dict(old_row)
        if style != "canonical":
            old_text = old_row["style_text"]
            new_text = variants[sid][style]
            new_row["style_text"] = new_text
            if sid not in seen_seed_ids:
                report_lines.append(f"## {sid} ({old_row['category']})\n\n")
                report_lines.append(f"**Canonical:** {old_row['canonical_text']}\n\n")
            report_lines.append(f"- **{style}**\n  - old: {old_text}\n  - new: {new_text}\n")
        new_row["needs_human_review"] = "true"
        new_rows.append(new_row)
        seen_seed_ids.add(sid)

    with open(SEEDS_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(new_rows)
    with open(SEEDS_JSONL, "w", encoding="utf-8") as f:
        for row in new_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    style_variants_out = {sid: {s: variants[sid][s] for s in STYLES} for sid in variants}
    STYLE_VARIANTS_OUT.write_text(
        json.dumps(style_variants_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    REPORT_OUT.write_text("".join(report_lines), encoding="utf-8")

    print(f"Wrote {len(new_rows)} rows -> {SEEDS_CSV.name} / {SEEDS_JSONL.name}")
    print(f"Wrote regenerated variants -> {STYLE_VARIANTS_OUT.name}")
    print(f"Wrote human-review report -> {REPORT_OUT.name}")


if __name__ == "__main__":
    main()
