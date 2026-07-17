"""Merge EN + AR + FR into one multilingual benchmark file for run_inference.py.

run_inference.py already supports per-language columns via --language/--text-column
on a single --benchmark file (select_input_text() tries f"{language}_text" first).
This script produces that single file by joining on (seed_id, style):

  - seeds_en_v1.csv                                  -> en_text (style_text, canonical
                                                         text for style="canonical")
  - translation_handoff/arabic_translation_with_egyptian_slang.csv
                                                       -> ar_text (MSA), ar_text_slang_eg
  - translation_handoff/fr_agency_handoff_translated_fr.csv
                                                       -> fr_text

ar_text_effective picks a register-appropriate default per style (clinical/canonical
-> MSA; layperson/indirect_cultural/ambiguous/emotionally_concerned -> Egyptian slang),
matching the register-preservation principle in fr_agency_brief.md -- but both raw AR
columns are kept so the team can override.

Every row also carries ar_validation_status / fr_house_validation_status forward
unchanged, so it stays visible whether a row has been reviewed. As of this run, ALL
540 rows are pending review in both languages -- text-complete, not yet validated.

Usage (typical per-language inference calls against the merged output):
    python scripts/run_inference.py --language en --text-column en_text ...
    python scripts/run_inference.py --language ar --text-column ar_text_effective ...
    python scripts/run_inference.py --language fr --text-column fr_text ...

Run: python scripts/merge_multilingual_benchmark.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

SEED_DIR = Path(__file__).resolve().parent.parent / "HerHealthGPT-LU_seed"
EN_PATH = SEED_DIR / "seeds_en_v1.csv"
_HANDOFF = SEED_DIR / "translation_handoff"


def _first_existing(*names: str) -> Path:
    for name in names:
        path = _HANDOFF / name
        if path.exists():
            return path
    raise FileNotFoundError(f"none of {names} found under {_HANDOFF}")


# Newest-first: team-reviewed deliveries supersede raw/v2/v1 translations.
AR_PATH = _first_existing(
    "arabic_translation_with_egyptian_slang_reviewed.csv",
    "arabic_translation_with_egyptian_slang.csv",
)
FR_PATH = _first_existing(
    "fr_agency_handoff_translated_fr_reviewed.csv",
    "fr_agency_handoff_translated_fr_v2.csv",
    "fr_agency_handoff_translated_fr.csv",
)
OUT_CSV = SEED_DIR / "benchmark_multilingual_v1.csv"
OUT_JSONL = SEED_DIR / "benchmark_multilingual_v1.jsonl"

# Register-preservation default: formal styles get MSA, colloquial styles get Egyptian slang.
MSA_STYLES = {"clinical", "canonical"}

FIELDNAMES = [
    "seed_id", "category", "style",
    "en_text", "ar_text", "ar_text_slang_eg", "ar_text_effective", "fr_text",
    "gold_condition", "gold_risk_level", "gold_action", "evidence_quote", "source_url",
    "requires_clarification", "needs_human_review",
    "ar_validation_status", "fr_house_validation_status",
]


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    en_rows = read_csv(EN_PATH)
    ar_by_key = {(r["seed_id"], r["style"]): r for r in read_csv(AR_PATH)}
    fr_by_key = {(r["seed_id"], r["style"]): r for r in read_csv(FR_PATH)}

    merged = []
    missing_ar, missing_fr = [], []
    for en in en_rows:
        key = (en["seed_id"], en["style"])
        ar = ar_by_key.get(key)
        fr = fr_by_key.get(key)
        if ar is None:
            missing_ar.append(key)
        if fr is None:
            missing_fr.append(key)

        ar_text = ar["ar_text"] if ar else ""
        ar_slang = ar["ar_text_slang_eg"] if ar else ""
        ar_effective = ar_text if en["style"] in MSA_STYLES else (ar_slang or ar_text)

        merged.append({
            "seed_id": en["seed_id"],
            "category": en["category"],
            "style": en["style"],
            "en_text": en["canonical_text"] if en["style"] == "canonical" else en["style_text"],
            "ar_text": ar_text,
            "ar_text_slang_eg": ar_slang,
            "ar_text_effective": ar_effective,
            "fr_text": fr["fr_text"] if fr else "",
            "gold_condition": en.get("gold_condition", ""),
            "gold_risk_level": en.get("gold_risk_level", ""),
            "gold_action": en.get("gold_action", ""),
            "evidence_quote": en.get("evidence_quote", ""),
            "source_url": en.get("source_url", ""),
            "requires_clarification": en.get("requires_clarification", ""),
            "needs_human_review": en.get("needs_human_review", ""),
            "ar_validation_status": ar["ar_validation_status"] if ar else "missing",
            "fr_house_validation_status": fr["fr_house_validation_status"] if fr else "missing",
        })

    if missing_ar or missing_fr:
        print(f"WARNING: {len(missing_ar)} rows missing AR match, {len(missing_fr)} rows missing FR match")
        for k in (missing_ar + missing_fr)[:10]:
            print("  missing:", k)

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(merged)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    ar_pending = sum(1 for r in merged if r["ar_validation_status"] not in ("approved", "accepted"))
    fr_pending = sum(1 for r in merged if r["fr_house_validation_status"] not in ("approved", "accepted"))
    print(f"Wrote {len(merged)} rows -> {OUT_CSV.name} / {OUT_JSONL.name}")
    print(f"AR: {len(merged) - ar_pending}/{len(merged)} reviewed, {ar_pending} still pending review")
    print(f"FR: {len(merged) - fr_pending}/{len(merged)} reviewed, {fr_pending} still pending review")
    print(
        "\nReady for run_inference.py: --language en --text-column en_text | "
        "--language ar --text-column ar_text_effective | --language fr --text-column fr_text"
    )


if __name__ == "__main__":
    main()
