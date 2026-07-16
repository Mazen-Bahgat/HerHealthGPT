"""Export translator/agency handoff files.

Translation ownership decision (2026-07-09): Arabic is translated in-house
by team members who are fluent native/near-native speakers (not machine
translation + external validation); French is outsourced to a professional
translation and localization agency. This replaces the design spec's
original translate_benchmark.py plan (GPT/NLLB machine-translation first
pass) -- there is no MT step for either language now.

Re-runnable: reads whatever is current in seeds_en_v1.csv (run again after
style-variant regeneration to refresh the source text agencies/team
translate from).

Output (all under HerHealthGPT-LU_seed/translation_handoff/):
  ar_handoff.csv       for the team -- source EN + blank AR + reviewer columns
  fr_agency_handoff.csv for the localization agency -- source EN + blank FR
  fr_agency_brief.md    written brief to send with fr_agency_handoff.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

SEED_DIR = Path(__file__).resolve().parent.parent / "HerHealthGPT-LU_seed"
SEEDS_CSV = SEED_DIR / "seeds_en_v1.csv"
OUT_DIR = SEED_DIR / "translation_handoff"

AR_COLUMNS = [
    "item_id", "seed_id", "category", "style", "source_text_en",
    "gold_condition", "ar_text", "ar_translator_id", "ar_reviewer_id",
    "ar_validation_status", "ar_notes",
]
FR_COLUMNS = [
    "item_id", "seed_id", "category", "style", "source_text_en",
    "gold_condition", "fr_text", "fr_agency_translator_id", "fr_agency_qa_id",
    "fr_house_reviewer_id", "fr_house_validation_status", "fr_notes",
]

FR_BRIEF = """# Translation brief -- HerHealthGPT-LU (LUHME 2026) English -> French

## What this is

`fr_agency_handoff.csv` contains **synthetic, patient-style text** for a multilingual
language-understanding research benchmark. It is NOT real patient data -- each row is
either a de-identified/paraphrased seed sourced from public health-QA datasets, or a
style variant generated to test how language models interpret different registers of
the same underlying health concern (menstrual health, PCOS/hormonal symptoms, fertility).

## Critical requirement: preserve the register (`style` column), not just the meaning

Each row is deliberately written in ONE of five registers, and the register itself is
part of what we are measuring:

| `style` value | What it means | Translation guidance |
|---|---|---|
| `canonical` | Original patient wording, lightly cleaned | Translate naturally, preserve original tone |
| `clinical` | Clinical/chart-note phrasing | Keep formal, clinical French register |
| `layperson` | Everyday non-medical phrasing | Keep informal, everyday French -- do NOT upgrade to medical terminology |
| `indirect_cultural` | Indirect/euphemistic phrasing, avoids naming the condition | Preserve the indirectness -- do NOT make it more explicit or clinical |
| `ambiguous` | Deliberately vague, missing detail | Preserve the vagueness -- do NOT add specificity that isn't in the English |
| `emotionally_concerned` | Same content, worried/anxious tone | Preserve the emotional register |

**Please do not normalize register toward formal medical French across all rows.** A
`layperson` row translated into clinical French, or an `ambiguous` row translated with
added specificity, breaks the research design even if the translation is otherwise
accurate and natural.

## What we need back

Fill in `fr_text` for every row. Leave `fr_agency_translator_id` / `fr_agency_qa_id` with
your internal translator/QA identifiers if you track per-item ownership -- otherwise leave
blank. The `fr_house_*` columns are for our own team's spot-check pass after delivery and
should be left blank.

## Timeline

Please confirm turnaround time on receipt so we can schedule our in-house review window.

## Contact

Hana (Language lead) -- coordinates all translation handoffs for this project.
"""


def main() -> None:
    if not SEEDS_CSV.exists():
        raise SystemExit(f"{SEEDS_CSV} not found")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(SEEDS_CSV.open(encoding="utf-8", newline="")))

    ar_rows, fr_rows = [], []
    for r in rows:
        item_id = f"{r['seed_id']}_{r['style']}"
        source_text = r["canonical_text"] if r["style"] == "canonical" else r["style_text"]
        common = {
            "item_id": item_id,
            "seed_id": r["seed_id"],
            "category": r["category"],
            "style": r["style"],
            "source_text_en": source_text,
            "gold_condition": r.get("gold_condition", ""),
        }
        ar_rows.append({**common, "ar_text": "", "ar_translator_id": "", "ar_reviewer_id": "",
                         "ar_validation_status": "pending", "ar_notes": ""})
        fr_rows.append({**common, "fr_text": "", "fr_agency_translator_id": "",
                         "fr_agency_qa_id": "", "fr_house_reviewer_id": "",
                         "fr_house_validation_status": "pending", "fr_notes": ""})

    with open(OUT_DIR / "ar_handoff.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AR_COLUMNS)
        w.writeheader()
        w.writerows(ar_rows)

    with open(OUT_DIR / "fr_agency_handoff.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FR_COLUMNS)
        w.writeheader()
        w.writerows(fr_rows)

    (OUT_DIR / "fr_agency_brief.md").write_text(FR_BRIEF, encoding="utf-8")

    print(f"Wrote {len(ar_rows)} rows -> {OUT_DIR / 'ar_handoff.csv'} (in-house AR translation)")
    print(f"Wrote {len(fr_rows)} rows -> {OUT_DIR / 'fr_agency_handoff.csv'} (agency FR translation)")
    print(f"Wrote brief -> {OUT_DIR / 'fr_agency_brief.md'}")
    print(
        "\nNOTE: source_text_en currently reflects the CURRENT seeds_en_v1.csv. If style "
        "variants have not been regenerated yet (see regenerate_style_variants_and_gold.py), "
        "re-run this script after regeneration so translators work from corrected text."
    )


if __name__ == "__main__":
    main()
