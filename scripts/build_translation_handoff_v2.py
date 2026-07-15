"""Build FR/AR translation handoff templates for the v2 FT corpus.

Exports the leakage-cleaned train+val rows (exactly the rows that
prepare_ft_data_v2.py keeps -- translators never see a row we'd drop) to one
CSV per language with empty Question_translated / Answer_translated columns.
row_id is stable (split + zero-padded index in the ORIGINAL canonical file)
so returned files join back deterministically.

Run: python scripts/build_translation_handoff_v2.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_ft_data_v2 as prep  # noqa: E402

OUT_DIR = prep.DATA_DIR / "translation_handoff_v2"
LANGS = ["fr", "ar"]
FIELDS = ["row_id", "split", "Question", "Answer", "Topic", "Keywords",
          "Question_translated", "Answer_translated"]

README = """# Translation handoff v2 (FR / AR)

## Handoff contract

- Fill only `Question_translated` and `Answer_translated` for every row.
- Do not edit or reorder any other column; `row_id` must remain unchanged.
- Preserve meaning and register (for example, layperson wording stays layperson
  and ambiguous wording stays ambiguous). Translation is not fact-checking.
- Keep the file as UTF-8 CSV. `Topic` and `Keywords` are English metadata and
  are not translated.
- Return one completed file per language (`fr.csv`, `ar.csv`).

## French v2 delivery (2026-07-15)

`fr.csv` is complete: 3,580 silver fine-tuning rows (2,862 train and 718
validation). It is not the 540-row evaluation benchmark, and no French
benchmark-quality claim is made.

The French text was generated with OpenAI's Responses API using
`gpt-5.6-sol`, low reasoning effort, Structured Outputs, and `store: false`.
Style/register was recovered from each stable `row_id`; 3,573 unique
register-aware questions were translated. The 600 unique English answers were
translated once and reused wherever the source answer was identical.

Quality control included corpus-wide schema/identity/UTF-8/language checks,
triage of 404 unique source-vs-translation review jobs, and AI-assisted
stratified review of 180 rows (5.03%) spanning every style and topic. Seventy-one
fail-closed correction rules changed only translation cells. The final
validator result is 0 blocking errors across all 3,580 rows. Natural
cross-language number and punctuation changes remain non-blocking review flags.

Native-French human review is still pending. Known ambiguous source corruptions
are listed in `fr_translation_qa_report.json` instead of being silently
corrected. Full model, prompt, usage, hash, and review provenance is recorded in
`fr_translation_provenance.json`.

## French ingest

The returned handoff contains both splits, so pass the same file as `--train`
and `--val`:

```powershell
python scripts/prepare_ft_data_v2.py --lang fr `
  --train Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/translation_handoff_v2/fr.csv `
  --val Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/translation_handoff_v2/fr.csv
```

Verified ingest output is 2,858 train and 718 validation rows. The leakage
guard drops four training rows as `train_val_dup` because synonymous English
questions (for example, “menses” versus “period”) correctly collapse to the
same natural French wording across splits.

Arabic remains a separate handoff and was not changed by the French workflow.
"""


def build_handoff_rows(train_rows: list[dict], val_rows: list[dict],
                       bench_questions: set[str]) -> list[dict]:
    # tag original indices BEFORE cleaning so row_id survives future edits
    for i, r in enumerate(train_rows):
        r["row_id"], r["split"] = f"train-{i:04d}", "train"
    for i, r in enumerate(val_rows):
        r["row_id"], r["split"] = f"val-{i:04d}", "val"
    ctrain, cval, _log = prep.clean_splits(train_rows, val_rows, bench_questions)
    out = []
    for r in ctrain + cval:
        out.append({
            "row_id": r["row_id"], "split": r["split"],
            "Question": r["Question"], "Answer": r["Answer"],
            "Topic": r.get("Topic", ""), "Keywords": r.get("Keywords", ""),
            "Question_translated": "", "Answer_translated": "",
        })
    return out


def main() -> None:
    train_rows = prep.read_csv(prep.DEFAULT_TRAIN)
    val_rows = prep.read_csv(prep.DEFAULT_VAL)
    bench_questions = {prep.norm_q(r["Question"])
                       for r in prep.read_csv(prep.DEFAULT_BENCHMARK)}
    rows = build_handoff_rows(train_rows, val_rows, bench_questions)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for lang in LANGS:
        path = OUT_DIR / f"{lang}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"{len(rows)} rows -> {path}")
    (OUT_DIR / "README.md").write_text(README, encoding="utf-8")


if __name__ == "__main__":
    main()
