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

- Fill ONLY `Question_translated` and `Answer_translated` for every row.
- Do NOT edit or reorder any other column; `row_id` must come back untouched.
- Translate meaning and register (layperson tone stays layperson).
- Keep the file CSV, UTF-8 encoded. Topic/Keywords are metadata -- leave in English.
- Return one completed file per language (fr.csv, ar.csv).
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
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"{len(rows)} rows -> {path}")
    (OUT_DIR / "README.md").write_text(README, encoding="utf-8")


if __name__ == "__main__":
    main()
