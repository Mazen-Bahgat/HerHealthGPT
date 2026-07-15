"""Prepare the 200_Seed_Dataset styled QA splits for Qwen QLoRA training.

Reads Train/train_canonical_styled.csv / validate/validation_canonical_styled.csv
(Question/Answer/Topic/Keywords/Style), drops rows whose Question appears
verbatim (case-insensitive) in the styled benchmark
(Test/gold_seeds_styled_labeled.csv), dedups questions shared between train
and val (val wins so the validation set stays untouched), and emits Qwen
chat-message JSONL in the same record shape as prepare_ft_data.py. Every
dropped row is written to leakage_log.csv with its reason.

FR/AR: the team translates the styled CSVs in place (same schema), so pass
--lang fr --train <fr train csv> --val <fr val csv> and the files are used
directly. If a file instead carries Question_translated / Answer_translated
columns (handoff-template style), those are mapped onto Question/Answer
first, with split membership from the row_id prefix when train and val are
the same file.

Run (EN):
    python scripts/prepare_ft_data_v2.py --lang en
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_inference as inf  # noqa: E402

DATA_DIR = Path("Used_Datasets/Consolidated_Datasets/200_Seed_Dataset")
DEFAULT_TRAIN = DATA_DIR / "Train" / "train_canonical_styled.csv"
DEFAULT_VAL = DATA_DIR / "validate" / "validation_canonical_styled.csv"
DEFAULT_BENCHMARK = DATA_DIR / "Test" / "gold_seeds_styled_labeled.csv"

SYSTEM_PROMPT = (
    "You are HerHealthGPT, a women's-health assistant. Answer the user's "
    "question about menstrual health, PCOS, or fertility accurately, clearly, "
    "and safely, and advise seeing a doctor when appropriate."
)


def norm_q(q: str) -> str:
    return (q or "").strip().casefold()


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))


def apply_translation(rows: list[dict]) -> list[dict]:
    """Map *_translated columns onto Question/Answer; fail on empty cells."""
    out = []
    bad = []
    for r in rows:
        qt = (r.get("Question_translated") or "").strip()
        at = (r.get("Answer_translated") or "").strip()
        if not qt or not at:
            bad.append(r.get("row_id") or r.get("Question", "?"))
            continue
        mapped = dict(r)
        mapped["Question"], mapped["Answer"] = qt, at
        out.append(mapped)
    if bad:
        raise ValueError(f"{len(bad)} rows with empty translations, e.g. {bad[:5]}")
    return out


def is_degenerate_ambiguous(row: dict) -> bool:
    """Ambiguity rewrites that erased the content words entirely
    ("What is something? I'm not really sure...") — training on these pairs a
    content-free question with a confident specific answer."""
    if (row.get("Style") or "").strip().lower() != "ambiguous":
        return False
    q = row.get("Question", "")
    return bool(re.search(r"\bsomething\b", q, re.I)) and "not really sure" in q.lower()


def clean_splits(train_rows: list[dict], val_rows: list[dict],
                 bench_questions: set[str]) -> tuple[list[dict], list[dict], list[dict]]:
    log: list[dict] = []

    def drop_degenerate(rows: list[dict], split: str) -> list[dict]:
        kept = []
        for r in rows:
            if is_degenerate_ambiguous(r):
                log.append({"split": split, "reason": "degenerate_ambiguous",
                            "Question": r["Question"]})
            else:
                kept.append(r)
        return kept

    train_rows = drop_degenerate(train_rows, "train")
    val_rows = drop_degenerate(val_rows, "val")

    def drop_leaks(rows: list[dict], split: str) -> list[dict]:
        kept = []
        for r in rows:
            if norm_q(r["Question"]) in bench_questions:
                log.append({"split": split, "reason": "benchmark_leak",
                            "Question": r["Question"]})
            else:
                kept.append(r)
        return kept

    train_rows = drop_leaks(train_rows, "train")
    val_rows = drop_leaks(val_rows, "val")

    val_qs = {norm_q(r["Question"]) for r in val_rows}
    kept_train = []
    for r in train_rows:
        if norm_q(r["Question"]) in val_qs:
            log.append({"split": "train", "reason": "train_val_dup",
                        "Question": r["Question"]})
        else:
            kept_train.append(r)
    return kept_train, val_rows, log


def to_chat_record(row: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["Question"].strip()},
            {"role": "assistant", "content": row["Answer"].strip()},
        ],
        "category": inf.normalize_category(row.get("Topic")),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(to_chat_record(r), ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["en", "fr", "ar"])
    ap.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--val", type=Path, default=DEFAULT_VAL)
    ap.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or Path(f"data/ft/{args.lang}_v2")

    train_rows = read_csv(args.train)
    val_rows = read_csv(args.val)
    if args.lang != "en":
        if args.train == DEFAULT_TRAIN:
            raise SystemExit("--lang fr/ar requires --train pointing at the translated CSV")
        has_handoff_cols = train_rows and "Question_translated" in train_rows[0]
        if has_handoff_cols and args.train == args.val:
            # single returned handoff file: split membership from row_id prefix
            all_rows = apply_translation(read_csv(args.train))
            train_rows = [r for r in all_rows if r["row_id"].startswith("train-")]
            val_rows = [r for r in all_rows if r["row_id"].startswith("val-")]
        elif has_handoff_cols:
            train_rows = apply_translation(train_rows)
            val_rows = apply_translation(val_rows)
        # else: fully translated styled CSVs in the standard schema — use directly

    bench_questions = {norm_q(r["Question"]) for r in read_csv(args.benchmark)}
    ctrain, cval, log = clean_splits(train_rows, val_rows, bench_questions)

    write_jsonl(out_dir / "train.jsonl", ctrain)
    write_jsonl(out_dir / "val.jsonl", cval)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "leakage_log.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "reason", "Question"])
        w.writeheader()
        w.writerows(log)

    # hard guarantee: no benchmark question survives into the output
    for r in ctrain + cval:
        assert norm_q(r["Question"]) not in bench_questions
    print(f"lang={args.lang} train={len(ctrain)} val={len(cval)} "
          f"dropped={len(log)} -> {out_dir}")


if __name__ == "__main__":
    main()
