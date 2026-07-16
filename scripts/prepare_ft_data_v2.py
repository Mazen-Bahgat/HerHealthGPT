"""Prepare the 200_Seed_Dataset styled QA splits for Qwen QLoRA training.

Reads Train/train_canonical_styled.csv / validate/validation_canonical_styled.csv
(Question/Answer/Topic/Keywords/Style), drops rows whose Question appears
verbatim (case-insensitive) in the styled benchmark
(Test/gold_seeds_styled_labeled.csv), dedups questions shared between train
and val (val wins so the validation set stays untouched), and emits Qwen
chat-message JSONL in the same record shape as prepare_ft_data.py. Every
dropped row is written to leakage_log.csv with its reason.

FR/AR: pass translated styled CSVs directly, or pass a returned handoff file
carrying Question_translated / Answer_translated columns. Handoff translations
are mapped onto Question/Answer first, with split membership recovered from the
row_id prefix when --train and --val point to the same combined file. Translation
provenance is documented beside each returned handoff; the ingest path does not
assume a particular human or model translation provider.

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
from build_ft_mix_v2 import risk_heuristic, _first_sentence  # noqa: E402

CLARIFYING_QUESTION = (
    "Could you tell me a bit more — for example how long this has been going on "
    "and any other symptoms you've noticed — so I can help more accurately?"
)

CATEGORY_ENUM = {"menstrual", "pcos", "fertility", "other"}


def enum_category(topic: str | None) -> str:
    """Clamp the 13-way Topic taxonomy onto the model's 4-way enum so training
    targets never carry an out-of-enum predicted_category."""
    c = inf.normalize_category(topic)
    if c in CATEGORY_ENUM:
        return c
    if c in {"menstruation", "menarche", "menopause", "pms"}:
        return "menstrual"
    return "other"

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


def load_style_by_row_id() -> dict[str, str]:
    """row_id -> Style from the canonical EN styled sources. The FR/AR
    handoff schema carries translations but no Style column, and every
    downstream clarify decision (oversampling, asks_clarification) keys
    on Style."""
    return {r["row_id"]: r["Style"]
            for r in read_csv(DEFAULT_TRAIN) + read_csv(DEFAULT_VAL)}


def recover_styles(rows: list[dict], style_by_row_id: dict[str, str]) -> list[dict]:
    """Attach Style by row_id join; rows that already carry a Style keep it."""
    missing = [r.get("row_id", "?") for r in rows
               if not (r.get("Style") or "").strip()
               and r.get("row_id") not in style_by_row_id]
    if missing:
        raise ValueError(
            f"{len(missing)} rows with no recoverable Style, e.g. {missing[:5]}")
    out = []
    for r in rows:
        if (r.get("Style") or "").strip():
            out.append(r)
        else:
            mapped = dict(r)
            mapped["Style"] = style_by_row_id[r["row_id"]]
            out.append(mapped)
    return out


def is_degenerate_ambiguous(row: dict) -> bool:
    """Ambiguity rewrites that erased the content words entirely
    ("What is something? I'm not really sure...") — training on these pairs a
    content-free question with a confident specific answer."""
    if (row.get("Style") or "").strip().lower() != "ambiguous":
        return False
    q = row.get("Question", "")
    return bool(re.search(r"\bsomething\b", q, re.I)) and "not really sure" in q.lower()


def load_degenerate_row_ids() -> set[str]:
    """Row_ids of degenerate-ambiguous rows in the canonical EN styled source.
    Used to drop the matching row in ANY language: a French/Arabic translation
    of a content-erased ambiguous rewrite has the same emptied-content
    problem, but is_degenerate_ambiguous's English-word regex can't detect it
    once Question has already been overwritten with translated text."""
    en_train = read_csv(DEFAULT_TRAIN)
    en_val = read_csv(DEFAULT_VAL)
    return {r["row_id"] for r in en_train + en_val if is_degenerate_ambiguous(r)}


def clean_splits(train_rows: list[dict], val_rows: list[dict],
                 bench_questions: set[str],
                 degenerate_ids: set[str] | None = None) -> tuple[list[dict], list[dict], list[dict]]:
    log: list[dict] = []
    degenerate_ids = degenerate_ids or set()

    def drop_degenerate(rows: list[dict], split: str) -> list[dict]:
        kept = []
        for r in rows:
            if is_degenerate_ambiguous(r) or r.get("row_id") in degenerate_ids:
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


def oversample_ambiguous(rows: list[dict], factor: int) -> list[dict]:
    """Repeat ambiguous-style (clarification) rows `factor` times so the
    clarify signal is strong enough to survive fine-tuning. factor=1 is a
    no-op. Non-ambiguous rows are untouched; order is preserved."""
    if factor <= 1:
        return rows
    out = []
    for r in rows:
        out.append(r)
        if (r.get("Style") or "").strip().lower() == "ambiguous":
            out.extend([r] * (factor - 1))
    return out


def to_chat_record(row: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["Question"].strip()},
            {"role": "assistant", "content": row["Answer"].strip()},
        ],
        "category": inf.normalize_category(row.get("Topic")),
    }


def to_json_record(row: dict) -> dict:
    """Eval-shaped training target: user = FIXED_PROMPT_TEMPLATE(question),
    assistant = the 8-key JSON object evaluate.py scores. Ambiguous-style rows
    model clarification behaviour (matching the benchmark's requires_clarification
    label for that style); all other styles answer directly."""
    q = row["Question"].strip()
    a = row["Answer"].strip()
    cat = enum_category(row.get("Topic"))
    if (row.get("Style") or "").strip().lower() == "ambiguous":
        obj = {
            "predicted_category": cat,
            "interpreted_symptom": "The description is too vague to interpret confidently",
            "predicted_risk": "see-doctor",
            "recommended_action": "Ask for more detail, then advise consulting a clinician",
            "asks_clarification": True,
            "clarifying_question": CLARIFYING_QUESTION,
            "unsafe_response": False,
            "response_text": CLARIFYING_QUESTION,
        }
    else:
        obj = {
            "predicted_category": cat,
            "interpreted_symptom": _first_sentence(a),
            "predicted_risk": risk_heuristic(a),
            "recommended_action": _first_sentence(a),
            "asks_clarification": False,
            "clarifying_question": "",
            "unsafe_response": False,
            "response_text": a[:800],
        }
    return {"messages": [
        {"role": "user", "content": inf.FIXED_PROMPT_TEMPLATE.format(text=q)},
        {"role": "assistant", "content": json.dumps(obj, ensure_ascii=False, indent=2)},
    ], "category": cat}


def write_jsonl(path: Path, rows: list[dict], fmt: str = "chat") -> None:
    builder = to_json_record if fmt == "json" else to_chat_record
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(builder(r), ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["en", "fr", "ar"])
    ap.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--val", type=Path, default=DEFAULT_VAL)
    ap.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    ap.add_argument("--format", choices=["chat", "json"], default="chat",
                    help="chat = plain Q->A; json = eval-shaped JSON triage target")
    ap.add_argument("--oversample-clarify", type=int, default=1,
                    help="repeat ambiguous/clarify train rows N times (json format only)")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    suffix = "_json" if args.format == "json" else ""
    out_dir = args.out_dir or Path(f"data/ft/{args.lang}_v2{suffix}")

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
        styles = load_style_by_row_id()
        train_rows = recover_styles(train_rows, styles)
        val_rows = recover_styles(val_rows, styles)

    bench_questions = {norm_q(r["Question"]) for r in read_csv(args.benchmark)}
    degenerate_ids = load_degenerate_row_ids()
    ctrain, cval, log = clean_splits(train_rows, val_rows, bench_questions, degenerate_ids)

    if args.format == "json" and args.oversample_clarify > 1:
        before = len(ctrain)
        ctrain = oversample_ambiguous(ctrain, args.oversample_clarify)
        print(f"oversample-clarify x{args.oversample_clarify}: train {before} -> {len(ctrain)}")

    write_jsonl(out_dir / "train.jsonl", ctrain, args.format)
    write_jsonl(out_dir / "val.jsonl", cval, args.format)
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
