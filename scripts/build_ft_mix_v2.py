"""Build data-mix v2 for the M3-v2 fine-tune (Experiment B).

Combines the leakage-clean v1 corpus with three targeted components, each fixing
a measured M3 regression (see specs/2026-07-13-m3v2-data-mix-finetune-design.md):
JSON-task examples (parse_ok), clarification examples (clarification recall),
style-augmented inputs (cross-style consistency). Deterministic (seed 42).

Run: python scripts/build_ft_mix_v2.py
Outputs (gitignored): data/ft/en_v2/{train,val,probe}.jsonl + mix_stats.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_inference as inf  # noqa: E402
import prepare_ft_data as prep  # noqa: E402

CORPUS = Path("HerHealthGPT-LU_seed/ft_corpus_v1.jsonl")
CLARIFS = Path("HerHealthGPT-LU_seed/clarification_variants_v2.json")
STYLES = Path("HerHealthGPT-LU_seed/style_rewrites_v2.json")
OUT_DIR = Path("data/ft/en_v2")

URGENT_WORDS = ["emergency", "immediately", "call 911", "go to the er",
                "severe bleeding", "unbearable"]
CONSULT_WORDS = ["doctor", "clinician", "gynecologist", "provider",
                 "medical attention", "check-up", "checkup", "healthcare professional"]
N_JSON_PER_CAT = 300
N_PROBE_TRIAGE = 75
N_PROBE_CLARIF = 15


def risk_heuristic(answer: str) -> str:
    low = answer.lower()
    if any(w in low for w in URGENT_WORDS):
        return "urgent"
    if any(w in low for w in CONSULT_WORDS):
        return "see-doctor"
    return "routine"


def _first_sentence(text: str, limit: int = 120) -> str:
    head = text.strip().split(".")[0][:limit]
    return head or text[:limit]


def _json_answer(row: dict, asks: bool = False, clarifying: str = "") -> str:
    obj = {
        "predicted_category": row["category"],
        "interpreted_symptom": _first_sentence(row["output"]),
        "predicted_risk": risk_heuristic(row["output"]),
        "recommended_action": _first_sentence(row["output"]),
        "asks_clarification": asks,
        "clarifying_question": clarifying,
        "unsafe_response": False,
        "response_text": row["output"][:500],
    }
    return json.dumps(obj, ensure_ascii=False, indent=2)


def make_json_example(row: dict) -> dict:
    return {"messages": [
        {"role": "user", "content": inf.FIXED_PROMPT_TEMPLATE.format(text=row["input"])},
        {"role": "assistant", "content": _json_answer(row)},
    ], "category": row["category"]}


def make_clarification_example(v: dict) -> dict:
    if v["register"] == "json":
        obj = {
            "predicted_category": v["category"],
            "interpreted_symptom": "The description is too vague to interpret safely",
            "predicted_risk": "see-doctor",
            "recommended_action": "Ask for more detail, then consult a clinician",
            "asks_clarification": True,
            "clarifying_question": v["clarifying_question"],
            "unsafe_response": False,
            "response_text": v["clarifying_question"],
        }
        answer = json.dumps(obj, ensure_ascii=False, indent=2)
        user = inf.FIXED_PROMPT_TEMPLATE.format(text=v["vague_question"])
    else:
        answer = v["chat_reply"]
        user = v["vague_question"]
    return {"messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": answer}],
            "category": v["category"]}


def make_style_example(v: dict, answer_by_id: dict) -> dict:
    src = answer_by_id[v["source_row_id"]]
    return {"messages": [
        {"role": "system", "content": src["instruction"]},
        {"role": "user", "content": v["rewritten_question"]},
        {"role": "assistant", "content": src["output"]},
    ], "category": v["category"]}


def build(corpus: list[dict], clarifs: list[dict], styles: list[dict], seed: int) -> dict:
    rng = random.Random(seed)
    train_rows, val_rows = prep.split_train_val(corpus, val_frac=0.05, seed=42)
    # Keyed off train rows ONLY: style rewrites whose source lives in the val
    # split are silently dropped below, so no val answer can leak into train.
    answer_by_id = {f"{r['source_dataset']}:{r['source_row_id']}": r for r in train_rows}

    chat = [prep.to_chat_record(r) for r in train_rows]

    by_cat: dict[str, list[dict]] = {}
    for r in train_rows:
        by_cat.setdefault(r["category"], []).append(r)
    json_examples = []
    for cat in sorted(by_cat):
        pool = sorted(by_cat[cat], key=lambda r: str(r["source_row_id"]))
        rng.shuffle(pool)
        json_examples += [make_json_example(r) for r in pool[:N_JSON_PER_CAT]]

    clarif_sorted = sorted(clarifs, key=lambda v: v["source_row_id"])
    rng.shuffle(clarif_sorted)
    probe_clarif_src = clarif_sorted[:N_PROBE_CLARIF]
    train_clarifs = [make_clarification_example(v) for v in clarif_sorted[N_PROBE_CLARIF:]]

    style_examples = [make_style_example(v, answer_by_id) for v in styles
                      if v["source_row_id"] in answer_by_id]

    train = chat + json_examples + train_clarifs + style_examples
    rng.shuffle(train)

    probe = []
    val_pool = sorted(val_rows, key=lambda r: str(r["source_row_id"]))
    rng.shuffle(val_pool)
    for r in val_pool[:N_PROBE_TRIAGE]:
        probe.append({"messages": [{"role": "user",
                                    "content": inf.FIXED_PROMPT_TEMPLATE.format(text=r["input"])}],
                      "category": r["category"], "probe_kind": "triage",
                      "gold_risk_hint": risk_heuristic(r["output"])})
    for v in probe_clarif_src:
        probe.append({"messages": [{"role": "user",
                                    "content": inf.FIXED_PROMPT_TEMPLATE.format(text=v["vague_question"])}],
                      "category": v["category"], "probe_kind": "clarification"})

    stats = {"chat": len(chat), "json": len(json_examples),
             "clarification": len(train_clarifs), "style": len(style_examples)}
    return {"train": train, "val": [prep.to_chat_record(r) for r in val_rows],
            "probe": probe, "stats": stats}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    corpus = [json.loads(l) for l in CORPUS.open(encoding="utf-8")]
    clarifs = json.loads(CLARIFS.read_text(encoding="utf-8"))
    styles = json.loads(STYLES.read_text(encoding="utf-8"))
    out = build(corpus, clarifs, styles, args.seed)
    breakdown = out.pop("stats")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in out.items():
        with (OUT_DIR / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {name: len(rows) for name, rows in out.items()}
    stats["components"] = breakdown
    (OUT_DIR / "mix_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
