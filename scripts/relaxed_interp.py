"""Relaxed, clinically-acceptable interpretation metric for HerHealthEval.

Strict interpretation accuracy credits a prediction only when
`predicted_category == gold_category`. But the benchmark's three categories
(menstrual, PCOS, fertility) overlap in real patient language: ~half the
menstrual seeds are conception-motivated (menstrual by symptom, fertility by
intent), and irregular-periods-with-polycystic-ovaries is simultaneously
menstrual and PCOS (see result/error_analysis_menstrual.md).

The *relaxed* metric credits a prediction when it falls in a per-item
**acceptable set** = {gold} plus any clinically-adjacent category that the case
content actually justifies. Adjacency is content-gated, not blanket: an adjacent
category is only acceptable if the case text contains that category's clinical
markers.

The acceptable set is a property of the *case*, which is language-independent, so
markers are detected on the **English source text per seed** (aggregated over that
seed's styles) and applied uniformly across EN/FR/AR. This mirrors the
risk-by-row_id label recovery used elsewhere in the pipeline.

Usage:
  python scripts/relaxed_interp.py \
    --dir Used_Datasets/Consolidated_Datasets/200_Seed_Dataset \
    --model M2ml --model M3ml --model M3ml_v2 --langs en,fr,ar
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # noqa: E402

CATEGORIES = ("menstrual", "pcos", "fertility")

# Content markers per category (matched case-insensitively on the English source).
MARKERS = {
    "fertility": re.compile(
        r"pregnan|concei|conceiv|trying (?:for|to)|ivf|fertil|get pregnant|"
        r"want (?:a |to )?(?:child|baby)|desperate.*child|planning.*pregnan",
        re.I,
    ),
    "pcos": re.compile(
        r"pcos|polycystic|cyst|ovar|hirsut|androgen|hormonal imbalance|"
        r"facial hair|excess hair", re.I,
    ),
    "menstrual": re.compile(
        r"period|menstru|cycle|bleed|flow|spotting|menses", re.I,
    ),
}


def seed_markers(en_recs: list[dict]) -> dict[str, set]:
    """seed_id -> set of categories whose content markers appear in that seed's
    English text (union over all of the seed's styles)."""
    text_by_seed: dict[str, list[str]] = defaultdict(list)
    for r in en_recs:
        sid = r.get("seed_id")
        text_by_seed[sid].append(r.get("input_text") or "")
    out: dict[str, set] = {}
    for sid, texts in text_by_seed.items():
        blob = " \n ".join(texts)
        out[sid] = {c for c, rx in MARKERS.items() if rx.search(blob)}
    return out


def acceptable_set(seed_id: str, gold: str, markers: dict[str, set]) -> set:
    """{gold} plus adjacent categories justified by the case content."""
    acc = {gold}
    present = markers.get(seed_id, set())
    for c in CATEGORIES:
        if c != gold and c in present:
            acc.add(c)
    return acc


def load(path: Path) -> list[dict]:
    return [ev.score_record(json.loads(l)) for l in path.open(encoding="utf-8") if l.strip()]


def score_relaxed(recs: list[dict], markers: dict[str, set]) -> dict:
    ok = [r for r in recs if r.get("parse_ok")]
    strict = sum(1 for r in ok if r.get("category_correct"))
    relaxed = 0
    relaxed_by_cat: dict[str, list] = defaultdict(lambda: [0, 0])  # [correct, n]
    for r in ok:
        gold = r.get("gold_category")
        acc = acceptable_set(r.get("seed_id"), gold, markers)
        hit = r.get("predicted_category") in acc
        relaxed += 1 if hit else 0
        relaxed_by_cat[gold][1] += 1
        relaxed_by_cat[gold][0] += 1 if hit else 0
    n = len(ok)
    return {
        "n": n,
        "strict": strict / n if n else None,
        "relaxed": relaxed / n if n else None,
        "by_cat": {c: (v[0] / v[1] if v[1] else None) for c, v in relaxed_by_cat.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--langs", default="en,fr,ar")
    args = ap.parse_args()
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    print("## Strict vs relaxed (clinically-acceptable) interpretation")
    print("| model | lang | n | strict | relaxed | Δ | menstrual (strict→relaxed) |")
    print("|---|---|---|---|---|---|---|")
    for model in args.model:
        # markers from this model's own EN file if aligned; else from M2ml_en
        en_path = args.dir / f"{model}_en.jsonl"
        src = en_path if en_path.exists() else args.dir / "M2ml_en.jsonl"
        en_recs = load(src)
        # guard: if the model's EN file is the legacy gss set, fall back to M2ml_en
        if {r.get("seed_id") for r in en_recs} and any(
            str(s).startswith("gss") for s in {r.get("seed_id") for r in en_recs}
        ):
            en_recs = load(args.dir / "M2ml_en.jsonl")
        markers = seed_markers(en_recs)
        for lang in langs:
            p = args.dir / f"{model}_{lang}.jsonl"
            if not p.exists():
                continue
            recs = load(p)
            if any(str(r.get("seed_id")).startswith("gss") for r in recs):
                continue  # legacy set, not on aligned benchmark
            s = score_relaxed(recs, markers)
            men = s["by_cat"].get("menstrual")
            # strict menstrual for the arrow
            ok = [r for r in recs if r.get("parse_ok") and r.get("gold_category") == "menstrual"]
            men_strict = sum(1 for r in ok if r.get("category_correct")) / len(ok) if ok else None
            d = (s["relaxed"] - s["strict"]) if s["relaxed"] is not None else None
            print(f"| {model} | {lang} | {s['n']} | {s['strict']:.3f} | {s['relaxed']:.3f} | "
                  f"+{d:.3f} | {men_strict:.3f}→{men:.3f} |")


if __name__ == "__main__":
    main()
