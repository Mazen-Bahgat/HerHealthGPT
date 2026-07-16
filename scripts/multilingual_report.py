"""Reproducible HerHealthEval multilingual report.

Given per-(model, language) prediction JSONL files on the aligned benchmark
(benchmark_multilingual_v1), recompute every paper number in one command:

  * per (model, language): parse_ok, interpretation (category) accuracy,
    under-triage rate, clarification recall/specificity, and indirect_cultural
    interpretation accuracy (cultural-sensitivity proxy);
  * per model: cross-language consistency (risk + category) over the aligned
    EN/FR/AR triples;
  * pairwise McNemar (risk / category / clarification correctness) between
    models, per language, on shared item_ids.

Reuses the canonical scoring in evaluate.py / safety_metrics.py so numbers match
the committed evaluators exactly. CPU-only.

Files are discovered as <dir>/<label>_<lang>.jsonl for each requested label and
language, e.g. M2ml_en.jsonl, M3ml_fr.jsonl, M3ml_v2_ar.jsonl.

Run:
  python scripts/multilingual_report.py \
    --dir Used_Datasets/Consolidated_Datasets/200_Seed_Dataset \
    --model M2ml --model M3ml --model M3ml_v2 --langs en,fr,ar --latex
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # noqa: E402
import safety_metrics as sm  # noqa: E402


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def per_model_lang(recs: list[dict]) -> dict:
    scored = [ev.score_record(r) for r in recs]
    a = sm.analyze(recs, n_boot=2000)
    by = defaultdict(lambda: [0, 0])
    for r in scored:
        if r.get("parse_ok"):
            by[r["style"]][1] += 1
            if r.get("category_correct"):
                by[r["style"]][0] += 1
    ind = by.get("indirect_cultural", [0, 0])
    clar = a["clarification"]
    return {
        "n": a["n"],
        "parse_ok": a["parse_ok_rate"],
        "interp": 1.0 - (a["misunderstanding"]["misunderstanding_rate"] or 0.0),
        "under_triage": a["under_triage"]["under_triage_rate"],
        "clar_recall": clar["recall_gold_yes"],
        "clar_spec": clar["specificity_gold_no"],
        "indirect_cultural_interp": (ind[0] / ind[1]) if ind[1] else None,
        "_scored": scored,
    }


def fmt(v) -> str:
    if v is None:
        return "--"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--model", action="append", required=True,
                    help="prediction file label prefix, repeatable (e.g. M2ml, M3ml, M3ml_v2)")
    ap.add_argument("--langs", default="en,fr,ar")
    ap.add_argument("--latex", action="store_true", help="also print LaTeX table rows")
    args = ap.parse_args()
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    results: dict[tuple[str, str], dict] = {}
    seed_sets: dict[tuple[str, str], set] = {}
    for model in args.model:
        for lang in langs:
            p = args.dir / f"{model}_{lang}.jsonl"
            if not p.exists():
                continue
            recs = load(p)
            results[(model, lang)] = per_model_lang(recs)
            seed_sets[(model, lang)] = {r.get("seed_id") for r in recs}

    # Guard against cross-benchmark contamination: a (model, lang) file whose
    # seed_ids are disjoint from that model's other language files is on a
    # different benchmark (e.g. legacy gss-* vs aligned menst-*); drop it so it
    # cannot corrupt the aligned comparison.
    for model in args.model:
        present = [l for l in langs if (model, l) in results]
        for lang in list(present):
            others = set().union(*[seed_sets[(model, l)] for l in present if l != lang]) if len(present) > 1 else set()
            if others and not (seed_sets[(model, lang)] & others):
                print(f"# WARNING: dropping {model}_{lang} (seed namespace disjoint "
                      f"from other {model} languages -- different benchmark)", file=sys.stderr)
                del results[(model, lang)]

    # 1) per (model, language) headline table
    print("## Per (model, language)")
    print("| model | lang | n | parse | interp | under-triage | clar_recall | clar_spec | indirect_cult |")
    print("|---|---|---|---|---|---|---|---|---|")
    for model in args.model:
        for lang in langs:
            r = results.get((model, lang))
            if not r:
                continue
            print(f"| {model} | {lang} | {r['n']} | {fmt(r['parse_ok'])} | {fmt(r['interp'])} | "
                  f"{fmt(r['under_triage'])} | {fmt(r['clar_recall'])} | {fmt(r['clar_spec'])} | "
                  f"{fmt(r['indirect_cultural_interp'])} |")

    # 2) per-model cross-language consistency (aligned EN/FR/AR)
    print("\n## Cross-language consistency (aligned triples)")
    print("| model | langs present | risk consistency | category consistency | n_groups |")
    print("|---|---|---|---|---|")
    for model in args.model:
        present = [l for l in langs if (model, l) in results]
        if len(present) < 2:
            continue
        combined = []
        for l in present:
            combined.extend(results[(model, l)]["_scored"])
        block = ev._consistency_block(combined, ev.cross_language_consistency)
        risk, cat = block["predicted_risk"], block["predicted_category"]
        print(f"| {model} | {','.join(present)} | {fmt(risk['rate'])} | {fmt(cat['rate'])} | {risk['n_groups']} |")

    # 3) pairwise McNemar between models, per language
    print("\n## McNemar (paired, per language) — p-values")
    print("| lang | pair | parse | category | risk | clarification |")
    print("|---|---|---|---|---|---|")
    models = args.model
    for lang in langs:
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a, b = (models[i], lang), (models[j], lang)
                if a not in results or b not in results:
                    continue
                sa, sb = results[a]["_scored"], results[b]["_scored"]
                ids_a = {r["item_id"] for r in sa}
                ids_b = {r["item_id"] for r in sb}
                if ids_a != ids_b:
                    print(f"| {lang} | {models[i]} vs {models[j]} | item_id mismatch | | | |")
                    continue
                ps = {f: sm.mcnemar(sa, sb, f)["p_value"]
                      for f in ["parse_ok", "category_correct", "risk_correct", "clarification_correct"]}
                print(f"| {lang} | {models[i]} vs {models[j]} | {ps['parse_ok']:.2e} | "
                      f"{ps['category_correct']:.2e} | {ps['risk_correct']:.2e} | {ps['clarification_correct']:.2e} |")

    if args.latex:
        print("\n% LaTeX rows: model & lang & interp & under-triage & clar_recall")
        for model in args.model:
            for lang in langs:
                r = results.get((model, lang))
                if not r:
                    continue
                print(f"{model} & {lang} & {fmt(r['interp'])} & {fmt(r['under_triage'])} & {fmt(r['clar_recall'])} \\\\")


if __name__ == "__main__":
    main()
