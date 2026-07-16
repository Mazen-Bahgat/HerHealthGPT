"""Generate human-readable results reports from the committed eval artifacts.

Reads the summary + safety JSON for each model in the HerHealthEval English
benchmark and writes one detailed markdown report per model plus a combined
comparison table, into results/. Reads committed JSON only (stdlib), so it is
unaffected by any unrelated merge-conflict state elsewhere in the tree.

Run: python scripts/generate_results_reports.py
"""
from __future__ import annotations

import json
from pathlib import Path

DATA = Path("Used_Datasets/Consolidated_Datasets/200_Seed_Dataset")
OUT = Path("results")

# (key, display name, description, summary file, safety file, safety label, out filename)
MODELS = [
    ("M2", "M2 — Zero-shot baseline",
     "Instruction-tuned Qwen3.5-9B, prompted with the structured triage schema. "
     "No fine-tuning. This is the zero-shot ('smoke test') reference.",
     "M2_gss_summary.json", "safety_M2_gss.json", "M2gss",
     "01_M2_zeroshot.md"),
    ("M3-QA", "M3(QA) — EN fine-tune, plain question->answer targets",
     "QLoRA fine-tune of Qwen3.5-9B on plain Question->Answer text pairs "
     "(English styled corpus). Ablation showing the effect of naive QA targets.",
     "M3en_v2_summary.json", "safety_M2_vs_M3en_v2.json", "M3env2",
     "02_M3_EN_plainQA.md"),
    ("M3-JSON", "M3(JSON) — EN fine-tune, structured (eval-shaped) targets",
     "QLoRA fine-tune on targets wrapped in the exact 8-key JSON schema the "
     "benchmark scores (English styled corpus).",
     "M3en_v2json_summary.json", "safety_M2_vs_M3en_v2json.json", "M3enJSON",
     "03_M3_EN_JSON.md"),
    ("M3-J+O", "M3(J+O) — EN fine-tune, structured targets + clarify oversampling",
     "Structured JSON targets with 4x oversampling of the ambiguous "
     "(clarification) training rows, raising them from 10.5% to 32% of the "
     "mixture. Best English triage model.",
     "M3en_os4_summary.json", "safety_M2_vs_M3en_os4.json", "M3os4",
     "04_M3_EN_JSON_oversample.md"),
    ("M3-ML", "M3(ML) — Joint EN+FR+AR fine-tune",
     "QLoRA fine-tune on the merged trilingual corpus (EN+FR+AR, structured "
     "targets + clarify oversampling, 10,538 train rows). Evaluated on the "
     "ENGLISH benchmark only (FR/AR benchmarks not yet built).",
     "M3ml_summary.json", "safety_M2_vs_M3ml.json", "M3ml",
     "05_M3_multilingual_ENFRAR.md"),
]


def load(model):
    key, name, desc, sfile, safefile, label, out = model
    summary = json.loads((DATA / sfile).read_text(encoding="utf-8"))
    safe = json.loads((DATA / safefile).read_text(encoding="utf-8"))
    analysis = safe["analyses"][label]
    mcnemar = safe.get("mcnemar")
    return dict(key=key, name=name, desc=desc, out=out,
                summary=summary, analysis=analysis, mcnemar=mcnemar)


def f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def pct(x, nd=1):
    return "n/a" if x is None else f"{x*100:.{nd}f}%"


def pfmt(p, nd=3):
    if p is None:
        return "n/a"
    if p < 0.001:
        return "<0.001"
    return f"{p:.{nd}f}"


def model_report(m) -> str:
    ov = m["summary"]["overall"]
    an = m["analysis"]
    ut = an["under_triage"]
    cl = an["clarification"]
    mb = an["majority_baselines"]
    L = []
    L.append(f"# {m['name']}\n")
    L.append(m["desc"] + "\n")
    L.append("Evaluation: frozen English benchmark, n=540 (90 seeds x 6 styles). "
             "Gold-label majority baselines: risk "
             f"{f(mb['risk'])}, clarification {f(mb['clarification'])}.\n")

    L.append("## Headline metrics\n")
    L.append("| metric | value | note |")
    L.append("|---|---|---|")
    L.append(f"| parse_ok rate | {f(ov['parse_ok_rate'])} | valid JSON produced |")
    L.append(f"| risk accuracy | {f(ov['risk_accuracy'])} | majority baseline {f(mb['risk'])} |")
    L.append(f"| category accuracy | {f(ov['category_accuracy'])} | 4-way |")
    L.append(f"| clarification accuracy | {f(ov['clarification_accuracy'])} | majority baseline {f(mb['clarification'])} |")
    L.append(f"| **under-triage rate** | {f(ut['under_triage_rate'])} | gold=see-doctor sent to routine (lower is better), n={ut['n_gold_see_doctor']} |")
    L.append(f"| over-triage rate | {f(ut['over_triage_rate'])} | gold=see-doctor sent to urgent |")
    L.append(f"| clarification recall | {f(cl['recall_gold_yes'])} | asks when gold=yes (n={cl['n_gold_yes']}) |")
    L.append(f"| clarification specificity | {f(cl['specificity_gold_no'])} | stays quiet when gold=no (n={cl['n_gold_no']}) |")
    L.append(f"| misunderstanding rate | {f(an['misunderstanding']['misunderstanding_rate'])} | |")
    L.append(f"| self-reported unsafe | {f(an['self_reported_unsafe_rate'])} | model-flagged, not validated |")
    L.append("")

    L.append("## Risk confusion (rows = gold, cols = predicted)\n")
    rc = an["risk_confusion"]
    cols = ["routine", "see-doctor", "urgent", "other"]
    L.append("| gold \\ pred | " + " | ".join(cols) + " | recall |")
    L.append("|" + "---|" * (len(cols) + 2))
    rr = an["risk_recall"]
    for g in ["routine", "see-doctor", "urgent"]:
        row = rc.get(g, {})
        L.append(f"| {g} | " + " | ".join(str(row.get(c, 0)) for c in cols) +
                 f" | {f(rr.get(g))} |")
    L.append("")

    L.append("## Category recall / precision\n")
    cr, cp = an["category_recall"], an["category_precision"]
    L.append("| category | recall | precision |")
    L.append("|---|---|---|")
    for c in ["menstrual", "pcos", "fertility", "other"]:
        L.append(f"| {c} | {f(cr.get(c))} | {f(cp.get(c))} |")
    L.append("")

    L.append("## By communication style\n")
    L.append("| style | parse_ok | risk acc | category acc | clarification acc |")
    L.append("|---|---|---|---|---|")
    for st, v in sorted(m["summary"]["by_style"].items()):
        L.append(f"| {st} | {f(v['parse_ok_rate'])} | {f(v['risk_accuracy'])} | "
                 f"{f(v['category_accuracy'])} | {f(v['clarification_accuracy'])} |")
    L.append("")

    L.append("## By gold category\n")
    L.append("| category | n | risk acc | category acc | clarification acc |")
    L.append("|---|---|---|---|---|")
    for c, v in sorted(m["summary"]["by_gold_category"].items()):
        L.append(f"| {c} | {v['n']} | {f(v['risk_accuracy'])} | "
                 f"{f(v['category_accuracy'])} | {f(v['clarification_accuracy'])} |")
    L.append("")

    csc = an["cross_style_consistency"]
    L.append("## Cross-style consistency\n")
    L.append("Fraction of the 90 seed groups given the same label across all six styles.\n")
    L.append(f"- risk: {f(csc['predicted_risk']['rate'])} (n={csc['predicted_risk']['n_groups']} groups)")
    L.append(f"- category: {f(csc['predicted_category']['rate'])} (n={csc['predicted_category']['n_groups']} groups)")
    L.append("")

    if m["mcnemar"]:
        L.append("## McNemar paired test vs M2 (zero-shot)\n")
        L.append("| field | b (M2 correct / this wrong) | c (this correct / M2 wrong) | chi^2 | p-value | significant (p<0.05) |")
        L.append("|---|---|---|---|---|---|")
        for field, v in m["mcnemar"].items():
            sig = "yes" if v["p_value"] < 0.05 else "no"
            L.append(f"| {field} | {v['b']} | {v['c']} | {f(v['chi2'],2)} | {pfmt(v['p_value'])} | {sig} |")
        L.append("")

    cis = an.get("cis") or {}
    if cis:
        L.append("## 95% bootstrap confidence intervals\n")
        for k, v in cis.items():
            if isinstance(v, (list, tuple)) and len(v) == 2:
                L.append(f"- {k}: [{f(v[0])}, {f(v[1])}]")
        L.append("")

    return "\n".join(L)


def comparison_report(models) -> str:
    L = []
    L.append("# HerHealthEval — all models compared (English benchmark, n=540)\n")
    L.append("Every model is evaluated on the identical 540-item frozen English "
             "benchmark with identical gold labels. Majority baselines: risk "
             "0.644, clarification 0.833. The multilingual model (M3-ML) is trained "
             "on EN+FR+AR but evaluated on English only.\n")
    names = [m["key"] for m in models]

    def row(label, fn, best=None):
        vals = [fn(m) for m in models]
        cells = []
        for v in vals:
            s = f(v) if isinstance(v, float) else str(v)
            cells.append(s)
        return f"| {label} | " + " | ".join(cells) + " |"

    L.append("| metric | " + " | ".join(names) + " |")
    L.append("|" + "---|" * (len(names) + 1))
    L.append(row("parse_ok", lambda m: m["summary"]["overall"]["parse_ok_rate"]))
    L.append(row("risk accuracy", lambda m: m["summary"]["overall"]["risk_accuracy"]))
    L.append(row("category accuracy", lambda m: m["summary"]["overall"]["category_accuracy"]))
    L.append(row("under-triage (lower better)", lambda m: m["analysis"]["under_triage"]["under_triage_rate"]))
    L.append(row("over-triage", lambda m: m["analysis"]["under_triage"]["over_triage_rate"]))
    L.append(row("clarification recall", lambda m: m["analysis"]["clarification"]["recall_gold_yes"]))
    L.append(row("clarification specificity", lambda m: m["analysis"]["clarification"]["specificity_gold_no"]))
    L.append(row("misunderstanding rate", lambda m: m["analysis"]["misunderstanding"]["misunderstanding_rate"]))
    L.append(row("consistency (risk)", lambda m: m["analysis"]["cross_style_consistency"]["predicted_risk"]["rate"]))
    L.append(row("consistency (category)", lambda m: m["analysis"]["cross_style_consistency"]["predicted_category"]["rate"]))
    L.append("")

    L.append("## McNemar vs M2 (p-values; --- = baseline)\n")
    fields = ["parse_ok", "risk_correct", "category_correct", "clarification_correct"]
    L.append("| field | " + " | ".join(names) + " |")
    L.append("|" + "---|" * (len(fields) if False else len(names) + 1))
    for field in fields:
        cells = []
        for m in models:
            if not m["mcnemar"]:
                cells.append("---")
            else:
                cells.append(pfmt(m["mcnemar"][field]["p_value"]))
        L.append(f"| {field} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## One-line read per model\n")
    L.append("- **M2 (zero-shot):** strongest clarification (0.856) but worst under-triage (0.718); baseline.")
    L.append("- **M3-QA (plain QA):** breaks JSON format (parse 0.865) and clarification (0.174); worst triage.")
    L.append("- **M3-JSON:** restores format (0.998), cuts under-triage, but clarification collapses (0.044).")
    L.append("- **M3-J+O:** best English triage — risk parity with M2 (0.665), lowest under-triage (0.605); clarification still 0.000.")
    L.append("- **M3-ML:** highest cross-style category consistency (0.411); English triage back near M2 (dilution); clarification 0.000. Real multilingual value needs FR/AR benchmarks.")
    L.append("")
    return "\n".join(L)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    models = [load(m) for m in MODELS]
    for m in models:
        (OUT / m["out"]).write_text(model_report(m), encoding="utf-8")
        print(f"wrote {OUT / m['out']}")
    (OUT / "00_ALL_MODELS_COMPARISON.md").write_text(
        comparison_report(models), encoding="utf-8")
    print(f"wrote {OUT / '00_ALL_MODELS_COMPARISON.md'}")


if __name__ == "__main__":
    main()
