# HerHealthEval — Results

Consolidated results for the multilingual (EN/FR/AR) female-health
language-understanding study, N=540 items/language.

**Two evaluation benchmarks are represented here** (see
`EXPERIMENT_LINEAGE_AUDIT.md` in the repo root for why):
- **menst** (`benchmark_multilingual_v1`, 100% see-doctor gold) — the deep-dive
  analyses below (baseline report, relaxed metric, menstrual error analysis).
- **gss** (`gold_seeds_styled*`, mixed gold) — **the camera-ready paper's
  benchmark**. `gss_v2_recovery.md` reports the M3-ML-RC correction on gss, which
  was integrated into the paper at
  `camera_ready_updated_results/` (repo root) / `E:\camera_ready\camera_ready_updated_results\`.

## Contents
| Path | What it is |
|---|---|
| **`HerHealthEval_baseline_report.md`** / **`paper/HerHealthEval_baseline_report.pdf`** | **In-depth baseline & full-results report for supervisor review** — the document to build the paper on. Per-register, per-category, cross-style, bootstrap-CI, and McNemar detail with interpretation and a suggested paper structure. |
| `error_analysis_menstrual.md` / `paper/error_analysis_menstrual.pdf` | Error analysis of the menstrual interpretation gap — shows it is largely category-boundary ambiguity (conception-/PCOS-themed seeds), not a model blind spot, with confusion tables and verbatim examples. |
| `paper/HerHealthEval.pdf` | Compiled 10-page paper draft (final results). |
| `HerHealthEval_methodology_results.md` | Living methodology + results doc (protocol, metric taxonomy, all numbers, contribution statement, limitations). |
| `multilingual_report.md` | Auto-generated reproducible report (headline table, cross-language consistency, all McNemar p-values) from `scripts/multilingual_report.py`. |
| **`gss_v2_recovery.md`** | **The gss-benchmark result used in the paper:** M3-ML-RC recovers FR/AR under-triage 0.994→0.572/0.558 (see-doctor McNemar p≈10⁻²¹/10⁻²³). Directly comparable to the paper's M2/M3-ML. |
| `predictions/` | menst benchmark: 9 raw JSONLs (3 models × EN/FR/AR). |
| `predictions_gss/` | gss benchmark: M2_gss_en + M3-ML-RC EN/FR/AR (the new gss eval). M2_gss_fr/ar and M3-ML(v1) gss live on `origin/main`. |

> **Start here for review:** `paper/HerHealthEval_baseline_report.pdf` (a
> professor-ready deep dive) or its editable source `HerHealthEval_baseline_report.md`.

> Note: `predictions/M3ml_en.jsonl` is a **legacy English-only set** (`gss-*`
> seeds) on a *different* benchmark; it is excluded from all aligned multilingual
> claims (the report's seed-namespace guard drops it automatically). Kept only
> for provenance.

## Models
- **M2** — base Qwen3.5-9B, zero-shot (the baseline).
- **M3ml-v1** — QLoRA fine-tune with a risk-labeling bug (English consult-word
  heuristic on translated answers → FR/AR mislabeled routine). Ablation arm.
- **M3ml-v2** — same recipe, risk labels recovered language-independently from the
  English source by `row_id`. The corrected arm.

## Headline results (interp ↑ / under-triage ↓ / clarification recall)
| Model | EN | FR | AR |
|---|---|---|---|
| M2 (base) | 0.617 / 0.463 / 0.208 | 0.614 / 0.440 / 0.333 | 0.617 / 0.420 / 0.375 |
| M3ml-v1 | — | 0.605 / **1.000** / 0.000 | 0.618 / **0.998** / 0.000 |
| M3ml-v2 | **0.644** / 0.439 / 0.000 | 0.617 / 0.450 / 0.000 | **0.646** / 0.443 / 0.000 |

## Cross-language consistency (aligned EN/FR/AR triples, n=540)
| Model | risk | category |
|---|---|---|
| M2 (base) | 0.711 | 0.824 |
| M3ml-v1 (FR↔AR) | 0.994* | 0.880 |
| M3ml-v2 | 0.681 | 0.837 |

\* artifactual — both languages collapsed to `routine`.

## Key findings
1. **Naive multilingual fine-tuning degraded safety below baseline.** M3ml-v1
   understood FR/AR but under-triaged ~100% of see-doctor cases. McNemar vs base:
   b=289/298, c=0, p<10⁻⁶³ (entirely one-directional).
2. **Root cause = language-dependent label quality**, not comprehension (English
   consult-word heuristic mislabeled 99.8% of FR/AR training rows routine).
3. **The correction fully worked.** M3ml-v2 vs v1 risk: b=0, c=292 (FR) / 296
   (AR), p<10⁻⁶⁴. v2 restored non-English safety to **indistinguishable from
   baseline** (M2-vs-v2 risk p=0.88 FR / 0.94 AR) and posted the **best
   interpretation accuracy in every language**.
4. **Consistency became genuine again** (v2 risk 0.681 / category 0.837, like the
   base model; unlike v1's collapsed 0.994).
5. **Open limitation:** clarification recall stays 0.000 across all fine-tunes
   (deferred by design).

## Reproduce
```
python scripts/multilingual_report.py \
  --dir Used_Datasets/Consolidated_Datasets/200_Seed_Dataset \
  --model M2ml --model M3ml --model M3ml_v2 --langs en,fr,ar --latex
```
Produces `multilingual_report.md` verbatim.
