# RESULTS_AUDIT — number → source provenance

Every important numerical value in the updated manuscript, mapped to its source
result file, branch, benchmark, and verification status. **All values are on the
`gss` evaluation benchmark** (540 items/language; gold risk 174 see-doctor / 348
routine / 18 urgent; 90 clarification-yes), scored by `scripts/evaluate.py` +
`scripts/safety_metrics.py`.

Legend — branch/machine: `origin/main` = PC1 (user `sw2`, the paper's runs);
`hassan-pc` = PC2 (user `hassan`, the new corrected run). "Verified" = recomputed
from the source file with the current scorer.

## Table `tab:main` (English pilot) — UNCHANGED (paper's existing numbers)
| Value(s) | Source file (branch) | Verified |
|---|---|---|
| M2 col (parse 1.000, risk 0.667, u-triage 0.718, clar 0.856, cat 0.539, cons-risk 0.644, cons-cat 0.156) | `M2_gss_en.jsonl` (origin/main, gss) | ✓ (u-triage 0.718, clar 0.856 reproduced) |
| M3 col | `M3en_v2json_en.jsonl` (shared, gss) | paper value (not re-run) |
| M3+O col | `M3en_os4_en.jsonl` (shared, gss) | paper value (not re-run) |
| M3-ML col (English) | `M3ml_en.jsonl` (origin/main, gss) | ✓ (u-triage 0.711, cat 0.510 reproduced) |

## Table `tab:multilingual` — M2 & M3-ML UNCHANGED, M3-ML-RC NEW
| Metric | M2 (EN/FR/AR) | M3-ML (EN/FR/AR) | **M3-ML-RC (EN/FR/AR)** |
|---|---|---|---|
| parse_ok | 1.000/1.000/1.000 | 0.998/0.998/0.998 | **0.994/0.991/0.989** |
| risk accuracy | 0.667/0.685/0.674 | 0.655/0.646/0.646 | **0.663/0.677/0.684** |
| under-triage ↓ | 0.718/0.632/0.638 | 0.711/0.994/0.994 | **0.590/0.572/0.558** |
| clarif recall | 0.856/0.811/0.889 | 0.000/0.000/0.011 | **0.000/0.000/0.000** |
| category acc | 0.539/0.561/0.533 | 0.510/0.512/0.531 | **0.531/0.523/0.545** |
| consistency (risk) | 0.644/0.544/0.456 | 0.500/0.978/0.967 | **0.367/0.467/0.422** |

| Column | Source files (branch) | Verified |
|---|---|---|
| M2 (EN/FR/AR) | `M2_gss_en/fr/ar.jsonl` (origin/main, gss) | ✓ every cell reproduced exactly (incl. FR cons 0.544, AR cons 0.456) |
| M3-ML (EN/FR/AR) | `M3ml_en/fr/ar.jsonl` (origin/main, gss) | ✓ every cell reproduced (u-triage 0.994/0.994; cons 0.978/0.967) |
| **M3-ML-RC (EN/FR/AR)** | `M3ml_v2_gss_en/fr/ar.jsonl` (hassan-pc, gss, **new run**) | ✓ new; same prompt+decoding as paper (adapter `qwen3.5-9b-herhealth-enfrar-lora-v2`) |

## In-text numbers (Results §Cross-Lingual — recovery paragraph)
| Statement | Value | Source |
|---|---|---|
| FR under-triage recovery | 0.994 → 0.572 | M3ml_fr vs M3ml_v2_gss_fr |
| AR under-triage recovery | 0.994 → 0.558 | M3ml_ar vs M3ml_v2_gss_ar |
| EN under-triage recovery | 0.711 → 0.590 | M3ml_en vs M3ml_v2_gss_en |
| FR see-doctor McNemar | escalates 74 vs 1; p=4.0×10⁻²¹ | see-doctor subset (173 items), exact binomial |
| AR see-doctor McNemar | escalates 75 vs 0; p=5.3×10⁻²³ | see-doctor subset (172 items), exact binomial |
| RC cross-style risk consistency | 0.467 FR / 0.422 AR | `ev.cross_style_consistency` on M3ml_v2_gss_fr/ar |

## In-text numbers (Abstract / Introduction) — UNCHANGED + additions
| Statement | Value | Source | Note |
|---|---|---|---|
| clarification 0.856 → ~0 | 0.856 | M2_gss_en | unchanged |
| FR under-triage rises to 0.994 | 0.994 | M3ml_fr | **unchanged; verified this IS under-triage** (173/174 see-doctor→routine), not consistency |
| correction 0.994 → 0.572/0.558 | new | M3ml_v2_gss_fr/ar | added |

## The `0.994` provenance (was explicitly flagged for verification)
`0.994` is the **under-triage rate** of M3-ML on French and Arabic gss — confirmed
by `safety_M2_vs_M3ml_fr.md` (`under_triage_rate 0.632 → 0.994`) and by direct
recomputation (M3-ML routes 173/174 FR and 173/174 AR see-doctor cases to
`routine`). It is **not** a consistency value and **not** a copy error. The
manuscript's use of 0.994 as under-triage is correct.

## Numbers NOT changed but worth noting
- "M3-ML predicts routine for 538/540 French and 537/540 Arabic cases" (Results) —
  paper's existing count, unchanged (describes M3-ML, unaffected by RC).
- Majority baselines (risk 0.644, clarification 0.833) — property of the gss
  benchmark, unchanged.

## Reproducibility
- New file generation: `scripts/run_local_inference.py --adapter
  models/qwen3.5-9b-herhealth-enfrar-lora-v2 --benchmark
  gold_seeds_styled{,_fr,_ar}.jsonl --language {en,fr,ar} --gen-max-time 150`.
- Scoring + McNemar: `scripts/evaluate.py`, `scripts/safety_metrics.py`; see-doctor
  McNemar on gold=see-doctor items only (exact binomial).
- Full lineage and benchmark-identity evidence: `EXPERIMENT_LINEAGE_AUDIT.md`
  (code repository, branch `hassan-pc`).
