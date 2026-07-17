# M3ml-v2 on the gss benchmark — label-corrected recovery (Option B rerun)

Per `EXPERIMENT_LINEAGE_AUDIT.md`, the paper is built on the **gss** benchmark. To
add the v2 label-correction *validly*, M3ml-v2 was evaluated on gss (EN/FR/AR) with
the **same prompt and greedy decoding** as the paper's runs (only the adapter
differs). New files: `M3ml_v2_gss_{en,fr,ar}.jsonl`.

**Pipeline parity check (validation):** re-scoring the paper's own `M2_gss_*` and
`M3ml_*` files with the current scorer reproduces the paper's published numbers
exactly (M2 EN under-triage 0.718; M3ml-v1 FR/AR 0.994/0.994) — so the new v2
numbers are directly comparable to the manuscript.

## All-gss comparison (n=540/lang; 174 see-doctor gold)

| Model | Lang | parse | under-triage ↓ | clar recall | category |
|---|---|---|---|---|---|
| M2 (paper) | EN | 1.000 | 0.718 | 0.856 | 0.539 |
| M2 (paper) | FR | 1.000 | 0.632 | 0.811 | 0.561 |
| M2 (paper) | AR | 1.000 | 0.638 | 0.889 | 0.533 |
| M3ml-v1 (paper) | EN | 0.998 | 0.711 | 0.000 | 0.510 |
| M3ml-v1 (paper) | FR | 0.998 | **0.994** | 0.000 | 0.512 |
| M3ml-v1 (paper) | AR | 0.998 | **0.994** | 0.011 | 0.531 |
| **M3ml-v2 (new)** | EN | 0.994 | **0.590** | 0.000 | 0.531 |
| **M3ml-v2 (new)** | FR | 0.991 | **0.572** | 0.000 | 0.523 |
| **M3ml-v2 (new)** | AR | 0.989 | **0.558** | 0.000 | 0.545 |

## The finding
- **v2 recovers the cross-lingual under-triage collapse on gss.** FR/AR under-triage
  falls from v1's **0.994 → 0.572 / 0.558** — below even the M2 baseline
  (0.632/0.638). English also improves (0.711 → 0.590). The label-corrected
  re-adaptation fixes the safety regression on the paper's own benchmark.
- **The recovery is statistically decisive on the see-doctor subset** (McNemar,
  escalation = "did not route to routine"):
  - FR: v2 escalates **74** see-doctor cases v1 under-triaged; v1 escalates **1** the
    other way; **p = 4.0×10⁻²¹**. Under-triaged see-doctor count v1 **173** → v2 **99**.
  - AR: v2 escalates **75**; v1 escalates **0**; **p = 5.3×10⁻²³**. v1 **173** → v2 **96**.
- **It is invisible to aggregate risk accuracy.** McNemar on overall `risk_correct`
  is *non-significant* (FR p=0.27, AR p=0.17), because on gss's mixed gold v2 trades
  routine-accuracy for see-doctor-accuracy. This is *not* a weakness of v2 — it is a
  direct instance of the paper's thesis that aggregate accuracy masks safety-critical
  behavior; under-triage on the see-doctor subset is the honest metric.
- **Clarification is not restored** (v2 recall 0.000, like v1) — the deferred
  clarification-collapse limitation stands unchanged.

## What this enables (for the manuscript — not yet applied)
The paper's `tab:multilingual` (M2 vs M3-ML on gss) can gain a **third column,
M3ml-v2**, all on the same gss benchmark and denominator. The narrative extends
cleanly: adaptation collapses non-English under-triage (0.632→0.994) → traced to a
language-dependent risk-label bug → **label-corrected re-adaptation recovers safety
(0.994→0.57), while clarification remains unsolved** — with the recovery quantified
by the see-doctor-subset McNemar and correctly shown to be hidden from aggregate
accuracy.

## Provenance / reproducibility
- Adapter: `models/qwen3.5-9b-herhealth-enfrar-lora-v2` (hassan-pc).
- Benchmarks: `gold_seeds_styled{,_fr,_ar}.jsonl` (FR/AR extracted read-only from
  `origin/main`, placed at canonical path; EN already present).
- Inference: `scripts/run_local_inference.py` (greedy, `--gen-max-time 150`).
- Scoring: `scripts/evaluate.py` + `scripts/safety_metrics.py`; under-triage McNemar
  computed on gold=see-doctor items only (exact binomial).
