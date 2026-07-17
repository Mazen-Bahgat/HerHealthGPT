# HerHealthEval — Baseline & Full Results Report (in depth)

**Study:** *Do language models understand women's-health communication across
languages and cultures?*
**Object of study:** language *understanding* (interpretation, triage calibration,
clarification, cross-language/register consistency) — **not** answer generation.
**Prepared for supervisor review, to serve as the empirical basis for the paper.**

All numbers are computed deterministically by the committed evaluators
(`scripts/evaluate.py`, `scripts/safety_metrics.py`, `scripts/multilingual_report.py`)
on the aligned benchmark `benchmark_multilingual_v1`, N = 540 items per language.
Every figure in this report is reproducible with one command (see §9).

---

## 1. Executive summary

- The **baseline** is **M2 = base Qwen3.5-9B, zero-shot** (no fine-tuning),
  prompted with our structured triage schema. It is *also* the multilingual
  reference, because Qwen3.5-9B is natively multilingual.
- The baseline **interprets** the concern (category accuracy) at **0.617 / 0.614 /
  0.617** in EN / FR / AR — remarkably **stable across languages** (well above the
  0.333 majority baseline).
- Its safety-critical weakness is **under-triage** — routing a *see-doctor* case to
  *routine* — at **0.463 / 0.440 / 0.420** (EN/FR/AR): the base model misses
  ~42–46% of cases that should see a clinician, but does so *consistently* across
  languages.
- Two strong structure effects the paper is built on:
  1. **Register (surface form) drives interpretation.** Interpretation falls from
     ~0.66–0.68 on *clinical* phrasing to ~0.46–0.51 on *ambiguous* phrasing — a
     ~20-point swing on the **same underlying case**. Cross-style consistency is
     only **0.433** for risk (the model changes its risk verdict across the six
     phrasings of a case more than half the time).
  2. **Condition category matters enormously.** Fertility items are interpreted at
     ~0.89–0.92 but menstrual items at only ~0.39–0.40 — a ~50-point gap that is
     stable across all three languages.
- **Cross-language consistency** is genuine: for the same seed, the base model
  gives the same category verdict across EN/FR/AR **82.4%** of the time and the
  same risk verdict **71.1%** of the time — interpretation is largely
  language-invariant, and the residual disagreement is the honest cross-lingual
  instability the benchmark is designed to expose.
- The baseline retains **non-trivial clarification behaviour** (recall 0.21–0.38 on
  genuinely ambiguous items) — a capability both fine-tunes lose entirely.

**Why this matters for the paper.** The baseline establishes that a strong,
natively-multilingual model (i) understands the concern moderately well and
language-invariantly, yet (ii) is surface-form-sensitive and (iii) unsafe on ~44%
of high-risk cases. Everything else in the study — how English and multilingual
fine-tuning move these numbers — is measured *against this reference*.

---

## 2. The benchmark (`benchmark_multilingual_v1`)

| Property | Value |
|---|---|
| Seed clinical conditions | 90 |
| Categories (balanced) | menstrual, PCOS, fertility → 180 items each per language |
| Communication registers | canonical, clinical, layperson, indirect_cultural, ambiguous, emotionally_concerned (6) |
| Items per language | 90 seeds × 6 registers = **540** |
| Languages | English, French, Arabic → **1,620 parallel items** |
| Gold labels | concern **category**, **risk level**, recommended **action**, **requires-clarification** |

**Parallel design.** Each `menst-XXX` (etc.) seed appears in all 3 languages × 6
registers with the *same* gold labels — labels are a property of the underlying
case, not the surface form. This is what makes cross-language consistency and
cross-register consistency directly measurable.

**Gold-label distribution** (identical across languages by design):
- **risk_level: 100% see-doctor** → risk majority baseline = 1.0. The honest
  headline is therefore **under-triage rate**, not risk accuracy.
- **requires_clarification: 24 yes / 516 no** → clarification recall is measured on
  the 24 genuinely-underspecified (ambiguous) items; specificity on the other 516.
- **category: balanced 180/180/180** → category (interpretation) accuracy is
  informative, majority baseline 0.333.

**Translation-validation protocol.** GPT-assisted initial translation → (1)
native-speaker review for fluency/naturalness → (2) linguistic-professional review
for meaning preservation, ambiguity, and cultural appropriateness. Per-item
validation status is carried in the benchmark.

**Benchmark hygiene (important).** An earlier English-only set (`gss-*` seeds) is a
*different* question set (2/540 text overlap) with different clarification labels.
It is **excluded** from all multilingual claims; the report's seed-namespace guard
drops it automatically. (This is why M3ml-v1 has no aligned EN row.)

---

## 3. Metric definitions

All metrics are computed from the model's structured JSON prediction against gold.

| Metric | Definition | Notes |
|---|---|---|
| **parse_ok** | fraction of responses that parse into the required schema | format compliance |
| **Interpretation accuracy** | predicted concern category == gold category | majority baseline 0.333 |
| **Under-triage rate** ↓ | P(predict `routine` \| gold `see-doctor`) | the safety-critical error; lower is better |
| **Clarification recall** | of the 24 gold "needs clarification" items, fraction where the model asks | measured only on genuine ambiguity |
| **Clarification specificity** | of the 516 "no clarification" items, fraction where the model does *not* ask | guards against always-asking |
| **Indirect-cultural interpretation** | interpretation accuracy restricted to the `indirect_cultural` register | cultural-sensitivity proxy |
| **Cross-language consistency** | for a fixed seed, fraction of seeds where the verdict agrees across EN/FR/AR | meaning preservation |
| **Cross-style consistency** | for a fixed seed, fraction of seeds where the verdict is identical across all 6 registers | surface-form robustness |

**Statistics.** Bootstrap 95% CIs on rates (2,000 resamples); **McNemar's paired
test** on shared items for model-vs-model comparisons (reported later). Because
gold risk is uniformly `see-doctor`, we always report under-triage and per-class
behaviour rather than a single risk-accuracy number.

---

## 4. Baseline (M2) results — in depth

### 4.1 Overall, per language (with 95% CIs)
| Lang | parse_ok | interpretation [95% CI] | under-triage ↓ [95% CI] | clar. recall | clar. spec |
|---|---|---|---|---|---|
| EN | 1.000 | 0.617 [0.576, 0.657] | 0.463 [0.422, 0.506] | 0.208 | 0.878 |
| FR | 0.998 | 0.614 [0.573, 0.655] | 0.440 [0.399, 0.482] | 0.333 | 0.862 |
| AR | 1.000 | 0.617 [0.574, 0.656] | 0.420 [0.380, 0.463] | 0.375 | 0.849 |

**Reading:** interpretation is statistically identical across the three languages
(overlapping CIs); the base model understands the concern equally well regardless
of language. Under-triage is high everywhere (~0.42–0.46) but, again, consistent.

### 4.2 By communication register (the surface-form effect)
Interpretation accuracy / under-triage per register (averaged behaviour; each cell n≈90):

| Register | EN interp | FR interp | AR interp | EN u-triage | FR u-triage | AR u-triage |
|---|---|---|---|---|---|---|
| canonical | 0.667 | 0.652 | 0.644 | 0.444 | 0.416 | 0.444 |
| clinical | 0.678 | 0.667 | 0.667 | 0.478 | 0.522 | 0.467 |
| layperson | 0.644 | 0.611 | 0.611 | 0.489 | 0.433 | 0.433 |
| indirect_cultural | 0.611 | 0.600 | 0.611 | 0.478 | 0.389 | 0.367 |
| **ambiguous** | **0.456** | **0.511** | **0.500** | 0.533 | 0.500 | 0.478 |
| emotionally_concerned | 0.644 | 0.644 | 0.667 | 0.356 | 0.378 | 0.333 |

**Key finding (RQ1).** Holding the clinical case fixed, moving from *clinical* to
*ambiguous* phrasing drops interpretation by ~20 points in every language, and
*ambiguous* also carries the highest under-triage. Indirect/cultural phrasing costs
a smaller but consistent ~5–7 points. The model's understanding is **driven by
surface form, not just clinical content**.

### 4.3 By condition category (a large, stable disparity)
Interpretation accuracy per category (n≈180 each):

| Category | EN | FR | AR |
|---|---|---|---|
| fertility | 0.917 | 0.894 | 0.889 |
| PCOS | 0.539 | 0.553 | 0.561 |
| **menstrual** | **0.394** | **0.394** | **0.400** |

**Key finding.** The base model recognises fertility concerns almost perfectly but
menstrual concerns poorly (~0.39), with PCOS in between — a ~50-point gap that
holds across all three languages.

**Error analysis (see `error_analysis_menstrual.md`) shows this is largely a
category-boundary artifact, not a blind spot.** Of the wrong menstrual items,
**87% are mapped to an adjacent category** (fertility or PCOS) and only 13% to
"other". The reason: **14 of the 30 menstrual seeds (47%) explicitly concern
conception/pregnancy** (e.g. *"Do irregular periods influence the ability to get
pregnant?"*, *"trying for child since 1 yr"*) — labeled `menstrual` by presenting
symptom but *fertility* by patient intent, which is what the model surfaces. One
seed (irregular periods + polycystic ovaries on ultrasound) is a textbook PCOS
presentation the model arguably reads *more* correctly than the gold. The
confusion is stable across all six registers and all three languages, and
fine-tuning merely shifts it (base → fertility; fine-tunes → PCOS) without
resolving it.

**Relaxed clinically-acceptable metric (implemented; `scripts/relaxed_interp.py`).**
Crediting content-justified adjacent reads (conception→fertility, cysts→PCOS,
gated on the language-independent English case content) raises the base model's
interpretation to **~0.88** (menstrual 0.39→0.81):

Reported as three brackets — **strict** (exact), **relaxed (gated)** = the
defensible headline, **loose** = upper bound crediting any clinical-category read:

| Model | Lang | strict | relaxed (gated) | loose (upper) |
|---|---|---|---|---|
| M2 (base) | EN | 0.617 | **0.878** | 0.930 |
| M2 (base) | FR | 0.614 | **0.881** | 0.952 |
| M2 (base) | AR | 0.617 | **0.883** | 0.952 |
| M3ml-v1 | FR/AR | 0.605/0.618 | 0.803/0.805 | 1.000/0.998 |
| M3ml-v2 | EN | 0.644 | 0.806 | 0.996 |
| M3ml-v2 | FR | 0.617 | 0.810 | 0.994 |
| M3ml-v2 | AR | 0.646 | 0.829 | 0.996 |

(The loose column is near-ceiling for the fine-tunes because they never abstain to
`other`; treat gated-relaxed as the headline and loose as an upper bracket.)

**Important nuance for the paper:** the strict-accuracy ranking *flips* under
clinical gating. By strict accuracy M3ml-v2 leads (0.644 > 0.617); by the
content-gated relaxed metric the **base model leads** (0.878 > 0.806), because
fine-tuning shifted menstrual errors toward PCOS (usually *not* justified by the
case) whereas the base model's errors go to fertility (*justified* by the
conception-themed seeds). So the fine-tune's strict gain partly reflects a
category-prior shift, not better clinical interpretation. See
`error_analysis_menstrual.md` for the full table and mechanism.

### 4.4 Cross-style consistency (same case, six phrasings)
| Lang | risk consistency | category consistency |
|---|---|---|
| EN | 0.433 | 0.489 |
| FR | 0.433 | 0.478 |
| AR | 0.433 | 0.522 |

**Reading.** For only ~43% of seeds does the base model give the *same* risk
verdict across all six phrasings, and ~half for category. This is direct evidence
that interpretation is unstable under paraphrase — the central motivation for a
register-controlled benchmark.

### 4.5 Cross-language consistency (same seed, three languages)
Risk **0.711**, category **0.824** (n = 540 aligned triples). Interpretation is
largely language-invariant (82% agreement); the residual ~18% category and ~29%
risk disagreement is genuine cross-lingual instability over a *live* risk
distribution — not an artifact.

### 4.6 Clarification behaviour
The base model **does** ask for clarification on genuinely ambiguous input — recall
0.21 (EN) rising to 0.33–0.38 (FR/AR) — at high specificity (0.85–0.88, i.e. it
rarely over-asks). This calibrated "ask when unsure" capability is a baseline
strength; §6 shows both fine-tunes destroy it.

---

## 5. Full study context — how the fine-tunes move the baseline

The baseline is the reference for a two-step adaptation study.

| Model | EN interp / u-triage | FR interp / u-triage | AR interp / u-triage |
|---|---|---|---|
| **M2 (base)** | 0.617 / 0.463 | 0.614 / 0.440 | 0.617 / 0.420 |
| M3ml-v1 (buggy labels) | — (legacy set) | 0.605 / **1.000** | 0.618 / **0.998** |
| **M3ml-v2 (corrected)** | **0.644** / 0.439 | 0.617 / 0.450 | **0.646** / 0.443 |

- **M3ml-v1** = QLoRA multilingual fine-tune whose risk labels were derived by an
  English consult-word heuristic run on *translated* answers, mislabeling 99.8% of
  FR/AR training rows `routine`. Result: it understands FR/AR as well as the base
  model but **under-triages ~100%** of high-risk cases — i.e. naive multilingual
  fine-tuning made the model *less safe than the baseline*.
- **M3ml-v2** = identical recipe with risk labels recovered language-independently
  from the English source by `row_id`. Result: under-triage returns to **0.45/0.44
  ≈ baseline**, and interpretation becomes the **best of any configuration** in
  every language.

### Statistical confirmation (McNemar, paired, shared 540 items; b = first-model right & second wrong, c = reverse)
| Comparison | Lang | Measure | b | c | p |
|---|---|---|---|---|---|
| M2 vs M3ml-v1 | FR | risk | 289 | 0 | 2.2e-64 |
| M2 vs M3ml-v1 | AR | risk | 298 | 0 | 2.5e-66 |
| M3ml-v1 vs M3ml-v2 | FR | risk | 0 | 292 | 5.0e-65 |
| M3ml-v1 vs M3ml-v2 | AR | risk | 0 | 296 | 6.7e-66 |
| M2 vs M3ml-v2 | FR | risk | 84 | 87 | 0.88 |
| M2 vs M3ml-v2 | AR | risk | 83 | 81 | 0.94 |
| M2 vs M3ml-v2 | EN | risk | 63 | 96 | 0.011 |
| (all) | all | category | — | — | n.s. (0.06–1.0) |

**Interpretation:** the base-vs-v1 risk discordance is total and one-directional
(c = 0): the baseline is safer wherever they disagree. The correction (v1→v2) flips
it completely (b = 0). After the fix, v2 vs the baseline is statistically
**indistinguishable** on risk in FR/AR (p = 0.88/0.94) and slightly safer in EN —
i.e. corrected supervision restored baseline safety — while category interpretation
is unchanged throughout. **Clarification** is the exception: both fine-tunes drop to
0.000 recall, so the baseline retains a real advantage there (p ~ 1e-11).

---

## 6. What the baseline tells us (for writing the paper)

1. **A strong multilingual base model is only moderately safe out of the box**
   (~44% under-triage on curated high-risk items) — motivating the whole
   "understanding ≠ safety" framing.
2. **Interpretation is language-invariant but form-sensitive.** Cross-language
   consistency is high (0.82 category) while cross-style consistency is low (0.43
   risk). This dissociation is the paper's cleanest empirical hook for RQ1/RQ2.
3. **Category and register are the two axes of failure**: menstrual concerns and
   ambiguous phrasing are where interpretation collapses. Both are stable across
   languages, so they generalise.
4. **The baseline sets the safety bar that fine-tuning must not fall below** — and
   the study's headline is precisely that naive fine-tuning *did* fall below it,
   and corrected supervision climbed back to it.
5. **Clarification is the open problem** at every stage (baseline modest, fine-tunes
   zero) — the clearest direction for future work.

---

## 7. Threats to validity / honest limitations
- **Gold risk is uniformly `see-doctor`** (curated high-stakes set). This supports
  under-triage analysis but not full triage calibration (no `routine`/`urgent`
  ground truth to score specificity of the risk head).
- **Cultural sensitivity is proxied** by `indirect_cultural`-register performance,
  not by human cultural-appropriateness ratings.
- **Response helpfulness/clarity** is not yet automatically scored (LLM-judge
  candidate).
- The **menstrual interpretation gap** needs a qualitative error analysis before
  strong claims (is it a labeling artifact or a genuine model blind spot?).
- Single fine-tuning seed (3407); no multi-seed variance yet.

---

## 8. Suggested paper structure grounded in these results
1. **Framing:** understanding ≠ safe answering; register + language as controlled axes.
2. **Benchmark contribution:** the parallel EN/FR/AR × 6-register design (§2).
3. **Baseline analysis (this report):** §4 — language-invariance, register
   sensitivity, category disparity, cross-style instability.
4. **Adaptation study:** §5 — the v1 safety regression and its v2 correction, with
   the McNemar ablation isolating *label quality* as the mechanism.
5. **Error analysis:** menstrual gap + ambiguous-register collapse + clarification loss.
6. **Limitations & ethics:** §7.

---

## 9. Reproducibility
Every number here regenerates from committed code and data:
```bash
# headline table + cross-language consistency + all McNemar p-values
python scripts/multilingual_report.py \
  --dir Used_Datasets/Consolidated_Datasets/200_Seed_Dataset \
  --model M2ml --model M3ml --model M3ml_v2 --langs en,fr,ar --latex
```
Raw predictions: `result/predictions/*.jsonl` (also under
`Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/`).
Per-register / per-category / cross-style / CI breakdowns in §4 were computed with
`scripts/evaluate.py` + `scripts/safety_metrics.py` scorers over those same files.

*All artifacts are on branch `hassan-pc`. The compiled paper is
`result/paper/HerHealthEval.pdf`; the living results doc is
`result/HerHealthEval_methodology_results.md`.*
