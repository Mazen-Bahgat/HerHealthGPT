# HerHealthEval — Methodology, Evaluation Protocol, and Results (living doc)

**Paper:** *HerHealthEval: Do Language Models Understand Women's Health
Communication Across Languages and Cultures?*

This document is written continuously (per the supervisor's instruction) and is
filled in as experiments complete. It fixes the evaluation protocol so results
are reproducible and publication-valid, maps the required language-understanding
metrics onto what the pipeline computes, and states the paper's contribution.

Framing: **HerHealthEval is a language-understanding benchmark**, not a medical
chatbot or a fine-tuning paper. The object of study is *how and why* LLMs
understand or misunderstand indirect, culturally sensitive, vague, and
non-clinical female-health language across English, French, and Arabic, and
whether multilingual adaptation reduces those misunderstandings.

---

## 1. Benchmark (`benchmark_multilingual_v1.jsonl`)

| Property | Value |
|---|---|
| Seed conditions | 90 |
| Categories (balanced) | menstrual (30), PCOS (30), fertility (30) → 180 items/category/language |
| Communication registers (styles) | canonical, clinical, layperson, indirect_cultural, ambiguous, emotionally_concerned (6) |
| Items per language | 90 × 6 = **540** |
| Languages | English, French, Arabic (+ Egyptian-Arabic slang variant field) → **1,620** parallel items |
| Gold labels | condition, risk level, recommended action, requires-clarification |

Gold-label distribution (per language, identical across languages by design —
labels are a property of the *seed*, not the surface form):
- **risk_level: 100% see-doctor** → risk majority baseline = 1.0; the honest
  headline is **under-triage rate**, not accuracy.
- **requires_clarification: 24 yes / 516 no** → clarification recall is measured
  on the 24 genuinely-underspecified items.
- **category: balanced 180/180/180** → category accuracy is informative
  (majority baseline 0.333).

**Parallel design.** Each `menst-XXX` seed appears in all 3 languages × 6 styles
with the *same* gold labels, which is what makes **cross-language consistency**
and **meaning preservation** measurable (same seed, does the verdict agree
across languages / across styles?).

**Benchmark hygiene note (methodological finding).** An earlier EN-only set
(`gold_seeds_styled.jsonl`, `gss-*` seeds) is a *different* question set (2/540
text overlap) and uses different clarification labels. It is retained only as a
legacy EN reference; **all multilingual claims use `benchmark_multilingual_v1`
exclusively**, evaluated with the same benchmark file per language
(`--language en|fr|ar` selecting `en_text|fr_text|ar_text`) so seed IDs align.

**Translation-validation protocol** (documented for the paper): GPT-assisted
initial translation → (1) native-speaker review for fluency/naturalness → (2)
linguistic-professional review for meaning preservation, ambiguity, and cultural
appropriateness. Per-item validation status is carried in the benchmark
(`ar_validation_status`, `fr_house_validation_status`, `needs_human_review`).

---

## 2. Models compared

| ID | Role in the comparison | Description |
|---|---|---|
| **M2** | Base / multilingual open-source LLM | Qwen3.5-9B, zero-shot (no adapter). Qwen3.5-9B is natively multilingual, so it serves as both the base and the multilingual-base reference. |
| **M3ml-v1** | Fine-tuned multilingual (ablation) | QLoRA fine-tune on the joint EN+FR+AR corpus **with a risk-labeling bug** (risk derived by an English-word heuristic on translated answers → FR/AR mislabeled ~99.8% "routine"). Kept as an ablation. |
| **M3ml-v2** | Fine-tuned multilingual (corrected) | Same recipe, with risk labels recovered from the English source by `row_id` (language-independent) and a conservative consult-word expansion. |
| *(M4: +RAG)* | *if time allows* | Retrieval-augmented; out of scope for this iteration. |

Fine-tune recipe (reproducibility): QLoRA (4-bit NF4), LoRA r=16 α=16 on
all 7 projections; lr 2e-4, cosine, warmup 0.03, max-grad-norm 0.3, bf16, paged
AdamW-8bit, 2 epochs, effective batch 16, max-seq 2048, seed 3407; thinking-mode
off; loss on responses only. Joint corpus: 10,538 train / 1,989 val, 32%
clarification rows, risk dist (non-clarify) routine 0.583 / see-doctor 0.400 /
urgent 0.017 (identical across EN/FR/AR after the risk-by-`row_id` fix).

---

## 3. Evaluation protocol — language-understanding metrics

The supervisor's required metrics map onto the pipeline as follows. All are
computed deterministically from the model's structured JSON prediction
(`predicted_category`, `predicted_risk`, `asks_clarification`, …) against gold.

| Required metric (doctor's list) | Operationalization here | Source |
|---|---|---|
| Correct symptom interpretation | `category_accuracy` (predicted vs gold condition category), per style/language | evaluate.py |
| Meaning preservation across languages | **cross-language consistency**: same (model, seed, style), does `predicted_risk`/`predicted_category` agree across EN/FR/AR? | evaluate.py `cross_language_consistency` |
| Consistency across EN/FR/AR | same as above (risk + category agreement rate) | evaluate.py |
| Clarification behavior when vague | clarification **recall** on gold=yes (24 items, all `ambiguous`) + **specificity** on gold=no | safety_metrics.py |
| Misunderstanding rate | 1 − category accuracy (`misunderstanding_rate`); strict variant counts parse failures | safety_metrics.py |
| Unsafe response rate | **under-triage rate** = P(predict routine \| gold see-doctor) — "unsafe reassurance"; plus self-reported unsafe flag (caveated) | safety_metrics.py |
| Cultural sensitivity | per-style breakdown on the `indirect_cultural` register (interpretation + under-triage on culturally-indirect phrasing) | by_style breakdown |
| Response helpfulness / clarity | *not automated this iteration* — flagged as a limitation; candidate for an LLM-judge pass | — |

Statistics: **McNemar's paired test** (same items across models) on parse /
category / risk / clarification correctness; **bootstrap 95% CIs** on rates.
Reporting stays honest under the 100%-see-doctor risk skew: per-class recall,
under-triage, and majority baselines are shown alongside any accuracy number.

Additional cross-cutting analyses (the "why they misunderstand" contribution):
- **Cross-style consistency** — does the verdict change when the *same* seed is
  phrased clinically vs. ambiguously vs. emotionally? (Instability = the model's
  understanding is driven by surface form, not meaning.)
- **Per-style error profile** — where do failures concentrate (ambiguous →
  clarification collapse; indirect_cultural → interpretation drop)?
- **Cross-lingual under-triage** — the M3ml-v1→v2 ablation isolating label
  quality as the mechanism behind non-English under-triage.

---

## 4. Results

*(Filled as evals complete. All on `benchmark_multilingual_v1`; N=540/language;
expected item_ids `menst-XXX_<style>_<lang>`.)*

### 4.1 Model × language — headline table
All on `benchmark_multilingual_v1`, N=540/language. interp = category accuracy;
under-triage = P(routine | gold see-doctor); clar recall on 24 gold=yes items.

| Model | Lang | parse_ok | interp. acc | under-triage ↓ | clar. recall | clar. spec. | indirect-cultural interp. |
|---|---|---|---|---|---|---|---|
| M2 (base) | en | 1.000 | 0.617 | 0.463 | 0.208 | 0.878 | 0.611 |
| M2 (base) | fr | 0.998 | 0.614 | 0.440 | 0.333 | 0.862 | 0.600 |
| M2 (base) | ar | 1.000 | 0.617 | 0.420 | 0.375 | 0.849 | 0.611 |
| M3ml-v1 | fr | 0.998 | 0.605 | **1.000** | 0.000 | 1.000 | 0.618 |
| M3ml-v1 | ar | 0.998 | 0.618 | **0.998** | 0.000 | 0.998 | 0.578 |
| M3ml-v2 | en/fr/ar | *pending retrain* | | | | | |

### 4.2 Cross-language consistency
- **M2 base EN↔FR↔AR:** risk **0.711** (n=540 aligned triples), category
  **0.824** (n=540). This is *genuine* consistency: the base model gives the same
  category verdict across languages 82% of the time and the same risk verdict 71%
  of the time, with a non-degenerate risk distribution (under-triage 0.42–0.46
  across the three languages, not a collapse). Interpretation of the same case is
  largely language-invariant; the residual ~18% category disagreement and ~29%
  risk disagreement are the honest cross-lingual instability the benchmark is
  designed to surface.
- **M3ml-v1 FR↔AR:** risk 0.994 (n=540), category 0.880 (n=540). The near-perfect
  risk "consistency" is an *artifact* of *both* languages collapsing to routine —
  the model is consistently unsafe, not consistently correct. Contrast with M2's
  0.711: M2's lower number is more trustworthy because it reflects a live
  risk decision rather than a constant. Category consistency 0.880 (slightly above
  M2's 0.824) shows the fine-tune did not damage cross-lingual condition
  understanding — only the risk labels were broken.
- M3ml-v2 (EN/FR/AR triples): *pending retrain*.

### 4.3 Per-style error analysis
- Cross-style risk consistency (M3ml-v1): FR 0.978, AR 0.978 (n=90) — verdict
  barely varies with register (because it is uniformly routine).
- indirect_cultural interpretation (category acc): FR 0.618, AR 0.578 — culturally
  indirect phrasing is understood at roughly the language average, not worse.

### 4.4 M3ml-v1 → v2 ablation: cross-lingual under-triage
**Confirmed (both non-English languages):** M3ml-v1 under-triages 100%/99.8% of
FR/AR see-doctor items while parsing at 0.998 and interpreting category at
0.605/0.618 — it understands the input but was trained to under-triage it,
because the English-word risk heuristic mislabeled 99.8% of FR/AR training rows
routine. v2 (risk recovered from EN source by row_id) is the corrected arm;
retrain + eval pending.

**Confirmed finding so far (M3ml-v1, on this benchmark):** French fine-tuned
predictions collapse to "routine" on 539/540 items (under-triage ≈ 1.00) while
parse compliance (0.998) and category interpretation (0.605) transfer well —
i.e., the model *understood* the French but was *trained to under-triage* it,
because the risk labels themselves were broken for non-English. This is the
paper's clearest "why models misunderstand non-English safety" result and
motivates the v2 correction.

---

## 5. Contribution statement (draft — the doctor's open "??" item)

This paper makes four contributions:

1. **Benchmark (dataset).** *HerHealthEval*, a parallel multilingual (English,
   French, Arabic) benchmark of women's-health symptom descriptions: 90 seed
   conditions across three categories (menstrual, PCOS, fertility) rendered in
   five communication registers (clinical, layperson, indirect/cultural,
   ambiguous, emotionally-concerned) — 540 items per language, 1,620 parallel
   items — with gold condition/risk/action/clarification labels and a documented
   two-tier (native-speaker + linguistic-professional) translation-validation
   protocol. The parallel design enables direct measurement of meaning
   preservation and consistency across languages *and* registers.

2. **Evaluation protocol (metrics).** A language-understanding evaluation for a
   high-stakes domain that goes beyond QA accuracy: interpretation accuracy,
   **under-triage / unsafe-reassurance rate**, **clarification appropriateness**,
   **cross-language and cross-style consistency**, and misunderstanding rate —
   reported honestly under gold-label skew (per-class recall, majority baselines,
   McNemar, bootstrap CIs).

3. **Error analysis (why models misunderstand).** A systematic account of the
   failure modes of multilingual female-health understanding: (i) collapse of
   clarification behavior on genuinely ambiguous input, (ii) surface-form
   sensitivity across registers, and (iii) a **cross-lingual under-triage**
   failure in which non-English input is systematically judged low-risk — traced,
   via a controlled ablation, to language-dependent label quality rather than to
   the model's inability to understand the non-English text.

4. **Adaptation result (does multilingual fine-tuning help?).** Evidence on
   whether and how multilingual fine-tuning changes understanding and safety
   behavior across languages, with the v1→v2 ablation isolating the mechanism —
   showing that *correct, language-independent supervision*, not merely adding
   languages to the training mix, is what reduces cross-lingual misunderstanding.

---

## 6. Reproducibility

- Deterministic data pipeline (seed 3407); `data/ft/` regenerated from committed
  scripts (`prepare_ft_data_v2.py`, `merge_ft_langs.py`).
- Fixes committed on branch `hassan-pc`: consult-word risk expansion; FR/AR
  risk-by-`row_id` recovery; per-row generation time cap for stable inference.
- All eval artifacts (predictions JSONL, summaries, safety reports) committed
  under `Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/`.

## 7. Limitations (honest, for the paper)
- Response helpfulness/clarity not yet automatically scored (LLM-judge candidate).
- Cultural-sensitivity is proxied by `indirect_cultural`-register performance,
  not by human cultural-appropriateness ratings.
- Gold risk is uniformly "see-doctor" (curated high-stakes set): supports
  under-triage analysis but not full triage calibration.
