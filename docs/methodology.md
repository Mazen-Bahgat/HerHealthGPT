# HerHealthEval — methodology and design choices

This document records the methodology and the design decisions made during the
evaluation and adaptation work, so the paper's Methods section and any future
contributor can trace *what was done and why*. It complements
`docs/m3_json_design_choices.md` (the fine-tune target-format deep-dive) and the
per-model reports in `results/`.

## 1. Task and benchmark

We evaluate **understanding**, not answer generation. Given a patient message,
a model must produce a structured interpretation: concern **category**
(menstrual / pcos / fertility / other), **risk** level (routine / see-doctor /
urgent), whether it should **ask for clarification**, and a reply.

The frozen benchmark is **540 items = 90 seeds × 6 communication styles**
(`Test/gold_seeds_styled_labeled.csv`). Each seed is one clinical situation
rendered in six registers: canonical, clinical, layperson, indirect_cultural,
ambiguous, emotionally_concerned. The six styles share **one gold
interpretation**, which is what makes cross-style consistency measurable: any
variation in a model's output across the six phrasings is a failure of
understanding, since the underlying situation is identical.

**Gold labels are language-invariant.** Risk, condition, and clarification-need
depend on the clinical situation, not the language it is expressed in. This is
the central design principle for the multilingual extension (§6): translations
change the *text*, never the *labels*.

## 2. Gold labels

Gold `gold_risk_level`, `gold_condition`, `gold_action`, and
`requires_clarification` are assigned by a deterministic, auditable heuristic and
frozen. The distribution is skewed (routine-heavy; ~1 in 6 items requires
clarification, matching the one ambiguous style per seed), so we **never report
raw accuracy alone**: every accuracy number is shown against its majority-class
baseline (risk 0.644, clarification 0.833), and the honest headline metrics are
per-class recall, under-triage, and clarification recall.

## 3. Leakage control (separation of benchmark and adaptation data)

Adaptation (fine-tune) data is drawn from the same seed pool but must be disjoint
from the benchmark. `scripts/prepare_ft_data_v2.py` enforces this and logs every
drop to `leakage_log.csv`:

- **Benchmark leak** — any training question equal (case-insensitive) to a
  benchmark question is removed.
- **Train/val dedup** — questions shared between train and val are removed from
  train; validation is left untouched.
- **Degenerate-ambiguous drop** — a flaw in the ambiguity rewrites replaced
  content words with "something" ("What is something? I'm not really sure…"),
  producing content-free questions paired with confident specific answers. These
  ~250 rows are dropped **by row_id** so the drop applies identically across
  languages (the English word-regex cannot detect them once text is translated).

## 4. Model progression

All models are Qwen3.5-9B, evaluated on the identical 540-item English benchmark.

| Model | What it is |
|---|---|
| **M2** | Zero-shot baseline (instruction-tuned, structured prompt, no fine-tune). |
| **M3_QA** | QLoRA fine-tune on plain question→answer targets. |
| **M3_JSON** | QLoRA fine-tune on eval-shaped structured (JSON) targets. |
| **M3_J+O** | M3_JSON + 4× oversampling of clarification (ambiguous) rows. |
| **M3_ML** | Joint EN+FR+AR fine-tune (same recipe as M3_J+O). |

Findings, in one line each (full numbers in `results/00_ALL_MODELS_COMPARISON.md`):

- Zero-shot **under-triages 72%** of care-warranting cases and interprets the same
  situation differently across phrasings (risk consistency 0.644, category 0.156).
- Plain-QA fine-tuning **breaks** JSON format and clarification.
- Structured (JSON) targets **restore format**, cut under-triage, and double
  cross-style category consistency.
- Oversampling clarification rows gives the **best triage** (risk parity with
  zero-shot, lowest under-triage) but does **not** restore clarification recall,
  which collapses under every fine-tune variant.
- The multilingual model gives the **highest cross-style category consistency**
  but, on English, does not beat the English-only model (expected — its value is
  in FR/AR, not yet measurable).

The target-format ablation is the core adaptation result; see
`docs/m3_json_design_choices.md`.

## 5. Evaluation metrics

Computed by `scripts/evaluate.py` and `scripts/safety_metrics.py`:

- **parse_ok** — fraction producing schema-valid JSON.
- **risk / category / clarification accuracy** — reported with majority baselines.
- **under-triage rate** — gold=see-doctor predicted routine (the key safety
  metric); **over-triage** — gold=see-doctor predicted urgent.
- **clarification recall / specificity** — asks when it should / stays quiet when
  it shouldn't.
- **misunderstanding rate**, **self-reported-unsafe** (model-flagged, not
  independently validated — labeled as such).
- **cross-style consistency** — same label across all 6 styles of a seed.
- **cross-language consistency** — same label across languages of a seed
  (currently 0/n=0 because only English exists; unlocked by §6).
- **McNemar paired tests** vs M2 and **95% bootstrap CIs**.

## 6. Multilingual adaptation (EN + FR + AR)

The team translated the *adaptation* data (train/val) into French and Arabic via
per-language handoff files. Two design points make the multilingual corpus a
faithful parallel of the English one:

- **Style recovery** — the handoff schema carries `row_id` but not `Style`, and
  every clarify decision keys on Style. `load_style_by_row_id` / `recover_styles`
  rejoin Style from the aligned English row by `row_id`. Without this, translated
  rows all look non-ambiguous and the clarification signal silently vanishes
  (verified: 0 clarify examples before the fix, 1,124 / 32% after).
- **Identical cleaning across languages** — leakage, dedup, and degenerate-row
  drops are applied by row_id so all three languages keep exactly the same
  surviving rows.

The three per-language corpora (`--format json --oversample-clarify 4`) are
merged and shuffled deterministically by `scripts/merge_ft_langs.py` into a
10,538-row joint corpus, then fine-tuned with the M3_J+O recipe.

## 7. FR/AR *benchmark* translation

To report genuine multilingual results and cross-language consistency, the 540
benchmark **questions** (not answers, and never the labels) are translated into
FR and AR and validated before use. The translated benchmarks are now built:
`gold_seeds_styled_fr.jsonl` and `gold_seeds_styled_ar.jsonl`, 540 items each,
carrying the **canonical English gold labels** unchanged (verified: gold
identical to English row-for-row; all FR rows in Latin script, all AR rows in
Arabic script). Evaluation of M2 and M3-ML on both languages is in progress.

The pipeline is:

1. `scripts/build_benchmark_translation_handoff.py` → per-language handoff CSVs
   (English question + empty translation column, keyed by seed+style), for the
   team to translate and validate.
2. `scripts/build_translated_benchmark.py` → per-language benchmark JSONL
   (translated question + **canonical English gold**, joined by seed+style).
3. `scripts/run_local_inference.py --language {fr,ar}` → run any model on the
   translated benchmark.
4. Evaluate as in §5; cross-language consistency becomes measurable.

Gold labels remain canonical in the English Test file throughout; translation
only ever supplies the question text the model reads.
