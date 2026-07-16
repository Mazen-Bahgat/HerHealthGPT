# M3 fine-tune — target-format design choices

This document explains **how** the M3 adaptation data is built and **why** each
choice was made. It is the design rationale behind
`scripts/prepare_ft_data_v2.py` (the `--format json` path) and
`scripts/train_qlora.py`.

The short version: *what you fine-tune on must match what you evaluate.* Naive
question→answer fine-tuning silently destroys the structured triage behavior the
benchmark scores. Wrapping the identical training content in the evaluation's own
JSON schema fixes format and triage, but does **not** by itself restore
clarification-seeking.

## The problem that forced these choices

The benchmark scores a model on a **structured triage task**: given a patient
message, the model must emit a JSON object with a concern category, a risk level,
whether it should ask a clarifying question, and a free-text reply
(`scripts/run_inference.py:FIXED_PROMPT_TEMPLATE`). Our first fine-tune
(`M3_QA`) trained on plain `Question → Answer` text. Result on the 540-item
English benchmark:

| | parse_ok | under-triage | clarification recall |
|---|---|---|---|
| M2 (zero-shot) | 1.000 | 0.718 | 0.856 |
| M3_QA (plain answers) | 0.865 | 0.800 | 0.174 |

Fine-tuning on answers taught the model to answer in prose and **stop following
the structured instruction** — JSON compliance dropped, and clarification-seeking
collapsed. This motivated everything below.

## Design choice 1 — train on eval-shaped JSON targets

Each training row's **user turn** is the exact `FIXED_PROMPT_TEMPLATE` used at
evaluation, and the **assistant turn** is the 8-key JSON object the benchmark
parses (`to_json_record`). Training and evaluation now speak the same format.

Effect (`M3_JSON`): parse_ok returns to 0.998 and under-triage drops below the
zero-shot baseline (0.647). Format is a solved problem once training matches eval.

## Design choice 2 — derive the structured labels, don't invent a new schema

The training seeds have only `Question / Answer / Topic / Keywords / Style`; they
carry no gold risk/clarification labels (those exist only on the frozen
benchmark). We derive the JSON fields as **silver labels**, reusing the same
deterministic logic as the benchmark's own labeler so train and eval are on the
same scale:

- `predicted_category`: `enum_category(Topic)` — see choice 3.
- `predicted_risk`: `risk_heuristic(Answer)` — reused verbatim from
  `scripts/build_ft_mix_v2.py` (urgent/see-doctor/routine keyword rules on the
  answer text). No reimplementation.
- `interpreted_symptom` / `recommended_action`: first sentence of the answer.
- `response_text`: the answer (truncated).

These are silver labels used **only for training**; they are never used to score
the benchmark. The benchmark keeps its independent, frozen gold.

## Design choice 3 — clamp the category to the model's 4-way enum

The source `Topic` field is a 13-way taxonomy (menstruation, menarche, menopause,
PMS, symptoms, …) but the model's schema allows only
`{menstrual, pcos, fertility, other}`. Left unclamped, 386 training targets
carried out-of-enum categories like `"menstruation"`, which would teach the model
to emit invalid values. `enum_category()` maps the stragglers
(menstruation/menarche/menopause/pms → menstrual; everything else → other). This
is why **100% of training targets parse** through the real evaluation parser — a
check we run before every training run.

## Design choice 4 — ambiguous style ⇒ ask for clarification

The benchmark labels the **ambiguous** communication style as
`requires_clarification = yes` (one style per seed). To model that behavior, every
ambiguous-style training row sets `asks_clarification = True`, a fixed
`clarifying_question`, `predicted_risk = see-doctor`, and `response_text` = the
clarifying question. All other styles answer directly. This mirrors the
benchmark's own label rule rather than inventing a separate clarification
signal.

## Design choice 5 — oversample the clarification rows

In the natural mixture, ambiguous rows are only ~10.5% of training, and even in
the correct JSON format the fine-tune drove clarification recall to 0.044. We
added `--oversample-clarify N`, which repeats ambiguous rows `N×` (only the
ambiguous ones; order preserved). At `N=4` the clarify signal rises to **32%** of
the corpus.

Empirical result — oversampling helps triage but **not** clarification:

| | parse_ok | risk acc | under-triage | clarif. recall | consistency (cat.) |
|---|---|---|---|---|---|
| M3_JSON | 0.998 | 0.623 | 0.647 | 0.044 | 0.344 |
| M3_J+O (4×) | 0.996 | **0.665** | **0.605** | **0.000** | **0.378** |

`M3_J+O` reaches risk parity with the zero-shot baseline (0.665 vs 0.667), the
lowest under-triage of any model, and the highest cross-style category
consistency — but clarification recall does not recover; it falls further, to
0.000. This is a **finding, not a bug**: clarification-seeking behaves as a
fragile, emergent instruction-following capability of the base model that
answer-style supervised fine-tuning erases, and that naive data rebalancing does
not restore. Recovering it likely needs an objective that rewards
abstention/asking (e.g. preference optimization), not more demonstrations.

## Training configuration (`train_qlora.py`)

QLoRA on Qwen3.5-9B: 4-bit NF4, LoRA r=16 α=16, lr 2e-4, batch 2 × grad-accum 8,
2 epochs, thinking-mode off, trained on responses only. A `--max-steps 5` smoke
gate runs before every full run. The English J+O corpus is 2,672 train / 663 val
(after leakage/degenerate cleaning) → 3,515 after 4× clarify oversampling.

## Pre-flight checks encoded in the pipeline

1. **Zero benchmark leakage** — any training question matching a benchmark
   question verbatim is dropped and logged (`leakage_log.csv`).
2. **Degenerate ambiguous rows dropped** — ambiguity rewrites that erased the
   content ("What is *something*? I'm not really sure…") are removed by row_id
   across all languages.
3. **All targets parse** — every emitted JSON target is validated against the
   real evaluation parser.
4. **Clarify ratio ≈ 32%** — verified per language before training, so the
   clarification signal is never silently lost (see the multilingual note in
   `docs/methodology.md`).
