# LUHME Weekly Brief

## 1. A short literature summary from each of you.

### Summary A (Benchmark and Safety Perspective)
Recent medical NLP literature shows that domain adaptation alone does not guarantee clinically safe behavior; robust evaluation requires explicit tests of understanding, ambiguity handling, and calibration. For LUHME, this supports framing the project as a multilingual language-understanding benchmark rather than only a response-generation system. The benchmark should prioritize realistic patient wording, meaning preservation under paraphrase and translation, and strict leakage control between evaluation and fine-tuning tracks.

### Summary B (Multilingual and Low-Resource Perspective)
Multilingual health QA studies indicate that translation quality and annotation consistency are major error drivers, especially for symptom nuance and culturally specific phrasing. Cross-lingual transfer is improved when benchmark items are parallelized across languages and validated by human reviewers with adjudication. For LUHME, this justifies a freeze-first workflow: lock EN benchmark lineage, translate to AR/FR, apply dual validation, and evaluate models under identical prompts and scoring criteria.

## 2. A dataset comparison table.

| Dataset | Domain Coverage | Size / Format | Strengths | Risks / Constraints | Fit for LUHME Role |
|---|---|---|---|---|---|
| MENST (`training2K.csv`, `train24K.csv`) | Menstrual health, PCOS, fertility-relevant education | CSV, structured QA (`~42K` in `train24K`) | Topic relevance, curated structure, clear metadata fields | Schema variation across files; must exclude held-out test lineage | Primary source for balanced fine-tuning pool construction |
| HealthCareMagic-100k-Chat-Format-en | Broad clinical Q&A dialogs | JSONL (`~112K` lines), `<human>/<bot>` text format | Large scale, realistic patient language | Requires parsing and strict filtering to LUHME categories | High-yield augmentation source after filtering and provenance tracking |
| ChatDoctor-iCliniq | General medical Q&A with parallel answer fields | Parquet (`7,321` examples) | Multiple response references per question | Not women-health-specific by default; potential domain drift | Supplemental source for candidate mining and phrasing diversity |
| HerHealthGPT-LU seed set (`seeds_en_v1.csv`) | LUHME-focused benchmark nucleus (`menstrual`, `pcos`, `fertility`) | 90 seeds / 540 style rows | Frozen benchmark lineage, style diversity, audit-ready provenance | Not for training once frozen; leakage risk if lineage not blocked | Benchmark v1 nucleus and evaluation anchor |

## 3. A selected primary dataset.

The selected primary dataset for the next phase is the **existing LUHME English benchmark nucleus** (`seeds_en_v1.csv`) for evaluation, paired with the **non-benchmark MENST-centered pool** for fine-tuning data assembly. This preserves the benchmark objective while keeping enough category-specific volume for adaptation.

- Benchmark v1 is fixed at **90 seeds / 540 rows** (30 seeds per category across six styles).
- Fine-tuning target is finalized at **800-1,000 per category** (**2,400-3,000 total**), with **900/category (2,700 total)** as the recommended default.
- Leakage prevention is mandatory: block any item sharing `source_dataset + source_row_id` with benchmark lineage, and maintain this in `leakage_note.md`.

## 4. A draft dataset schema.

The following draft schema is proposed for the unified multilingual benchmark and fine-tuning manifests.

| Field | Type | Description | Required |
|---|---|---|---|
| `item_id` | string | Stable benchmark/fine-tuning item identifier | Yes |
| `seed_id` | string | Seed lineage ID (`menst-*`, `pcos-*`, `fert-*`) when applicable | Yes |
| `split_track` | enum | `benchmark` or `finetune` | Yes |
| `split_name` | enum | `benchmark_v1`, `train`, `validation` | Yes |
| `language` | enum | `en`, `ar`, `fr` | Yes |
| `category` | enum | `menstrual`, `pcos`, `fertility` | Yes |
| `style` | enum | `canonical` plus style variants | Yes |
| `text` | string | User question text in the target language | Yes |
| `source_dataset` | string | Upstream dataset name | Yes |
| `source_row_id` | string | Upstream row/line identifier | Yes |
| `leakage_key` | string | Deterministic key from `source_dataset + source_row_id` | Yes |
| `grounding_label` | string | Draft evidence condition or `NEEDS_GROUNDING` | Recommended |
| `needs_grounding_flag` | boolean | Grounding uncertainty flag | Recommended |
| `translator_id` | string | Translator identity for AR/FR | AR/FR only |
| `validator_1`, `validator_2` | string | Dual human validator IDs | AR/FR only |
| `validation_status` | enum | `accepted`, `needs_revision`, `adjudicated` | AR/FR only |

## 5. An Arabic and French translation-validation plan.

1. Freeze the EN benchmark lineage (`benchmark_v1`) before translation begins.
2. Translate each EN item to AR and FR while preserving intent, symptom semantics, ambiguity level, and style register.
3. Run dual independent human validation per translated item with a compact checklist:
   - meaning preserved
   - naturalness acceptable
   - symptom/intent retained
   - ambiguity preserved where intended
   - no unsafe added clinical claims
4. If validators disagree, trigger adjudication and record rationale.
5. Only accepted/adjudicated items enter the multilingual frozen benchmark release.
6. Store all validation decisions with item-level provenance and reviewer IDs for auditability.

## 6. A preliminary experiment design.

### Planned split (current direction)

| Split | Per-category target | Total target | Notes |
|---|---:|---:|---|
| Benchmark v1 (frozen) | 30 seeds | 90 seeds / 540 rows | Evaluation only; never used for training |
| Validation (non-benchmark) | small balanced set (TBD) | TBD | Built after leakage exclusion; locked before training |
| Fine-tuning train (range) | 800-1,000 | 2,400-3,000 | Finalized target range |
| Fine-tuning train (recommended default) | 900 | 2,700 | First concrete run setting |

### Model conditions and protocol

- Compare at least three conditions: baseline model, multilingual baseline model, and fine-tuned model.
- Keep prompts, decoding policy, and scoring rubric fixed across conditions.
- Evaluate on the same frozen EN/AR/FR benchmark IDs.
- Report both aggregate and category/language/style-stratified performance.
- Conduct item-level error analysis using a misunderstanding taxonomy (e.g., category confusion, unsafe diagnostic leap, clarification failure, translation-induced drift).

## 7. A rough Introduction drafted in Overleaf.

```latex
\section{Introduction}
Women's health question answering remains challenging for language models because user queries are often ambiguous, colloquial, and clinically under-specified. These challenges become more pronounced in multilingual settings, where translation and cultural phrasing can alter symptom meaning and safety-relevant nuance. To address this gap, we position LUHME (HerHealthGPT-LU) as a multilingual language-understanding benchmark focused on three practical categories: menstruation, polycystic ovary syndrome (PCOS), and fertility.

Our benchmark design separates evaluation from adaptation by freezing an English benchmark nucleus (90 seeds; 540 style-augmented rows) and enforcing leakage prevention through source-level provenance keys (\texttt{source\_dataset + source\_row\_id}). This protocol reduces contamination risk when constructing non-benchmark validation and fine-tuning splits. For adaptation, we target 800--1,000 instances per category (2,400--3,000 total), with 900 per category as the recommended default for the first run.

We further extend the benchmark to Arabic and French through translation with dual human validation and adjudication, enabling controlled cross-lingual evaluation under matched item identities. This setup supports systematic comparison across base, multilingual, and fine-tuned model conditions while preserving fairness, traceability, and safety-oriented analysis. Beyond aggregate accuracy, LUHME emphasizes misunderstanding patterns relevant to women's health communication, including meaning drift, over-confident diagnostic inference, and missed clarification opportunities.
```

## 8. A list of candidate evaluation metrics.

- **Category accuracy** (`menstrual`, `pcos`, `fertility`) on frozen benchmark items.
- **Intent understanding score** (rubric-based, item-level).
- **Safety violation rate** (unsupported diagnosis, unsafe reassurance, harmful omission).
- **Clarification appropriateness** (asks for clarification when uncertainty is clinically relevant).
- **Evidence-grounding compliance** (aligned with approved grounding labels when available).
- **Cross-lingual consistency** (EN vs AR vs FR decision agreement on parallel items).
- **Style robustness** (performance variance across style variants per seed).
- **Calibration-oriented confidence quality** (if confidence is exposed): ECE/Brier-like diagnostics.
- **Error taxonomy distribution** (e.g., category confusion, translation-induced drift, grounding failure).
