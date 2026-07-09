# LUHME / HerHealthGPT-LU Direction Workflow

## Project Framing

This direction should be treated as a **language understanding benchmark project**, not only as a chatbot-building exercise. The core goal is to evaluate whether models can correctly understand women's health questions across **English, Arabic, and French**, preserve meaning across phrasing variants, avoid unsafe leaps beyond the evidence, and remain robust to realistic patient language.

The current seed deliverable already provides a strong English starting point:

- **90 English seeds**
- **540 total rows** (`90 seeds x 6 styles`)
- **30 seeds per category** across `menstrual`, `pcos`, and `fertility`
- A currently known balanced-pool ceiling of **1,154 per category** from the eligible Clear pool, because `fertility` is the smallest eligible category

For planning purposes, the project should keep two data tracks separate from the beginning:

1. **Benchmark data**: frozen evaluation items used to compare models fairly
2. **Fine-tuning data**: larger training pool used to adapt models

Those two tracks must never be mixed after the benchmark is frozen.

## Core Data Separation Rules

### Benchmark track

Use this track for:

- English benchmark seeds
- English style variants
- Future gold labels, rubric labels, and human-reviewed multilingual benchmark items

This track is for **evaluation only** after freeze.

### Fine-tuning track

Use this track for:

- Additional non-benchmark rows from the same source repositories
- Train/validation data for model adaptation
- Optional RAG document support sets if used separately

This track is for **training and tuning**, not for final benchmark scoring.

### Leakage prevention rule

The simplest leakage rule is:

> If a row appears in the benchmark lineage, then any row with the same `source_dataset + source_row_id` must be excluded from fine-tuning and validation pulls.

Operational block list reference: `leakage_note.md`.

Example:

- `HealthCareMagic-100k + line/row X`
- `MENST_train24K + row Y`
- `ChatDoctor-iCliniq + row Z`

Even if multiple style variants are generated from one seed, they all point back to the same original provenance key. That provenance key is what should be blocked from train/val selection.

## Recommended Working Split

This is a **recommended practical split**, not a fixed truth.

Finalized fine-tuning decision line: **800-1,000 per category** (**2,400-3,000 total**) for **SFT / multilingual adaptation**; recommended default **900 per category (2,700 total)**.

### Canonical split table (current benchmark vs future fine-tuning)

| Split | Per-category count | Total count | Purpose |
|------|-------------------:|------------:|---------|
| Benchmark seed v1 (current nucleus, frozen lineage) | 30 | 90 seeds / 540 rows | Benchmark/evaluation only |
| Validation (non-benchmark only) | TBD (small balanced set) | TBD | Tuning-time validation |
| Fine-tuning train (non-benchmark only) | 800-1,000 | 2,400-3,000 | SFT / multilingual adaptation |
| Fine-tuning train (recommended default) | 900 | 2,700 | Default first fine-tuning run |

### Recommended benchmark direction

- Keep the current **90 English seeds** as the initial benchmark nucleus
- Keep all **540 English style rows** as benchmark-linked material
- Keep benchmark nucleus separate from future fine-tuning target sizing; benchmark v1 remains **90 seeds / 540 rows**
- After translation and dual human validation, freeze matched **EN/AR/FR benchmark sets** as the main multilingual evaluation suite

### Recommended train/val direction

From the **non-benchmark** eligible pool only:

- **Validation**: build a small, high-quality locked set first
- **Training**: use the remaining balanced pool for model adaptation

Practical recommendation for the current phase:

- Reserve the benchmark lineage first
- Build a **balanced validation set** from the remaining eligible pool
- Use the rest as training for **SFT / multilingual adaptation**, targeting **800-1,000/category** with **900/category** as default

Because the current eligible Clear pool is:

- `menstrual`: **2,724**
- `pcos`: **1,393**
- `fertility`: **1,154**

the current balanced upper ceiling is **1,154 per category** if full balancing is required. In practice, the team may choose a smaller balanced training subset for easier iteration, then expand later.

## 1. Candidate Pool (EN, real sources only)

### Purpose

Create the English candidate pool from real patient or dataset-backed sources only, with enough metadata to support later deduplication, balancing, grounding, and leakage prevention.

### Inputs

- `MENST/training2K.csv`
- `MENST/train24K.csv`
- `HealthCareMagic-100k-Chat-Format-en`
- `ChatDoctor-iCliniq`
- Existing seed pipeline artifacts in this folder

### Outputs

- Raw English candidate pool with category assignment
- Provenance fields for each row
- Initial source-level inclusion/exclusion notes

### Key Rules

- Use only **real source-backed English items**
- Preserve provenance as `source_dataset` and `source_row_id`
- Exclude known held-out test material such as `MENST/test.csv`
- Keep benchmark-oriented sourcing focused on realistic patient language
- Do not treat generated style variants as raw-source candidates

### Immediate Action Items

- Confirm the current accepted source list in the project notes
- Keep a single candidate-pool schema across all sources
- Ensure every imported row has `source_dataset` and `source_row_id`
- Keep source licensing notes visible for later publication decisions

## 2. Dedup + Provenance Checks

### Purpose

Remove exact and near-duplicate items, and make sure each surviving row can still be traced back to its original dataset location.

### Inputs

- Raw English candidate pool
- Existing dedup artifacts such as `phase2_dedup_drops.jsonl`

### Outputs

- Deduplicated candidate pool
- Dedup drop log
- Stable provenance mapping for each kept item

### Key Rules

- Dedup before benchmark selection and before train/val splitting
- Keep the drop log for auditability
- Never lose provenance during cleaning
- Use the provenance key for leakage blocking later

### Immediate Action Items

- Reuse the existing dedup evidence already produced
- Retain near-duplicate clusters through `dedup_group_id` where available
- Keep a simple rule sheet explaining why a candidate was kept or removed

## 3. Category Balancing + Quality Filtering (Clear/Borderline)

### Purpose

Filter the deduplicated pool into a reliable benchmark/training candidate set, while keeping category balance visible and preserving a separate borderline bucket.

### Inputs

- Deduplicated English pool
- Existing quality decisions
- `borderline_bucket.csv`
- Current counts from the seed work

### Outputs

- Eligible Clear pool
- Borderline bucket with reasons
- Category-level capacity estimates

### Key Rules

- Main benchmark pool should use **Clear** items first
- Borderline items should remain separately documented, not silently mixed in
- Category balance should be computed against the smallest safe category
- Quality should win over volume

### Current Known Numbers

- Clear raw after dedup:
  - `menstrual`: **4,000**
  - `pcos`: **1,550**
  - `fertility`: **1,219**
- Eligible Clear after quality filter:
  - `menstrual`: **2,724**
  - `pcos`: **1,393**
  - `fertility`: **1,154**
- Practical balanced ceiling from the current eligible pool: **1,154 per category**

### Immediate Action Items

- Keep the current Clear-first policy
- Preserve borderline rows with explicit keep/drop reasons
- Reconfirm whether any borderline promotion is needed only if a later phase becomes yield-constrained

## 4. Benchmark Freeze (EN seeds + style variants)

### Purpose

Lock the initial English benchmark nucleus so later model training does not contaminate evaluation.

### Inputs

- Eligible Clear pool
- Selected English seeds
- Style-variant generation rules
- Existing `seeds_en_v1.csv` / `seeds_en_v1.jsonl`

### Outputs

- Locked English benchmark seed list
- Locked English style variants
- Benchmark exclusion list for training

### Key Rules

- The current benchmark nucleus is:
  - **90 English seeds**
  - **540 rows**
  - **30 per category**
- Benchmark items must remain separate from all train/val construction
- Style variants are benchmark-linked and inherit the same provenance exclusion
- No future training sample may share the same `source_dataset + source_row_id`

### Immediate Action Items

- Treat `seeds_en_v1` as the current English benchmark freeze candidate
- Export or maintain a clean benchmark-provenance exclusion table
- Decide whether the English benchmark freeze is now official or still one review away

## 5. Gold-Label Grounding Draft + Safety/Clarification Rubric Draft

### Purpose

Define what counts as a correct, safe, evidence-grounded model response and when the model should clarify instead of over-claiming.

### Inputs

- English benchmark seeds
- Existing draft grounding fields
- Allowed evidence pages already referenced in project notes

### Outputs

- Draft gold-label grounding sheet
- Draft safety/clarification rubric
- List of benchmark items that still need grounding review

### Key Rules

- Do not invent clinical facts or unsupported claims
- Gold labels should stay grounded only to approved evidence sources
- If a claim cannot be safely grounded, label it as needing clarification or further review
- The benchmark should test understanding, not reward hallucinated medical advice

### Current Known Notes

- Current seed file contains `gold_condition` and `needs_grounding_flag`
- **29 seeds** currently have `needs_grounding_flag=true`
- Existing draft mappings already restrict grounding to approved pages rather than free-form invention

### Immediate Action Items

- Draft a benchmark annotation sheet with fields such as:
  - intent understood
  - category understood
  - symptom meaning preserved
  - unsafe assumption made or not
  - clarification needed or not
  - grounded condition/supporting evidence
- Resolve the currently flagged grounding gaps before the final benchmark freeze

## 6. Train/Val Split from Non-Benchmark Pool

### Purpose

Create the fine-tuning dataset from the remaining English pool without contaminating the benchmark.

### Inputs

- Eligible Clear pool
- Benchmark provenance exclusion list
- Category-balance policy

### Outputs

- Training split
- Validation split
- Split manifest documenting exclusion logic

### Key Rules

- Exclude all benchmark-linked provenance keys first
- Split only from the **non-benchmark pool**
- Prefer a balanced validation set across the three categories
- Keep validation locked once created

### Recommended Practical Direction

- Build validation first as a smaller, carefully reviewed set
- Then build training from the remaining balanced pool using finalized target **800-1,000/category** (default **900/category**)
- If iteration speed matters, start with a smaller balanced training subset and scale later

### Immediate Action Items

- Generate the non-benchmark eligible pool after provenance blocking
- Count the remaining category capacities
- Propose one small validation set and one larger training set for supervisor review

## 7. Arabic/French Translation + Dual Human Validation

### Purpose

Extend the benchmark beyond English while preserving meaning, tone, ambiguity level, and safety constraints.

### Inputs

- Frozen English benchmark items
- Translation guidelines
- Human validation checklist

### Outputs

- Arabic benchmark translations
- French benchmark translations
- Human validation records for both languages

### Key Rules

- Translation should preserve the original patient meaning, not rewrite the medical content
- Style variants should remain aligned with their English source intent
- Each translated item should receive **dual human validation**
- If translators disagree, record an adjudication decision explicitly

### Immediate Action Items

- Prepare a compact translator packet with benchmark IDs, source text, style label, and validation fields
- Define what validators should check:
  - meaning preservation
  - naturalness
  - symptom intent retention
  - ambiguity preservation where intended
  - no unsafe added content

## 8. Multilingual Benchmark Freeze (EN/AR/FR locked)

### Purpose

Lock the multilingual benchmark once English grounding and Arabic/French validation are complete.

### Inputs

- English benchmark freeze
- Arabic validated translations
- French validated translations
- Rubric/gold-label draft

### Outputs

- Final multilingual benchmark release package
- Locked evaluation manifest
- Benchmark card for internal use and paper methods

### Key Rules

- Freeze only after translation validation is complete
- Keep parallel item identity across languages
- Do not modify benchmark content after model evaluation begins unless versioned clearly
- Benchmark versions should be named explicitly

### Immediate Action Items

- Decide the version naming format now
- Produce one benchmark manifest containing language, benchmark ID, style, category, and provenance block key
- Mark the benchmark as locked for all downstream model comparisons

## 9. Model Runs (base, multilingual, fine-tuned, +RAG optional)

### Purpose

Run controlled model comparisons to see whether multilingual pretraining, fine-tuning, and optional retrieval improve understanding on the benchmark.

### Inputs

- Locked multilingual benchmark
- Fine-tuning train/val splits
- Base model shortlist
- Optional RAG document set if used

### Outputs

- Model run table
- Prompting and decoding settings log
- Prediction archive for later error analysis

### Key Rules

- Keep benchmark identical across all model runs
- Log prompt template, temperature, context policy, and checkpoint identity
- Separate model families clearly:
  - base
  - multilingual base
  - fine-tuned
  - optional RAG-assisted
- If RAG is used, report it as a separate condition, not mixed invisibly with non-RAG results

### Immediate Action Items

- Define the initial comparison matrix
- Freeze evaluation prompts before broad comparison
- Save outputs in a format that supports item-level error review

## 10. Final Eval + Error Analysis (misunderstanding taxonomy)

### Purpose

Move beyond overall accuracy and identify the actual types of misunderstanding the models make.

### Inputs

- Model predictions
- Gold/rubric labels
- Benchmark metadata

### Outputs

- Final benchmark results
- Error analysis tables
- Misunderstanding taxonomy for the paper

### Key Rules

- Evaluate both correctness and safety behavior
- Track failure modes by category, language, and style
- Keep the taxonomy simple enough to annotate consistently

### Suggested Misunderstanding Taxonomy

- category confusion
- symptom meaning drift
- over-specific diagnosis leap
- missed need for clarification
- unsupported reassurance
- translation-induced misunderstanding
- style sensitivity failure
- evidence-grounding failure

### Immediate Action Items

- Draft the taxonomy before the first full run so labels stay consistent
- Store item-level prediction notes, not only aggregate scores
- Compare failures across EN vs AR vs FR and across style variants

## 11. Paper Writing in Parallel (Intro/Related/Methods evolving continuously)

### Purpose

Keep the paper draft moving while the benchmark and experiments are still being built, so the team does not postpone writing until after all modeling work is done.

### Inputs

- Workflow decisions
- Data curation notes
- Benchmark versions
- Experimental logs

### Outputs

- Evolving paper draft
- Methods notes ready for Overleaf
- Tables and figure placeholders

### Key Rules

- Writing should run in parallel with dataset and evaluation development
- Methods decisions should be documented immediately, not reconstructed later
- Keep benchmark design, leakage prevention, translation validation, and error taxonomy writeups current

### Immediate Action Items

- Start the Methods skeleton now
- Maintain one running note for benchmark construction decisions
- Draft Intro and Related Work as living sections rather than waiting for final results

## This Week Deliverables

To match the current supervisor-facing direction, this week should ideally produce:

1. A clean planning document for the LUHME / HerHealthGPT-LU benchmark direction
2. Confirmation that the current English seed freeze is the benchmark nucleus:
   - **90 seeds**
   - **540 rows**
   - **30 per category**
3. A simple benchmark-vs-training separation rule based on `source_dataset + source_row_id`
4. A proposed train/val plan from the non-benchmark pool
5. A draft grounding and safety/clarification rubric structure
6. A translation and dual-validation plan for Arabic and French
7. A first-pass model comparison plan for base, multilingual, fine-tuned, and optional RAG conditions

The train/fine-tune target for this planning cycle is finalized at **800-1,000 per category** (**2,400-3,000 total**) with default **900/category**.

## Open Decisions for Next Meeting

1. Should the current `seeds_en_v1` freeze be treated as the official English benchmark, or as a pre-freeze pending one more review?
2. What exact size should the first validation set use from the non-benchmark pool?
3. Confirm whether the team will use the recommended default **900/category** in the first fine-tuning run.
4. Which model families are mandatory for the first comparison table?
5. Will RAG be included in the first paper results or deferred to an extension experiment?
6. Who will perform the dual human validation for Arabic and French, and how will adjudication be recorded?
7. Which rubric fields are required for scoring understanding versus safety versus clarification behavior?

## Recommended Immediate Next Step

The most practical next step is to **officially freeze the current English benchmark nucleus and export its provenance exclusion list**, because that decision unlocks every downstream phase: grounding, train/val construction, translation, and fair model evaluation.

In parallel, the team should draft the first annotation sheet for **gold grounding + safety/clarification labels** so the benchmark becomes not just a collection of questions, but a usable evaluation instrument for LUHME.
