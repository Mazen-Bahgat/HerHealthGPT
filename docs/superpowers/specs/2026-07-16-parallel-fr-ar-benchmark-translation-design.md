# Parallel French and Arabic Benchmark Translation Design

**Date:** 2026-07-16

**Status:** Approved for implementation planning

**Scope:** The 540-question evaluation benchmark handoffs in
`Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/Test/translation_handoff_benchmark/`

## Objective

Produce complete French and Modern Standard Arabic (MSA) versions of the
540-item English evaluation benchmark, preserving the meaning and communication
register of every question while leaving all English gold data unchanged. Run
the two language workflows concurrently, validate every item, build the two
translated benchmark JSONL files, and retain a reproducible audit trail.

These translations remain provisional evaluation artifacts until native
French and native Arabic reviewers approve meaning preservation, naturalness,
medical terminology, and register. AI-assisted QA does not satisfy that human
validation gate.

## Inputs and Outputs

Inputs:

- Canonical English benchmark:
  `Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/gold_seeds_styled.jsonl`
- French handoff:
  `Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/Test/translation_handoff_benchmark/fr.csv`
- Arabic handoff:
  `Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/Test/translation_handoff_benchmark/ar.csv`

Each handoff has exactly 540 unique `item_key` values and 90 questions for each
of the six styles: `canonical`, `clinical`, `layperson`,
`indirect_cultural`, `ambiguous`, and `emotionally_concerned`.

Final outputs:

- The two filled handoff CSV files, with only `Question_translated` changed.
- `Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/gold_seeds_styled_fr.jsonl`
- `Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/gold_seeds_styled_ar.jsonl`
- Per-language machine-readable generation provenance and QA reports beside
  the handoff CSV files.
- Updated handoff documentation recording the exact method and the pending
  native-speaker validation requirement.

## Translation Policy

Both languages use OpenAI's Responses API with `gpt-5.6-sol`, low reasoning
effort, Structured Outputs, and `store: false`. Requests contain only the
handoff context needed for translation: `item_key`, `style`, `Topic`, and the
English `Question`; gold labels and gold answers are never sent or translated.

French uses natural, internationally understandable French and established
medical terminology. Arabic uses MSA only. It must not switch to Egyptian or
another dialect for informal styles; instead, style is expressed through
wording, sentence structure, directness, and emotional tone within idiomatic
MSA.

For both languages:

- Preserve the source meaning, uncertainty, urgency, negation, person, tense,
  and quantitative content.
- Preserve the six communication registers rather than flattening them.
- Keep ambiguous questions genuinely vague. Do not diagnose, clarify, repair,
  expand, or fact-check the source.
- Keep clinical questions clinical and lay questions accessible.
- Preserve brands, URLs, measurements, numeral values, and medical acronyms
  when changing them would alter identity or meaning.
- Return only the translation associated with each `item_key`; do not add
  explanations, alternatives, labels, or translator notes inside the cell.

## Parallel Generation Architecture

A single benchmark translation driver owns common CSV parsing, reference
validation, batching, cache persistence, and provenance. It launches one
independent language pipeline for French and one for Arabic concurrently.

Each language pipeline:

1. Validates the blank handoff against the canonical benchmark before sending
   any request.
2. Creates deterministic jobs keyed by `item_key` and source-text hash.
3. Sends batches of at most 45 questions with language-specific prompts, with
   up to two in-flight batches per language and six bounded retry attempts.
4. Writes every successful batch atomically to a resumable cache under
   `C:/tmp`; completed jobs are never requested twice.
5. Rejects missing, duplicate, unknown, or reordered response IDs and rejects
   blank translations.
6. Writes a staged filled CSV only after all 540 jobs are present.

French and Arabic caches, staging paths, prompts, and outputs remain separate,
so a failure in one language cannot corrupt or block completed work in the
other. The shared driver exits unsuccessfully if either final language fails
validation.

## Validation and Review

Validation has four gates.

### 1. Reference integrity

For every CSV, verify exact UTF-8/UTF-8-BOM readability, header, 540-row count,
unique and ordered `item_key` values, and byte-equivalent values for
`item_key`, `seed_id`, `style`, `Topic`, and `Question`. Only
`Question_translated` may differ from the blank reference.

### 2. Deterministic translation checks

Reject blank cells, malformed CSV, replacement characters, control characters,
unexpected line-count changes, and lost URLs. Produce review flags, rather
than automatic failures, for numeral surface-form changes, punctuation changes,
very short translations, Latin-script leakage in Arabic, weak French-language
signal, and unexpected Arabic dialect markers.

### 3. Full semantic QA

Run a separate structured QA request over all 540 source/translation pairs in
each language. The reviewer classifies meaning preservation, negation,
urgency, uncertainty, numerical fidelity, medical terminology, and register.
It must explicitly check that ambiguous questions were not made more specific.

Any proposed correction is applied only through a fail-closed rule containing
the language, `item_key`, exact old text, exact new text, reason, and expected
match count. Corrections are followed by the complete deterministic and
semantic validation passes again. Critical safety or meaning issues block the
language artifact until corrected.

### 4. Native-speaker gate

The README and provenance must state `pending` until native French and native
Arabic reviewers sign off. No output may be described as human-validated,
gold, or benchmark-ready solely because automated and AI-assisted checks pass.

## Benchmark Build and Round-Trip Verification

After both handoffs pass automated and AI-assisted QA, run
`scripts/build_translated_benchmark.py` separately for `fr` and `ar`. Verify:

- Exactly 540 JSONL records per language.
- `style_text` is the only canonical benchmark value replaced.
- Every other key and value, including all gold labels, actions, answers,
  risks, and clarification targets, is identical to the English benchmark.
- The record order and `(seed_id, style)` sequence are identical across
  English, French, and Arabic.
- `scripts/run_local_inference.py --language fr` and `--language ar` can read
  the built files through their pure, non-model-loading helpers.

## Testing and Failure Handling

Add focused tests before implementation for reference drift, partial caches,
wrong response IDs, Arabic MSA policy, translation-only mutations, semantic QA
result parsing, fail-closed correction counts, and gold-preserving JSONL builds.
Run the existing translation and local-inference tests together with the new
suite.

All writes use temporary files followed by atomic replacement. A failed API
batch, malformed response, failed review, or failed build leaves the original
handoff untouched and preserves the resumable cache. No automatic fallback to
another model, machine-translation provider, Arabic dialect, or older
translation asset is allowed.

## Success Criteria

- French and MSA Arabic handoffs each contain 540 nonblank translations.
- Only `Question_translated` changes in either handoff.
- All six styles and all source topics retain their original coverage.
- Deterministic validation has zero blocking errors for both languages.
- Full AI-assisted semantic QA covers all 1,080 translated questions, with no
  unresolved critical meaning, safety, or register errors.
- Both translated benchmark JSONL files contain 540 records and preserve every
  English gold field exactly.
- Generation and QA provenance records exact prompts, model configuration,
  usage, source/output hashes, corrections, and native-review status.
- Native-speaker review remains visibly pending until performed by the team.

## Non-Goals

- Translating gold labels, gold answers, topics, or metadata.
- Correcting factual or linguistic defects in the English benchmark.
- Producing Egyptian Arabic or a mixed MSA/dialect benchmark.
- Running GPU model inference or reporting multilingual evaluation results.
- Claiming clinical, native-speaker, or benchmark-quality validation.
