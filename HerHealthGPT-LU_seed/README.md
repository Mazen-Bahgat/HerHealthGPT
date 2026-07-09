# HerHealthGPT-LU English seed dataset (v1)

Week-1 categorized **English** seed set for HerHealthGPT-LU / LUHME 2026.

## Status (2026-07-09)

- **Style variants are being regenerated.** `build_seed.py`'s `generate_styles()` is a
  fixed-template rewriter and was verified to lose clinical content (e.g. `menst-001`
  and `menst-002` — different canonical questions, one about post-miscarriage timing,
  one about fertility — both collapsed to the identical templated `clinical` variant
  "I have irregular periods."). `../scripts/regenerate_style_variants_and_gold.py`
  replaces these with LLM-regenerated, meaning-preserving variants and also fills in
  `gold_risk_level`, `gold_action`, `requires_clarification` (previously blank —
  `draft_grounding()` only ever set `gold_condition`). Requires `OPENAI_API_KEY` (model: `gpt-5.5`);
  not yet run as of this note. **A human must review `regeneration_report.md` before
  the benchmark freezes** — this is an AI first pass, not a substitute for the
  meaning-preservation review the design spec requires.
- **Translation ownership decided:** Arabic is translated in-house by team members
  (fluent speakers); French is outsourced to a professional translation/localization
  agency. See `translation_handoff/` (built by `../scripts/build_translator_handoff.py`)
  and `translation_handoff/fr_agency_brief.md`. This replaces the earlier
  GPT/NLLB-machine-translation-first-pass plan.
- **FT corpus v1 built:** `../scripts/build_ft_corpus.py` produced `ft_corpus_v1.jsonl`
  (2,700 pairs, 900/category) with a **dual** leakage key — `leakage_note.md` now also
  carries `seed_answer_hash` per seed (MENST paraphrase-family key), and
  `leakage_log.csv` is the build-time audit trail of every row excluded and why.
- **Grounding sources:** 5 of the 6 NHS/CDC/NICHD pages in `build_seed.py`'s
  `GROUNDING` dict are fetched under `grounding_sources/` (see
  `../scripts/scrape_grounding_sources.py`). CDC's `common-concerns.html` returns 403
  to every fetch method available in this environment (browser UA, WebFetch tool) —
  needs a manual fetch from someone with normal browser access, then drop the HTML at
  `grounding_sources/cdc_reproductive.html` and re-run the regeneration script.

## Finalized split decision (canonical)

Finalized fine-tuning target: **800-1,000 per category** (**2,400-3,000 total**) for **SFT / multilingual adaptation**.  
Recommended concrete default: **900 per category (2,700 total)**.

| Split | Per-category count | Total count | Purpose |
|------|-------------------:|------------:|---------|
| Benchmark seed v1 (current, frozen nucleus) | 30 | 90 seeds / 540 rows | Multilingual benchmark/evaluation lineage (EN seed nucleus before AR/FR expansion) |
| Validation (non-benchmark only) | TBD (small balanced set) | TBD | Tuning-time model selection and early stopping |
| Fine-tuning train (non-benchmark only) | 800-1,000 | 2,400-3,000 | SFT / multilingual adaptation |
| Fine-tuning train (recommended default) | 900 | 2,700 | Default run plan for first concrete fine-tuning cycle |

Leakage rule: any row sharing the same `source_dataset + source_row_id` with benchmark lineage is excluded from validation and fine-tuning. See `leakage_note.md`.

## How to read deliverables

| File | Purpose |
|------|---------|
| `seeds_en_v1.jsonl` / `seeds_en_v1.csv` | Main deliverable: one row per (seed × style) |
| `stats_summary.md` | Counts, source mix, dedup, grounding |
| `leakage_note.md` | `(source_dataset, source_row_id)` to exclude from FT pulls |
| `decisions_log.md` | Target counts, filters, license notes |
| `FINETUNING_TARGETS.md` | Canonical split targets and finalized fine-tuning decision |
| `borderline_bucket.csv` | Borderline candidates + keep/drop reason |
| `build_seed.py` | Reproducible pipeline |
| `phase1_*` / `phase2_*` / `phase3_*` | Intermediate dumps |

## Schema (`seeds_en_v1.*`)

- `seed_id` — `menst-NNN` / `pcos-NNN` / `fert-NNN`
- `category` — `menstrual` | `pcos` | `fertility`
- `source_dataset`, `source_row_id` — provenance
- `confidence_tier` — Clear (main set)
- `dedup_group_id` — set when near-dup cluster existed
- `canonical_text` — original patient question (light clean only)
- `style`, `style_text` — register variant (includes `canonical`)
- `gold_condition` — draft map to allowed NHS/CDC/NICHD page, or `NEEDS_GROUNDING`
- `needs_grounding_flag` — true when mapping uncertain

## Rebuild

```bash
python build_seed.py
```
