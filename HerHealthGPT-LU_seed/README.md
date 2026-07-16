# HerHealthGPT-LU English seed dataset (v1)

Week-1 categorized **English** seed set for HerHealthGPT-LU / LUHME 2026.

## Status (2026-07-09)

- **Style variants regenerated (done 2026-07-09).** `build_seed.py`'s `generate_styles()`
  was a fixed-template rewriter, verified to lose clinical content (`menst-001` and
  `menst-002` — different canonical questions, one about post-miscarriage timing, one
  about fertility — both collapsed to the identical templated `clinical` variant
  "I have irregular periods."). All 450 style rows (90 seeds x 5 styles) were rewritten
  directly by Claude, one at a time, under the meaning-preservation rubric — no external
  LLM API call, since the OpenAI account behind `regenerate_style_variants_and_gold.py`
  hit a billing/quota block. Verified before merging: **100% distinct-string ratio**
  across all 450 variants (up from ~51 distinct strings under the old template), zero
  collisions with canonical text. Merged via `../scripts/merge_manual_style_variants.py`.
  `regenerate_style_variants_and_gold.py` is kept as an alternative/second-pass path if
  `OPENAI_API_KEY` becomes available later; not required for the current data.
  **A human on the team must still review `regeneration_report.md` before the benchmark
  freezes** — this is an AI first pass (a different AI than the one that wrote the
  original broken templates), not a substitute for the team's own meaning-preservation
  review. `gold_risk_level`/`gold_action`/`requires_clarification` were already filled
  in separately and deterministically by `complete_gold_labels.py` (see below) and were
  not touched by this step.
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
| `style_variants_manual.json` | Claude-authored style variants (source for the merge below) |
| `style_variants.json` | Merged/current style variants (post-regeneration) |
| `regeneration_report.md` | Old vs new style_text per seed, for human review |
| `gold_label_completion_report.md` | Deterministic gold-label completion detail |
| `ft_corpus_v1.jsonl` / `ft_corpus_stats.md` | Silver fine-tuning corpus + build stats |
| `leakage_log.csv` | FT-build-time audit trail of excluded rows |
| `translation_handoff/` | AR (in-house) and FR (agency) translator handoff files |

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
