# HerHealthGPT-LU English seed dataset (v1)

Week-1 categorized **English** seed set for HerHealthGPT-LU / LUHME 2026.

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
