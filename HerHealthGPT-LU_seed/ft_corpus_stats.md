# FT corpus v1 (English) — build stats

**Total after §2C steps 1-7:** 2700  
**Target:** 900/category (2700 total)

## Yield per stage

| Stage | Count |
|---|---:|
| Raw normalized (Step 1) | 136109 |
| Domain-categorized (Step 2) | 17939 |
| Quality-filtered (Step 4) | 15510 |
| Leakage-excluded (Step 6) | 317 |
| Final FT corpus (Step 7, balanced) | 2700 |

## Final category balance

| Category | Count |
|---|---:|
| menstrual | 900 |
| pcos | 900 |
| fertility | 900 |

## Quality-filter drop reasons

| Reason | Count |
|---|---:|
| exact_qa_dup | 2279 |
| answer_length | 150 |

## Leakage

- Dual key applied: `(source_dataset, source_row_id)` + `seed_answer_hash` (MENST-only). 317 rows excluded, see `leakage_log.csv`.
- Zero rows in `ft_corpus_v1.jsonl` share a `seed_answer_hash` or `(source_dataset, source_row_id)` with any of the 90 frozen benchmark seeds.
