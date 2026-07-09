# Fine-Tuning Targets (Canonical)

## Finalized decision

Finalized fine-tuning target: **800-1,000 per category** (**2,400-3,000 total**) for **SFT / multilingual adaptation**.  
Recommended concrete default: **900 per category (2,700 total)**.

## Canonical split table

| Split | Per-category count | Total count | Purpose |
|------|-------------------:|------------:|---------|
| Benchmark seed v1 (current nucleus, frozen lineage) | 30 | 90 seeds / 540 rows | Benchmark/evaluation lineage only |
| Validation (non-benchmark only) | TBD (small balanced set) | TBD | Tuning-time validation |
| Fine-tuning train (non-benchmark only) | 800-1,000 | 2,400-3,000 | SFT / multilingual adaptation |
| Fine-tuning train (recommended default) | 900 | 2,700 | Default first fine-tuning run |

## Leakage exclusion reminder

If an item appears in benchmark lineage, any row with the same `source_dataset + source_row_id` must be excluded from validation and fine-tuning pulls.  
Use `leakage_note.md` as the operational block-list reference.
