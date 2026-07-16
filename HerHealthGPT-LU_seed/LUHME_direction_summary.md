# LUHME / HerHealthGPT-LU Direction Summary

## Direction in One Page

The LUHME / HerHealthGPT-LU direction should be framed as a **multilingual language-understanding benchmark** for women's health questions, not only as a chatbot project. The purpose is to test whether models can understand realistic patient wording, preserve meaning across style variation and translation, stay evidence-grounded, and avoid unsafe over-interpretation.

The current English seed work already gives a usable benchmark nucleus:

- **90 English seeds**
- **540 rows total**
- **30 seeds per category**
- Categories: `menstrual`, `pcos`, `fertility`

Finalized decision for fine-tuning split: **800-1,000 per category** (**2,400-3,000 total**) for **SFT / multilingual adaptation**, with recommended default **900 per category (2,700 total)**.

## Canonical Split Table (Benchmark vs Val vs Fine-Tuning)

| Split | Per-category count | Total count | Purpose |
|------|-------------------:|------------:|---------|
| Benchmark seed v1 (current nucleus) | 30 | 90 seeds / 540 rows | Multilingual benchmark/eval lineage |
| Validation (non-benchmark only) | TBD (small balanced set) | TBD | Tuning-time validation |
| Fine-tuning train (non-benchmark only) | 800-1,000 | 2,400-3,000 | SFT / multilingual adaptation |
| Fine-tuning train (recommended default) | 900 | 2,700 | Default first run |

The current eligible Clear pool after filtering is:

- `menstrual`: **2,724**
- `pcos`: **1,393**
- `fertility`: **1,154**

This means the current fully balanced ceiling is **1,154 per category** from the eligible non-benchmark pool.

## Recommended Workflow

1. Build the English candidate pool from real source-backed data only.
2. Deduplicate and preserve provenance for every row.
3. Filter to Clear/Borderline and keep category balance visible.
4. Freeze the English benchmark seeds plus style variants.
5. Draft gold grounding and a safety/clarification rubric.
6. Create train/validation splits only from the **non-benchmark** pool.
7. Translate the benchmark to Arabic and French.
8. Perform dual human validation for both translated versions.
9. Freeze the multilingual benchmark across EN/AR/FR.
10. Run base, multilingual, fine-tuned, and optional RAG comparisons.
11. Report final evaluation with error analysis and a misunderstanding taxonomy.

## Benchmark vs Fine-Tuning Rule

The project should keep two tracks separate from the start:

- **Benchmark track**: frozen evaluation items only
- **Fine-tuning track**: train/val data only

Leakage prevention should use one simple rule:

> If an item is in the benchmark lineage, any row with the same `source_dataset + source_row_id` must be excluded from training and validation.

Operational reference for blocked keys: `leakage_note.md`.

This keeps the evaluation fair even if benchmark style variants were generated from the same original patient question.

## Practical Recommendation

For the current direction, the most practical plan is:

- Treat the current **90-seed English set** as the benchmark nucleus
- Keep benchmark size and purpose explicit: **90 seeds / 540 rows** is the current v1 benchmark nucleus, not the fine-tuning target
- Freeze its provenance keys
- Build validation first from the remaining eligible pool
- Use the rest as balanced fine-tuning data targeting **800-1,000/category** (default **900/category**)
- Then translate the benchmark into Arabic and French with dual human validation

## This Week Priority

The immediate next step is to **officially freeze the current English benchmark nucleus and export the provenance exclusion list**, then draft the first gold-grounding and safety/clarification annotation sheet. That decision unlocks train/val construction, multilingual benchmark building, and fair model comparison.
