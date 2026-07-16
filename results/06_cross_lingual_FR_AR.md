# Cross-lingual evaluation — English / French / Arabic

Matched 540-item evaluation set in three languages (same cases, translated
questions, **identical language-invariant gold labels**). Two models evaluated:
the zero-shot baseline (M2) and the joint multilingual fine-tune (M3-ML). All
numbers are from the committed eval artifacts under
`Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/`.

## Headline table

| metric | M2 EN | M2 FR | M2 AR | M3-ML EN | M3-ML FR | M3-ML AR |
|---|---|---|---|---|---|---|
| parse_ok | 1.000 | 1.000 | 1.000 | 0.998 | 0.998 | 0.998 |
| risk accuracy | 0.667 | 0.685 | 0.674 | 0.655 | 0.646 | 0.646 |
| under-triage ↓ | 0.718 | 0.632 | 0.638 | 0.711 | 0.994 | 0.994 |
| clarification recall | 0.856 | 0.811 | 0.889 | 0.000 | 0.000 | 0.011 |
| category accuracy | 0.539 | 0.561 | 0.533 | 0.510 | 0.512 | 0.531 |

## Cross-language agreement (same case, same predicted label across languages)

| pair | M2 risk | M2 category | M3-ML risk | M3-ML category |
|---|---|---|---|---|
| EN–FR | 0.904 | 0.898 | 0.806 | 0.896 |
| EN–AR | 0.900 | 0.902 | 0.811 | 0.856 |
| FR–AR | 0.889 | 0.917 | 0.991* | 0.870 |
| 3-way EN=FR=AR | 0.846 | 0.859 | 0.804 | 0.813 |

\* M3-ML's near-perfect FR–AR risk agreement is an **artifact** of predicting
`routine` for ~all items in both languages, not genuine robustness.

## Risk prediction distribution (the collapse, made concrete)

| model | routine | see-doctor | urgent |
|---|---|---|---|
| M2 EN | 437 | 101 | 1 |
| M2 FR | 423 | 109 | 8 |
| M2 AR | 423 | 107 | 10 |
| M3-ML EN | 437 | 101 | 1 |
| **M3-ML FR** | **538** | **0** | **0** |
| **M3-ML AR** | **537** | **2** | **0** |

## Reading

- **The zero-shot baseline transfers cleanly** to French and Arabic — comparable
  (slightly better) triage and clarification, and high cross-language agreement
  (3-way risk 0.846).
- **The multilingual fine-tune collapses on both non-English languages**: it
  down-classifies essentially every care-warranting FR/AR case to `routine`
  (under-triage 0.994), stops asking for clarification, and is *less*
  cross-lingually consistent than the untuned model (3-way risk 0.846 → 0.804).
- **Root cause:** the FR/AR *training* risk labels were produced by an
  English-keyed keyword heuristic on translated text, skewing them toward
  `routine`; joint fine-tuning amplified this into a "non-English → routine"
  shortcut. The base model is competent in FR/AR; the failure is in the
  language-asymmetric adaptation labels. Fix (future work): derive risk from the
  English source answer and attach to translated rows by `row_id`.
