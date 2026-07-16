# HerHealthEval — all models compared (English benchmark, n=540)

Every model is evaluated on the identical 540-item frozen English benchmark with identical gold labels. Majority baselines: risk 0.644, clarification 0.833. The multilingual model (M3-ML) is trained on EN+FR+AR but evaluated on English only.

| metric | M2 | M3-QA | M3-JSON | M3-J+O | M3-ML |
|---|---|---|---|---|---|
| parse_ok | 1.000 | 0.865 | 0.998 | 0.996 | 0.998 |
| risk accuracy | 0.667 | 0.625 | 0.623 | 0.665 | 0.655 |
| category accuracy | 0.539 | 0.499 | 0.549 | 0.550 | 0.510 |
| under-triage (lower better) | 0.718 | 0.800 | 0.647 | 0.605 | 0.711 |
| over-triage | 0.017 | 0.000 | 0.000 | 0.000 | 0.000 |
| clarification recall | 0.856 | 0.174 | 0.044 | 0.000 | 0.000 |
| clarification specificity | 0.862 | 1.000 | 0.998 | 1.000 | 1.000 |
| misunderstanding rate | 0.461 | 0.501 | 0.451 | 0.450 | 0.490 |
| consistency (risk) | 0.644 | 0.456 | 0.433 | 0.444 | 0.500 |
| consistency (category) | 0.156 | 0.189 | 0.344 | 0.378 | 0.411 |

## McNemar vs M2 (p-values; --- = baseline)

| field | M2 | M3-QA | M3-JSON | M3-J+O | M3-ML |
|---|---|---|---|---|---|
| parse_ok | --- | <0.001 | 1.000 | 0.480 | 1.000 |
| risk_correct | --- | <0.001 | 0.033 | 0.920 | 0.550 |
| category_correct | --- | <0.001 | 0.707 | 0.704 | 0.164 |
| clarification_correct | --- | <0.001 | 0.302 | 0.205 | 0.205 |

## One-line read per model

- **M2 (zero-shot):** strongest clarification (0.856) but worst under-triage (0.718); baseline.
- **M3-QA (plain QA):** breaks JSON format (parse 0.865) and clarification (0.174); worst triage.
- **M3-JSON:** restores format (0.998), cuts under-triage, but clarification collapses (0.044).
- **M3-J+O:** best English triage — risk parity with M2 (0.665), lowest under-triage (0.605); clarification still 0.000.
- **M3-ML:** highest cross-style category consistency (0.411); English triage back near M2 (dilution); clarification 0.000. Real multilingual value needs FR/AR benchmarks.
