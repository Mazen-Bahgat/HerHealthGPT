# HerHealthEval — all models compared (English benchmark, n=540)

Every model below is evaluated on the identical 540-item frozen **English** benchmark with identical gold labels. Majority baselines: risk 0.644, clarification 0.833. The multilingual model (M3-ML) is trained on EN+FR+AR; its **French and Arabic** evaluation (also 540 items each) is in [`06_cross_lingual_FR_AR.md`](06_cross_lingual_FR_AR.md) and summarized at the bottom of this file.

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
- **M3-ML:** highest cross-style category consistency (0.411); English triage back near M2 (dilution); clarification 0.000. **On French and Arabic it collapses** — see below.

## Cross-lingual summary (M2 vs M3-ML, matched EN/FR/AR, 540 each)

Full detail in [`06_cross_lingual_FR_AR.md`](06_cross_lingual_FR_AR.md).

| metric | M2 EN | M2 FR | M2 AR | M3-ML EN | M3-ML FR | M3-ML AR |
|---|---|---|---|---|---|---|
| under-triage ↓ | 0.718 | 0.632 | 0.638 | 0.711 | **0.994** | **0.994** |
| clarification recall | 0.856 | 0.811 | 0.889 | 0.000 | 0.000 | 0.011 |
| risk accuracy | 0.667 | 0.685 | 0.674 | 0.655 | 0.646 | 0.646 |

- The **zero-shot baseline transfers** to French and Arabic (comparable, slightly safer); 3-way EN/FR/AR risk agreement 0.846.
- The **multilingual fine-tune collapses** on both non-English languages: routine for 538/540 (FR) and 537/540 (AR), under-triage 0.994; 3-way risk agreement drops to 0.804. 95% bootstrap CIs for M3-ML under-triage are [0.981, 1.000] in FR/AR, disjoint from the baseline.
- **Root cause:** FR/AR training risk labels were derived by an English-only keyword heuristic on translated text, skewing them toward `routine`; joint fine-tuning amplified a "non-English → routine" shortcut. Fix (future work): derive risk from the English source, attach to translated rows by `row_id`.
