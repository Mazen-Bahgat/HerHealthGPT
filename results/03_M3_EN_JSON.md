# M3(JSON) — EN fine-tune, structured (eval-shaped) targets

QLoRA fine-tune on targets wrapped in the exact 8-key JSON schema the benchmark scores (English styled corpus).

Evaluation: frozen English benchmark, n=540 (90 seeds x 6 styles). Gold-label majority baselines: risk 0.644, clarification 0.833.

## Headline metrics

| metric | value | note |
|---|---|---|
| parse_ok rate | 0.998 | valid JSON produced |
| risk accuracy | 0.623 | majority baseline 0.644 |
| category accuracy | 0.549 | 4-way |
| clarification accuracy | 0.839 | majority baseline 0.833 |
| **under-triage rate** | 0.647 | gold=see-doctor sent to routine (lower is better), n=173 |
| over-triage rate | 0.000 | gold=see-doctor sent to urgent |
| clarification recall | 0.044 | asks when gold=yes (n=90) |
| clarification specificity | 0.998 | stays quiet when gold=no (n=449) |
| misunderstanding rate | 0.451 | |
| self-reported unsafe | 0.000 | model-flagged, not validated |

## Risk confusion (rows = gold, cols = predicted)

| gold \ pred | routine | see-doctor | urgent | other | recall |
|---|---|---|---|---|---|
| routine | 274 | 73 | 1 | 0 | 0.787 |
| see-doctor | 112 | 61 | 0 | 0 | 0.353 |
| urgent | 10 | 7 | 1 | 0 | 0.056 |

## Category recall / precision

| category | recall | precision |
|---|---|---|
| menstrual | 0.917 | 0.434 |
| pcos | 0.683 | 0.854 |
| fertility | 0.413 | 0.796 |
| other | n/a | 0.000 |

## By communication style

| style | parse_ok | risk acc | category acc | clarification acc |
|---|---|---|---|---|
| ambiguous | 1.000 | 0.622 | 0.322 | 0.044 |
| canonical | 1.000 | 0.667 | 0.667 | 1.000 |
| clinical | 0.989 | 0.618 | 0.607 | 1.000 |
| emotionally_concerned | 1.000 | 0.611 | 0.611 | 1.000 |
| indirect_cultural | 1.000 | 0.633 | 0.522 | 0.989 |
| layperson | 1.000 | 0.589 | 0.567 | 1.000 |

## By gold category

| category | n | risk acc | category acc | clarification acc |
|---|---|---|---|---|
| fertility | 180 | 0.547 | 0.413 | 0.832 |
| menarche | 18 | 0.722 | 0.000 | 0.778 |
| menopause | 18 | 0.833 | 0.000 | 0.833 |
| menstrual | 108 | 0.759 | 0.917 | 0.833 |
| menstruation | 12 | 0.750 | 0.000 | 0.833 |
| pcos | 180 | 0.567 | 0.683 | 0.850 |
| pms | 12 | 0.500 | 0.000 | 0.833 |
| symptoms | 12 | 0.917 | 0.000 | 0.917 |

## Cross-style consistency

Fraction of the 90 seed groups given the same label across all six styles.

- risk: 0.433 (n=90 groups)
- category: 0.344 (n=90 groups)

## McNemar paired test vs M2 (zero-shot)

| field | b (M2 correct / this wrong) | c (this correct / M2 wrong) | chi^2 | p-value | significant (p<0.05) |
|---|---|---|---|---|---|
| parse_ok | 1 | 0 | 0.00 | 1.000 | no |
| category_correct | 54 | 59 | 0.14 | 0.707 | no |
| risk_correct | 70 | 46 | 4.56 | 0.033 | yes |
| clarification_correct | 74 | 61 | 1.07 | 0.302 | no |

## 95% bootstrap confidence intervals

- parse_ok_rate: [0.994, 1.000]
- category_accuracy: [0.506, 0.590]
