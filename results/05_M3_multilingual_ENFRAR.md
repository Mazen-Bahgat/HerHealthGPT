# M3(ML) — Joint EN+FR+AR fine-tune

QLoRA fine-tune on the merged trilingual corpus (EN+FR+AR, structured targets + clarify oversampling, 10,538 train rows). Evaluated on the ENGLISH benchmark only (FR/AR benchmarks not yet built).

Evaluation: frozen English benchmark, n=540 (90 seeds x 6 styles). Gold-label majority baselines: risk 0.644, clarification 0.833.

## Headline metrics

| metric | value | note |
|---|---|---|
| parse_ok rate | 0.998 | valid JSON produced |
| risk accuracy | 0.655 | majority baseline 0.644 |
| category accuracy | 0.510 | 4-way |
| clarification accuracy | 0.833 | majority baseline 0.833 |
| **under-triage rate** | 0.711 | gold=see-doctor sent to routine (lower is better), n=173 |
| over-triage rate | 0.000 | gold=see-doctor sent to urgent |
| clarification recall | 0.000 | asks when gold=yes (n=90) |
| clarification specificity | 1.000 | stays quiet when gold=no (n=449) |
| misunderstanding rate | 0.490 | |
| self-reported unsafe | 0.000 | model-flagged, not validated |

## Risk confusion (rows = gold, cols = predicted)

| gold \ pred | routine | see-doctor | urgent | other | recall |
|---|---|---|---|---|---|
| routine | 303 | 44 | 1 | 0 | 0.871 |
| see-doctor | 123 | 50 | 0 | 0 | 0.289 |
| urgent | 11 | 7 | 0 | 0 | 0.000 |

## Category recall / precision

| category | recall | precision |
|---|---|---|
| menstrual | 0.935 | 0.388 |
| pcos | 0.694 | 0.856 |
| fertility | 0.274 | 0.803 |
| other | n/a | n/a |

## By communication style

| style | parse_ok | risk acc | category acc | clarification acc |
|---|---|---|---|---|
| ambiguous | 1.000 | 0.667 | 0.389 | 0.000 |
| canonical | 1.000 | 0.700 | 0.611 | 1.000 |
| clinical | 1.000 | 0.667 | 0.544 | 1.000 |
| emotionally_concerned | 0.989 | 0.584 | 0.539 | 1.000 |
| indirect_cultural | 1.000 | 0.644 | 0.411 | 1.000 |
| layperson | 1.000 | 0.667 | 0.567 | 1.000 |

## By gold category

| category | n | risk acc | category acc | clarification acc |
|---|---|---|---|---|
| fertility | 180 | 0.598 | 0.274 | 0.832 |
| menarche | 18 | 0.667 | 0.000 | 0.833 |
| menopause | 18 | 0.833 | 0.000 | 0.833 |
| menstrual | 108 | 0.806 | 0.935 | 0.833 |
| menstruation | 12 | 0.583 | 0.000 | 0.833 |
| pcos | 180 | 0.600 | 0.694 | 0.833 |
| pms | 12 | 0.417 | 0.000 | 0.833 |
| symptoms | 12 | 1.000 | 0.000 | 0.833 |

## Cross-style consistency

Fraction of the 90 seed groups given the same label across all six styles.

- risk: 0.500 (n=90 groups)
- category: 0.411 (n=90 groups)

## McNemar paired test vs M2 (zero-shot)

| field | b (M2 correct / this wrong) | c (this correct / M2 wrong) | chi^2 | p-value | significant (p<0.05) |
|---|---|---|---|---|---|
| parse_ok | 1 | 0 | 0.00 | 1.000 | no |
| category_correct | 66 | 50 | 1.94 | 0.164 | no |
| risk_correct | 54 | 47 | 0.36 | 0.550 | no |
| clarification_correct | 78 | 62 | 1.61 | 0.205 | no |

## 95% bootstrap confidence intervals

- parse_ok_rate: [0.994, 1.000]
- category_accuracy: [0.468, 0.553]
