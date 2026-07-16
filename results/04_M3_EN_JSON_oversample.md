# M3(J+O) — EN fine-tune, structured targets + clarify oversampling

Structured JSON targets with 4x oversampling of the ambiguous (clarification) training rows, raising them from 10.5% to 32% of the mixture. Best English triage model.

Evaluation: frozen English benchmark, n=540 (90 seeds x 6 styles). Gold-label majority baselines: risk 0.644, clarification 0.833.

## Headline metrics

| metric | value | note |
|---|---|---|
| parse_ok rate | 0.996 | valid JSON produced |
| risk accuracy | 0.665 | majority baseline 0.644 |
| category accuracy | 0.550 | 4-way |
| clarification accuracy | 0.835 | majority baseline 0.833 |
| **under-triage rate** | 0.605 | gold=see-doctor sent to routine (lower is better), n=172 |
| over-triage rate | 0.000 | gold=see-doctor sent to urgent |
| clarification recall | 0.000 | asks when gold=yes (n=89) |
| clarification specificity | 1.000 | stays quiet when gold=no (n=449) |
| misunderstanding rate | 0.450 | |
| self-reported unsafe | 0.000 | model-flagged, not validated |

## Risk confusion (rows = gold, cols = predicted)

| gold \ pred | routine | see-doctor | urgent | other | recall |
|---|---|---|---|---|---|
| routine | 289 | 59 | 0 | 0 | 0.830 |
| see-doctor | 104 | 68 | 0 | 0 | 0.395 |
| urgent | 10 | 7 | 1 | 0 | 0.056 |

## Category recall / precision

| category | recall | precision |
|---|---|---|
| menstrual | 0.907 | 0.438 |
| pcos | 0.750 | 0.877 |
| fertility | 0.354 | 0.716 |
| other | n/a | n/a |

## By communication style

| style | parse_ok | risk acc | category acc | clarification acc |
|---|---|---|---|---|
| ambiguous | 0.989 | 0.629 | 0.404 | 0.000 |
| canonical | 1.000 | 0.767 | 0.633 | 1.000 |
| clinical | 1.000 | 0.667 | 0.533 | 1.000 |
| emotionally_concerned | 0.989 | 0.663 | 0.629 | 1.000 |
| indirect_cultural | 1.000 | 0.567 | 0.522 | 1.000 |
| layperson | 1.000 | 0.700 | 0.578 | 1.000 |

## By gold category

| category | n | risk acc | category acc | clarification acc |
|---|---|---|---|---|
| fertility | 180 | 0.607 | 0.354 | 0.837 |
| menarche | 18 | 0.722 | 0.000 | 0.833 |
| menopause | 18 | 0.722 | 0.000 | 0.833 |
| menstrual | 108 | 0.769 | 0.907 | 0.833 |
| menstruation | 12 | 0.833 | 0.000 | 0.833 |
| pcos | 180 | 0.622 | 0.750 | 0.833 |
| pms | 12 | 0.583 | 0.000 | 0.833 |
| symptoms | 12 | 1.000 | 0.000 | 0.833 |

## Cross-style consistency

Fraction of the 90 seed groups given the same label across all six styles.

- risk: 0.444 (n=90 groups)
- category: 0.378 (n=90 groups)

## McNemar paired test vs M2 (zero-shot)

| field | b (M2 correct / this wrong) | c (this correct / M2 wrong) | chi^2 | p-value | significant (p<0.05) |
|---|---|---|---|---|---|
| parse_ok | 2 | 0 | 0.50 | 0.480 | no |
| category_correct | 53 | 58 | 0.14 | 0.704 | no |
| risk_correct | 50 | 48 | 0.01 | 0.920 | no |
| clarification_correct | 78 | 62 | 1.61 | 0.205 | no |

## 95% bootstrap confidence intervals

- parse_ok_rate: [0.991, 1.000]
- category_accuracy: [0.509, 0.591]
