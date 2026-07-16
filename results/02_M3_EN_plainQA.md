# M3(QA) — EN fine-tune, plain question->answer targets

QLoRA fine-tune of Qwen3.5-9B on plain Question->Answer text pairs (English styled corpus). Ablation showing the effect of naive QA targets.

Evaluation: frozen English benchmark, n=540 (90 seeds x 6 styles). Gold-label majority baselines: risk 0.644, clarification 0.833.

## Headline metrics

| metric | value | note |
|---|---|---|
| parse_ok rate | 0.865 | valid JSON produced |
| risk accuracy | 0.625 | majority baseline 0.644 |
| category accuracy | 0.499 | 4-way |
| clarification accuracy | 0.848 | majority baseline 0.833 |
| **under-triage rate** | 0.800 | gold=see-doctor sent to routine (lower is better), n=160 |
| over-triage rate | 0.000 | gold=see-doctor sent to urgent |
| clarification recall | 0.174 | asks when gold=yes (n=86) |
| clarification specificity | 1.000 | stays quiet when gold=no (n=381) |
| misunderstanding rate | 0.501 | |
| self-reported unsafe | 0.000 | model-flagged, not validated |

## Risk confusion (rows = gold, cols = predicted)

| gold \ pred | routine | see-doctor | urgent | other | recall |
|---|---|---|---|---|---|
| routine | 260 | 32 | 0 | 0 | 0.890 |
| see-doctor | 128 | 32 | 0 | 0 | 0.200 |
| urgent | 10 | 5 | 0 | 0 | 0.000 |

## Category recall / precision

| category | recall | precision |
|---|---|---|
| menstrual | 0.854 | 0.443 |
| pcos | 0.574 | 0.931 |
| fertility | 0.424 | 0.761 |
| other | n/a | 0.000 |

## By communication style

| style | parse_ok | risk acc | category acc | clarification acc |
|---|---|---|---|---|
| ambiguous | 0.956 | 0.593 | 0.314 | 0.174 |
| canonical | 0.778 | 0.671 | 0.586 | 1.000 |
| clinical | 0.844 | 0.684 | 0.500 | 1.000 |
| emotionally_concerned | 0.922 | 0.602 | 0.602 | 1.000 |
| indirect_cultural | 0.878 | 0.595 | 0.443 | 1.000 |
| layperson | 0.811 | 0.616 | 0.575 | 1.000 |

## By gold category

| category | n | risk acc | category acc | clarification acc |
|---|---|---|---|---|
| fertility | 180 | 0.612 | 0.424 | 0.848 |
| menarche | 18 | 0.667 | 0.000 | 0.833 |
| menopause | 18 | 0.692 | 0.000 | 0.923 |
| menstrual | 108 | 0.740 | 0.854 | 0.833 |
| menstruation | 12 | 0.583 | 0.000 | 0.833 |
| pcos | 180 | 0.539 | 0.574 | 0.858 |
| pms | 12 | 0.500 | 0.000 | 0.833 |
| symptoms | 12 | 1.000 | 0.000 | 0.800 |

## Cross-style consistency

Fraction of the 90 seed groups given the same label across all six styles.

- risk: 0.456 (n=90 groups)
- category: 0.189 (n=90 groups)

## McNemar paired test vs M2 (zero-shot)

| field | b (M2 correct / this wrong) | c (this correct / M2 wrong) | chi^2 | p-value | significant (p<0.05) |
|---|---|---|---|---|---|
| parse_ok | 73 | 0 | 71.01 | <0.001 | yes |
| category_correct | 77 | 19 | 33.84 | <0.001 | yes |
| risk_correct | 87 | 19 | 42.35 | <0.001 | yes |
| clarification_correct | 127 | 58 | 24.99 | <0.001 | yes |

## 95% bootstrap confidence intervals

- parse_ok_rate: [0.835, 0.893]
- category_accuracy: [0.454, 0.544]
