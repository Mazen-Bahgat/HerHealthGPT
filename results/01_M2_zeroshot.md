# M2 — Zero-shot baseline

Instruction-tuned Qwen3.5-9B, prompted with the structured triage schema. No fine-tuning. This is the zero-shot ('smoke test') reference.

Evaluation: frozen English benchmark, n=540 (90 seeds x 6 styles). Gold-label majority baselines: risk 0.644, clarification 0.833.

## Headline metrics

| metric | value | note |
|---|---|---|
| parse_ok rate | 1.000 | valid JSON produced |
| risk accuracy | 0.667 | majority baseline 0.644 |
| category accuracy | 0.539 | 4-way |
| clarification accuracy | 0.861 | majority baseline 0.833 |
| **under-triage rate** | 0.718 | gold=see-doctor sent to routine (lower is better), n=174 |
| over-triage rate | 0.017 | gold=see-doctor sent to urgent |
| clarification recall | 0.856 | asks when gold=yes (n=90) |
| clarification specificity | 0.862 | stays quiet when gold=no (n=450) |
| misunderstanding rate | 0.461 | |
| self-reported unsafe | 0.006 | model-flagged, not validated |

## Risk confusion (rows = gold, cols = predicted)

| gold \ pred | routine | see-doctor | urgent | other | recall |
|---|---|---|---|---|---|
| routine | 312 | 32 | 4 | 0 | 0.897 |
| see-doctor | 125 | 46 | 3 | 0 | 0.264 |
| urgent | 10 | 6 | 2 | 0 | 0.111 |

## Category recall / precision

| category | recall | precision |
|---|---|---|
| menstrual | 0.759 | 0.539 |
| pcos | 0.694 | 0.947 |
| fertility | 0.467 | 0.816 |
| other | n/a | 0.000 |

## By communication style

| style | parse_ok | risk acc | category acc | clarification acc |
|---|---|---|---|---|
| ambiguous | 1.000 | 0.611 | 0.178 | 0.856 |
| canonical | 1.000 | 0.744 | 0.656 | 0.933 |
| clinical | 1.000 | 0.656 | 0.611 | 0.967 |
| emotionally_concerned | 1.000 | 0.644 | 0.644 | 0.856 |
| indirect_cultural | 1.000 | 0.689 | 0.511 | 0.656 |
| layperson | 1.000 | 0.656 | 0.633 | 0.900 |

## By gold category

| category | n | risk acc | category acc | clarification acc |
|---|---|---|---|---|
| fertility | 180 | 0.622 | 0.467 | 0.939 |
| menarche | 18 | 0.611 | 0.000 | 0.667 |
| menopause | 18 | 0.722 | 0.000 | 0.833 |
| menstrual | 108 | 0.694 | 0.759 | 0.861 |
| menstruation | 12 | 0.750 | 0.000 | 0.500 |
| pcos | 180 | 0.678 | 0.694 | 0.822 |
| pms | 12 | 0.500 | 0.000 | 0.917 |
| symptoms | 12 | 1.000 | 0.000 | 0.917 |

## Cross-style consistency

Fraction of the 90 seed groups given the same label across all six styles.

- risk: 0.644 (n=90 groups)
- category: 0.156 (n=90 groups)

## 95% bootstrap confidence intervals

- parse_ok_rate: [1.000, 1.000]
- category_accuracy: [0.496, 0.580]
