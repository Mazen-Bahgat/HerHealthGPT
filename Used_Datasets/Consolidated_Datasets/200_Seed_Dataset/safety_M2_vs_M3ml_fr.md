# Safety metrics — M2fr vs M3mlfr

Gold-label skew caveat: risk majority-class baseline 0.644; clarification majority-class baseline 0.833. Accuracy-style numbers are shown next to majority baselines; per-class recall and under-triage are the honest headlines.

| metric | M2fr | M3mlfr | Δ(M3mlfr−M2fr) |
|---|---|---|---|
| parse_ok_rate | 1.000 | 0.998 | -0.002 |
| under_triage_rate (gold=see-doctor -> routine) | 0.632 | 0.994 | +0.362 |
| over_triage_rate (gold=see-doctor -> urgent) | 0.011 | 0.000 | -0.011 |
| clarification_recall (gold=yes) | 0.811 | 0.000 | -0.811 |
| clarification_specificity (gold=no) | 0.862 | 0.998 | +0.136 |
| misunderstanding_rate | 0.439 | 0.488 | +0.049 |
| strict_misunderstanding_rate | 0.439 | 0.489 | +0.050 |
| self_reported_unsafe_rate (caveat: model-generated) | 0.002 | 0.000 | -0.002 |

### M2fr: majority baselines — risk 0.644, clarification 0.833
### M2fr: risk confusion (gold=see-doctor row): {"routine": 110, "see-doctor": 62, "urgent": 2, "other": 0}
### M2fr: category recall: {"menstrual": 0.796, "pcos": 0.717, "fertility": 0.489, "other": null}
### M2fr: 95% bootstrap CIs — parse_ok (1.0, 1.0), category_acc (0.519, 0.602)
### M2fr: cross-style consistency: {"predicted_risk": {"rate": 0.5444444444444444, "n_groups": 90}, "predicted_category": {"rate": 0.16666666666666666, "n_groups": 90}}

### M3mlfr: majority baselines — risk 0.644, clarification 0.833
### M3mlfr: risk confusion (gold=see-doctor row): {"routine": 173, "see-doctor": 1, "urgent": 0, "other": 0}
### M3mlfr: category recall: {"menstrual": 0.944, "pcos": 0.689, "fertility": 0.283, "other": null}
### M3mlfr: 95% bootstrap CIs — parse_ok (0.994, 1.0), category_acc (0.469, 0.555)
### M3mlfr: cross-style consistency: {"predicted_risk": {"rate": 0.9777777777777777, "n_groups": 90}, "predicted_category": {"rate": 0.35555555555555557, "n_groups": 90}}

## McNemar's paired tests (M2fr vs M3mlfr)

| field | b (first✓/second✗) | c | χ² | p |
|---|---|---|---|---|
| parse_ok | 1 | 0 | 0.00 | 1.00e+00 |
| category_correct | 73 | 46 | 5.68 | 1.72e-02 |
| risk_correct | 66 | 44 | 4.01 | 4.53e-02 |
| clarification_correct | 73 | 60 | 1.08 | 2.98e-01 |
