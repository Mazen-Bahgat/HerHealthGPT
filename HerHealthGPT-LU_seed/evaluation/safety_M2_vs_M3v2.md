# Safety metrics — M2 vs M3v2

Gold-label skew caveat: risk majority-class baseline 1.000; clarification majority-class baseline 0.956. Accuracy-style numbers are shown next to majority baselines; per-class recall and under-triage are the honest headlines.

| metric | M2 | M3v2 | Δ(M3v2−M2) |
|---|---|---|---|
| parse_ok_rate | 0.998 | 0.998 | +0.000 |
| under_triage_rate (gold=see-doctor -> routine) | 0.456 | 0.035 | -0.421 |
| over_triage_rate (gold=see-doctor -> urgent) | 0.045 | 0.000 | -0.045 |
| clarification_recall (gold=yes) | 0.208 | 0.958 | +0.750 |
| clarification_specificity (gold=no) | 0.878 | 0.487 | -0.390 |
| misunderstanding_rate | 0.390 | 0.349 | -0.041 |
| strict_misunderstanding_rate | 0.391 | 0.350 | -0.041 |
| self_reported_unsafe_rate (caveat: model-generated) | 0.052 | 0.000 | -0.052 |

### M2: majority baselines — risk 1.000, clarification 0.956
### M2: risk confusion (gold=see-doctor row): {"routine": 246, "see-doctor": 269, "urgent": 24, "other": 0}
### M2: category recall: {"menstrual": 0.389, "pcos": 0.531, "fertility": 0.911, "other": null}
### M2: 95% bootstrap CIs — parse_ok (0.994, 1.0), category_acc (0.57, 0.651)
### M2: cross-style consistency: {"predicted_risk": {"rate": 0.43333333333333335, "n_groups": 90}, "predicted_category": {"rate": 0.4888888888888889, "n_groups": 90}}

### M3v2: majority baselines — risk 1.000, clarification 0.956
### M3v2: risk confusion (gold=see-doctor row): {"routine": 19, "see-doctor": 520, "urgent": 0, "other": 0}
### M3v2: category recall: {"menstrual": 0.251, "pcos": 0.906, "fertility": 0.794, "other": null}
### M3v2: 95% bootstrap CIs — parse_ok (0.994, 1.0), category_acc (0.61, 0.692)
### M3v2: cross-style consistency: {"predicted_risk": {"rate": 0.8888888888888888, "n_groups": 90}, "predicted_category": {"rate": 0.5, "n_groups": 90}}

## McNemar's paired tests (M2 vs M3v2)

| field | b (first✓/second✗) | c | χ² | p |
|---|---|---|---|---|
| parse_ok | 1 | 1 | 0.50 | 4.80e-01 |
| category_correct | 56 | 78 | 3.29 | 6.97e-02 |
| risk_correct | 5 | 256 | 239.46 | 5.15e-54 |
| clarification_correct | 217 | 34 | 131.97 | 1.52e-30 |
