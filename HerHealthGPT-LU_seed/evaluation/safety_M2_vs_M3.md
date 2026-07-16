# Safety metrics — M2 vs M3

Gold-label skew caveat: risk gold is 100% see-doctor; clarification gold ~95.6% no. Accuracy-style numbers are shown next to majority baselines; per-class recall and under-triage are the honest headlines.

| metric | M2 | M3 | Δ(M3−M2) |
|---|---|---|---|
| parse_ok_rate | 0.998 | 0.915 | -0.083 |
| under_triage_rate (gold=see-doctor -> routine) | 0.456 | 0.862 | +0.406 |
| over_triage_rate (gold=see-doctor -> urgent) | 0.045 | 0.128 | +0.083 |
| clarification_recall (gold=yes) | 0.208 | 0.000 | -0.208 |
| clarification_specificity (gold=no) | 0.878 | 0.949 | +0.071 |
| misunderstanding_rate | 0.390 | 0.393 | +0.003 |
| strict_misunderstanding_rate | 0.391 | 0.444 | +0.054 |
| self_reported_unsafe_rate (caveat: model-generated) | 0.052 | 0.000 | -0.052 |

### M2: majority baselines — risk 1.000, clarification 0.956
### M2: risk confusion (gold=see-doctor row): {"routine": 246, "see-doctor": 269, "urgent": 24, "other": 0}
### M2: category recall: {"menstrual": 0.389, "pcos": 0.531, "fertility": 0.911, "other": null}
### M2: 95% bootstrap CIs — parse_ok (0.994, 1.0), category_acc (0.57, 0.651)
### M2: cross-style consistency: {"predicted_risk": {"rate": 0.43333333333333335, "n_groups": 90}, "predicted_category": {"rate": 0.4888888888888889, "n_groups": 90}}

### M3: majority baselines — risk 1.000, clarification 0.956
### M3: risk confusion (gold=see-doctor row): {"routine": 426, "see-doctor": 5, "urgent": 63, "other": 0}
### M3: category recall: {"menstrual": 0.643, "pcos": 0.329, "fertility": 0.846, "other": null}
### M3: 95% bootstrap CIs — parse_ok (0.891, 0.937), category_acc (0.565, 0.65)
### M3: cross-style consistency: {"predicted_risk": {"rate": 0.45555555555555555, "n_groups": 90}, "predicted_category": {"rate": 0.37777777777777777, "n_groups": 90}}

## McNemar's paired tests (M2 vs M3)

| field | b (first✓/second✗) | c | χ² | p |
|---|---|---|---|---|
| parse_ok | 46 | 1 | 41.19 | 1.38e-10 |
| category_correct | 76 | 47 | 6.37 | 1.16e-02 |
| risk_correct | 265 | 1 | 260.03 | 1.69e-58 |
| clarification_correct | 54 | 44 | 0.83 | 3.63e-01 |
