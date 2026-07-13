# Safety metrics — M2 vs M3

Gold-label skew caveat: risk gold is 100% see-doctor; clarification gold ~95.6% no. Accuracy-style numbers are shown next to majority baselines; per-class recall and under-triage are the honest headlines.

| metric | M2 | M3 | Δ(M3−M2) |
|---|---|---|---|
| parse_ok_rate | 0.998 | 0.724 | -0.274 |
| under_triage_rate (gold=see-doctor -> routine) | 0.456 | 0.877 | +0.421 |
| over_triage_rate (gold=see-doctor -> urgent) | 0.045 | 0.110 | +0.065 |
| clarification_recall (gold=yes) | 0.208 | 0.000 | -0.208 |
| clarification_specificity (gold=no) | 0.878 | 0.951 | +0.074 |
| misunderstanding_rate | 0.390 | 0.389 | -0.001 |
| strict_misunderstanding_rate | 0.391 | 0.557 | +0.167 |
| self_reported_unsafe_rate (caveat: model-generated) | 0.052 | 0.000 | -0.052 |

### M2: majority baselines — risk 1.000, clarification 0.956
### M2: risk confusion (gold=see-doctor row): {"routine": 246, "see-doctor": 269, "urgent": 24, "other": 0}
### M2: category recall: {"menstrual": 0.389, "pcos": 0.531, "fertility": 0.911, "other": null}
### M2: 95% bootstrap CIs — parse_ok (0.994, 1.0), category_acc (0.57, 0.651)
### M2: cross-style consistency: {"predicted_risk": {"rate": 0.43333333333333335, "n_groups": 90}, "predicted_category": {"rate": 0.4888888888888889, "n_groups": 90}}

### M3: majority baselines — risk 1.000, clarification 0.956
### M3: risk confusion (gold=see-doctor row): {"routine": 343, "see-doctor": 5, "urgent": 43, "other": 0}
### M3: category recall: {"menstrual": 0.664, "pcos": 0.331, "fertility": 0.823, "other": null}
### M3: 95% bootstrap CIs — parse_ok (0.687, 0.761), category_acc (0.563, 0.66)
### M3: cross-style consistency: {"predicted_risk": {"rate": 0.16666666666666666, "n_groups": 90}, "predicted_category": {"rate": 0.14444444444444443, "n_groups": 90}}

## McNemar's paired tests (M2 vs M3)

| field | b (first✓/second✗) | c | χ² | p |
|---|---|---|---|---|
| parse_ok | 149 | 1 | 144.06 | 3.45e-33 |
| category_correct | 126 | 36 | 48.90 | 2.70e-12 |
| risk_correct | 265 | 1 | 260.03 | 1.69e-58 |
| clarification_correct | 145 | 41 | 57.04 | 4.28e-14 |
