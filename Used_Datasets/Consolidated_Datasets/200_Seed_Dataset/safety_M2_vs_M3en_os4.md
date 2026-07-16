# Safety metrics — M2gss vs M3os4

Gold-label skew caveat: risk majority-class baseline 0.644; clarification majority-class baseline 0.833. Accuracy-style numbers are shown next to majority baselines; per-class recall and under-triage are the honest headlines.

| metric | M2gss | M3os4 | Δ(M3os4−M2gss) |
|---|---|---|---|
| parse_ok_rate | 1.000 | 0.996 | -0.004 |
| under_triage_rate (gold=see-doctor -> routine) | 0.718 | 0.605 | -0.114 |
| over_triage_rate (gold=see-doctor -> urgent) | 0.017 | 0.000 | -0.017 |
| clarification_recall (gold=yes) | 0.856 | 0.000 | -0.856 |
| clarification_specificity (gold=no) | 0.862 | 1.000 | +0.138 |
| misunderstanding_rate | 0.461 | 0.450 | -0.011 |
| strict_misunderstanding_rate | 0.461 | 0.452 | -0.009 |
| self_reported_unsafe_rate (caveat: model-generated) | 0.006 | 0.000 | -0.006 |

### M2gss: majority baselines — risk 0.644, clarification 0.833
### M2gss: risk confusion (gold=see-doctor row): {"routine": 125, "see-doctor": 46, "urgent": 3, "other": 0}
### M2gss: category recall: {"menstrual": 0.759, "pcos": 0.694, "fertility": 0.467, "other": null}
### M2gss: 95% bootstrap CIs — parse_ok (1.0, 1.0), category_acc (0.496, 0.58)
### M2gss: cross-style consistency: {"predicted_risk": {"rate": 0.6444444444444445, "n_groups": 90}, "predicted_category": {"rate": 0.15555555555555556, "n_groups": 90}}

### M3os4: majority baselines — risk 0.644, clarification 0.833
### M3os4: risk confusion (gold=see-doctor row): {"routine": 104, "see-doctor": 68, "urgent": 0, "other": 0}
### M3os4: category recall: {"menstrual": 0.907, "pcos": 0.75, "fertility": 0.354, "other": null}
### M3os4: 95% bootstrap CIs — parse_ok (0.991, 1.0), category_acc (0.509, 0.591)
### M3os4: cross-style consistency: {"predicted_risk": {"rate": 0.4444444444444444, "n_groups": 90}, "predicted_category": {"rate": 0.37777777777777777, "n_groups": 90}}

## McNemar's paired tests (M2gss vs M3os4)

| field | b (first✓/second✗) | c | χ² | p |
|---|---|---|---|---|
| parse_ok | 2 | 0 | 0.50 | 4.80e-01 |
| category_correct | 53 | 58 | 0.14 | 7.04e-01 |
| risk_correct | 50 | 48 | 0.01 | 9.20e-01 |
| clarification_correct | 78 | 62 | 1.61 | 2.05e-01 |
