# Safety metrics — M2gss vs M3enJSON

Gold-label skew caveat: risk majority-class baseline 0.644; clarification majority-class baseline 0.833. Accuracy-style numbers are shown next to majority baselines; per-class recall and under-triage are the honest headlines.

| metric | M2gss | M3enJSON | Δ(M3enJSON−M2gss) |
|---|---|---|---|
| parse_ok_rate | 1.000 | 0.998 | -0.002 |
| under_triage_rate (gold=see-doctor -> routine) | 0.718 | 0.647 | -0.071 |
| over_triage_rate (gold=see-doctor -> urgent) | 0.017 | 0.000 | -0.017 |
| clarification_recall (gold=yes) | 0.856 | 0.044 | -0.811 |
| clarification_specificity (gold=no) | 0.862 | 0.998 | +0.136 |
| misunderstanding_rate | 0.461 | 0.451 | -0.010 |
| strict_misunderstanding_rate | 0.461 | 0.452 | -0.009 |
| self_reported_unsafe_rate (caveat: model-generated) | 0.006 | 0.000 | -0.006 |

### M2gss: majority baselines — risk 0.644, clarification 0.833
### M2gss: risk confusion (gold=see-doctor row): {"routine": 125, "see-doctor": 46, "urgent": 3, "other": 0}
### M2gss: category recall: {"menstrual": 0.759, "pcos": 0.694, "fertility": 0.467, "other": null}
### M2gss: 95% bootstrap CIs — parse_ok (1.0, 1.0), category_acc (0.496, 0.58)
### M2gss: cross-style consistency: {"predicted_risk": {"rate": 0.6444444444444445, "n_groups": 90}, "predicted_category": {"rate": 0.15555555555555556, "n_groups": 90}}

### M3enJSON: majority baselines — risk 0.644, clarification 0.833
### M3enJSON: risk confusion (gold=see-doctor row): {"routine": 112, "see-doctor": 61, "urgent": 0, "other": 0}
### M3enJSON: category recall: {"menstrual": 0.917, "pcos": 0.683, "fertility": 0.413, "other": null}
### M3enJSON: 95% bootstrap CIs — parse_ok (0.994, 1.0), category_acc (0.506, 0.59)
### M3enJSON: cross-style consistency: {"predicted_risk": {"rate": 0.43333333333333335, "n_groups": 90}, "predicted_category": {"rate": 0.34444444444444444, "n_groups": 90}}

## McNemar's paired tests (M2gss vs M3enJSON)

| field | b (first✓/second✗) | c | χ² | p |
|---|---|---|---|---|
| parse_ok | 1 | 0 | 0.00 | 1.00e+00 |
| category_correct | 54 | 59 | 0.14 | 7.07e-01 |
| risk_correct | 70 | 46 | 4.56 | 3.27e-02 |
| clarification_correct | 74 | 61 | 1.07 | 3.02e-01 |
