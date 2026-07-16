# Safety metrics — M2gss vs M3ml

Gold-label skew caveat: risk majority-class baseline 0.644; clarification majority-class baseline 0.833. Accuracy-style numbers are shown next to majority baselines; per-class recall and under-triage are the honest headlines.

| metric | M2gss | M3ml | Δ(M3ml−M2gss) |
|---|---|---|---|
| parse_ok_rate | 1.000 | 0.998 | -0.002 |
| under_triage_rate (gold=see-doctor -> routine) | 0.718 | 0.711 | -0.007 |
| over_triage_rate (gold=see-doctor -> urgent) | 0.017 | 0.000 | -0.017 |
| clarification_recall (gold=yes) | 0.856 | 0.000 | -0.856 |
| clarification_specificity (gold=no) | 0.862 | 1.000 | +0.138 |
| misunderstanding_rate | 0.461 | 0.490 | +0.029 |
| strict_misunderstanding_rate | 0.461 | 0.491 | +0.030 |
| self_reported_unsafe_rate (caveat: model-generated) | 0.006 | 0.000 | -0.006 |

### M2gss: majority baselines — risk 0.644, clarification 0.833
### M2gss: risk confusion (gold=see-doctor row): {"routine": 125, "see-doctor": 46, "urgent": 3, "other": 0}
### M2gss: category recall: {"menstrual": 0.759, "pcos": 0.694, "fertility": 0.467, "other": null}
### M2gss: 95% bootstrap CIs — parse_ok (1.0, 1.0), category_acc (0.496, 0.58)
### M2gss: cross-style consistency: {"predicted_risk": {"rate": 0.6444444444444445, "n_groups": 90}, "predicted_category": {"rate": 0.15555555555555556, "n_groups": 90}}

### M3ml: majority baselines — risk 0.644, clarification 0.833
### M3ml: risk confusion (gold=see-doctor row): {"routine": 123, "see-doctor": 50, "urgent": 0, "other": 0}
### M3ml: category recall: {"menstrual": 0.935, "pcos": 0.694, "fertility": 0.274, "other": null}
### M3ml: 95% bootstrap CIs — parse_ok (0.994, 1.0), category_acc (0.468, 0.553)
### M3ml: cross-style consistency: {"predicted_risk": {"rate": 0.5, "n_groups": 90}, "predicted_category": {"rate": 0.4111111111111111, "n_groups": 90}}

## McNemar's paired tests (M2gss vs M3ml)

| field | b (first✓/second✗) | c | χ² | p |
|---|---|---|---|---|
| parse_ok | 1 | 0 | 0.00 | 1.00e+00 |
| category_correct | 66 | 50 | 1.94 | 1.64e-01 |
| risk_correct | 54 | 47 | 0.36 | 5.50e-01 |
| clarification_correct | 78 | 62 | 1.61 | 2.05e-01 |
