# Safety metrics — M2ar vs M3mlar

Gold-label skew caveat: risk majority-class baseline 0.644; clarification majority-class baseline 0.833. Accuracy-style numbers are shown next to majority baselines; per-class recall and under-triage are the honest headlines.

| metric | M2ar | M3mlar | Δ(M3mlar−M2ar) |
|---|---|---|---|
| parse_ok_rate | 1.000 | 0.998 | -0.002 |
| under_triage_rate (gold=see-doctor -> routine) | 0.638 | 0.994 | +0.356 |
| over_triage_rate (gold=see-doctor -> urgent) | 0.023 | 0.000 | -0.023 |
| clarification_recall (gold=yes) | 0.889 | 0.011 | -0.878 |
| clarification_specificity (gold=no) | 0.827 | 1.000 | +0.173 |
| misunderstanding_rate | 0.467 | 0.469 | +0.003 |
| strict_misunderstanding_rate | 0.467 | 0.470 | +0.004 |
| self_reported_unsafe_rate (caveat: model-generated) | 0.006 | 0.000 | -0.006 |

### M2ar: majority baselines — risk 0.644, clarification 0.833
### M2ar: risk confusion (gold=see-doctor row): {"routine": 111, "see-doctor": 59, "urgent": 4, "other": 0}
### M2ar: category recall: {"menstrual": 0.796, "pcos": 0.683, "fertility": 0.439, "other": null}
### M2ar: 95% bootstrap CIs — parse_ok (1.0, 1.0), category_acc (0.491, 0.576)
### M2ar: cross-style consistency: {"predicted_risk": {"rate": 0.45555555555555555, "n_groups": 90}, "predicted_category": {"rate": 0.15555555555555556, "n_groups": 90}}

### M3mlar: majority baselines — risk 0.644, clarification 0.833
### M3mlar: risk confusion (gold=see-doctor row): {"routine": 173, "see-doctor": 1, "urgent": 0, "other": 0}
### M3mlar: category recall: {"menstrual": 0.925, "pcos": 0.711, "fertility": 0.328, "other": null}
### M3mlar: 95% bootstrap CIs — parse_ok (0.994, 1.0), category_acc (0.488, 0.573)
### M3mlar: cross-style consistency: {"predicted_risk": {"rate": 0.9666666666666667, "n_groups": 90}, "predicted_category": {"rate": 0.32222222222222224, "n_groups": 90}}

## McNemar's paired tests (M2ar vs M3mlar)

| field | b (first✓/second✗) | c | χ² | p |
|---|---|---|---|---|
| parse_ok | 1 | 0 | 0.00 | 1.00e+00 |
| category_correct | 63 | 61 | 0.01 | 9.28e-01 |
| risk_correct | 63 | 47 | 2.05 | 1.53e-01 |
| clarification_correct | 80 | 78 | 0.01 | 9.37e-01 |
