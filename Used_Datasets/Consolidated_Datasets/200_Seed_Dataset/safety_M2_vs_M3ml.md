# Safety metrics — M2gss vs M3ml

Gold-label skew caveat: risk majority-class baseline 0.644; clarification majority-class baseline 0.833. Accuracy-style numbers are shown next to majority baselines; per-class recall and under-triage are the honest headlines.

| metric | M2gss | M3ml | Δ(M3ml−M2gss) |
|---|---|---|---|
<<<<<<< HEAD
| parse_ok_rate | 1.000 | 0.996 | -0.004 |
=======
| parse_ok_rate | 1.000 | 0.998 | -0.002 |
>>>>>>> e4a617716e0e7b05a27616d3dfd2c34415a8f25e
| under_triage_rate (gold=see-doctor -> routine) | 0.718 | 0.711 | -0.007 |
| over_triage_rate (gold=see-doctor -> urgent) | 0.017 | 0.000 | -0.017 |
| clarification_recall (gold=yes) | 0.856 | 0.000 | -0.856 |
| clarification_specificity (gold=no) | 0.862 | 1.000 | +0.138 |
<<<<<<< HEAD
| misunderstanding_rate | 0.461 | 0.507 | +0.046 |
| strict_misunderstanding_rate | 0.461 | 0.509 | +0.048 |
=======
| misunderstanding_rate | 0.461 | 0.490 | +0.029 |
| strict_misunderstanding_rate | 0.461 | 0.491 | +0.030 |
>>>>>>> e4a617716e0e7b05a27616d3dfd2c34415a8f25e
| self_reported_unsafe_rate (caveat: model-generated) | 0.006 | 0.000 | -0.006 |

### M2gss: majority baselines — risk 0.644, clarification 0.833
### M2gss: risk confusion (gold=see-doctor row): {"routine": 125, "see-doctor": 46, "urgent": 3, "other": 0}
### M2gss: category recall: {"menstrual": 0.759, "pcos": 0.694, "fertility": 0.467, "other": null}
### M2gss: 95% bootstrap CIs — parse_ok (1.0, 1.0), category_acc (0.496, 0.58)
### M2gss: cross-style consistency: {"predicted_risk": {"rate": 0.6444444444444445, "n_groups": 90}, "predicted_category": {"rate": 0.15555555555555556, "n_groups": 90}}

### M3ml: majority baselines — risk 0.644, clarification 0.833
### M3ml: risk confusion (gold=see-doctor row): {"routine": 123, "see-doctor": 50, "urgent": 0, "other": 0}
<<<<<<< HEAD
### M3ml: category recall: {"menstrual": 0.935, "pcos": 0.672, "fertility": 0.242, "other": null}
### M3ml: 95% bootstrap CIs — parse_ok (0.991, 1.0), category_acc (0.45, 0.535)
### M3ml: cross-style consistency: {"predicted_risk": {"rate": 0.5111111111111111, "n_groups": 90}, "predicted_category": {"rate": 0.4222222222222222, "n_groups": 90}}
=======
### M3ml: category recall: {"menstrual": 0.935, "pcos": 0.694, "fertility": 0.274, "other": null}
### M3ml: 95% bootstrap CIs — parse_ok (0.994, 1.0), category_acc (0.468, 0.553)
### M3ml: cross-style consistency: {"predicted_risk": {"rate": 0.5, "n_groups": 90}, "predicted_category": {"rate": 0.4111111111111111, "n_groups": 90}}
>>>>>>> e4a617716e0e7b05a27616d3dfd2c34415a8f25e

## McNemar's paired tests (M2gss vs M3ml)

| field | b (first✓/second✗) | c | χ² | p |
|---|---|---|---|---|
<<<<<<< HEAD
| parse_ok | 2 | 0 | 0.50 | 4.80e-01 |
| category_correct | 76 | 50 | 4.96 | 2.59e-02 |
| risk_correct | 55 | 45 | 0.81 | 3.68e-01 |
| clarification_correct | 79 | 62 | 1.82 | 1.78e-01 |
=======
| parse_ok | 1 | 0 | 0.00 | 1.00e+00 |
| category_correct | 66 | 50 | 1.94 | 1.64e-01 |
| risk_correct | 54 | 47 | 0.36 | 5.50e-01 |
| clarification_correct | 78 | 62 | 1.61 | 2.05e-01 |
>>>>>>> e4a617716e0e7b05a27616d3dfd2c34415a8f25e
