# Safety metrics — M2 vs M3 vs M3v2

Gold-label skew caveat: risk majority-class baseline 1.000; clarification majority-class baseline 0.956. Accuracy-style numbers are shown next to majority baselines; per-class recall and under-triage are the honest headlines.

| metric | M2 | M3 | M3v2 |
|---|---|---|---|
| parse_ok_rate | 0.998 | 0.915 | 0.998 |
| under_triage_rate (gold=see-doctor -> routine) | 0.456 | 0.862 | 0.035 |
| over_triage_rate (gold=see-doctor -> urgent) | 0.045 | 0.128 | 0.000 |
| clarification_recall (gold=yes) | 0.208 | 0.000 | 0.958 |
| clarification_specificity (gold=no) | 0.878 | 0.949 | 0.487 |
| misunderstanding_rate | 0.390 | 0.393 | 0.349 |
| strict_misunderstanding_rate | 0.391 | 0.444 | 0.350 |
| self_reported_unsafe_rate (caveat: model-generated) | 0.052 | 0.000 | 0.000 |

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

### M3v2: majority baselines — risk 1.000, clarification 0.956
### M3v2: risk confusion (gold=see-doctor row): {"routine": 19, "see-doctor": 520, "urgent": 0, "other": 0}
### M3v2: category recall: {"menstrual": 0.251, "pcos": 0.906, "fertility": 0.794, "other": null}
### M3v2: 95% bootstrap CIs — parse_ok (0.994, 1.0), category_acc (0.61, 0.692)
### M3v2: cross-style consistency: {"predicted_risk": {"rate": 0.8888888888888888, "n_groups": 90}, "predicted_category": {"rate": 0.5, "n_groups": 90}}

