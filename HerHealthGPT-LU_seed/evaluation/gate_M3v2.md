# Safety metrics — M3v2

Gold-label skew caveat: risk majority-class baseline 0.578; clarification majority-class baseline 0.833. Accuracy-style numbers are shown next to majority baselines; per-class recall and under-triage are the honest headlines.

| metric | M3v2 |
|---|---|
| parse_ok_rate | 1.000 |
| under_triage_rate (gold=see-doctor -> routine) | 0.077 |
| over_triage_rate (gold=see-doctor -> urgent) | 0.000 |
| clarification_recall (gold=yes) | 0.800 |
| clarification_specificity (gold=no) | 0.840 |
| misunderstanding_rate | 0.011 |
| strict_misunderstanding_rate | 0.011 |
| self_reported_unsafe_rate (caveat: model-generated) | 0.000 |

### M3v2: majority baselines — risk 0.578, clarification 0.833
### M3v2: risk confusion (gold=see-doctor row): {"routine": 4, "see-doctor": 48, "urgent": 0, "other": 0}
### M3v2: category recall: {"menstrual": 1.0, "pcos": 1.0, "fertility": 0.962, "other": null}
### M3v2: 95% bootstrap CIs — parse_ok (1.0, 1.0), category_acc (0.967, 1.0)
### M3v2: cross-style consistency: {"predicted_risk": {"rate": 0.0, "n_groups": 0}, "predicted_category": {"rate": 0.0, "n_groups": 0}}

