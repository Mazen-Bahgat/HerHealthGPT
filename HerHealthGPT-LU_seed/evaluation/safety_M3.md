# Safety metrics — M3

Gold-label skew caveat: risk majority-class baseline 1.000; clarification majority-class baseline 0.956. Accuracy-style numbers are shown next to majority baselines; per-class recall and under-triage are the honest headlines.

| metric | M3 |
|---|---|
| parse_ok_rate | 0.724 |
| under_triage_rate (gold=see-doctor -> routine) | 0.877 |
| over_triage_rate (gold=see-doctor -> urgent) | 0.110 |
| clarification_recall (gold=yes) | 0.000 |
| clarification_specificity (gold=no) | 0.951 |
| misunderstanding_rate | 0.389 |
| strict_misunderstanding_rate | 0.557 |
| self_reported_unsafe_rate (caveat: model-generated) | 0.000 |

### M3: majority baselines — risk 1.000, clarification 0.956
### M3: risk confusion (gold=see-doctor row): {"routine": 343, "see-doctor": 5, "urgent": 43, "other": 0}
### M3: category recall: {"menstrual": 0.664, "pcos": 0.331, "fertility": 0.823, "other": null}
### M3: 95% bootstrap CIs — parse_ok (0.687, 0.761), category_acc (0.563, 0.66)
### M3: cross-style consistency: {"predicted_risk": {"rate": 0.16666666666666666, "n_groups": 90}, "predicted_category": {"rate": 0.14444444444444443, "n_groups": 90}}

