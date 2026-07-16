# Safety metrics — M2gss

Gold-label skew caveat: risk gold is 100% see-doctor; clarification gold ~95.6% no. Accuracy-style numbers are shown next to majority baselines; per-class recall and under-triage are the honest headlines.

| metric | M2gss |
|---|---|
| parse_ok_rate | 1.000 |
| under_triage_rate (gold=see-doctor -> routine) | 0.750 |
| over_triage_rate (gold=see-doctor -> urgent) | 0.022 |
| clarification_recall (gold=yes) | 0.856 |
| clarification_specificity (gold=no) | 0.862 |
| misunderstanding_rate | 0.461 |
| strict_misunderstanding_rate | 0.461 |
| self_reported_unsafe_rate (caveat: model-generated) | 0.006 |

### M2gss: majority baselines — risk 0.633, clarification 0.833
### M2gss: risk confusion (gold=see-doctor row): {"routine": 135, "see-doctor": 41, "urgent": 4, "other": 0}
### M2gss: category recall: {"menstrual": 0.759, "pcos": 0.694, "fertility": 0.467, "other": null}
### M2gss: 95% bootstrap CIs — parse_ok (1.0, 1.0), category_acc (0.496, 0.58)
### M2gss: cross-style consistency: {"predicted_risk": {"rate": 0.6444444444444445, "n_groups": 90}, "predicted_category": {"rate": 0.15555555555555556, "n_groups": 90}}

