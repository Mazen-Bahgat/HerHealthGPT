# Safety metrics — M2gss vs M3env2

Gold-label skew caveat: risk majority-class baseline 0.644; clarification majority-class baseline 0.833. Accuracy-style numbers are shown next to majority baselines; per-class recall and under-triage are the honest headlines.

| metric | M2gss | M3env2 | Δ(M3env2−M2gss) |
|---|---|---|---|
| parse_ok_rate | 1.000 | 0.865 | -0.135 |
| under_triage_rate (gold=see-doctor -> routine) | 0.718 | 0.800 | +0.082 |
| over_triage_rate (gold=see-doctor -> urgent) | 0.017 | 0.000 | -0.017 |
| clarification_recall (gold=yes) | 0.856 | 0.174 | -0.681 |
| clarification_specificity (gold=no) | 0.862 | 1.000 | +0.138 |
| misunderstanding_rate | 0.461 | 0.501 | +0.040 |
| strict_misunderstanding_rate | 0.461 | 0.569 | +0.107 |
| self_reported_unsafe_rate (caveat: model-generated) | 0.006 | 0.000 | -0.006 |

### M2gss: majority baselines — risk 0.644, clarification 0.833
### M2gss: risk confusion (gold=see-doctor row): {"routine": 125, "see-doctor": 46, "urgent": 3, "other": 0}
### M2gss: category recall: {"menstrual": 0.759, "pcos": 0.694, "fertility": 0.467, "other": null}
### M2gss: 95% bootstrap CIs — parse_ok (1.0, 1.0), category_acc (0.496, 0.58)
### M2gss: cross-style consistency: {"predicted_risk": {"rate": 0.6444444444444445, "n_groups": 90}, "predicted_category": {"rate": 0.15555555555555556, "n_groups": 90}}

### M3env2: majority baselines — risk 0.644, clarification 0.833
### M3env2: risk confusion (gold=see-doctor row): {"routine": 128, "see-doctor": 32, "urgent": 0, "other": 0}
### M3env2: category recall: {"menstrual": 0.854, "pcos": 0.574, "fertility": 0.424, "other": null}
### M3env2: 95% bootstrap CIs — parse_ok (0.835, 0.893), category_acc (0.454, 0.544)
### M3env2: cross-style consistency: {"predicted_risk": {"rate": 0.45555555555555555, "n_groups": 90}, "predicted_category": {"rate": 0.18888888888888888, "n_groups": 90}}

## McNemar's paired tests (M2gss vs M3env2)

| field | b (first✓/second✗) | c | χ² | p |
|---|---|---|---|---|
| parse_ok | 73 | 0 | 71.01 | 3.55e-17 |
| category_correct | 77 | 19 | 33.84 | 5.97e-09 |
| risk_correct | 87 | 19 | 42.35 | 7.64e-11 |
| clarification_correct | 127 | 58 | 24.99 | 5.75e-07 |
