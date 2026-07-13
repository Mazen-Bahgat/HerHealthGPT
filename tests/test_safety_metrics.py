import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
import safety_metrics as sm  # noqa: E402

RISK_LABELS = ["routine", "see-doctor", "urgent"]


def _scored(gold_risk="see-doctor", pred_risk="routine", parse_ok=True,
            gold_cat="menstrual", pred_cat="menstrual",
            req_clar=False, asks=False, item="i1"):
    return {"item_id": item, "parse_ok": parse_ok,
            "gold_risk_level": gold_risk, "predicted_risk": pred_risk,
            "gold_category": gold_cat, "predicted_category": pred_cat,
            "requires_clarification_bool": req_clar, "asks_clarification": asks,
            "category_correct": parse_ok and gold_cat == pred_cat,
            "risk_correct": parse_ok and gold_risk == pred_risk,
            "clarification_correct": parse_ok and req_clar == asks,
            "unsafe_response": False}


def test_confusion_matrix_counts_and_other_bucket():
    rows = [_scored(pred_risk="routine"), _scored(pred_risk="routine"),
            _scored(pred_risk="urgent"), _scored(pred_risk="weird-label"),
            _scored(parse_ok=False)]  # excluded
    cm = sm.confusion_matrix(rows, "gold_risk_level", "predicted_risk", RISK_LABELS)
    assert cm["see-doctor"]["routine"] == 2
    assert cm["see-doctor"]["urgent"] == 1
    assert cm["see-doctor"]["other"] == 1
    assert sum(sum(r.values()) for r in cm.values()) == 4  # parse-fail excluded


def test_per_class_recall_none_when_no_gold():
    cm = {"see-doctor": {"routine": 3, "see-doctor": 1, "urgent": 0, "other": 0},
          "routine": {"routine": 0, "see-doctor": 0, "urgent": 0, "other": 0}}
    rec = sm.per_class_recall(cm)
    assert rec["see-doctor"] == 0.25
    assert rec["routine"] is None


def test_under_triage_separates_over_triage():
    rows = ([_scored(pred_risk="routine")] * 7 + [_scored(pred_risk="urgent")] * 2
            + [_scored(pred_risk="see-doctor")] * 1)
    ut = sm.under_triage(rows)
    assert ut["n_gold_see_doctor"] == 10
    assert abs(ut["under_triage_rate"] - 0.7) < 1e-9
    assert abs(ut["over_triage_rate"] - 0.2) < 1e-9


def test_clarification_zero_recall_case():
    rows = ([_scored(req_clar=True, asks=False)] * 4      # all misses
            + [_scored(req_clar=False, asks=False)] * 90
            + [_scored(req_clar=False, asks=True)] * 6)   # false alarms
    cs = sm.clarification_stats(rows)
    assert cs["recall_gold_yes"] == 0.0
    assert cs["n_gold_yes"] == 4
    assert cs["false_alarms"] == 6
    assert abs(cs["specificity_gold_no"] - 90 / 96) < 1e-9


def test_majority_baseline_on_skew():
    rows = [_scored(req_clar=False)] * 95 + [_scored(req_clar=True)] * 5
    assert abs(sm.majority_baseline(rows, "requires_clarification_bool") - 0.95) < 1e-9


def test_misunderstanding_plain_and_strict():
    rows = ([_scored(pred_cat="menstrual")] * 8 + [_scored(pred_cat="pcos")] * 2
            + [_scored(parse_ok=False)] * 10)
    m = sm.misunderstanding(rows, n_total=20)
    assert abs(m["misunderstanding_rate"] - 0.2) < 1e-9          # 2/10 parseable
    assert abs(m["strict_misunderstanding_rate"] - 0.6) < 1e-9   # (2+10)/20


def test_mcnemar_discordant_counts():
    a = [_scored(item=f"i{k}", pred_cat="menstrual") for k in range(10)]          # all correct
    b = ([_scored(item=f"i{k}", pred_cat="pcos") for k in range(6)]               # 6 wrong
         + [_scored(item=f"i{k}", pred_cat="menstrual") for k in range(6, 10)])
    r = sm.mcnemar(a, b, "category_correct")
    assert r["b"] == 6 and r["c"] == 0      # a-correct/b-wrong = 6
    assert r["chi2"] > 3.84                  # significant at 0.05
    assert r["p_value"] < 0.05


def test_bootstrap_ci_brackets_mean_and_is_deterministic():
    vals = [1.0] * 70 + [0.0] * 30
    lo, hi = sm.bootstrap_ci(vals, n_boot=2000, seed=42)
    assert lo < 0.7 < hi and 0.6 < lo and hi < 0.8
    assert (lo, hi) == sm.bootstrap_ci(vals, n_boot=2000, seed=42)


def test_render_report_two_labels_has_delta_and_dash():
    a1 = {"parse_ok_rate": 1.0, "under_triage": {"under_triage_rate": 0.8, "over_triage_rate": 0.1, "n_gold_see_doctor": 10},
          "clarification": {"recall_gold_yes": None, "specificity_gold_no": 1.0, "n_gold_yes": 0, "n_gold_no": 10, "false_alarms": 0, "confusion": {}},
          "misunderstanding": {"misunderstanding_rate": 0.2, "strict_misunderstanding_rate": 0.3},
          "self_reported_unsafe_rate": 0.0, "majority_baselines": {}, "category_recall": {}, "category_precision": {},
          "risk_confusion": {}, "cross_style_consistency": {}, "cross_language_consistency": {}, "cis": {}}
    b1 = dict(a1, parse_ok_rate=0.7)
    md = sm.render_report({"M2": a1, "M3": b1}, pair_tests=None)
    assert "M2" in md and "M3" in md
    assert "-" in md            # None rendered as dash
    assert "-0.300" in md or "−0.300" in md or "-0.30" in md  # delta on parse_ok
