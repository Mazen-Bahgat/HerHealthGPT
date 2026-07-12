import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
import evaluate as ev  # noqa: E402


def valid_fields(**overrides):
    fields = {"predicted_category": "menstrual", "interpreted_symptom": "symptom",
              "predicted_risk": "see-doctor", "recommended_action": "see clinician",
              "asks_clarification": False, "clarifying_question": "",
              "unsafe_response": False, "response_text": "Please see a clinician."}
    fields.update(overrides)
    return fields


def test_score_record_tracks_category_risk_clarification_and_parse_status():
    record = {
        **valid_fields(asks_clarification=True),
        "item_id": "menst-001_canonical_en",
        "model_label": "M2_zero_shot",
        "language": "en",
        "style": "canonical",
        "gold_category": "menstrual",
        "predicted_category": "menstrual",
        "gold_risk_level": "see-doctor",
        "predicted_risk": "see-doctor",
        "requires_clarification": "yes",
        "asks_clarification": True,
        "unsafe_response": False,
        "_parse_error": "",
    }

    scored = ev.score_record(record)

    assert scored["category_correct"] is True
    assert scored["risk_correct"] is True
    assert scored["clarification_correct"] is True
    assert scored["unsafe_response"] is False
    assert scored["parse_ok"] is True


def test_summarize_reports_overall_and_breakdowns():
    records = [
        {
            **valid_fields(),
            "item_id": "a",
            "model_label": "M2_zero_shot",
            "language": "en",
            "style": "canonical",
            "gold_category": "menstrual",
            "predicted_category": "menstrual",
            "gold_risk_level": "see-doctor",
            "predicted_risk": "see-doctor",
            "requires_clarification": "no",
            "asks_clarification": False,
            "unsafe_response": False,
            "_parse_error": "",
        },
        {
            **valid_fields(predicted_category="fertility", predicted_risk="routine", unsafe_response=True),
            "item_id": "b",
            "model_label": "M2_zero_shot",
            "language": "en",
            "style": "ambiguous",
            "gold_category": "pcos",
            "predicted_category": "fertility",
            "gold_risk_level": "see-doctor",
            "predicted_risk": "routine",
            "requires_clarification": "yes",
            "asks_clarification": False,
            "unsafe_response": True,
            "_parse_error": "",
        },
    ]

    summary = ev.summarize(records)

    assert summary["overall"]["n"] == 2
    assert summary["overall"]["category_accuracy"] == 0.5
    assert summary["overall"]["risk_accuracy"] == 0.5
    assert summary["overall"]["clarification_accuracy"] == 0.5
    assert summary["overall"]["self_reported_unsafe_rate"] == 0.5
    assert summary["by_style"]["canonical"]["category_accuracy"] == 1.0
    assert summary["by_style"]["ambiguous"]["category_accuracy"] == 0.0


def test_validate_records_rejects_duplicates_and_mixed_groups():
    base = {"item_id": "a", "model_label": "M1", "model": "m", "language": "en"}
    with pytest.raises(ValueError, match="duplicate"):
        ev.validate_records([base, dict(base)])
    with pytest.raises(ValueError, match="mixed"):
        ev.validate_records([base, {**base, "item_id": "b", "language": "fr"}])
    ev.validate_records([base, {**base, "item_id": "b", "language": "fr"}], multi_group=True)


def test_validate_records_checks_expected_count():
    with pytest.raises(ValueError, match="expected 2"):
        ev.validate_records([{"item_id": "a", "model_label": "M", "model": "m", "language": "en"}], expected_count=2)


def test_summary_uses_valid_prediction_denominators_and_reports_coverage():
    good = {
        **valid_fields(predicted_category="pcos", predicted_risk="routine"),
        "item_id": "a", "model_label": "M", "model": "m", "language": "en", "style": "canonical",
        "gold_category": "pcos", "predicted_category": "pcos", "gold_risk_level": "routine",
        "predicted_risk": "routine", "requires_clarification": "no", "asks_clarification": False,
        "unsafe_response": False, "_parse_error": "", "_error": "",
    }
    failed = {**good, "item_id": "b", "predicted_category": None, "asks_clarification": None,
              "_parse_error": "response_contract_error", "_error": "null content", "_unparsed_response": ""}
    overall = ev.summarize([good, failed])["overall"]
    assert overall["attempted_count"] == 2
    assert overall["valid_prediction_count"] == 1
    assert overall["request_error_count"] == 0
    assert overall["parse_schema_error_count"] == 1
    assert overall["unparsed_response_count"] == 1
    assert overall["prediction_coverage"] == 0.5
    assert overall["category_accuracy"] == 1.0
    assert overall["category_error_rate"] == 0.0
    assert overall["risk_error_rate"] == 0.0
    assert "unsafe_response_rate" not in overall
    assert overall["self_reported_unsafe_metric_warning"]


def test_score_record_rejects_schema_incomplete_record_even_without_error_marker():
    scored = ev.score_record({"item_id": "legacy", "_parse_error": "", "_error": ""})
    assert scored["parse_ok"] is False
    assert scored["parse_schema_error"] is True
    assert scored["asks_clarification"] is None
    assert scored["unsafe_response"] is None


def test_empty_category_is_excluded_from_scoring_and_counted_as_schema_error():
    record = {
        **valid_fields(predicted_category=""), "item_id": "empty", "gold_category": "other",
        "gold_risk_level": "see-doctor", "requires_clarification": "no",
        "_parse_error": "", "_error": "",
    }
    scored = ev.score_record(record)
    overall = ev.summarize([record])["overall"]
    assert scored["parse_ok"] is False
    assert scored["category_correct"] is False
    assert overall["valid_prediction_count"] == 0
    assert overall["parse_schema_error_count"] == 1
    assert overall["prediction_coverage"] == 0.0
