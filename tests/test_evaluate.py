import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
import evaluate as ev  # noqa: E402


def test_score_record_tracks_category_risk_clarification_and_parse_status():
    record = {
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
    assert summary["overall"]["unsafe_response_rate"] == 0.5
    assert summary["by_style"]["canonical"]["category_accuracy"] == 1.0
    assert summary["by_style"]["ambiguous"]["category_accuracy"] == 0.0
