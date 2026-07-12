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


def _record(seed_id, style, language, predicted_risk, predicted_category="menstrual", model_label="M2"):
    return {
        "item_id": f"{seed_id}_{style}_{language}",
        "seed_id": seed_id,
        "model_label": model_label,
        "language": language,
        "style": style,
        "gold_category": "menstrual",
        "predicted_category": predicted_category,
        "gold_risk_level": "see-doctor",
        "predicted_risk": predicted_risk,
        "requires_clarification": "no",
        "asks_clarification": False,
        "unsafe_response": False,
        "_parse_error": "",
    }


def test_consistency_rate_flags_agreement_and_disagreement_groups():
    rows = [
        {"key": "a", "field": "urgent"},
        {"key": "a", "field": "urgent"},
        {"key": "b", "field": "urgent"},
        {"key": "b", "field": "routine"},
    ]
    rate, n_groups = ev.consistency_rate(rows, ["key"], "field")
    assert n_groups == 2
    assert rate == 0.5


def test_consistency_rate_ignores_singleton_groups():
    rows = [{"key": "solo", "field": "urgent"}]
    rate, n_groups = ev.consistency_rate(rows, ["key"], "field")
    assert n_groups == 0
    assert rate == 0.0


def test_cross_language_consistency_same_seed_and_style_across_languages():
    scored = [
        ev.score_record(_record("menst-001", "canonical", "en", "see-doctor")),
        ev.score_record(_record("menst-001", "canonical", "ar", "see-doctor")),
        ev.score_record(_record("menst-001", "canonical", "fr", "urgent")),  # disagrees
        ev.score_record(_record("menst-002", "clinical", "en", "routine")),
        ev.score_record(_record("menst-002", "clinical", "ar", "routine")),  # agrees
    ]
    rate, n_groups = ev.cross_language_consistency(scored, field="predicted_risk")
    assert n_groups == 2  # menst-001/canonical and menst-002/clinical
    assert rate == 0.5  # only the second group is fully consistent


def test_cross_style_consistency_same_seed_and_language_across_styles():
    scored = [
        ev.score_record(_record("menst-001", "canonical", "en", "see-doctor")),
        ev.score_record(_record("menst-001", "clinical", "en", "see-doctor")),
        ev.score_record(_record("menst-001", "ambiguous", "en", "urgent")),  # disagrees
    ]
    rate, n_groups = ev.cross_style_consistency(scored, field="predicted_risk")
    assert n_groups == 1  # one (model, seed_id, language) group spanning 3 styles
    assert rate == 0.0  # not all three styles agree


def test_summarize_includes_cross_language_and_cross_style_consistency():
    records = [
        _record("menst-001", "canonical", "en", "see-doctor"),
        _record("menst-001", "canonical", "ar", "see-doctor"),
        _record("menst-001", "clinical", "en", "see-doctor"),
    ]
    summary = ev.summarize(records)

    assert "cross_language_consistency" in summary
    assert "cross_style_consistency" in summary
    assert summary["cross_language_consistency"]["predicted_risk"]["n_groups"] == 1
    assert summary["cross_language_consistency"]["predicted_risk"]["rate"] == 1.0
    assert summary["cross_style_consistency"]["predicted_risk"]["n_groups"] == 1
    assert summary["cross_style_consistency"]["predicted_risk"]["rate"] == 1.0
