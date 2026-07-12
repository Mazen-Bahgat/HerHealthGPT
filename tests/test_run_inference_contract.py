import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
import run_inference as ri  # noqa: E402


def test_build_output_record_preserves_gold_and_normalizes_prediction():
    row = {
        "seed_id": "menst-001",
        "category": "menstrual",
        "style": "ambiguous",
        "style_text": "Something feels off with my cycle.",
        "gold_risk_level": "see-doctor",
        "gold_action": "See a GP.",
        "gold_condition": "NEEDS_GROUNDING",
        "requires_clarification": "yes",
    }
    parsed = {
        "predicted_category": "Menstrual",
        "predicted_risk": "see doctor",
        "recommended_action": "See a doctor for assessment.",
        "asks_clarification": "True",
        "unsafe_response": "false",
        "response_text": "Can you share how long this has been happening?",
    }

    record = ri.build_output_record(
        row=row,
        parsed=parsed,
        raw_response=json.dumps(parsed),
        model="Qwen/Qwen3.5-9B",
        model_label="M2_zero_shot",
        language="en",
        row_number=1,
    )

    assert record["item_id"] == "menst-001_ambiguous_en"
    assert record["predicted_category"] == "menstrual"
    assert record["predicted_risk"] == "see-doctor"
    assert record["asks_clarification"] is True
    assert record["unsafe_response"] is False
    assert record["gold_category"] == "menstrual"
    assert record["gold_risk_level"] == "see-doctor"
    assert record["requires_clarification"] == "yes"
    assert record["raw_response"] == json.dumps(parsed)


def test_parse_model_content_extracts_json_from_fenced_response():
    content = """```json
{"predicted_category": "pcos", "predicted_risk": "routine"}
```"""

    parsed = ri.parse_model_content(content)

    assert parsed["_parse_error"] == ""
    assert parsed["predicted_category"] == "pcos"
    assert parsed["predicted_risk"] == "routine"
