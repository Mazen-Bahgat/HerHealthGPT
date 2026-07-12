import json
import pathlib
import sys

import pytest

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
{"predicted_category":"pcos","interpreted_symptom":"irregular cycles","predicted_risk":"routine","recommended_action":"track symptoms","asks_clarification":false,"clarifying_question":"","unsafe_response":false,"response_text":"Track your symptoms."}
```"""

    parsed = ri.parse_model_content(content)

    assert parsed["_parse_error"] == ""
    assert parsed["predicted_category"] == "pcos"
    assert parsed["predicted_risk"] == "routine"


def valid_prediction(**overrides):
    result = {
        "predicted_category": "pcos", "interpreted_symptom": "irregular cycles",
        "predicted_risk": "routine", "recommended_action": "track symptoms",
        "asks_clarification": False, "clarifying_question": "",
        "unsafe_response": False, "response_text": "Track your symptoms.",
    }
    result.update(overrides)
    return result


def test_call_endpoint_disables_thinking_by_default(monkeypatch):
    seen = {}
    class Response:
        def raise_for_status(self): pass
        def json(self): return {"choices": [{"message": {"content": "{}"}}]}
    def post(*args, **kwargs):
        seen.update(kwargs["json"])
        return Response()
    monkeypatch.setattr(ri.requests, "post", post)
    ri.call_endpoint("http://localhost/v1", "model", "key", "prompt", 1, 10)
    assert seen["chat_template_kwargs"] == {"enable_thinking": False}


def test_fixed_prompt_requires_exact_enum_values():
    assert '"predicted_risk" must be exactly one of' in ri.FIXED_PROMPT_TEMPLATE


@pytest.mark.parametrize("content", [None, 3, {"not": "text"}])
def test_parse_model_content_handles_null_missing_or_non_string(content):
    parsed = ri.parse_model_content(content)
    assert parsed["_parse_error"] == "response_contract_error"
    assert "content" in parsed["_error"]


@pytest.mark.parametrize("payload,error", [
    ("not json", "malformed_json"),
    ("[]", "non_object_json"),
    (json.dumps({"predicted_category": "pcos"}), "incomplete_schema"),
    (json.dumps(valid_prediction(predicted_category="cardiac")), "invalid_enum"),
    (json.dumps(valid_prediction(predicted_category="")), "invalid_enum"),
    (json.dumps(valid_prediction(predicted_risk="")), "invalid_enum"),
    (json.dumps(valid_prediction(asks_clarification="maybe")), "invalid_boolean"),
    (json.dumps(valid_prediction(unsafe_response="maybe")), "invalid_boolean"),
    (json.dumps(valid_prediction(response_text=3)), "incomplete_schema"),
])
def test_parse_model_content_strictly_validates_schema(payload, error):
    assert ri.parse_model_content(payload)["_parse_error"] == error


def test_select_input_text_rejects_empty_text():
    with pytest.raises(ValueError, match="non-empty"):
        ri.select_input_text({"text": "   "}, None, "en")


def test_resume_replaces_failed_record_without_duplicate(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps({"item_id": "x", "_parse_error": "malformed_json"}) + "\n")
    records = ri.load_resume_records(path)
    assert "x" not in ri.successful_item_ids(records)
    ri.upsert_output_record(path, records, {"item_id": "x", **valid_prediction(), "_parse_error": "", "_error": ""})
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["_parse_error"] == ""


@pytest.mark.parametrize("field,value", [
    ("predicted_category", ""), ("predicted_category", None),
    ("predicted_risk", ""), ("predicted_risk", "critical"),
    ("asks_clarification", "maybe"), ("unsafe_response", None),
    ("recommended_action", 4),
])
def test_success_validator_rejects_raw_schema_boundaries(field, value):
    record = {**valid_prediction(), "_parse_error": "", "_error": "", field: value}
    assert ri.is_successful_record(record) is False


def test_parse_model_content_normalizes_false_like_booleans():
    parsed = ri.parse_model_content(json.dumps(valid_prediction(
        asks_clarification="no", unsafe_response="0"
    )))
    assert parsed["_parse_error"] == ""
    assert parsed["asks_clarification"] is False
    assert parsed["unsafe_response"] is False


def test_main_retries_and_replaces_existing_empty_category(monkeypatch, tmp_path):
    benchmark = tmp_path / "benchmark.csv"
    benchmark.write_text(
        "seed_id,category,style,style_text,gold_risk_level,requires_clarification\n"
        "x,pcos,canonical,My cycles are irregular,routine,no\n",
        encoding="utf-8",
    )
    output = tmp_path / "predictions.jsonl"
    invalid = {
        "item_id": "x_canonical_en", **valid_prediction(predicted_category=""),
        "_parse_error": "", "_error": "",
    }
    output.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(ri, "call_endpoint", lambda *args: calls.append(args) or {
        "choices": [{"message": {"content": json.dumps(valid_prediction())}}]
    })
    monkeypatch.setattr(ri.time, "sleep", lambda _: None)
    monkeypatch.setattr(sys, "argv", [
        "run_inference.py", "--base-url", "http://local/v1", "--model", "m",
        "--model-label", "M", "--benchmark", str(benchmark), "--output", str(output),
        "--sleep", "0",
    ])

    ri.main()

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(calls) == 1
    assert len(rows) == 1
    assert rows[0]["predicted_category"] == "pcos"
    assert ri.is_successful_record(rows[0])
