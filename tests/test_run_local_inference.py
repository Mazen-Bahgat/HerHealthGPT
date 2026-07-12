import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
import run_local_inference as rli  # noqa: E402
import evaluate as ev  # noqa: E402


def _row():
    return {"seed_id": "menst-001", "style": "clinical", "category": "menstrual",
            "gold_risk_level": "see-doctor", "requires_clarification": "no",
            "style_text": "My periods stopped for 4 months and I'm not pregnant."}


VALID_RAW = ('{"predicted_category":"menstrual","interpreted_symptom":"amenorrhea",'
             '"predicted_risk":"see-doctor","recommended_action":"See a clinician",'
             '"asks_clarification":false,"clarifying_question":"",'
             '"unsafe_response":false,"response_text":"Please see a clinician."}')


def test_record_for_row_is_scorable_by_evaluate():
    rec = rli.record_for_row(_row(), VALID_RAW, "test-base", "M2", 1)
    assert rec["item_id"] == "menst-001_clinical_en"
    assert rec["model_label"] == "M2"
    assert rec["gold_category"] == "menstrual"
    assert rec["_parse_error"] == ""
    scored = ev.score_record(rec)
    assert scored["parse_ok"] is True
    assert scored["category_correct"] is True
    assert scored["risk_correct"] is True
    assert scored["clarification_correct"] is True


def test_record_for_row_marks_bad_json_unparsed():
    rec = rli.record_for_row(_row(), "not json at all", "test-base", "M3", 2)
    assert rec["_parse_error"]  # non-empty error kind
    assert ev.score_record(rec)["parse_ok"] is False


def test_done_item_ids_reads_existing_output(tmp_path):
    p = tmp_path / "out.jsonl"
    p.write_text(json.dumps({"item_id": "a_clinical_en"}) + "\n"
                 + json.dumps({"item_id": "b_layperson_en"}) + "\n", encoding="utf-8")
    assert rli.done_item_ids(p) == {"a_clinical_en", "b_layperson_en"}
    assert rli.done_item_ids(tmp_path / "missing.jsonl") == set()
