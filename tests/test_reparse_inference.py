import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
import reparse_inference as rp  # noqa: E402
from test_run_inference_contract import valid_prediction  # noqa: E402


def failed_record(raw_response: str) -> dict:
    return {
        "item_id": "menst-001_canonical_en",
        "seed_id": "menst-001",
        "model_label": "M3",
        "language": "en",
        "gold_category": "menstrual",
        "gold_risk_level": "see-doctor",
        "raw_response": raw_response,
        "_parse_error": "malformed_json",
        "_error": "Expecting ',' delimiter",
        "_unparsed_response": raw_response,
    }


def test_reparse_recovers_missing_brace_and_preserves_identity_and_gold():
    raw = json.dumps(valid_prediction())[:-1] + "\n"
    updated, recovered = rp.reparse_record(failed_record(raw))
    assert recovered is True
    assert updated["_parse_error"] == ""
    assert updated["_json_repaired"] is True
    assert updated["predicted_category"] == "pcos"
    assert updated["item_id"] == "menst-001_canonical_en"
    assert updated["gold_risk_level"] == "see-doctor"
    assert updated["raw_response"] == raw
    assert "_unparsed_response" not in updated


def test_reparse_leaves_successful_rows_untouched():
    record = {"item_id": "x", "_parse_error": "", "predicted_category": "pcos"}
    updated, recovered = rp.reparse_record(record)
    assert recovered is False
    assert updated is record


def test_reparse_keeps_unrecoverable_failures_intact():
    record = failed_record("not json at all")
    updated, recovered = rp.reparse_record(record)
    assert recovered is False
    assert updated["_parse_error"] == "malformed_json"
