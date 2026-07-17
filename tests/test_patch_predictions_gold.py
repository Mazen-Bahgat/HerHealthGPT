import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import patch_predictions_gold as ppg


def _bench_row(seed_id="gss-000", style="canonical", q="How is PCOS diagnosed?"):
    return {"seed_id": seed_id, "style": style, "style_text": q,
            "category": "pcos", "gold_risk_level": "routine",
            "gold_action": "self-care ok", "gold_condition": "PCOS",
            "requires_clarification": "no"}


def _pred(seed_id="gss-000", style="canonical", q="How is PCOS diagnosed?"):
    return {"item_id": f"{seed_id}_{style}_en", "seed_id": seed_id,
            "style": style, "language": "en", "input_text": q,
            "gold_category": "pcos", "gold_risk_level": "see-doctor",
            "gold_action": "", "gold_condition": "old", "requires_clarification": "yes",
            "raw_response": "{...}", "predicted_risk": "routine"}


def test_patch_overwrites_gold_preserves_predictions():
    out = ppg.patch_records([_pred()], [_bench_row()])
    assert out[0]["gold_risk_level"] == "routine"
    assert out[0]["gold_action"] == "self-care ok"
    assert out[0]["gold_condition"] == "PCOS"
    assert out[0]["requires_clarification"] == "no"
    assert out[0]["raw_response"] == "{...}"
    assert out[0]["predicted_risk"] == "routine"


def test_patch_fails_on_unknown_item_id():
    with pytest.raises(ValueError, match="gss-999"):
        ppg.patch_records([_pred(seed_id="gss-999")], [_bench_row()])


def test_patch_fails_on_missing_prediction():
    bench = [_bench_row(), _bench_row(style="clinical", q="Clinical Q?")]
    with pytest.raises(ValueError, match="clinical"):
        ppg.patch_records([_pred()], bench)


def test_patch_fails_on_question_text_mismatch():
    with pytest.raises(ValueError, match="input_text"):
        ppg.patch_records([_pred(q="Different question?")], [_bench_row()])
