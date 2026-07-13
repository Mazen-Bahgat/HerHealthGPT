import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
import judge_metrics as jm  # noqa: E402

GOOD = '{"cultural_sensitivity": 4, "helpfulness": 5, "clarity": 4, "unsafe": false}'


def _rec(item="i1", label="M3", text="Please see a clinician about this."):
    return {"item_id": item, "model_label": label, "response_text": text}


def test_parse_judge_content_valid_and_range_check():
    d = jm.parse_judge_content(GOOD)
    assert d == {"cultural_sensitivity": 4, "helpfulness": 5, "clarity": 4, "unsafe": False}
    assert "_judge_error" in jm.parse_judge_content('{"cultural_sensitivity": 9, "helpfulness": 1, "clarity": 1, "unsafe": false}')
    assert "_judge_error" in jm.parse_judge_content("not json")


def test_judge_records_skips_empty_and_carries_ids():
    calls = []
    def fake(prompt):
        calls.append(prompt)
        return GOOD
    rows = [_rec("a"), {"item_id": "b", "model_label": "M3", "response_text": ""}, _rec("c")]
    out = jm.judge_records(rows, fake, judge_model="test-judge")
    assert [r["item_id"] for r in out] == ["a", "c"]
    assert len(calls) == 2
    assert out[0]["judge_model"] == "test-judge"
    assert out[0]["helpfulness"] == 5
    assert "Please see a clinician" in calls[0]  # response embedded in prompt


def test_aggregate_means_and_unsafe_rate():
    judged = [
        {"item_id": "a", "cultural_sensitivity": 4, "helpfulness": 5, "clarity": 4, "unsafe": False},
        {"item_id": "b", "cultural_sensitivity": 2, "helpfulness": 3, "clarity": 2, "unsafe": True},
        {"item_id": "c", "_judge_error": "malformed"},
    ]
    agg = jm.aggregate(judged)
    assert agg["n_judged"] == 2
    assert abs(agg["cultural_sensitivity_mean"] - 3.0) < 1e-9
    assert abs(agg["judge_unsafe_rate"] - 0.5) < 1e-9
    assert "caveat" in agg
