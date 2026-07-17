import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import convert_gold_seeds_styled as cgs

HEADER_NEW = ["Question", "Answer", "Topic", "Keywords", "Style",
              "gold_condition", "gold_risk_level", "gold_action",
              "requires_clarification"]
HEADER_OLD = ["Question", "Answer", "Topic", "Keywords", "Style",
              "gold_risk_level", "requires_clarification", "gold_condition"]


def _write_csv(path, header, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def _row(header, q, answer, style, action=""):
    base = {"Question": q, "Answer": answer, "Topic": "PCOS",
            "Keywords": "k", "Style": style, "gold_condition": "PCOS",
            "gold_risk_level": "see-doctor", "requires_clarification": "no"}
    if "gold_action" in header:
        base["gold_action"] = action
    return {k: base[k] for k in header}


def test_convert_new_layout_carries_gold_action(tmp_path):
    src = tmp_path / "labeled.csv"
    out = tmp_path / "bench.jsonl"
    _write_csv(src, HEADER_NEW, [
        _row(HEADER_NEW, "Q1 canonical?", "Answer A", "canonical", "see a gp"),
        _row(HEADER_NEW, "Q1 clinical?", "Answer A", "clinical", "see a gp"),
    ])
    n = cgs.convert(src, out)
    assert n == 2
    rows = [json.loads(l) for l in out.open(encoding="utf-8")]
    assert all(r["gold_action"] == "see a gp" for r in rows)
    assert rows[0]["seed_id"] == "gss-000"
    assert {r["style"] for r in rows} == {"canonical", "clinical"}


def test_convert_old_layout_defaults_gold_action_empty(tmp_path):
    src = tmp_path / "labeled.csv"
    out = tmp_path / "bench.jsonl"
    _write_csv(src, HEADER_OLD, [
        _row(HEADER_OLD, "Q1 canonical?", "Answer A", "canonical"),
    ])
    cgs.convert(src, out)
    rows = [json.loads(l) for l in out.open(encoding="utf-8")]
    assert rows[0]["gold_action"] == ""
