import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
import build_ft_mix_v2 as bm  # noqa: E402
import run_inference as inf  # noqa: E402


def test_risk_heuristic_mapping():
    assert bm.risk_heuristic("Go to the ER immediately.") == "urgent"
    assert bm.risk_heuristic("You should talk to your doctor about this.") == "see-doctor"
    assert bm.risk_heuristic("This is a normal part of the cycle.") == "routine"
    assert bm.risk_heuristic("See a gynecologist; if severe bleeding occurs call 911") == "urgent"  # urgent wins


def _corpus_row(cat="menstrual", n=0, answer="Please see a doctor about this."):
    return {"instruction": "sys", "input": f"question {n}", "output": answer,
            "category": cat, "source_dataset": "MENST", "source_row_id": str(n)}


def test_make_json_example_is_schema_valid():
    rec = bm.make_json_example(_corpus_row())
    assert rec["messages"][0]["role"] == "user"          # eval-style: no system turn
    assert "Respond with ONLY a JSON object" in rec["messages"][0]["content"]
    obj = json.loads(rec["messages"][1]["content"])
    normalized, kind, _ = inf.validate_prediction_object(obj)
    assert normalized is not None, kind
    assert normalized["predicted_risk"] == "see-doctor"
    assert normalized["predicted_category"] == "menstrual"


def test_make_clarification_example_json_register():
    v = {"source_row_id": "MENST:1", "category": "pcos", "register": "json",
         "vague_question": "my hormones feel off??", "clarifying_question": "Which symptoms have you noticed, and for how long?", "chat_reply": None}
    rec = bm.make_clarification_example(v)
    obj = json.loads(rec["messages"][1]["content"])
    assert obj["asks_clarification"] is True
    assert obj["clarifying_question"] == v["clarifying_question"]
    normalized, kind, _ = inf.validate_prediction_object(obj)
    assert normalized is not None, kind


def test_build_counts_balance_and_determinism(tmp_path):
    corpus = ([_corpus_row("menstrual", i) for i in range(400)]
              + [_corpus_row("pcos", 1000 + i) for i in range(400)]
              + [_corpus_row("fertility", 2000 + i) for i in range(400)])
    clarifs = [{"source_row_id": f"MENST:{c}{i}", "category": cat, "register": "json" if i % 3 else "chat",
                "vague_question": f"vague {cat} {i}", "clarifying_question": f"clarify {cat} {i}?",
                "chat_reply": None if i % 3 else f"Could you tell me more, {i}? A clinician can help."}
               for cat, c in [("menstrual", 1), ("pcos", 2), ("fertility", 3)] for i in range(9)]
    styles = [{"source_row_id": f"MENST:{100 + i}", "category": "menstrual", "style": "layperson",
               "rewritten_question": f"rewritten {i}"} for i in range(6)]
    out1 = bm.build(corpus, clarifs, styles, seed=42)
    out2 = bm.build(corpus, clarifs, styles, seed=42)
    assert out1.keys() == {"train", "val", "probe"}
    assert json.dumps(out1["train"][:5]) == json.dumps(out2["train"][:5])  # deterministic
    kinds = [r.get("probe_kind") for r in out1["probe"]]
    assert "clarification" in kinds and "triage" in kinds
    probe_texts = {r["messages"][0]["content"] for r in out1["probe"]}
    train_texts = {r["messages"][0]["content"] for r in out1["train"]}
    assert not (probe_texts & train_texts)  # probe disjoint from train
