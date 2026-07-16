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


def test_risk_heuristic_catches_genuine_consult_language():
    # Conservative expansion: answers that clearly advise clinical follow-up
    # without the literal word "doctor" were previously mislabeled "routine",
    # driving under-triage. Each of these should now map to see-doctor.
    consult_answers = [
        "It's best to consult a healthcare provider about these symptoms.",
        "A gynaecologist can help you figure out the cause.",  # British spelling
        "You may need to see a specialist for further evaluation.",
        "Testing is needed to diagnose the underlying condition.",
        "Please seek medical advice if this continues.",
        "It's worth getting this checked to rule out anything serious.",
        "Book an appointment to have it properly evaluated.",
        "A nurse or health professional can advise on next steps.",
    ]
    for a in consult_answers:
        assert bm.risk_heuristic(a) == "see-doctor", a


def test_risk_heuristic_keeps_genuinely_routine_routine():
    # Guard against over-broadening: everyday reassurance must stay routine.
    routine_answers = [
        "Mild cramps in the first days of your period are completely normal.",
        "Using a warm compress and staying hydrated usually helps.",
        "This is a common experience and typically settles on its own.",
    ]
    for a in routine_answers:
        assert bm.risk_heuristic(a) == "routine", a


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
    assert out1.keys() == {"train", "val", "probe", "stats"}
    assert json.dumps(out1["train"][:5]) == json.dumps(out2["train"][:5])  # deterministic
    kinds = [r.get("probe_kind") for r in out1["probe"]]
    assert "clarification" in kinds and "triage" in kinds
    probe_texts = {r["messages"][0]["content"] for r in out1["probe"]}
    train_texts = {r["messages"][0]["content"] for r in out1["train"]}
    assert not (probe_texts & train_texts)  # probe disjoint from train


def test_no_val_answers_in_train():
    corpus = ([_corpus_row("menstrual", i, answer=f"Please see a doctor about issue {i}.") for i in range(400)]
              + [_corpus_row("pcos", 1000 + i, answer=f"Please see a doctor about issue {1000 + i}.") for i in range(400)]
              + [_corpus_row("fertility", 2000 + i, answer=f"Please see a doctor about issue {2000 + i}.") for i in range(400)])
    # style rewrites pointing at EVERY corpus row; val-sourced ones must be dropped
    styles = [{"source_row_id": f"MENST:{r['source_row_id']}", "category": r["category"],
               "style": "layperson", "rewritten_question": f"rw {r['source_row_id']}"}
              for r in corpus]
    import prepare_ft_data as prep
    _, val_rows = prep.split_train_val(corpus, val_frac=0.05, seed=42)
    val_answers = {r["output"] for r in val_rows}
    out = bm.build(corpus, [], styles, seed=42)
    train_answers = {rec["messages"][-1]["content"] for rec in out["train"]}
    assert not (val_answers & train_answers)
