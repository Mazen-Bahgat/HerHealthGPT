import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_ft_data_v2 as prep


def _row(q, a="An answer.", topic="PCOS"):
    return {"Question": q, "Answer": a, "Topic": topic, "Keywords": "k"}


def test_leaked_questions_dropped_from_both_splits():
    bench = {prep.norm_q("Leaky train Q?"), prep.norm_q("Leaky val Q?")}
    train = [_row("Leaky train Q?"), _row("Clean train Q?")]
    val = [_row("Leaky val Q?"), _row("Clean val Q?")]
    ctrain, cval, log = prep.clean_splits(train, val, bench)
    assert [r["Question"] for r in ctrain] == ["Clean train Q?"]
    assert [r["Question"] for r in cval] == ["Clean val Q?"]
    reasons = {(l["split"], l["reason"]) for l in log}
    assert ("train", "benchmark_leak") in reasons
    assert ("val", "benchmark_leak") in reasons


def test_train_val_dup_dropped_from_train_val_wins():
    train = [_row("Shared Q?"), _row("Train only Q?")]
    val = [_row("shared q?")]  # case-insensitive match
    ctrain, cval, log = prep.clean_splits(train, val, set())
    assert [r["Question"] for r in ctrain] == ["Train only Q?"]
    assert [r["Question"] for r in cval] == ["shared q?"]
    assert log[0]["reason"] == "train_val_dup"
    assert log[0]["split"] == "train"


def test_degenerate_ambiguous_rows_dropped_from_both_splits():
    deg = _row("What is something? I'm not really sure how to explain it exactly.")
    deg["Style"] = "ambiguous"
    ok = _row("My period is late and I'm not sure what that means. What should I do?")
    ok["Style"] = "ambiguous"
    deg_val = _row("Why is something important? I'm not really sure how to explain it exactly.")
    deg_val["Style"] = "ambiguous"
    ctrain, cval, log = prep.clean_splits([deg, ok], [deg_val], set())
    assert [r["Question"] for r in ctrain] == [ok["Question"]]
    assert cval == []
    reasons = {(l["split"], l["reason"]) for l in log}
    assert ("train", "degenerate_ambiguous") in reasons
    assert ("val", "degenerate_ambiguous") in reasons


def test_degenerate_filter_only_applies_to_ambiguous_style():
    # same wording but a non-ambiguous style is kept (filter is style-gated)
    row = _row("What is something? I'm not really sure how to explain it exactly.")
    row["Style"] = "layperson"
    ctrain, _, log = prep.clean_splits([row], [], set())
    assert len(ctrain) == 1 and log == []


def test_chat_record_shape():
    rec = prep.to_chat_record(_row("Q?", "A."))
    roles = [m["role"] for m in rec["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert rec["messages"][0]["content"] == prep.SYSTEM_PROMPT
    assert rec["messages"][1]["content"] == "Q?"
    assert rec["messages"][2]["content"] == "A."
    assert rec["category"] == "pcos"
