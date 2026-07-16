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


def test_degenerate_ids_drop_translated_rows_the_text_regex_cant_catch():
    # Simulates FR/AR: Question is already-translated text, so the English
    # "something"/"not really sure" regex can't match it -- only the
    # precomputed row_id set (from the EN source) can catch it.
    translated = _row("Qu'est-ce que quelque chose? Je ne suis pas sure.")
    translated["row_id"] = "train-0042"
    kept = _row("Quelle est la cause de mes crampes?")
    kept["row_id"] = "train-0043"
    ctrain, _, log = prep.clean_splits(
        [translated, kept], [], set(), degenerate_ids={"train-0042"}
    )
    assert [r["row_id"] for r in ctrain] == ["train-0043"]
    assert ("train", "degenerate_ambiguous") in {(l["split"], l["reason"]) for l in log}


def test_degenerate_ids_defaults_to_empty_when_omitted():
    # Backward compatibility: callers that don't pass degenerate_ids (e.g.
    # existing EN-only tests) get the same behavior as before this parameter
    # existed -- no row is dropped just for having an unmatched row_id.
    row = _row("Quelle est la cause de mes crampes?")
    row["row_id"] = "train-0099"
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


def test_json_record_non_ambiguous_is_valid_eval_schema():
    import json
    row = _row("How is PCOS diagnosed?", "A doctor will run blood tests and an ultrasound.")
    row["Style"] = "clinical"
    rec = prep.to_json_record(row)
    assert [m["role"] for m in rec["messages"]] == ["user", "assistant"]
    assert "Patient message:" in rec["messages"][0]["content"]  # FIXED_PROMPT_TEMPLATE
    obj = json.loads(rec["messages"][1]["content"])
    assert set(obj) == {"predicted_category", "interpreted_symptom", "predicted_risk",
                        "recommended_action", "asks_clarification", "clarifying_question",
                        "unsafe_response", "response_text"}
    assert obj["predicted_category"] == "pcos"
    assert obj["predicted_risk"] == "see-doctor"   # "doctor" in answer
    assert obj["asks_clarification"] is False
    assert obj["clarifying_question"] == ""


def test_oversample_ambiguous_repeats_only_ambiguous():
    amb = _row("Vague one?"); amb["Style"] = "ambiguous"
    direct = _row("Clear one?"); direct["Style"] = "clinical"
    out = prep.oversample_ambiguous([amb, direct], factor=3)
    styles = [r["Style"] for r in out]
    assert styles.count("ambiguous") == 3
    assert styles.count("clinical") == 1
    assert len(out) == 4


def test_oversample_factor_one_is_noop():
    amb = _row("Vague?"); amb["Style"] = "ambiguous"
    out = prep.oversample_ambiguous([amb], factor=1)
    assert out == [amb]


def test_json_record_ambiguous_asks_clarification():
    import json
    row = _row("My period is late and I'm not sure what that means. What should I do?",
               "Take a pregnancy test and see a doctor if it persists.")
    row["Style"] = "ambiguous"
    obj = json.loads(prep.to_json_record(row)["messages"][1]["content"])
    assert obj["asks_clarification"] is True
    assert obj["clarifying_question"].strip() != ""
    assert obj["response_text"] == obj["clarifying_question"]


<<<<<<< HEAD
def test_recover_styles_attaches_style_by_row_id():
    # Simulates the FR/AR handoff schema: rows carry row_id but no Style,
    # so clarify behaviour (oversampling + asks_clarification) silently
    # zeroes out unless Style is recovered from the EN styled source.
    handoff = {"row_id": "train-0004", "Question": "Question vague ?",
               "Answer": "Une réponse.", "Topic": "PCOS", "Keywords": "k"}
    styles = {"train-0004": "ambiguous"}
    out = prep.recover_styles([handoff], styles)
    assert out[0]["Style"] == "ambiguous"
    # and the recovered style drives both downstream clarify mechanisms
    assert len(prep.oversample_ambiguous(out, factor=2)) == 2
=======
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


def test_degenerate_ids_drop_translated_rows_the_text_regex_cant_catch():
    # Simulates FR/AR: Question is already-translated text, so the English
    # "something"/"not really sure" regex can't match it -- only the
    # precomputed row_id set (from the EN source) can catch it.
    translated = _row("Qu'est-ce que quelque chose? Je ne suis pas sure.")
    translated["row_id"] = "train-0042"
    kept = _row("Quelle est la cause de mes crampes?")
    kept["row_id"] = "train-0043"
    ctrain, _, log = prep.clean_splits(
        [translated, kept], [], set(), degenerate_ids={"train-0042"}
    )
    assert [r["row_id"] for r in ctrain] == ["train-0043"]
    assert ("train", "degenerate_ambiguous") in {(l["split"], l["reason"]) for l in log}


def test_degenerate_ids_defaults_to_empty_when_omitted():
    # Backward compatibility: callers that don't pass degenerate_ids (e.g.
    # existing EN-only tests) get the same behavior as before this parameter
    # existed -- no row is dropped just for having an unmatched row_id.
    row = _row("Quelle est la cause de mes crampes?")
    row["row_id"] = "train-0099"
    ctrain, _, log = prep.clean_splits([row], [], set())
    assert len(ctrain) == 1 and log == []


def test_attach_style_fills_only_missing_from_row_id_map():
    style_map = {"train-0001": "ambiguous", "train-0002": "clinical"}
    rows = [
        {"row_id": "train-0001", "Question": "q1"},                 # no Style -> filled
        {"row_id": "train-0002", "Style": "layperson", "Question": "q2"},  # has Style -> kept
        {"row_id": "train-9999", "Question": "q3"},                 # unknown -> empty
    ]
    out = prep.attach_style_by_row_id(rows, style_map)
    assert out[0]["Style"] == "ambiguous"
    assert out[1]["Style"] == "layperson"
    assert out[2]["Style"] == ""


def test_chat_record_shape():
    rec = prep.to_chat_record(_row("Q?", "A."))
    roles = [m["role"] for m in rec["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert rec["messages"][0]["content"] == prep.SYSTEM_PROMPT
    assert rec["messages"][1]["content"] == "Q?"
    assert rec["messages"][2]["content"] == "A."
    assert rec["category"] == "pcos"


def test_json_record_non_ambiguous_is_valid_eval_schema():
>>>>>>> b640ef475ca2efa0696587c7a6fbf5d413d74217
    import json
    obj = json.loads(prep.to_json_record(out[0])["messages"][1]["content"])
    assert obj["asks_clarification"] is True


def test_recover_styles_keeps_existing_style():
    row = {"row_id": "train-0001", "Question": "Q?", "Answer": "A.",
           "Topic": "PCOS", "Keywords": "k", "Style": "clinical"}
    out = prep.recover_styles([row], {"train-0001": "ambiguous"})
    assert out[0]["Style"] == "clinical"


def test_recover_styles_fails_on_unknown_row_id():
    import pytest
    row = {"row_id": "train-9999", "Question": "Q?", "Answer": "A.",
           "Topic": "PCOS", "Keywords": "k"}
    with pytest.raises(ValueError):
        prep.recover_styles([row], {"train-0001": "ambiguous"})
