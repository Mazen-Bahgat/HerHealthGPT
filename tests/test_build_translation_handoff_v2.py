import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_translation_handoff_v2 as handoff  # noqa: E402
import prepare_ft_data_v2 as prep  # noqa: E402


def _row(q, a="An answer.", topic="PCOS"):
    return {"Question": q, "Answer": a, "Topic": topic, "Keywords": "k"}


def test_handoff_contains_only_cleaned_rows_with_stable_ids():
    bench = {prep.norm_q("Leaky Q?")}
    train = [_row("Leaky Q?"), _row("Keep train Q?")]
    val = [_row("Keep val Q?")]
    rows = handoff.build_handoff_rows(train, val, bench)
    ids = [r["row_id"] for r in rows]
    assert ids == ["train-0001", "val-0000"]  # index from the ORIGINAL file
    assert all(r["Question_translated"] == "" == r["Answer_translated"] for r in rows)
    assert {r["split"] for r in rows} == {"train", "val"}


def test_roundtrip_filled_template_ingests_as_translation():
    bench = set()
    train = [_row("Keep train Q?")]
    rows = handoff.build_handoff_rows(train, [], bench)
    rows[0]["Question_translated"] = "Question traduite ?"
    rows[0]["Answer_translated"] = "Une réponse."
    mapped = prep.apply_translation(rows)
    assert mapped[0]["Question"] == "Question traduite ?"
    assert mapped[0]["Answer"] == "Une réponse."
    rec = prep.to_chat_record(mapped[0])
    assert rec["messages"][1]["content"] == "Question traduite ?"
