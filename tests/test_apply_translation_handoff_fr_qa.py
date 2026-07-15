import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import apply_translation_handoff_fr_qa as qa  # noqa: E402


def _row(row_id: str, question: str, answer: str) -> dict[str, str]:
    return {
        "row_id": row_id,
        "split": "train",
        "Question": question,
        "Answer": answer,
        "Topic": "Topic",
        "Keywords": "keywords",
        "Question_translated": f"FR: {question}",
        "Answer_translated": "Ancienne réponse.",
    }


def test_answer_rule_expands_to_every_reused_source_answer():
    rows = [
        _row("train-0000", "Question one?", "Shared answer."),
        _row("train-0001", "Question two?", "Shared answer."),
        _row("train-0002", "Question three?", "Different answer."),
    ]
    before = copy.deepcopy(rows)
    rule = qa.Rule(
        rule_id="shared-answer",
        seed_row_ids=("train-0000",),
        column="Answer_translated",
        old="Ancienne",
        new="Nouvelle",
        expected_cells=2,
        expected_replacements=2,
        expand_reused_answer=True,
    )

    corrected, audit = qa.apply_rules(rows, [rule])

    assert corrected[0]["Answer_translated"] == "Nouvelle réponse."
    assert corrected[1]["Answer_translated"] == "Nouvelle réponse."
    assert corrected[2]["Answer_translated"] == "Ancienne réponse."
    assert audit[0]["row_ids"] == ["train-0000", "train-0001"]
    assert audit[0]["replacement_count"] == 2
    for original, changed in zip(before, corrected, strict=True):
        assert {
            key: value
            for key, value in original.items()
            if key not in qa.TRANSLATION_FIELDS
        } == {
            key: value
            for key, value in changed.items()
            if key not in qa.TRANSLATION_FIELDS
        }


def test_question_rule_changes_only_named_rows():
    rows = [
        _row("train-0000", "Question one?", "Answer one."),
        _row("train-0001", "Question two?", "Answer two."),
    ]
    rule = qa.Rule(
        rule_id="one-question",
        seed_row_ids=("train-0001",),
        column="Question_translated",
        old="FR:",
        new="Question française :",
        expected_cells=1,
        expected_replacements=1,
    )

    corrected, audit = qa.apply_rules(rows, [rule])

    assert corrected[0]["Question_translated"] == "FR: Question one?"
    assert corrected[1]["Question_translated"] == (
        "Question française : Question two?"
    )
    assert audit[0]["row_ids"] == ["train-0001"]


def test_all_rows_rule_changes_every_matching_translation_cell():
    rows = [
        _row("train-0000", "Question one?", "Answer one."),
        _row("train-0001", "Question two?", "Answer two."),
    ]
    rule = qa.Rule(
        rule_id="global-prefix",
        seed_row_ids=(),
        column="Question_translated",
        old="FR:",
        new="FR naturel :",
        expected_cells=2,
        expected_replacements=2,
        all_rows=True,
    )

    corrected, audit = qa.apply_rules(rows, [rule])

    assert all(
        row["Question_translated"].startswith("FR naturel :") for row in corrected
    )
    assert audit[0]["row_ids"] == ["train-0000", "train-0001"]


def test_rule_fails_closed_when_expected_text_or_count_drifts():
    rows = [_row("train-0000", "Question?", "Answer.")]
    rule = qa.Rule(
        rule_id="drift",
        seed_row_ids=("train-0000",),
        column="Answer_translated",
        old="Texte absent",
        new="Nouveau texte",
        expected_cells=1,
        expected_replacements=1,
    )

    with pytest.raises(qa.CorrectionError, match="drift"):
        qa.apply_rules(rows, [rule])


def test_rejects_non_translation_columns():
    rows = [_row("train-0000", "Question?", "Answer.")]
    rule = qa.Rule(
        rule_id="immutable",
        seed_row_ids=("train-0000",),
        column="Question",
        old="Question",
        new="Changed",
        expected_cells=1,
        expected_replacements=1,
    )

    with pytest.raises(qa.CorrectionError, match="translation column"):
        qa.apply_rules(rows, [rule])


def test_normalize_translation_line_whitespace_preserves_source_and_line_breaks():
    rows = [_row("train-0000", "Source line one?\nSource line two?", "Answer.")]
    rows[0]["Question_translated"] = "Ligne une.  \nLigne deux.\t\nLigne trois."
    rows[0]["Answer_translated"] = "Réponse. \r\nSuite."
    before = copy.deepcopy(rows)

    normalized, audit = qa.normalize_translation_line_whitespace(rows)

    assert normalized[0]["Question_translated"] == (
        "Ligne une.\nLigne deux.\nLigne trois."
    )
    assert normalized[0]["Answer_translated"] == "Réponse.\r\nSuite."
    assert normalized[0]["Question_translated"].count("\n") == 2
    assert normalized[0]["Answer_translated"].count("\n") == 1
    assert audit == [
        {
            "row_id": "train-0000",
            "column": "Question_translated",
            "trimmed_line_count": 2,
        },
        {
            "row_id": "train-0000",
            "column": "Answer_translated",
            "trimmed_line_count": 1,
        },
    ]
    for field in qa.EXPECTED_HEADER[:6]:
        assert normalized[0][field] == before[0][field]
