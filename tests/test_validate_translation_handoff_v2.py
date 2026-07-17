import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import validate_translation_handoff_v2 as validator  # noqa: E402


FIELDS = [
    "row_id",
    "split",
    "Question",
    "Answer",
    "Topic",
    "Keywords",
    "Question_translated",
    "Answer_translated",
]


def _row(
    row_id: str,
    split: str,
    *,
    question: str = "What is menstruation?",
    answer: str = "Menstruation is a normal monthly process.",
    question_translated: str = "Qu'est-ce que la menstruation ?",
    answer_translated: str = "La menstruation est un processus mensuel normal.",
) -> dict[str, str]:
    return {
        "row_id": row_id,
        "split": split,
        "Question": question,
        "Answer": answer,
        "Topic": "Menstruation",
        "Keywords": "period, cycle",
        "Question_translated": question_translated,
        "Answer_translated": answer_translated,
    }


def _write_csv(
    path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str] | None = None,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _reference_rows() -> list[dict[str, str]]:
    train = _row("train-0000", "train")
    val = _row(
        "val-0000",
        "val",
        question="Can periods be painful?",
        answer="Periods can sometimes be painful.",
        question_translated="Les règles peuvent-elles être douloureuses ?",
        answer_translated="Les règles peuvent parfois être douloureuses.",
    )
    return [train, val]


def _blank_translations(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {**row, "Question_translated": "", "Answer_translated": ""}
        for row in rows
    ]


def _validate_single_row(tmp_path: Path, row: dict[str, str]):
    returned = tmp_path / "returned.csv"
    reference = tmp_path / "reference.csv"
    _write_csv(returned, [row])
    _write_csv(reference, _blank_translations([row]))
    return validator.validate_handoff(
        returned,
        reference_path=reference,
        expected_rows=1,
        min_french_signal_ratio=0,
    )


def test_valid_filled_handoff_matches_reference(tmp_path: Path):
    rows = _reference_rows()
    returned = tmp_path / "returned.csv"
    reference = tmp_path / "reference.csv"
    _write_csv(returned, rows)
    _write_csv(reference, _blank_translations(rows))

    report = validator.validate_handoff(
        returned,
        reference_path=reference,
        expected_rows=2,
    )

    assert report.ok, report.errors
    assert report.row_count == 2
    assert report.split_counts == {"train": 1, "val": 1}
    assert any("semantic accuracy" in warning for warning in report.warnings)


def test_rejects_wrong_column_order(tmp_path: Path):
    rows = [_reference_rows()[0]]
    returned = tmp_path / "returned.csv"
    reference = tmp_path / "reference.csv"
    wrong_order = FIELDS[:-2] + ["Answer_translated", "Question_translated"]
    _write_csv(returned, rows, wrong_order)
    _write_csv(reference, _blank_translations(rows))

    report = validator.validate_handoff(
        returned,
        reference_path=reference,
        expected_rows=1,
    )

    assert not report.ok
    assert any("header" in error and "exactly" in error for error in report.errors)


def test_rejects_a_filled_csv_as_a_non_pristine_reference(tmp_path: Path):
    rows = [_reference_rows()[0]]
    returned = tmp_path / "returned.csv"
    filled_reference = tmp_path / "filled-reference.csv"
    _write_csv(returned, rows)
    _write_csv(filled_reference, rows)

    with pytest.raises(validator.ValidationInputError, match="not pristine"):
        validator.validate_handoff(
            returned,
            reference_path=filled_reference,
            expected_rows=1,
        )


def test_rejects_reordering_and_immutable_source_edits(tmp_path: Path):
    expected = _reference_rows()
    changed = [dict(expected[1]), dict(expected[0])]
    changed[1]["Topic"] = "Changed topic"
    returned = tmp_path / "returned.csv"
    reference = tmp_path / "reference.csv"
    _write_csv(returned, changed)
    _write_csv(reference, _blank_translations(expected))

    report = validator.validate_handoff(
        returned,
        reference_path=reference,
        expected_rows=2,
    )

    assert any("row_id order" in error for error in report.errors)
    assert any("train-0000" in error and "Topic" in error for error in report.errors)


def test_rejects_duplicate_invalid_ids_split_mismatch_and_split_counts(tmp_path: Path):
    expected = _reference_rows()
    changed = [dict(expected[0]), dict(expected[1])]
    changed[0]["row_id"] = "training-0"
    changed[1]["row_id"] = "training-0"
    changed[1]["split"] = "train"
    returned = tmp_path / "returned.csv"
    reference = tmp_path / "reference.csv"
    _write_csv(returned, changed)
    _write_csv(reference, _blank_translations(expected))

    report = validator.validate_handoff(
        returned,
        reference_path=reference,
        expected_rows=2,
    )

    joined = "\n".join(report.errors)
    assert "duplicate row_id" in joined
    assert "invalid row_id" in joined
    assert "split counts" in joined


def test_rejects_blank_translations_and_inconsistent_repeated_answers(tmp_path: Path):
    expected = _reference_rows()
    expected[1]["Answer"] = expected[0]["Answer"]
    changed = [dict(row) for row in expected]
    changed[0]["Question_translated"] = "   "
    changed[1]["Answer_translated"] = "Il s'agit d'un processus mensuel normal."
    returned = tmp_path / "returned.csv"
    reference = tmp_path / "reference.csv"
    _write_csv(returned, changed)
    _write_csv(reference, _blank_translations(expected))

    report = validator.validate_handoff(
        returned,
        reference_path=reference,
        expected_rows=2,
    )

    joined = "\n".join(report.errors)
    assert "blank Question_translated" in joined
    assert "inconsistent Answer_translated" in joined


def test_rejects_invalid_utf8(tmp_path: Path):
    expected = [_reference_rows()[0]]
    returned = tmp_path / "returned.csv"
    reference = tmp_path / "reference.csv"
    returned.write_bytes(b"row_id,split,Question\ntrain-0000,train,\xff")
    _write_csv(reference, _blank_translations(expected))

    report = validator.validate_handoff(
        returned,
        reference_path=reference,
        expected_rows=1,
    )

    assert any("not valid UTF-8" in error for error in report.errors)


@pytest.mark.parametrize(
    ("bad_text", "expected_message"),
    [
        ("Une réponse avec\x00un NUL.", "NUL"),
        ("Une réponse avec le caractère \ufffd.", "U+FFFD"),
        ("Une rÃ©ponse manifestement mal encodÃ©e.", "mojibake"),
    ],
)
def test_rejects_text_encoding_red_flags(
    tmp_path: Path,
    bad_text: str,
    expected_message: str,
):
    expected = [_reference_rows()[0]]
    changed = [dict(expected[0])]
    changed[0]["Answer_translated"] = bad_text
    returned = tmp_path / "returned.csv"
    reference = tmp_path / "reference.csv"
    _write_csv(returned, changed)
    _write_csv(reference, _blank_translations(expected))

    report = validator.validate_handoff(
        returned,
        reference_path=reference,
        expected_rows=1,
    )

    assert any(expected_message in error for error in report.errors)


def test_allows_legitimate_french_capital_circumflex(tmp_path: Path):
    expected = [_reference_rows()[0]]
    changed = [dict(expected[0])]
    changed[0]["Question_translated"] = "Âge et menstruation : quel est le lien ?"
    changed[0]["Answer_translated"] = "Âge et cycle sont liés de manière complexe."
    returned = tmp_path / "returned.csv"
    reference = tmp_path / "reference.csv"
    _write_csv(returned, changed)
    _write_csv(reference, _blank_translations(expected))

    report = validator.validate_handoff(
        returned,
        reference_path=reference,
        expected_rows=1,
    )

    assert report.ok, report.errors


def test_rejects_translations_that_look_english(tmp_path: Path):
    expected = [_reference_rows()[0]]
    changed = [dict(expected[0])]
    changed[0]["Question_translated"] = "How does the monthly cycle normally work?"
    changed[0]["Answer_translated"] = "This monthly process is normal for most people."
    returned = tmp_path / "returned.csv"
    reference = tmp_path / "reference.csv"
    _write_csv(returned, changed)
    _write_csv(reference, _blank_translations(expected))

    report = validator.validate_handoff(
        returned,
        reference_path=reference,
        expected_rows=1,
    )

    joined = "\n".join(report.errors)
    assert "likely English" in joined
    assert "French-language signal" in joined


def test_allows_french_number_punctuation_when_digit_groups_are_preserved(
    tmp_path: Path,
):
    row = _row(
        "train-0000",
        "train",
        answer="Take 1,000 mg every 8.5 hours for 2 days.",
        answer_translated="Prenez 1 000 mg toutes les 8,5 heures pendant 2 jours.",
    )

    report = _validate_single_row(tmp_path, row)

    assert report.ok, report.errors


def test_warns_for_added_or_dropped_digit_groups_with_cell_location(tmp_path: Path):
    row = _row(
        "train-0000",
        "train",
        answer="Take 2 doses in 24 hours.",
        answer_translated="Prenez 3 doses en 48 heures.",
    )

    report = _validate_single_row(tmp_path, row)

    assert report.ok, report.errors
    matching = [
        warning for warning in report.warnings if "digit sequence mismatch" in warning
    ]
    assert matching
    assert "train-0000 Answer_translated" in matching[0]
    assert "missing" in matching[0] and "'2'" in matching[0] and "'24'" in matching[0]
    assert "added" in matching[0] and "'3'" in matching[0] and "'48'" in matching[0]


def test_rejects_changed_url_substrings_with_cell_location(tmp_path: Path):
    row = _row(
        "train-0000",
        "train",
        answer="Read https://example.org/medical-guide for details.",
        answer_translated="Consultez https://example.fr/medical-guide pour plus de details.",
    )

    report = _validate_single_row(tmp_path, row)

    matching = [error for error in report.errors if "URL mismatch" in error]
    assert matching
    assert "train-0000 Answer_translated" in matching[0]
    assert "https://example.org/medical-guide" in matching[0]
    assert "https://example.fr/medical-guide" in matching[0]


@pytest.mark.parametrize(
    ("answer", "answer_translated", "expected_message"),
    [
        (
            "First paragraph.\nSecond paragraph.",
            "Premier paragraphe. Deuxieme paragraphe.",
            "line-break count mismatch",
        ),
        (
            "First paragraph.\n\nSecond paragraph.",
            "Premier paragraphe.\nDeuxieme paragraphe.\nTroisieme paragraphe.",
            "blank-line count mismatch",
        ),
    ],
)
def test_rejects_changed_line_structure_with_cell_location(
    tmp_path: Path,
    answer: str,
    answer_translated: str,
    expected_message: str,
):
    row = _row(
        "train-0000",
        "train",
        answer=answer,
        answer_translated=answer_translated,
    )

    report = _validate_single_row(tmp_path, row)

    matching = [error for error in report.errors if expected_message in error]
    assert matching
    assert "train-0000 Answer_translated" in matching[0]


def test_warns_for_missing_source_question_or_exclamation_punctuation(tmp_path: Path):
    row = _row(
        "train-0000",
        "train",
        question="What helps?! Really?",
        question_translated="Qu'est-ce qui aide ? Vraiment ?",
    )

    report = _validate_single_row(tmp_path, row)

    assert report.ok, report.errors
    matching = [warning for warning in report.warnings if "'!' count mismatch" in warning]
    assert matching
    assert "train-0000 Question_translated" in matching[0]
    assert "expected 1" in matching[0] and "got 0" in matching[0]


def test_warns_for_sentence_punctuation_missing_immediately_after_a_url(
    tmp_path: Path,
):
    row = _row(
        "train-0000",
        "train",
        question="Is this page useful https://example.org/help?",
        question_translated="Cette page est-elle utile https://example.org/help",
    )

    report = _validate_single_row(tmp_path, row)

    assert report.ok, report.errors
    matching = [warning for warning in report.warnings if "'?' count mismatch" in warning]
    assert matching
    assert "train-0000 Question_translated" in matching[0]


def test_ignores_literal_unicode_escape_code_digits(tmp_path: Path):
    row = _row(
        "train-0000",
        "train",
        answer=r"Take 400\u00a0mg after reading \u219240% on the report.",
        answer_translated="Prenez 400 mg après avoir lu 40 % sur le compte rendu.",
    )

    report = _validate_single_row(tmp_path, row)

    assert report.ok, report.errors
    assert not any("digit sequence mismatch" in warning for warning in report.warnings)


def test_allows_french_punctuation_added_when_source_has_none(tmp_path: Path):
    row = _row(
        "train-0000",
        "train",
        answer="This is important.",
        answer_translated="C'est important !",
    )

    report = _validate_single_row(tmp_path, row)

    assert report.ok, report.errors


@pytest.mark.parametrize(
    ("english_acronym", "french_acronym"),
    [
        ("PCOS", "SOPK"),
        ("PCOD", "SOPK"),
        ("PMS", "SPM"),
        ("IVF", "FIV"),
        ("IUI", "IIU"),
    ],
)
def test_accepts_house_acronym_mappings_when_source_names_the_condition(
    tmp_path: Path,
    english_acronym: str,
    french_acronym: str,
):
    row = _row(
        "train-0000",
        "train",
        question=f"What is {english_acronym}?",
        question_translated=f"Qu'est-ce que {french_acronym} ?",
    )

    report = _validate_single_row(tmp_path, row)

    assert report.ok, report.errors


@pytest.mark.parametrize(
    ("english_acronym", "french_acronym"),
    [
        ("PCOS", "SOPK"),
        ("PCOD", "SOPK"),
        ("PMS", "SPM"),
        ("IVF", "FIV"),
        ("IUI", "IIU"),
    ],
)
def test_rejects_unmapped_english_acronyms_with_cell_location(
    tmp_path: Path,
    english_acronym: str,
    french_acronym: str,
):
    row = _row(
        "train-0000",
        "train",
        question=f"What is {english_acronym}?",
        question_translated=f"Qu'est-ce que {english_acronym} ?",
    )

    report = _validate_single_row(tmp_path, row)

    matching = [error for error in report.errors if "house acronym mismatch" in error]
    assert matching
    assert "train-0000 Question_translated" in matching[0]
    assert english_acronym in matching[0]
    assert french_acronym in matching[0]


def test_rejects_either_english_pcos_variant_when_sopk_is_required(tmp_path: Path):
    row = _row(
        "train-0000",
        "train",
        question="What is PCOS?",
        question_translated="Qu'est-ce que le SOPK (PCOD) ?",
    )

    report = _validate_single_row(tmp_path, row)

    matching = [error for error in report.errors if "house acronym mismatch" in error]
    assert matching
    assert "PCOD" in matching[0]
    assert "must be absent" in matching[0]


def test_does_not_require_an_acronym_for_an_unnamed_condition(tmp_path: Path):
    row = _row(
        "train-0000",
        "train",
        question="What can cause irregular ovulation?",
        question_translated="Qu'est-ce qui peut provoquer une ovulation irreguliere ?",
    )

    report = _validate_single_row(tmp_path, row)

    assert report.ok, report.errors


def test_acronym_rules_ignore_exact_urls(tmp_path: Path):
    row = _row(
        "train-0000",
        "train",
        answer="For PCOS, read https://example.org/PCOS.",
        answer_translated="Pour le SOPK, consultez https://example.org/PCOS.",
    )

    report = _validate_single_row(tmp_path, row)

    assert report.ok, report.errors


def test_warns_for_an_extreme_length_ratio_on_long_source_text(tmp_path: Path):
    row = _row(
        "train-0000",
        "train",
        answer=(
            "This detailed explanation contains enough source text to make a severe "
            "translation truncation suspicious."
        ),
        answer_translated="Oui.",
    )

    report = _validate_single_row(tmp_path, row)

    assert report.ok, report.errors
    matching = [warning for warning in report.warnings if "extreme length ratio" in warning]
    assert matching
    assert "train-0000 Answer_translated" in matching[0]


def test_warns_for_extreme_expansion_of_long_source_text(tmp_path: Path):
    row = _row(
        "train-0000",
        "train",
        answer="This source sentence is long enough for ratio review.",
        answer_translated=(
            "Cette traduction contient une expansion exceptionnellement longue qui "
            "repete beaucoup de contenu sans raison evidente et continue avec plusieurs "
            "phrases supplementaires afin de depasser tres largement la longueur du "
            "texte source fourni pour cette cellule de traduction francaise."
        ),
    )

    report = _validate_single_row(tmp_path, row)

    assert report.ok, report.errors
    matching = [warning for warning in report.warnings if "extreme length ratio" in warning]
    assert matching
    assert "train-0000 Answer_translated" in matching[0]


def test_does_not_warn_about_length_ratio_for_short_source_text(tmp_path: Path):
    row = _row(
        "train-0000",
        "train",
        answer="IUD.",
        answer_translated="Un dispositif intra-uterin utilise comme contraception.",
    )

    report = _validate_single_row(tmp_path, row)

    assert report.ok, report.errors
    assert not any("extreme length ratio" in warning for warning in report.warnings)


def test_can_reconstruct_expected_template_from_canonical_sources(tmp_path: Path):
    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    benchmark = tmp_path / "benchmark.csv"
    styled_fields = ["Question", "Answer", "Topic", "Keywords", "Style"]
    train_rows = [
        {"Question": "Leaky?", "Answer": "A", "Topic": "T", "Keywords": "K", "Style": "S"},
        {"Question": "Shared?", "Answer": "B", "Topic": "T", "Keywords": "K", "Style": "S"},
        {"Question": "Train only?", "Answer": "C", "Topic": "T", "Keywords": "K", "Style": "S"},
    ]
    val_rows = [
        {"Question": "shared?", "Answer": "D", "Topic": "T", "Keywords": "K", "Style": "S"},
        {"Question": "Also leaky?", "Answer": "E", "Topic": "T", "Keywords": "K", "Style": "S"},
        {"Question": "Val only?", "Answer": "F", "Topic": "T", "Keywords": "K", "Style": "S"},
    ]
    benchmark_rows = [
        {"Question": "leaky?", "Answer": "", "Topic": "", "Keywords": "", "Style": ""},
        {"Question": "also leaky?", "Answer": "", "Topic": "", "Keywords": "", "Style": ""},
    ]
    _write_csv(train, train_rows, styled_fields)
    _write_csv(val, val_rows, styled_fields)
    _write_csv(benchmark, benchmark_rows, styled_fields)

    returned_rows = [
        {
            **{key: train_rows[2][key] for key in styled_fields[:4]},
            "row_id": "train-0002",
            "split": "train",
            "Question_translated": "Une question réservée à l'entraînement ?",
            "Answer_translated": "La réponse est donnée en français.",
        },
        {
            **{key: val_rows[0][key] for key in styled_fields[:4]},
            "row_id": "val-0000",
            "split": "val",
            "Question_translated": "Une question partagée ?",
            "Answer_translated": "La réponse est aussi donnée en français.",
        },
        {
            **{key: val_rows[2][key] for key in styled_fields[:4]},
            "row_id": "val-0002",
            "split": "val",
            "Question_translated": "Une question réservée à la validation ?",
            "Answer_translated": "La réponse reste en français.",
        },
    ]
    returned = tmp_path / "returned.csv"
    _write_csv(returned, returned_rows)

    report = validator.validate_handoff(
        returned,
        train_path=train,
        val_path=val,
        benchmark_path=benchmark,
        expected_rows=3,
    )

    assert report.ok, report.errors
    assert report.split_counts == {"train": 1, "val": 2}


def test_cli_prints_actionable_failures(tmp_path: Path, capsys):
    expected = [_reference_rows()[0]]
    changed = [dict(expected[0])]
    changed[0]["Answer_translated"] = ""
    returned = tmp_path / "returned.csv"
    reference = tmp_path / "reference.csv"
    _write_csv(returned, changed)
    _write_csv(reference, _blank_translations(expected))

    exit_code = validator.main(
        [
            str(returned),
            "--reference",
            str(reference),
            "--expected-rows",
            "1",
        ]
    )

    output = capsys.readouterr()
    assert exit_code == 1
    assert "[ERROR]" in output.err
    assert "train-0000" in output.err
    assert "FAILED" in output.err


def test_cli_help_renders_without_argparse_formatting_error(capsys):
    with pytest.raises(SystemExit) as caught:
        validator.main(["--help"])

    output = capsys.readouterr()
    assert caught.value.code == 0
    assert "60%" in output.out
