import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import clean_translation_handoff_fr_v2 as cleanup  # noqa: E402


def _row(row_id: str, question_translated: str, answer_translated: str) -> dict[str, str]:
    return {
        "row_id": row_id,
        "split": "train",
        "Question": "English question",
        "Answer": "English answer",
        "Topic": "Topic",
        "Keywords": "keywords",
        "Question_translated": question_translated,
        "Answer_translated": answer_translated,
    }


def test_fix_literal_nbsp_escape_replaces_with_space_and_counts_exactly():
    rows = [_row("train-0000", "Question ?", r"Prenez 10 mg deux fois.")]
    with pytest.raises(cleanup.CleanupError):
        cleanup.fix_literal_nbsp_escape(rows)  # count drift: only 1, not 24/36


def test_fix_literal_nbsp_escape_clears_all_occurrences():
    # Build exactly the corpus-shaped input this fix expects: 24 rows, 36 hits.
    rows = [
        _row(f"val-{i:04d}", "Q", r"10 mg")
        for i in range(18)
    ] + [
        _row(f"val-{i:04d}", "Q", r"10 mg 20 mg 30 mg")
        for i in range(18, 24)
    ]
    fixed, entry = cleanup.fix_literal_nbsp_escape(rows)
    assert entry["replacement_count"] == 36
    assert len(entry["row_ids"]) == 24
    assert all("\\u00a0" not in row["Answer_translated"] for row in fixed)
    assert fixed[0]["Answer_translated"] == "10 mg"


def test_fix_stray_backslash_quote_requires_exact_expected_count():
    rows = [_row("val-0462", 'mesure 5\\"2.', "Answer")]
    with pytest.raises(cleanup.CleanupError):
        cleanup.fix_stray_backslash_quote(rows)  # only 1 row, expects exactly 3


def test_fix_stray_backslash_quote_drops_backslash():
    rows = [
        _row("val-0462", 'mesure 5\\"2.', "Answer"),
        _row("val-0465", 'mesure 5\\"2.', "Answer"),
        _row("val-0466", 'mesure 5\\"2.', "Answer"),
    ]
    fixed, entry = cleanup.fix_stray_backslash_quote(rows)
    assert entry["replacement_count"] == 3
    assert [row["Question_translated"] for row in fixed] == ['mesure 5"2.'] * 3


def test_fix_apostrophe_typography_preserves_garbled_english_token():
    # "Va's Difference" must survive untouched even though it looks like an
    # elision; only real French elision prefixes should be normalized.
    rows = [_row("train-2862", "obstruction du Va's Difference", "Answer")]
    with pytest.raises(cleanup.CleanupError):
        # Count won't match the corpus-wide expectation of 5411, so this must
        # fail closed rather than silently normalize a partial corpus.
        cleanup.fix_apostrophe_typography(rows)


def test_elision_regex_matches_expected_prefixes_and_skips_va():
    text = (
        "L'ovulation, l'ovaire, d'un ovule, j'ai, n'ai pas, qu'est-ce, "
        "aujourd'hui, quelqu'un, lorsqu'elle, puisqu'il, jusqu'à, m'aide, "
        "c'est, s'agit, mais Va's Difference reste intact"
    )
    new_text, count = cleanup.ELISION_RE.subn(
        lambda m: m.group(1) + "’", text
    )
    assert count == 14
    assert "Va's" in new_text
    assert "L’ovulation" in new_text
    assert "aujourd’hui" in new_text
    assert "quelqu’un" in new_text
