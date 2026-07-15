"""Validate a completed French v2 translation handoff CSV.

The validator checks the returned file against either an explicit pristine
handoff template (``--reference``) or an independently reconstructed template
from the canonical styled train/validation/benchmark CSVs.  Its French checks
are deliberately conservative heuristics; they cannot establish translation
meaning or medical accuracy.

Run against the repository's canonical sources::

    python scripts/validate_translation_handoff_v2.py path/to/returned_fr.csv

Run against a saved pristine template::

    python scripts/validate_translation_handoff_v2.py path/to/returned_fr.csv \
        --reference path/to/pristine_fr.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "Used_Datasets" / "Consolidated_Datasets" / "200_Seed_Dataset"
DEFAULT_TRAIN = DATA_DIR / "Train" / "train_canonical_styled.csv"
DEFAULT_VAL = DATA_DIR / "validate" / "validation_canonical_styled.csv"
DEFAULT_BENCHMARK = DATA_DIR / "Test" / "gold_seeds_styled_labeled.csv"

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
IMMUTABLE_FIELDS = FIELDS[:6]
TRANSLATION_FIELDS = FIELDS[6:]
DEFAULT_EXPECTED_ROWS = 3_580
DEFAULT_MIN_FRENCH_SIGNAL_RATIO = 0.60

ROW_ID_RE = re.compile(r"^(train|val)-\d{4}$")
WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿŒœÆæ]+", re.UNICODE)
FRENCH_SPECIFIC_RE = re.compile(r"[àâçéèêëîïôùûüÿœæ]", re.IGNORECASE)
MOJIBAKE_RE = re.compile(
    r"(?:Ã[\u0080-\u00bf]|Â[\u0080-\u00bf]|â€|ðŸ)"
)

DIGIT_SEQUENCE_RE = re.compile(r"\d+", re.UNICODE)
LITERAL_UNICODE_ESCAPE_RE = re.compile(r"\\u[0-9A-Fa-f]{4}")
LINE_BREAK_RE = re.compile(r"\r\n?|\n")
URL_RE = re.compile(r'''\b(?:https?://|www\.)[^\s<>"']+''', re.IGNORECASE)

SOURCE_TRANSLATION_FIELDS = (
    ("Question", "Question_translated"),
    ("Answer", "Answer_translated"),
)
HOUSE_ACRONYM_RULES = (
    (("PCOS", "PCOD"), "SOPK"),
    (("PMS",), "SPM"),
    (("IVF",), "FIV"),
    (("IUI",), "IIU"),
)
_ALL_HOUSE_ACRONYMS = {
    acronym
    for english_acronyms, french_acronym in HOUSE_ACRONYM_RULES
    for acronym in (*english_acronyms, french_acronym)
}
ACRONYM_PATTERNS = {
    acronym: re.compile(rf"(?<!\w){re.escape(acronym)}(?!\w)")
    for acronym in _ALL_HOUSE_ACRONYMS
}

# Character ratios are only a triage hint.  Short strings can legitimately
# expand substantially in French, so only conspicuously long/short results for
# a source cell with enough context generate a non-blocking warning.
LENGTH_RATIO_MIN_SOURCE_CHARS = 40
LENGTH_RATIO_LOW = 0.35
LENGTH_RATIO_HIGH = 3.0

# Function words make a useful corpus-level language signal.  The English set
# excludes ambiguous one-letter words such as "a", which is also French.
FRENCH_SIGNAL_WORDS = {
    "au",
    "aux",
    "avec",
    "ce",
    "ces",
    "cette",
    "comme",
    "comment",
    "dans",
    "de",
    "des",
    "doit",
    "doivent",
    "du",
    "elle",
    "elles",
    "en",
    "est",
    "et",
    "être",
    "il",
    "je",
    "la",
    "le",
    "les",
    "mais",
    "ne",
    "nous",
    "où",
    "par",
    "pas",
    "peut",
    "plus",
    "pour",
    "pourquoi",
    "quand",
    "que",
    "quel",
    "quelle",
    "quelles",
    "quels",
    "qui",
    "sa",
    "se",
    "ses",
    "son",
    "sont",
    "sur",
    "un",
    "une",
    "vos",
    "votre",
    "vous",
}
ENGLISH_SIGNAL_WORDS = {
    "and",
    "are",
    "be",
    "can",
    "does",
    "for",
    "from",
    "how",
    "is",
    "most",
    "of",
    "on",
    "should",
    "that",
    "the",
    "these",
    "this",
    "to",
    "what",
    "when",
    "which",
    "why",
    "with",
    "would",
    "you",
    "your",
}

SEMANTIC_WARNING = (
    "French-language checks are heuristic only; manual bilingual review is "
    "still required for semantic accuracy and medical fidelity."
)


class ValidationInputError(ValueError):
    """Raised when an expected/reference input cannot define the contract."""


@dataclass
class ValidationReport:
    """Machine-friendly result returned by :func:`validate_handoff`."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=lambda: [SEMANTIC_WARNING])
    row_count: int = 0
    split_counts: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class _ParsedHandoff:
    rows: list[dict[str, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _short(value: str, limit: int = 90) -> str:
    rendered = repr(value)
    return rendered if len(rendered) <= limit else rendered[: limit - 4] + "..." + rendered[-1]


def _decode_utf8(path: Path, label: str) -> tuple[str | None, list[str]]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, [f"{label} could not be read ({path}): {exc}"]
    try:
        text = raw.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        return None, [
            f"{label} is not valid UTF-8 at byte {exc.start}: {exc.reason} ({path})"
        ]

    errors: list[str] = []
    if "\x00" in text:
        errors.append(f"{label} contains a NUL byte/character; re-export it as clean UTF-8")
    if "\ufffd" in text:
        errors.append(
            f"{label} contains U+FFFD replacement characters, indicating lost or damaged text"
        )
    markers = sorted({match.group(0) for match in MOJIBAKE_RE.finditer(text)})
    if markers:
        errors.append(
            f"{label} contains likely mojibake marker(s) {markers!r}; check the UTF-8 export"
        )
    return text, errors


def _read_handoff(path: Path, label: str) -> _ParsedHandoff:
    text, errors = _decode_utf8(path, label)
    parsed = _ParsedHandoff(errors=errors)
    if text is None:
        return parsed

    try:
        records = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as exc:
        parsed.errors.append(f"{label} is not valid CSV: {exc}")
        return parsed
    if not records:
        parsed.errors.append(f"{label} is empty; expected the eight-column header and data rows")
        return parsed

    header = records[0]
    if header != FIELDS:
        parsed.errors.append(
            f"{label} header must be exactly {FIELDS!r} in that order; got {header!r}"
        )

    for record_number, record in enumerate(records[1:], start=2):
        if len(record) != len(FIELDS):
            parsed.errors.append(
                f"{label} CSV record {record_number} has {len(record)} columns; "
                f"expected exactly {len(FIELDS)}"
            )
            continue
        # Interpret by required position even after a header-order error so the
        # report can still expose additional damage in one run.
        parsed.rows.append(dict(zip(FIELDS, record, strict=True)))
    return parsed


def _read_styled_csv(
    path: Path,
    label: str,
    required_fields: Sequence[str],
) -> list[dict[str, str]]:
    text, decode_errors = _decode_utf8(path, label)
    if text is None or decode_errors:
        raise ValidationInputError("; ".join(decode_errors))
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        header = reader.fieldnames or []
        missing = [name for name in required_fields if name not in header]
        if missing:
            raise ValidationInputError(
                f"{label} is missing required column(s) {missing!r}; got {header!r}"
            )
        rows = list(reader)
    except csv.Error as exc:
        raise ValidationInputError(f"{label} is not valid CSV: {exc}") from exc
    for index, row in enumerate(rows, start=2):
        if None in row:
            raise ValidationInputError(
                f"{label} CSV record {index} has more cells than its header"
            )
    return rows


def _norm_question(value: str) -> str:
    return (value or "").strip().casefold()


def expected_rows_from_canonical(
    train_path: Path = DEFAULT_TRAIN,
    val_path: Path = DEFAULT_VAL,
    benchmark_path: Path = DEFAULT_BENCHMARK,
) -> list[dict[str, str]]:
    """Reconstruct the builder's leakage-clean rows without importing it."""

    styled_fields = ["Question", "Answer", "Topic", "Keywords"]
    train_rows = _read_styled_csv(Path(train_path), "canonical train CSV", styled_fields)
    val_rows = _read_styled_csv(Path(val_path), "canonical validation CSV", styled_fields)
    benchmark_rows = _read_styled_csv(
        Path(benchmark_path), "canonical benchmark CSV", ["Question"]
    )
    benchmark_questions = {_norm_question(row["Question"]) for row in benchmark_rows}

    indexed_val = [
        (index, row)
        for index, row in enumerate(val_rows)
        if _norm_question(row["Question"]) not in benchmark_questions
    ]
    surviving_val_questions = {_norm_question(row["Question"]) for _, row in indexed_val}
    indexed_train = [
        (index, row)
        for index, row in enumerate(train_rows)
        if _norm_question(row["Question"]) not in benchmark_questions
        and _norm_question(row["Question"]) not in surviving_val_questions
    ]

    expected: list[dict[str, str]] = []
    for split, indexed_rows in (("train", indexed_train), ("val", indexed_val)):
        for index, row in indexed_rows:
            expected.append(
                {
                    "row_id": f"{split}-{index:04d}",
                    "split": split,
                    "Question": row["Question"],
                    "Answer": row["Answer"],
                    "Topic": row.get("Topic") or "",
                    "Keywords": row.get("Keywords") or "",
                }
            )
    return expected


def expected_rows_from_reference(reference_path: Path) -> list[dict[str, str]]:
    """Load the immutable portion of an explicit pristine handoff template."""

    parsed = _read_handoff(Path(reference_path), "reference template")
    if parsed.errors:
        raise ValidationInputError("Reference template is invalid: " + "; ".join(parsed.errors))
    filled_cells = [
        f"{row['row_id'] or '<missing row_id>'} {name}"
        for row in parsed.rows
        for name in TRANSLATION_FIELDS
        if row[name].strip()
    ]
    if filled_cells:
        raise ValidationInputError(
            "Reference template is not pristine: translation columns must be empty; "
            f"found {len(filled_cells)} filled cell(s), e.g. {filled_cells[:10]!r}"
        )
    return [{name: row[name] for name in IMMUTABLE_FIELDS} for row in parsed.rows]


def _normalise_translation(value: str) -> str:
    return " ".join(value.split())


def _first_order_difference(actual: list[str], expected: list[str]) -> str:
    for position, (actual_id, expected_id) in enumerate(
        zip_longest(actual, expected, fillvalue="<missing>"), start=1
    ):
        if actual_id != expected_id:
            return (
                f"row_id order differs at data row {position}: "
                f"expected {expected_id!r}, got {actual_id!r}"
            )
    raise AssertionError("called only when row_id sequences differ")


def _check_identity_and_structure(
    actual_rows: list[dict[str, str]],
    expected_rows: list[dict[str, str]],
) -> list[str]:
    errors: list[str] = []
    actual_ids = [row["row_id"] for row in actual_rows]
    expected_ids = [row["row_id"] for row in expected_rows]

    duplicates = sorted(row_id for row_id, count in Counter(actual_ids).items() if count > 1)
    if duplicates:
        errors.append(
            f"duplicate row_id value(s): {duplicates[:10]!r}"
            + (f" (and {len(duplicates) - 10} more)" if len(duplicates) > 10 else "")
        )

    invalid_ids = sorted({row_id for row_id in actual_ids if not ROW_ID_RE.fullmatch(row_id)})
    if invalid_ids:
        errors.append(
            "invalid row_id value(s); expected 'train-NNNN' or 'val-NNNN': "
            f"{invalid_ids[:10]!r}"
        )

    split_mismatches = []
    for row in actual_rows:
        match = ROW_ID_RE.fullmatch(row["row_id"])
        if match and row["split"] != match.group(1):
            split_mismatches.append(
                f"{row['row_id']} has split {row['split']!r}, expected {match.group(1)!r}"
            )
    if split_mismatches:
        errors.append(
            f"{len(split_mismatches)} row_id/split mismatch(es), e.g. "
            + "; ".join(split_mismatches[:5])
        )

    if actual_ids != expected_ids:
        errors.append(_first_order_difference(actual_ids, expected_ids))

    expected_by_id = {row["row_id"]: row for row in expected_rows}
    unknown_ids = sorted({row_id for row_id in actual_ids if row_id not in expected_by_id})
    if unknown_ids:
        errors.append(
            f"row_id value(s) absent from the expected template: {unknown_ids[:10]!r}"
        )

    mismatch_messages: list[str] = []
    mismatch_count = 0
    for row in actual_rows:
        expected = expected_by_id.get(row["row_id"])
        if expected is None:
            continue
        for name in IMMUTABLE_FIELDS:
            if row[name] != expected[name]:
                mismatch_count += 1
                if len(mismatch_messages) < 20:
                    mismatch_messages.append(
                        f"{row['row_id']} field {name!r}: expected {_short(expected[name])}, "
                        f"got {_short(row[name])}"
                    )
    if mismatch_count:
        suffix = (
            f"; {mismatch_count - len(mismatch_messages)} more mismatch(es) omitted"
            if mismatch_count > len(mismatch_messages)
            else ""
        )
        errors.append(
            f"{mismatch_count} immutable-field mismatch(es): "
            + "; ".join(mismatch_messages)
            + suffix
        )
    return errors


def _check_translations(rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    for name in TRANSLATION_FIELDS:
        blank_ids = [row["row_id"] or f"data-row-{index}" for index, row in enumerate(rows, 1) if not row[name].strip()]
        if blank_ids:
            errors.append(
                f"{len(blank_ids)} row(s) have blank {name}, e.g. {blank_ids[:10]!r}"
            )

    translations_by_answer: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        translated = _normalise_translation(row["Answer_translated"])
        if translated:
            translations_by_answer[row["Answer"]][translated].append(row["row_id"])

    inconsistent = []
    for english_answer, variants in translations_by_answer.items():
        if len(variants) > 1:
            row_ids = [row_id for ids in variants.values() for row_id in ids]
            inconsistent.append(
                f"English Answer {_short(english_answer, 70)} has {len(variants)} translations "
                f"across row_id(s) {row_ids[:8]!r}"
            )
    if inconsistent:
        errors.append(
            f"{len(inconsistent)} repeated English Answer value(s) have inconsistent "
            f"Answer_translated text: {'; '.join(inconsistent[:10])}"
        )
    return errors


def _trim_url_match(raw_url: str) -> str:
    url = raw_url.rstrip(".,;:!?")
    # Sentence punctuation commonly closes a URL in parentheses or brackets.
    # Preserve balanced closing characters that are part of the URL, but
    # discard unmatched closers belonging to surrounding text.
    for opener, closer in (("(", ")"), ("[", "]"), ("{", "}")):
        while url.endswith(closer) and url.count(closer) > url.count(opener):
            url = url[:-1]
    return url


def _url_substrings(value: str) -> list[str]:
    return [
        url
        for match in URL_RE.finditer(value)
        if (url := _trim_url_match(match.group(0)))
    ]


def _without_urls(value: str) -> str:
    def preserve_sentence_punctuation(match: re.Match[str]) -> str:
        raw_url = match.group(0)
        url = _trim_url_match(raw_url)
        return " " + raw_url[len(url) :]

    return URL_RE.sub(preserve_sentence_punctuation, value)


def _line_counts(value: str) -> tuple[int, int]:
    lines = LINE_BREAK_RE.split(value)
    line_breaks = len(lines) - 1
    # A blank line is an empty/whitespace-only logical line between two line
    # breaks. A single leading or trailing newline is not itself a blank line.
    blank_lines = sum(not line.strip() for line in lines[1:-1])
    return line_breaks, blank_lines


def _counter_difference(expected: Counter[str], actual: Counter[str]) -> list[str]:
    return list((expected - actual).elements())


def _digit_sequences(value: str) -> Counter[str]:
    """Return visible digit groups, excluding digits inside literal ``\\uXXXX`` syntax."""

    return Counter(DIGIT_SEQUENCE_RE.findall(LITERAL_UNICODE_ESCAPE_RE.sub("", value)))


def _contains_acronym(value: str, acronym: str) -> bool:
    return bool(ACRONYM_PATTERNS[acronym].search(value))


def _compact_length(value: str) -> int:
    return len(" ".join(value.split()))


def _check_source_translation_invariants(
    rows: list[dict[str, str]],
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    for index, row in enumerate(rows, start=1):
        row_id = row["row_id"] or f"data-row-{index}"
        for source_name, translated_name in SOURCE_TRANSLATION_FIELDS:
            source = row[source_name]
            translated = row[translated_name]
            if not translated.strip():
                continue
            location = f"{row_id} {translated_name}"

            source_digits = _digit_sequences(source)
            translated_digits = _digit_sequences(translated)
            if source_digits != translated_digits:
                missing = _counter_difference(source_digits, translated_digits)
                added = _counter_difference(translated_digits, source_digits)
                warnings.append(
                    f"{location}: digit sequence mismatch versus {source_name}; "
                    f"missing {missing!r}, added {added!r}. French separators may change, "
                    "and number words or source artifacts may explain this; review numeric meaning"
                )

            source_urls = Counter(_url_substrings(source))
            translated_urls = Counter(_url_substrings(translated))
            if source_urls != translated_urls:
                missing = _counter_difference(source_urls, translated_urls)
                added = _counter_difference(translated_urls, source_urls)
                errors.append(
                    f"{location}: URL mismatch versus {source_name}; exact URL "
                    f"substring(s) missing {missing!r}, added {added!r}"
                )

            source_breaks, source_blank_lines = _line_counts(source)
            translated_breaks, translated_blank_lines = _line_counts(translated)
            if source_breaks != translated_breaks:
                errors.append(
                    f"{location}: line-break count mismatch versus {source_name}; "
                    f"expected {source_breaks}, got {translated_breaks}"
                )
            if source_blank_lines != translated_blank_lines:
                errors.append(
                    f"{location}: blank-line count mismatch versus {source_name}; "
                    f"expected {source_blank_lines}, got {translated_blank_lines}"
                )

            source_prose = _without_urls(source)
            translated_prose = _without_urls(translated)
            for punctuation in ("?", "!"):
                expected_count = source_prose.count(punctuation)
                actual_count = translated_prose.count(punctuation)
                if expected_count and actual_count != expected_count:
                    warnings.append(
                        f"{location}: {punctuation!r} count mismatch versus {source_name}; "
                        f"expected {expected_count}, got {actual_count}; review interrogative "
                        "or emotional force"
                    )

            for english_acronyms, french_acronym in HOUSE_ACRONYM_RULES:
                source_acronyms = [
                    acronym
                    for acronym in english_acronyms
                    if _contains_acronym(source_prose, acronym)
                ]
                if not source_acronyms:
                    continue
                french_missing = not _contains_acronym(
                    translated_prose, french_acronym
                )
                retained_acronyms = [
                    acronym
                    for acronym in english_acronyms
                    if _contains_acronym(translated_prose, acronym)
                ]
                if french_missing or retained_acronyms:
                    reasons = []
                    if french_missing:
                        reasons.append(f"French acronym {french_acronym} is missing")
                    if retained_acronyms:
                        reasons.append(
                            f"English acronym(s) {retained_acronyms!r} are retained"
                        )
                    forbidden = "/".join(english_acronyms)
                    errors.append(
                        f"{location}: house acronym mismatch for source "
                        f"{'/'.join(source_acronyms)}; expected {french_acronym}; "
                        f"{forbidden} must be absent from translated prose "
                        f"({'; '.join(reasons)})"
                    )

            source_length = _compact_length(source)
            translated_length = _compact_length(translated)
            if source_length >= LENGTH_RATIO_MIN_SOURCE_CHARS and translated_length:
                ratio = translated_length / source_length
                if ratio < LENGTH_RATIO_LOW or ratio > LENGTH_RATIO_HIGH:
                    warnings.append(
                        f"{location}: extreme length ratio versus {source_name}: "
                        f"{translated_length}/{source_length} characters ({ratio:.2f}x); "
                        "review for possible truncation or unintended expansion"
                    )

    return errors, warnings


def _tokens(value: str) -> list[str]:
    return [token.casefold() for token in WORD_RE.findall(value)]


def _check_french_language(
    rows: list[dict[str, str]],
    min_signal_ratio: float,
) -> list[str]:
    errors: list[str] = []
    unchanged: list[str] = []
    likely_english: list[str] = []
    assessable = 0
    french_signal = 0

    for row in rows:
        for english_name, translated_name in SOURCE_TRANSLATION_FIELDS:
            translated = row[translated_name].strip()
            if not translated:
                continue
            english = row[english_name].strip()
            location = f"{row['row_id']} {translated_name}"
            if (
                _normalise_translation(translated).casefold()
                == _normalise_translation(english).casefold()
                and any(character.isalpha() for character in english)
            ):
                unchanged.append(location)

            words = _tokens(translated)
            if len(words) < 3:
                continue
            assessable += 1
            french_word_count = sum(word in FRENCH_SIGNAL_WORDS for word in words)
            english_word_count = sum(word in ENGLISH_SIGNAL_WORDS for word in words)
            has_french_signal = bool(french_word_count or FRENCH_SPECIFIC_RE.search(translated))
            if has_french_signal:
                french_signal += 1
            if english_word_count >= 2 and english_word_count > french_word_count:
                likely_english.append(location)

    if unchanged:
        errors.append(
            f"{len(unchanged)} translation cell(s) are unchanged from English, e.g. "
            f"{unchanged[:10]!r}"
        )
    if likely_english:
        errors.append(
            f"{len(likely_english)} translation cell(s) look likely English rather than French, "
            f"e.g. {likely_english[:10]!r}"
        )
    if assessable:
        ratio = french_signal / assessable
        if ratio < min_signal_ratio:
            errors.append(
                f"French-language signal appears in {french_signal}/{assessable} assessable "
                f"translation cells ({ratio:.1%}); expected at least {min_signal_ratio:.1%}. "
                "This is a heuristic language check, not semantic validation."
            )
    return errors


def validate_handoff(
    returned_path: Path,
    *,
    reference_path: Path | None = None,
    train_path: Path = DEFAULT_TRAIN,
    val_path: Path = DEFAULT_VAL,
    benchmark_path: Path = DEFAULT_BENCHMARK,
    expected_rows: int = DEFAULT_EXPECTED_ROWS,
    min_french_signal_ratio: float = DEFAULT_MIN_FRENCH_SIGNAL_RATIO,
) -> ValidationReport:
    """Validate ``returned_path`` and return all detectable problems.

    ``reference_path`` takes precedence over the canonical source arguments.
    The default row count is intentionally fixed at 3,580 so unexpected source
    drift cannot silently redefine an acceptable handoff.
    """

    if expected_rows < 0:
        raise ValidationInputError("expected_rows must be zero or greater")
    if not 0.0 <= min_french_signal_ratio <= 1.0:
        raise ValidationInputError("min_french_signal_ratio must be between 0 and 1")

    parsed = _read_handoff(Path(returned_path), "returned CSV")
    report = ValidationReport(errors=list(parsed.errors), row_count=len(parsed.rows))
    actual_rows = parsed.rows

    if reference_path is not None:
        expected = expected_rows_from_reference(Path(reference_path))
    else:
        expected = expected_rows_from_canonical(
            Path(train_path), Path(val_path), Path(benchmark_path)
        )

    if len(expected) != expected_rows:
        report.errors.append(
            f"expected template/source reconstruction has {len(expected)} rows, but the "
            f"configured contract requires {expected_rows}"
        )
    if len(actual_rows) != expected_rows:
        report.errors.append(
            f"returned CSV has {len(actual_rows)} data rows; expected exactly {expected_rows}"
        )

    actual_split_counts = Counter(row["split"] for row in actual_rows)
    expected_split_counts = Counter(row["split"] for row in expected)
    report.split_counts = dict(sorted(actual_split_counts.items()))
    if actual_split_counts != expected_split_counts:
        report.errors.append(
            f"split counts differ: expected {dict(expected_split_counts)!r}, "
            f"got {dict(actual_split_counts)!r}"
        )

    report.errors.extend(_check_identity_and_structure(actual_rows, expected))
    report.errors.extend(_check_translations(actual_rows))
    invariant_errors, invariant_warnings = _check_source_translation_invariants(
        actual_rows
    )
    report.errors.extend(invariant_errors)
    report.warnings.extend(invariant_warnings)
    report.errors.extend(_check_french_language(actual_rows, min_french_signal_ratio))
    return report


def _ratio(value: str) -> float:
    try:
        ratio = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1") from exc
    if not 0.0 <= ratio <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return ratio


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a filled French v2 translation handoff. Structural and "
            "language checks do not replace bilingual semantic/medical review."
        )
    )
    parser.add_argument("returned_csv", type=Path, help="filled French handoff CSV")
    parser.add_argument(
        "--reference",
        type=Path,
        help=(
            "pristine eight-column handoff template; if omitted, reconstruct the "
            "expected rows from the canonical styled sources"
        ),
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=DEFAULT_EXPECTED_ROWS,
        help=f"exact required data-row count (default: {DEFAULT_EXPECTED_ROWS})",
    )
    parser.add_argument(
        "--min-french-signal-ratio",
        type=_ratio,
        default=DEFAULT_MIN_FRENCH_SIGNAL_RATIO,
        help=(
            "minimum fraction of assessable translation cells with a basic French "
            f"signal (default: {DEFAULT_MIN_FRENCH_SIGNAL_RATIO:.0%})".replace("%", "%%")
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_handoff(
            args.returned_csv,
            reference_path=args.reference,
            train_path=args.train,
            val_path=args.val,
            benchmark_path=args.benchmark,
            expected_rows=args.expected_rows,
            min_french_signal_ratio=args.min_french_signal_ratio,
        )
    except ValidationInputError as exc:
        print(f"[ERROR] Cannot establish expected handoff contract: {exc}", file=sys.stderr)
        print("FAILED: validator input/reference error", file=sys.stderr)
        return 2

    for error in report.errors:
        print(f"[ERROR] {error}", file=sys.stderr)
    for warning in report.warnings:
        print(f"[WARNING] {warning}", file=sys.stderr)

    if not report.ok:
        print(
            f"FAILED: {len(report.errors)} error(s), {len(report.warnings)} warning(s)",
            file=sys.stderr,
        )
        return 1

    splits = ", ".join(f"{name}={count}" for name, count in report.split_counts.items())
    print(f"PASS: {report.row_count} rows ({splits}) in {args.returned_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
