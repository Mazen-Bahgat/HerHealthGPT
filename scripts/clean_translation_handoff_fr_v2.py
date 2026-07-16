"""Clean up encoding/typography defects found by post-delivery translator review
of the finalized French v2 handoff (``fr.csv``).

This is a second, independent cleanup pass on top of the 71 rules already
applied by ``apply_translation_handoff_fr_qa.py``. It fixes three defects a
professional French-translation review surfaced in the shipped file:

1. Literal ``\\u00a0`` JSON-escape artifacts that leaked verbatim into
   ``Answer_translated`` instead of being rendered as a plain space.
2. A stray literal backslash before a straight double quote in
   ``Question_translated`` (source garbled height notation ``5\\"2``), kept in
   some rows but correctly dropped in sibling rows.
3. Inconsistent apostrophe typography: French elisions (``l'``, ``d'``,
   ``j'``, ``qu'``, ``aujourd'``, ...) rendered with a straight ASCII
   apostrophe instead of the typographic ``'`` used everywhere else in the
   corpus, sometimes even within the same sentence.

Every fix asserts its expected before/after counts and raises
``CleanupError`` on any drift, so silent partial application is impossible.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


TRANSLATION_FIELDS = ("Question_translated", "Answer_translated")
EXPECTED_HEADER = (
    "row_id",
    "split",
    "Question",
    "Answer",
    "Topic",
    "Keywords",
    *TRANSLATION_FIELDS,
)
BASE = Path("Used_Datasets/Consolidated_Datasets/200_Seed_Dataset")
DEFAULT_TARGET = BASE / "translation_handoff_v2" / "fr.csv"
DEFAULT_REPORT = BASE / "translation_handoff_v2" / "fr_translation_qa_report.json"
DEFAULT_PROVENANCE = (
    BASE / "translation_handoff_v2" / "fr_translation_provenance.json"
)
EXPECTED_TARGET_SHA256 = (
    "939bc0c6d568c57b27ec09cfb1fa75fc34e9c6cb5ec473327f8549a80f7ea7f1"
)
EXPECTED_ROW_COUNT = 3_580

BACKSLASH = "\\"
LITERAL_NBSP_ESCAPE = BACKSLASH + "u00a0"
LITERAL_BACKSLASH_QUOTE = BACKSLASH + '"'

# Every straight apostrophe in the corpus is either a French elision that
# should use the typographic apostrophe, or part of the two deliberately
# preserved, garbled English tokens "Va's Difference" (kept verbatim because
# the source term itself is corrupted/ambiguous). "va" is intentionally
# excluded from this prefix list so those tokens are left untouched.
ELISION_PREFIXES = (
    "jusqu",
    "aujourd",
    "quelqu",
    "lorsqu",
    "puisqu",
    "qu",
    "l",
    "d",
    "j",
    "n",
    "s",
    "m",
    "c",
)
ELISION_RE = re.compile(
    r"\b(" + "|".join(ELISION_PREFIXES) + r")'", re.IGNORECASE
)
PRESERVED_TOKEN_RE = re.compile(r"\bva's\b", re.IGNORECASE)


class CleanupError(ValueError):
    """Raised when a cleanup fix cannot be applied exactly as verified."""


def fix_literal_nbsp_escape(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Replace leaked ``\\u00a0`` escape sequences with a plain space."""

    fixed = [dict(row) for row in rows]
    changed_row_ids: list[str] = []
    total = 0
    for row in fixed:
        for column in TRANSLATION_FIELDS:
            value = row[column]
            count = value.count(LITERAL_NBSP_ESCAPE)
            if not count:
                continue
            row[column] = value.replace(LITERAL_NBSP_ESCAPE, " ")
            total += count
            if row["row_id"] not in changed_row_ids:
                changed_row_ids.append(row["row_id"])

    if len(changed_row_ids) != 24 or total != 36:
        raise CleanupError(
            "literal \\u00a0 cleanup drift: expected 24 rows / 36 replacements, "
            f"found {len(changed_row_ids)} rows / {total} replacements"
        )
    for row in fixed:
        for column in TRANSLATION_FIELDS:
            if LITERAL_NBSP_ESCAPE in row[column]:
                raise CleanupError(
                    f"{row['row_id']} {column}: literal \\u00a0 survived cleanup"
                )
    return fixed, {
        "fix_id": "literal-nbsp-escape",
        "row_ids": changed_row_ids,
        "replacement_count": total,
    }


def fix_stray_backslash_quote(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Drop the stray backslash before a literal ``"`` (garbled height notation)."""

    fixed = [dict(row) for row in rows]
    changed_row_ids: list[str] = []
    total = 0
    for row in fixed:
        for column in TRANSLATION_FIELDS:
            value = row[column]
            count = value.count(LITERAL_BACKSLASH_QUOTE)
            if not count:
                continue
            row[column] = value.replace(LITERAL_BACKSLASH_QUOTE, '"')
            total += count
            if row["row_id"] not in changed_row_ids:
                changed_row_ids.append(row["row_id"])

    if len(changed_row_ids) != 3 or total != 3:
        raise CleanupError(
            "stray backslash-quote cleanup drift: expected 3 rows / 3 replacements, "
            f"found {len(changed_row_ids)} rows / {total} replacements"
        )
    for row in fixed:
        for column in TRANSLATION_FIELDS:
            if LITERAL_BACKSLASH_QUOTE in row[column]:
                raise CleanupError(
                    f"{row['row_id']} {column}: stray backslash-quote survived cleanup"
                )
    return fixed, {
        "fix_id": "stray-backslash-quote",
        "row_ids": changed_row_ids,
        "replacement_count": total,
    }


def fix_apostrophe_typography(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Normalize straight elision apostrophes to the corpus's typographic apostrophe."""

    fixed = [dict(row) for row in rows]
    changed_row_ids: list[str] = []
    total = 0
    for row in fixed:
        for column in TRANSLATION_FIELDS:
            value = row[column]
            new_value, count = ELISION_RE.subn(
                lambda match: match.group(1) + "’", value
            )
            if not count:
                continue
            row[column] = new_value
            total += count
            if row["row_id"] not in changed_row_ids:
                changed_row_ids.append(row["row_id"])

    if total != 5_411:
        raise CleanupError(
            f"apostrophe normalization drift: expected 5411 replacements, found {total}"
        )

    remaining = 0
    preserved_tokens = 0
    for row in fixed:
        for column in TRANSLATION_FIELDS:
            value = row[column]
            remaining += value.count("'")
            preserved_tokens += len(PRESERVED_TOKEN_RE.findall(value))
    if remaining != 12 or preserved_tokens != 12:
        raise CleanupError(
            "apostrophe normalization left unexpected straight apostrophes: "
            f"{remaining} remaining, {preserved_tokens} matching the preserved "
            "'Va's Difference' token (expected 12 and 12)"
        )
    return fixed, {
        "fix_id": "apostrophe-typography",
        "row_count": len(changed_row_ids),
        "replacement_count": total,
    }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_csv(path: Path) -> tuple[bytes, list[dict[str, str]]]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CleanupError(f"{path}: not valid UTF-8") from exc
    reader = csv.DictReader(text.splitlines(keepends=True))
    if tuple(reader.fieldnames or ()) != EXPECTED_HEADER:
        raise CleanupError(
            f"{path}: header drift; expected {EXPECTED_HEADER!r}, got {reader.fieldnames!r}"
        )
    rows = list(reader)
    if len(rows) != EXPECTED_ROW_COUNT:
        raise CleanupError(f"{path}: expected {EXPECTED_ROW_COUNT} rows, found {len(rows)}")
    return raw, rows


def _write_csv_atomic(path: Path, rows: list[dict[str, str]]) -> bytes:
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPECTED_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    raw = temporary.read_bytes()
    temporary.replace(path)
    return raw


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_cleanup(
    rows: list[dict[str, str]]
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    audit: list[dict[str, object]] = []
    rows, entry = fix_literal_nbsp_escape(rows)
    audit.append(entry)
    rows, entry = fix_stray_backslash_quote(rows)
    audit.append(entry)
    rows, entry = fix_apostrophe_typography(rows)
    audit.append(entry)
    return rows, audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fix leaked escape artifacts and apostrophe typography in the "
            "finalized French v2 handoff."
        )
    )
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="verify the locked target and every fix without writing files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    target_path = args.target.resolve()
    report_path = args.report.resolve()
    provenance_path = args.provenance.resolve()

    raw, rows = _read_csv(target_path)
    input_sha256 = _sha256_bytes(raw)
    if input_sha256 != EXPECTED_TARGET_SHA256:
        raise CleanupError(
            f"{target_path}: SHA-256 drift; expected {EXPECTED_TARGET_SHA256}, "
            f"got {input_sha256}"
        )

    corrected, audit = run_cleanup(rows)

    for before, after in zip(rows, corrected, strict=True):
        for field in EXPECTED_HEADER[:6]:
            if before[field] != after[field]:
                raise CleanupError(f"{before['row_id']}: cleanup changed immutable {field}")
        if not after["Question_translated"].strip() or not after["Answer_translated"].strip():
            raise CleanupError(f"{before['row_id']}: cleanup produced a blank cell")

    changed_cells = {
        (before["row_id"], field)
        for before, after in zip(rows, corrected, strict=True)
        for field in TRANSLATION_FIELDS
        if before[field] != after[field]
    }
    total_replacements = sum(
        int(entry.get("replacement_count", 0)) for entry in audit
    )
    print(
        f"verified 3 fail-closed cleanup fixes; {len(changed_cells)} cells changed; "
        f"{total_replacements} replacements"
    )
    if args.dry_run:
        return 0

    output_raw = _write_csv_atomic(target_path, corrected)
    output_sha256 = _sha256_bytes(output_raw)

    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {}
    report["post_delivery_cleanup"] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trigger": "professional French-translator review of the shipped fr.csv",
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
        "changed_cell_count": len(changed_cells),
        "replacement_count": total_replacements,
        "fixes": audit,
    }
    _write_json_atomic(report_path, report)

    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance.setdefault("integrity", {})["post_cleanup_csv_sha256"] = output_sha256
        provenance.setdefault("review", {})["post_delivery_cleanup"] = {
            "status": "completed",
            "fixes_applied": len(audit),
            "changed_cells": len(changed_cells),
            "replacements": total_replacements,
            "report": report_path.relative_to(Path.cwd()).as_posix(),
        }
        _write_json_atomic(provenance_path, provenance)

    print(f"wrote cleaned CSV -> {target_path}")
    print(f"updated QA report -> {report_path}")
    if provenance_path.exists():
        print(f"updated provenance -> {provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
