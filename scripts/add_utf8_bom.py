"""Prefix a UTF-8 CSV handoff file with a UTF-8 BOM.

The translation handoff CSVs (``fr.csv``, ``ar.csv``) are valid, clean UTF-8
with no BOM. Windows tools that guess encoding from content instead of
respecting a declared one --- notably Excel's default "Open" behavior, and
some CSV-preview extensions --- fall back to the system ANSI code page (e.g.
Windows-1252) for a CSV with no BOM. That misreads every non-ASCII byte
sequence, turning accented letters and guillemets into mojibake on screen
(for example "rgles" becomes "rÃ¨gles" and the French guillemets become
"Â«"/"Â»") even though the bytes on disk are untouched and already correct.

A UTF-8 BOM (the three bytes ``EF BB BF``) makes those tools detect UTF-8
reliably. It is invisible in editors and tools that already handle UTF-8
correctly (Python's ``csv`` module, this repo's own validator, VS Code, git
diff), so adding one only helps and never regresses anything downstream.

This script never alters a single character of content: it only prepends the
BOM, and only after verifying the input decodes as strict UTF-8 and round-trips
byte-for-byte once the BOM is stripped back off.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


UTF8_BOM = b"\xef\xbb\xbf"


class BomFixError(ValueError):
    """Raised when a target file cannot be safely BOM-prefixed."""


def add_bom(path: Path) -> bool:
    """Prepend a UTF-8 BOM to ``path`` if missing. Returns True if changed."""

    raw = path.read_bytes()
    try:
        raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise BomFixError(f"{path}: not valid UTF-8, refusing to touch it") from exc

    if raw.startswith(UTF8_BOM):
        return False

    new_raw = UTF8_BOM + raw
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_bytes(new_raw)
    written = temporary.read_bytes()
    if written != new_raw or written[len(UTF8_BOM):] != raw:
        temporary.unlink(missing_ok=True)
        raise BomFixError(f"{path}: write verification failed; original left untouched")
    temporary.replace(path)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prefix one or more UTF-8 CSV files with a UTF-8 BOM."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in args.paths:
        changed = add_bom(path)
        if changed:
            print(f"added UTF-8 BOM -> {path}")
        else:
            print(f"already has a UTF-8 BOM, left unchanged -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
