"""Re-parse failed rows in an existing inference JSONL using the current parser.

Why: M3's fine-tune induced a JSON quirk -- 149/540 English responses were
complete except for the final closing brace, all marked malformed_json at
generation time. parse_model_content now carries a conservative repair for
exactly that shape (see _attempt_json_repair). This script re-runs the parser
over the stored raw_response of previously-failed rows so the already-paid-for
generations are recovered without touching the GPU.

Only rows whose _parse_error is non-empty are touched; successful rows pass
through byte-identical. Recovered rows get _json_repaired=true so the paper
can report the repair rate honestly.

Usage:
    python scripts/reparse_inference.py --input HerHealthGPT-LU_seed/inference/M3_en.jsonl
        [--in-place]   # default writes <stem>_reparsed.jsonl next to the input
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_inference as ri  # noqa: E402

# Fields owned by the parser; everything else in the record (identity, gold
# labels, raw_response) is preserved as-is on a successful re-parse.
PARSER_FIELDS = sorted(ri.REQUIRED_PREDICTION_FIELDS) + [
    "_parse_error", "_error", "_unparsed_response", "_json_repaired",
]


def reparse_record(record: dict) -> tuple[dict, bool]:
    """Return (record, recovered). Non-failed rows are returned untouched."""
    if not record.get("_parse_error"):
        return record, False
    raw = record.get("raw_response")
    if not isinstance(raw, str) or not raw.strip():
        return record, False
    reparsed = ri.parse_model_content(raw)
    if reparsed.get("_parse_error"):
        return record, False  # still failing -- keep the original failure intact
    updated = {key: value for key, value in record.items() if key not in PARSER_FIELDS}
    updated.update(reparsed)
    updated.setdefault("_error", "")
    return updated, True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--in-place", action="store_true",
                        help="overwrite the input file instead of writing <stem>_reparsed.jsonl")
    args = parser.parse_args()

    records = [json.loads(line) for line in args.input.open(encoding="utf-8") if line.strip()]
    failed_before = sum(1 for r in records if r.get("_parse_error"))

    out_records, recovered = [], 0
    for record in records:
        updated, was_recovered = reparse_record(record)
        out_records.append(updated)
        recovered += was_recovered

    output = args.input if args.in_place else args.input.with_name(args.input.stem + "_reparsed.jsonl")
    with output.open("w", encoding="utf-8") as f:
        for record in out_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    failed_after = sum(1 for r in out_records if r.get("_parse_error"))
    print(f"{args.input.name}: {len(records)} rows, {failed_before} previously failed, "
          f"{recovered} recovered by repair, {failed_after} still failing")
    print(f"-> {output}")


if __name__ == "__main__":
    main()
