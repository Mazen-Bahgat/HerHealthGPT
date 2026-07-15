"""Generate the 5 style variants (+ canonical) for a row-range batch of
train_canonical.csv / validation_canonical.csv.

WHY A HEURISTIC, NOT AN LLM: OPENAI_API_KEY is set but still returns
insufficient_quota (no billing on the account) -- verified live before writing
this. Manually hand-authoring ~2000 variants in one sitting is not feasible
under the same-day deadline, so this uses a disclosed, deterministic wrapper
transform instead of per-content LLM paraphrase:

- clinical / layperson / indirect_cultural / emotionally_concerned: prefix the
  ORIGINAL question (verbatim, so all clinical content is preserved by
  construction) with a register-appropriate opener, rotated deterministically
  by row index so nearby rows don't all get the same opener.
- ambiguous: redact the row's own Topic/Keywords terms from the question
  (first occurrence each, case-insensitive) and truncate to the first
  sentence, so the variant is actually vaguer (missing the specific
  condition/detail) rather than just re-wrapped -- matching the style's
  definition (a triage nurse would need to ask a follow-up).

This is a silver/heuristic pass, same disclosure standard already used in
this repo for gold_risk_level etc. (see complete_gold_labels_gss.py). It
should get an LLM or human quality pass later if budget/time allows -- flagged
in the output filename and this docstring, not silently presented as
LLM-quality paraphrase.

Writes a SEPARATE output file (does not touch the original train_canonical.csv
/ validation_canonical.csv) so it doesn't collide with a teammate's concurrent
edits to their own batch of the same source file. A later assembly step
concatenates all batches into the final styled file.

Usage:
    python scripts/generate_style_variants_batch.py \
        --input Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/train_canonical.csv \
        --start 160 --end 480 \
        --output Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/train_canonical_batch2_3_styled.csv

--start/--end are 0-indexed data-row positions (row "162" in a 1-indexed file
with a header = index 160), end-exclusive.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

STYLES = ("canonical", "clinical", "layperson", "indirect_cultural", "ambiguous", "emotionally_concerned")

CLINICAL_PREFIX = "Patient presents with the following concern: "

LAYPERSON_OPENERS = [
    "So here's my question:",
    "Just wanted to ask something:",
    "Quick question for you:",
    "I wanted to check something:",
]

INDIRECT_OPENERS = [
    "I'm not really comfortable talking about this kind of thing, but I have to ask.",
    "This is a bit awkward for me to bring up, but here goes.",
    "I don't usually talk about this stuff, but I need some advice.",
    "It's hard for me to put this into words, but I'll try.",
]

EMOTIONAL_OPENERS = [
    "I'm really worried about this and can't stop thinking about it.",
    "This is scaring me a lot and I need to know what's going on.",
    "I'm quite anxious about this, please help me understand.",
    "I've been really stressed about this and need some reassurance.",
]

AMBIGUOUS_PLACEHOLDERS = ["this", "it", "that", "something"]

_SENTENCE_SPLIT = re.compile(r"(?<=[.?!])\s+")
_MIN_SUBSTANTIVE_SENTENCE_LEN = 20
_AMBIGUOUS_WORD_FALLBACK = 25


def _rotate(options: list[str], index: int) -> str:
    return options[index % len(options)]


def make_clinical(question: str) -> str:
    return CLINICAL_PREFIX + question.strip()


def make_layperson(question: str, index: int) -> str:
    return f"{_rotate(LAYPERSON_OPENERS, index)} {question.strip()}"


def make_indirect(question: str, index: int) -> str:
    return f"{_rotate(INDIRECT_OPENERS, index)} {question.strip()}"


def make_emotional(question: str, index: int) -> str:
    return f"{_rotate(EMOTIONAL_OPENERS, index)} {question.strip()}"


def _redaction_terms(topic: str, keywords: str) -> list[str]:
    terms = [topic.strip()] + [k.strip() for k in keywords.split(",")]
    return sorted({t for t in terms if len(t) >= 4}, key=len, reverse=True)


def make_ambiguous(question: str, topic: str, keywords: str) -> str:
    q = question.strip()
    for term in _redaction_terms(topic, keywords):
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        match = pattern.search(q)
        if match:
            placeholder = AMBIGUOUS_PLACEHOLDERS[hash(term) % len(AMBIGUOUS_PLACEHOLDERS)]
            q = q[: match.start()] + placeholder + q[match.end():]
            break  # one redaction is enough to blur the specific condition

    # Pick the first sentence that's actually substantive -- short greeting
    # interjections ("Hello, Doctor!") must not become the whole "vague" variant,
    # or the redacted question loses all clinical content instead of just detail.
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(q) if s.strip()]
    core = next((s for s in sentences if len(s) >= _MIN_SUBSTANTIVE_SENTENCE_LEN), None)
    if core is None:
        words = q.split()
        core = " ".join(words[:_AMBIGUOUS_WORD_FALLBACK])
        if len(words) > _AMBIGUOUS_WORD_FALLBACK:
            core += "..."
    if not core.endswith(("?", ".", "!", "...")):
        core += "?"
    return core + " I'm not really sure how to explain it exactly."


def build_style_rows(row: dict, index: int) -> list[dict]:
    question, topic, keywords = row["Question"], row["Topic"], row["Keywords"]
    variants = {
        "canonical": question.strip(),
        "clinical": make_clinical(question),
        "layperson": make_layperson(question, index),
        "indirect_cultural": make_indirect(question, index),
        "ambiguous": make_ambiguous(question, topic, keywords),
        "emotionally_concerned": make_emotional(question, index),
    }
    return [
        {"Question": variants[style], "Answer": row["Answer"], "Topic": topic,
         "Keywords": keywords, "Style": style}
        for style in STYLES
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--start", required=True, type=int, help="0-indexed data row, inclusive")
    parser.add_argument("--end", required=True, type=int, help="0-indexed data row, exclusive")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    batch = rows[args.start:args.end]
    print(f"{args.input.name}: {len(rows)} total data rows, batch [{args.start}:{args.end}) = {len(batch)} seeds")

    out_rows = []
    for i, row in enumerate(batch, start=args.start):
        out_rows.extend(build_style_rows(row, i))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Question", "Answer", "Topic", "Keywords", "Style"])
        w.writeheader()
        w.writerows(out_rows)

    all_texts = [r["Question"] for r in out_rows]
    ratio = len(set(all_texts)) / len(all_texts) if all_texts else 0.0
    print(f"{len(out_rows)} rows ({len(batch)} seeds x 6 styles) -> {args.output}")
    print(f"distinct-string ratio: {ratio:.2%}")


if __name__ == "__main__":
    main()
