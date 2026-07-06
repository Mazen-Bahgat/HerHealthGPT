"""Stage 2: LLM-extract draft base entries + evidence passages from snapshots.

Idempotent: per-source API results are cached under data/base/extract_cache/,
so re-runs never re-pay for completed sources.

After running, a HUMAN reviews data/base/base_entries_draft.json, corrects it,
and saves the result as data/base/base_entries.json (the frozen file).
Then run with --validate-frozen to check the frozen file.
"""
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from femsympqa.extraction import ExtractionResult, RawEntry, build_extract_prompt, merge_entries
from femsympqa.html_clean import extract_sections
from femsympqa.io_utils import read_json, write_json
from femsympqa.schemas import BaseEntry
from femsympqa.sources import CONDITIONS, SOURCES, snapshot_path

MODEL = "claude-opus-4-8"
PROMPT_PATH = Path("prompts/extract_v1.md")
CACHE_DIR = Path("data/base/extract_cache")
DRAFT_PATH = Path("data/base/base_entries_draft.json")
FROZEN_PATH = Path("data/base/base_entries.json")
PASSAGES_PATH = Path("data/base/evidence_passages.json")


def condition_slug(condition: str) -> str:
    return next(slug for slug, name in CONDITIONS.items() if name == condition)


def extract_one_source(client: anthropic.Anthropic, source, template: str) -> list[dict]:
    cache_file = CACHE_DIR / f"{source.slug}.json"
    if cache_file.exists():
        print(f"skip  {source.slug} (cached)")
        return read_json(cache_file)

    html = snapshot_path(source.slug).read_text(encoding="utf-8")
    sections = extract_sections(html)
    prompt = build_extract_prompt(source.condition, source.url, sections, template)
    print(f"call  {source.slug} ({len(sections)} sections)")
    response = client.messages.parse(
        model=MODEL,
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
        output_format=ExtractionResult,
    )
    result: ExtractionResult = response.parsed_output
    rows = [e.model_dump() for e in result.entries]
    write_json(cache_file, rows)
    return rows


def build_passages() -> list[dict]:
    passages = []
    for source in SOURCES:
        html = snapshot_path(source.slug).read_text(encoding="utf-8")
        for section in extract_sections(html):
            passages.append({
                "condition": source.condition,
                "source_url": source.url,
                "heading": section.heading,
                "text": section.text,
            })
    return passages


def main() -> None:
    if "--validate-frozen" in sys.argv:
        rows = read_json(FROZEN_PATH)
        entries = [BaseEntry.model_validate(r) for r in rows]
        counts: dict[str, int] = {}
        for e in entries:
            counts[e.condition] = counts.get(e.condition, 0) + 1
        print(f"OK: {len(entries)} frozen base entries: {counts}")
        return

    load_dotenv()
    client = anthropic.Anthropic()
    template = PROMPT_PATH.read_text(encoding="utf-8")

    raw_entries: list[RawEntry] = []
    for source in SOURCES:
        for row in extract_one_source(client, source, template):
            raw_entries.append(RawEntry(condition=source.condition, source_url=source.url, **row))

    drafts: list[BaseEntry] = []
    for slug in CONDITIONS:
        condition_raws = [r for r in raw_entries if condition_slug(r.condition) == slug]
        drafts.extend(merge_entries(condition_raws, slug))

    write_json(DRAFT_PATH, [d.model_dump() for d in drafts])
    write_json(PASSAGES_PATH, build_passages())
    print(f"wrote {len(drafts)} draft entries -> {DRAFT_PATH}")
    print(f"wrote evidence passages -> {PASSAGES_PATH}")
    print("NEXT: human-review the draft, save corrected copy as "
          f"{FROZEN_PATH}, then run: python scripts/02_extract.py --validate-frozen")


if __name__ == "__main__":
    main()
