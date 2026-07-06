"""Stage 1: fetch source pages once and snapshot them under data/raw/.

Idempotent: a source whose snapshot file already exists is skipped, so the
committed snapshots stay stable even if the live pages change.
"""
from datetime import date

import requests

from femsympqa.io_utils import read_json, write_json
from femsympqa.sources import RAW_DIR, SOURCES, snapshot_path

MANIFEST = RAW_DIR / "manifest.json"
HEADERS = {"User-Agent": "HerHealthGPT-research/0.1 (Queen's University grad project)"}


def main() -> None:
    manifest = read_json(MANIFEST) if MANIFEST.exists() else []
    fetched_slugs = {m["slug"] for m in manifest}

    for source in SOURCES:
        path = snapshot_path(source.slug)
        if path.exists():
            print(f"skip  {source.slug} (snapshot exists)")
            continue
        print(f"fetch {source.slug} <- {source.url}")
        resp = requests.get(source.url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(resp.text, encoding="utf-8")
        if source.slug not in fetched_slugs:
            manifest.append({
                "slug": source.slug,
                "condition": source.condition,
                "url": source.url,
                "fetched_at": date.today().isoformat(),
            })

    write_json(MANIFEST, manifest)
    print(f"done: {len(manifest)} snapshots recorded in {MANIFEST}")


if __name__ == "__main__":
    main()
