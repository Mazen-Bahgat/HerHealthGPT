"""Fetch and clean the 6 NHS/CDC/NICHD grounding pages named in
HerHealthGPT-LU_seed/build_seed.py's GROUNDING dict (the project's own
stated grounding policy — see decisions_log.md).

Idempotent: an existing snapshot is skipped. Reuses femsympqa.html_clean
(generic HTML->sections extractor, no project coupling) for section text.

Output:
  HerHealthGPT-LU_seed/grounding_sources/{key}.html   raw snapshot
  HerHealthGPT-LU_seed/grounding_sources/evidence.json  cleaned sections per key
"""
from pathlib import Path

import requests

from femsympqa.html_clean import extract_sections
from femsympqa.io_utils import write_json

OUT = Path("HerHealthGPT-LU_seed/grounding_sources")
HEADERS = {"User-Agent": "HerHealthGPT-LU-research/0.1 (Queen's University grad project)"}
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Source of truth: HerHealthGPT-LU_seed/build_seed.py GROUNDING dict.
GROUNDING_SOURCES = {
    "nhs_pcos_symptoms": {
        "condition": "PCOS symptoms",
        "url": "https://www.nhs.uk/conditions/polycystic-ovary-syndrome-pcos/symptoms/",
    },
    "nhs_heavy_periods": {
        "condition": "Heavy periods (menorrhagia)",
        "url": "https://www.nhs.uk/conditions/heavy-periods/",
    },
    "nhs_infertility": {
        "condition": "Infertility",
        "url": "https://www.nhs.uk/conditions/infertility/",
    },
    "cdc_reproductive": {
        "condition": "Common reproductive health concerns",
        "url": "https://www.cdc.gov/reproductive-health/women-health/common-concerns.html",
    },
    "nichd_endo": {
        "condition": "Endometriosis (NICHD)",
        "url": "https://www.nichd.nih.gov/health/topics/endometriosis",
    },
    "nichd_infertility": {
        "condition": "Infertility (NICHD)",
        "url": "https://www.nichd.nih.gov/health/topics/infertility",
    },
}


def fetch(key: str, url: str) -> str:
    path = OUT / f"{key}.html"
    if path.exists():
        print(f"skip  {key} (snapshot exists)")
        return path.read_text(encoding="utf-8")
    print(f"fetch {key} <- {url}")
    resp = requests.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
    if resp.status_code == 403:
        resp = requests.get(url, headers=BROWSER_HEADERS, timeout=30, allow_redirects=True)
    resp.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(resp.text, encoding="utf-8")
    return resp.text


def main() -> None:
    evidence = {}
    failed = []
    for key, meta in GROUNDING_SOURCES.items():
        try:
            html = fetch(key, meta["url"])
        except requests.exceptions.HTTPError as e:
            print(f"FAILED {key}: {e}")
            failed.append(key)
            continue
        sections = extract_sections(html)
        evidence[key] = {
            "condition": meta["condition"],
            "url": meta["url"],
            "sections": [s.model_dump() for s in sections],
        }
        print(f"      {key}: {len(sections)} sections")
    write_json(OUT / "evidence.json", evidence)
    print(f"done: evidence for {len(evidence)}/{len(GROUNDING_SOURCES)} sources -> {OUT / 'evidence.json'}")
    if failed:
        print(f"UNRESOLVED (bot-blocked, needs manual fetch): {failed}")


if __name__ == "__main__":
    main()
