# FemSympQA Dataset Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the FemSympQA dataset pipeline — five staged Python scripts that turn 5 clinical web pages into ~3,000–4,500 multilingual (EN/FR/AR) patient-style symptom records with condition, risk, and action labels.

**Architecture:** Staged batch pipeline with committed intermediate artifacts: scrape → LLM extraction (+ human review gate) → compound-entry generation (+ review gate) → LLM paraphrasing via Batch API (+ embedding QC) → NLLB translation (+ back-translation QC) → assembly/validation/stats. Shared logic lives in `src/femsympqa/`; each stage is an idempotent script in `scripts/`.

**Tech Stack:** Python 3.12, pydantic v2, requests + BeautifulSoup, Anthropic SDK (`claude-opus-4-8`, Batch API, structured outputs), HuggingFace transformers + NLLB-200-distilled-600M (CPU), sentence-transformers, pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-07-05-femsympqa-dataset-pipeline-design.md` (approved 2026-07-05).

## Global Constraints

- Python **3.12**, Windows 11, venv at `.venv`. Run everything via `.\.venv\Scripts\python`.
- LLM model pinned to **`claude-opus-4-8`**. **Never pass `temperature`, `top_p`, or `top_k`** — this model returns HTTP 400 if any is present. Diversity comes from prompting.
- Paraphrase generation uses the **Message Batches API** (50% discount). Raw batch results are saved to disk **before** any parsing.
- Translation: **`facebook/nllb-200-distilled-600M`**, CPU, `num_beams=4, do_sample=False, max_new_tokens=256`. Language codes: `eng_Latn`, `fra_Latn`, `arb_Arab`. Sentence-split before translating.
- Embedder for QC: **`sentence-transformers/all-MiniLM-L6-v2`**. Thresholds: cosine **≥ 0.95** → drop as duplicate; **< 0.60** vs canonical → flag as meaning drift; back-translation similarity **< 0.70** → flag.
- Risk tiers: `low` / `moderate` / `high` per the spec's urgency-language mapping rule. When sources disagree, the **higher tier wins** and `conflict_note` records it.
- Record ID format: `fsq-{base_id}-{variant}-{lang}` (e.g. `fsq-pcos-s03-p07-ar`); `variant` is `can` or `p{NN}`. `parent_id` on translations = the `-en` record's id.
- Targets: 40–60 single-symptom base entries, ~20–30 compound entries, **18 paraphrases requested per entry**, ~3,000–4,500 final records.
- Every `open()` passes `encoding="utf-8"`. JSON written with `ensure_ascii=False`. JSONL = one compact JSON object per line.
- Every stage is idempotent: it skips work whose output already exists. Failed/flagged items go to `data/errors/`.
- Commit all data artifacts (`data/raw`, `data/base`, `data/expanded`, `data/final`) — never `.env`.
- Human review gates (Tasks 5 and 6) are performed by Mazen, not automated. The pipeline must not proceed past a gate until the frozen file exists.

---

### Task 1: Environment & project scaffolding

**Files:**
- Create: `pyproject.toml`, `src/femsympqa/__init__.py`, `tests/test_smoke.py`, `.env.example`, `README.md`
- Create (empty dirs via `.gitkeep`): `prompts/`, `scripts/`, `data/raw/`, `data/base/`, `data/expanded/`, `data/final/`, `data/errors/`, `tests/fixtures/`

**Interfaces:**
- Produces: an installed editable package `femsympqa` importable from tests and scripts; `.venv` with all dependencies.

- [ ] **Step 1: Install Python 3.12** (skip if `python --version` already reports 3.12)

Run in PowerShell:
```powershell
winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
```
Then **open a new shell** and verify: `python --version` → `Python 3.12.x`. If `python` still resolves to the Store stub, use `py -3.12` in the next step instead.

- [ ] **Step 2: Create venv and upgrade pip**

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
```

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "femsympqa"
version = "0.1.0"
description = "FemSympQA multilingual symptom dataset pipeline (HerHealthGPT sub-project 1)"
requires-python = ">=3.12"
dependencies = [
    "requests>=2.32",
    "beautifulsoup4>=4.12",
    "pydantic>=2.7",
    "anthropic>=0.92",
    "transformers>=4.44",
    "torch>=2.3",
    "sentencepiece>=0.2",
    "sentence-transformers>=3.0",
    "pandas>=2.2",
    "tabulate>=0.9",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 4: Create package skeleton and support files**

`src/femsympqa/__init__.py`:
```python
"""FemSympQA dataset pipeline — HerHealthGPT sub-project 1."""
```

`.env.example`:
```
ANTHROPIC_API_KEY=sk-ant-...
```

`README.md`:
```markdown
# HerHealthGPT — FemSympQA dataset pipeline

Multilingual (EN/FR/AR) symptom→condition/risk/action dataset built from NHS/CDC/NIH
sources. Spec: `docs/superpowers/specs/2026-07-05-femsympqa-dataset-pipeline-design.md`.

## Setup
1. Python 3.12, then: `python -m venv .venv`
2. `.\.venv\Scripts\python -m pip install -e .[dev]`
3. Copy `.env.example` to `.env` and add your Anthropic API key.

## Pipeline run order
(filled in as stages land — see scripts/)
```

Create placeholder dirs (PowerShell):
```powershell
"prompts","scripts","data/raw","data/base","data/expanded","data/final","data/errors","tests/fixtures" | ForEach-Object { New-Item -ItemType Directory -Force $_ | Out-Null; New-Item -ItemType File -Force "$_/.gitkeep" | Out-Null }
```

- [ ] **Step 5: Install the package** (torch download is ~200 MB CPU wheel; allow several minutes)

```powershell
.\.venv\Scripts\python -m pip install -e .[dev]
```
Expected: ends with `Successfully installed ... femsympqa-0.1.0 ...`

- [ ] **Step 6: Write and run smoke test**

`tests/test_smoke.py`:
```python
import femsympqa


def test_package_imports():
    assert femsympqa.__doc__ is not None
```

Run: `.\.venv\Scripts\python -m pytest tests/test_smoke.py -v`
Expected: `1 passed`

- [ ] **Step 7: Commit**

```powershell
git add pyproject.toml src tests .env.example README.md prompts scripts data
git commit -m "chore: scaffold femsympqa package, deps, and data directories"
```

---

### Task 2: Schemas, JSON I/O helpers, and risk mapping

**Files:**
- Create: `src/femsympqa/schemas.py`, `src/femsympqa/io_utils.py`, `src/femsympqa/risk_mapping.py`
- Test: `tests/test_schemas.py`, `tests/test_risk_mapping.py`

**Interfaces:**
- Produces:
  - `schemas.RiskLevel` (str Enum: `LOW="low"`, `MODERATE="moderate"`, `HIGH="high"`)
  - `schemas.Evidence(source_url: str, quote: str)`
  - `schemas.BaseEntry(base_id, condition, symptom, canonical_query, risk_level, recommended_action, evidence: list[Evidence], conflict_note: str | None)`
  - `schemas.Provenance(generated_by: str, generated_at: str, prompt_id: str)`
  - `schemas.FinalRecord(id, base_id, parent_id: str | None, lang: Literal["en","fr","ar"], variant_type: Literal["canonical","paraphrase","translation"], query, condition, risk_level, recommended_action, evidence, provenance)`
  - `io_utils.read_json(path) -> Any`, `io_utils.write_json(path, obj)`, `io_utils.read_jsonl(path) -> list[dict]`, `io_utils.write_jsonl(path, rows)`, `io_utils.append_jsonl(path, rows)`
  - `risk_mapping.map_urgency_to_risk(text: str) -> RiskLevel`
  - `risk_mapping.merge_risk(a: RiskLevel, b: RiskLevel) -> RiskLevel` (returns higher tier)

- [ ] **Step 1: Write failing tests**

`tests/test_schemas.py`:
```python
import pytest
from pydantic import ValidationError

from femsympqa.schemas import BaseEntry, Evidence, FinalRecord, Provenance, RiskLevel


def make_entry(**overrides):
    data = dict(
        base_id="pcos-s01",
        condition="PCOS",
        symptom="irregular periods",
        canonical_query="My periods have been really irregular for months.",
        risk_level="moderate",
        recommended_action="See a GP for assessment.",
        evidence=[Evidence(source_url="https://www.nhs.uk/x", quote="see a GP")],
    )
    data.update(overrides)
    return BaseEntry(**data)


def test_base_entry_valid():
    entry = make_entry()
    assert entry.risk_level is RiskLevel.MODERATE
    assert entry.conflict_note is None


def test_base_entry_rejects_bad_risk():
    with pytest.raises(ValidationError):
        make_entry(risk_level="critical")


def test_base_entry_requires_evidence():
    with pytest.raises(ValidationError):
        make_entry(evidence=[])


def test_final_record_roundtrip():
    rec = FinalRecord(
        id="fsq-pcos-s01-p03-fr",
        base_id="pcos-s01",
        parent_id="fsq-pcos-s01-p03-en",
        lang="fr",
        variant_type="translation",
        query="Mes règles sont très irrégulières depuis des mois.",
        condition="PCOS",
        risk_level="moderate",
        recommended_action="See a GP for assessment.",
        evidence=[Evidence(source_url="https://www.nhs.uk/x", quote="see a GP")],
        provenance=Provenance(
            generated_by="facebook/nllb-200-distilled-600M",
            generated_at="2026-07-05",
            prompt_id="nllb-beam4-v1",
        ),
    )
    again = FinalRecord.model_validate(rec.model_dump())
    assert again == rec


def test_final_record_rejects_bad_lang():
    with pytest.raises(ValidationError):
        FinalRecord(
            id="x", base_id="b", parent_id=None, lang="de",
            variant_type="canonical", query="q", condition="PCOS",
            risk_level="low", recommended_action="a",
            evidence=[Evidence(source_url="u", quote="q")],
            provenance=Provenance(generated_by="g", generated_at="d", prompt_id="p"),
        )
```

`tests/test_risk_mapping.py`:
```python
import pytest

from femsympqa.risk_mapping import map_urgency_to_risk, merge_risk
from femsympqa.schemas import RiskLevel


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Call 999 or go to A&E now", RiskLevel.HIGH),
        ("Ask for an urgent GP appointment", RiskLevel.HIGH),
        ("Go to your nearest emergency department immediately", RiskLevel.HIGH),
        ("See a GP if these symptoms persist", RiskLevel.MODERATE),
        ("Speak to a GP about your symptoms", RiskLevel.MODERATE),
        ("Talk to your doctor if you are worried", RiskLevel.MODERATE),
        ("This is common and usually nothing to worry about", RiskLevel.LOW),
        ("You can usually treat this yourself at home", RiskLevel.LOW),
    ],
)
def test_map_urgency_to_risk(text, expected):
    assert map_urgency_to_risk(text) is expected


def test_high_beats_moderate_in_same_text():
    text = "See a GP, but call 999 if you have severe pain"
    assert map_urgency_to_risk(text) is RiskLevel.HIGH


def test_merge_risk_returns_higher_tier():
    assert merge_risk(RiskLevel.LOW, RiskLevel.MODERATE) is RiskLevel.MODERATE
    assert merge_risk(RiskLevel.HIGH, RiskLevel.MODERATE) is RiskLevel.HIGH
    assert merge_risk(RiskLevel.LOW, RiskLevel.LOW) is RiskLevel.LOW
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python -m pytest tests/test_schemas.py tests/test_risk_mapping.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'femsympqa.schemas'`

- [ ] **Step 3: Implement**

`src/femsympqa/schemas.py`:
```python
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class Evidence(BaseModel):
    source_url: str
    quote: str


class BaseEntry(BaseModel):
    base_id: str
    condition: str
    symptom: str
    canonical_query: str
    risk_level: RiskLevel
    recommended_action: str
    evidence: list[Evidence] = Field(min_length=1)
    conflict_note: str | None = None


class Provenance(BaseModel):
    generated_by: str
    generated_at: str
    prompt_id: str


class FinalRecord(BaseModel):
    id: str
    base_id: str
    parent_id: str | None = None
    lang: Literal["en", "fr", "ar"]
    variant_type: Literal["canonical", "paraphrase", "translation"]
    query: str
    condition: str
    risk_level: RiskLevel
    recommended_action: str
    evidence: list[Evidence] = Field(min_length=1)
    provenance: Provenance
```

`src/femsympqa/io_utils.py`:
```python
import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _dump_lines(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(_dump_lines(rows))


def append_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(_dump_lines(rows))
```

`src/femsympqa/risk_mapping.py`:
```python
"""3-tier risk taxonomy mapped from clinical sources' own urgency language.

Rule (from the spec): high beats moderate beats low. Marker lists are
lowercase substrings matched against lowercased input.
"""
from femsympqa.schemas import RiskLevel

HIGH_MARKERS = [
    "999",
    "a&e",
    "emergency",
    "urgent gp",
    "urgent appointment",
    "immediately",
    "straight away",
]

MODERATE_MARKERS = [
    "see a gp",
    "see your gp",
    "speak to a gp",
    "speak to your gp",
    "gp appointment",
    "see a doctor",
    "talk to your doctor",
    "talk to a doctor",
    "contact a gp",
    "medical review",
]

_RISK_ORDER = {RiskLevel.LOW: 0, RiskLevel.MODERATE: 1, RiskLevel.HIGH: 2}


def map_urgency_to_risk(text: str) -> RiskLevel:
    lowered = text.lower()
    if any(marker in lowered for marker in HIGH_MARKERS):
        return RiskLevel.HIGH
    if any(marker in lowered for marker in MODERATE_MARKERS):
        return RiskLevel.MODERATE
    return RiskLevel.LOW


def merge_risk(a: RiskLevel, b: RiskLevel) -> RiskLevel:
    return a if _RISK_ORDER[a] >= _RISK_ORDER[b] else b
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python -m pytest tests/test_schemas.py tests/test_risk_mapping.py -v`
Expected: all PASS (13 tests)

- [ ] **Step 5: Commit**

```powershell
git add src/femsympqa tests
git commit -m "feat: add pydantic schemas, JSON I/O helpers, and 3-tier risk mapping"
```

---

### Task 3: Stage 1 — scrape source snapshots

**Files:**
- Create: `src/femsympqa/sources.py`, `scripts/01_scrape.py`
- Test: `tests/test_sources.py`

**Interfaces:**
- Produces:
  - `sources.SOURCES: list[Source]` where `Source` is a pydantic model with `slug: str`, `condition: str`, `url: str`
  - `sources.snapshot_path(slug: str) -> Path` → `data/raw/{slug}.html`
  - `sources.CONDITIONS: dict[str, str]` mapping condition slug → display name (`{"pcos": "PCOS", "heavy-periods": "Heavy periods (menorrhagia)", "infertility": "Infertility (female)", "endometriosis": "Endometriosis"}`)
  - Running `scripts/01_scrape.py` yields `data/raw/{slug}.html` per source + `data/raw/manifest.json` (list of `{slug, condition, url, fetched_at}`)

- [ ] **Step 1: Write failing tests**

`tests/test_sources.py`:
```python
from pathlib import Path

from femsympqa.sources import CONDITIONS, SOURCES, snapshot_path


def test_five_sources_defined():
    assert len(SOURCES) == 5
    assert len({s.slug for s in SOURCES}) == 5  # slugs unique


def test_all_conditions_covered():
    assert {s.condition for s in SOURCES} == set(CONDITIONS.values())
    assert set(CONDITIONS) == {"pcos", "heavy-periods", "infertility", "endometriosis"}


def test_snapshot_path():
    assert snapshot_path("nhs-pcos") == Path("data/raw/nhs-pcos.html")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python -m pytest tests/test_sources.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'femsympqa.sources'`

- [ ] **Step 3: Implement**

`src/femsympqa/sources.py`:
```python
from pathlib import Path

from pydantic import BaseModel

RAW_DIR = Path("data/raw")

CONDITIONS: dict[str, str] = {
    "pcos": "PCOS",
    "heavy-periods": "Heavy periods (menorrhagia)",
    "infertility": "Infertility (female)",
    "endometriosis": "Endometriosis",
}


class Source(BaseModel):
    slug: str
    condition: str
    url: str


SOURCES: list[Source] = [
    Source(slug="nhs-pcos", condition=CONDITIONS["pcos"],
           url="https://www.nhs.uk/conditions/polycystic-ovary-syndrome-pcos/symptoms/"),
    Source(slug="cdc-pcos", condition=CONDITIONS["pcos"],
           url="https://www.cdc.gov/diabetes/basics/pcos.html"),
    Source(slug="nhs-heavy-periods", condition=CONDITIONS["heavy-periods"],
           url="https://www.nhs.uk/conditions/heavy-periods/"),
    Source(slug="nhs-infertility", condition=CONDITIONS["infertility"],
           url="https://www.nhs.uk/conditions/infertility/"),
    Source(slug="nih-endometriosis", condition=CONDITIONS["endometriosis"],
           url="https://www.nichd.nih.gov/health/topics/endometri/conditioninfo"),
]


def snapshot_path(slug: str) -> Path:
    return RAW_DIR / f"{slug}.html"
```

`scripts/01_scrape.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python -m pytest tests/test_sources.py -v`
Expected: 3 PASS

- [ ] **Step 5: Run the scraper for real** (network required)

Run: `.\.venv\Scripts\python scripts/01_scrape.py`
Expected: five `fetch` lines then `done: 5 snapshots recorded`. Verify `data/raw/` contains 5 `.html` files, each non-trivially sized (> 10 KB). If a URL 404s or redirects (pages move), find the current URL for the same topic on the same site, update `SOURCES`, and note the change in the commit message.

- [ ] **Step 6: Commit (including snapshots)**

```powershell
git add src/femsympqa/sources.py scripts/01_scrape.py tests/test_sources.py data/raw
git commit -m "feat: stage 1 scraper with committed HTML snapshots of 5 clinical sources"
```

---

### Task 4: HTML cleaning module

**Files:**
- Create: `src/femsympqa/html_clean.py`, `tests/fixtures/sample_condition_page.html`
- Test: `tests/test_html_clean.py`

**Interfaces:**
- Consumes: raw HTML strings (from Task 3 snapshots).
- Produces: `html_clean.Section(heading: str, text: str)` (pydantic model) and `html_clean.extract_sections(html: str) -> list[Section]` — headings h1–h3 with the concatenated paragraph/list text under each; boilerplate (script/style/nav/header/footer/aside) removed; sections with < 40 chars of text dropped.

- [ ] **Step 1: Write fixture and failing test**

`tests/fixtures/sample_condition_page.html`:
```html
<!DOCTYPE html>
<html>
<head><title>Example condition</title><style>.x{color:red}</style></head>
<body>
<nav><a href="/">Home</a><a href="/conditions">Conditions</a></nav>
<header><p>Site-wide banner text that must not appear.</p></header>
<main>
  <h1>Example condition</h1>
  <p>Example condition is a common problem affecting many people worldwide.</p>
  <h2>Symptoms</h2>
  <p>The main symptoms include tiredness and irregular periods that persist.</p>
  <ul><li>heavy bleeding during periods</li><li>pain in the lower tummy</li></ul>
  <h2>When to see a GP</h2>
  <p>See a GP if these symptoms are affecting your daily life significantly.</p>
  <h3>Urgent advice</h3>
  <p>Call 999 if you have sudden severe pain and feel faint or collapse.</p>
  <h2>Tiny</h2>
  <p>Too short.</p>
</main>
<footer><p>Footer legal text that must not appear.</p></footer>
<script>console.log("junk")</script>
</body>
</html>
```

`tests/test_html_clean.py`:
```python
from pathlib import Path

from femsympqa.html_clean import extract_sections

FIXTURE = Path("tests/fixtures/sample_condition_page.html")


def _sections():
    return extract_sections(FIXTURE.read_text(encoding="utf-8"))


def test_headings_extracted_in_order():
    headings = [s.heading for s in _sections()]
    assert headings == ["Example condition", "Symptoms", "When to see a GP", "Urgent advice"]


def test_list_items_joined_into_section_text():
    symptoms = next(s for s in _sections() if s.heading == "Symptoms")
    assert "heavy bleeding during periods" in symptoms.text
    assert "pain in the lower tummy" in symptoms.text


def test_boilerplate_excluded():
    all_text = " ".join(s.text for s in _sections())
    assert "banner" not in all_text
    assert "Footer legal" not in all_text
    assert "console.log" not in all_text


def test_short_sections_dropped():
    assert "Tiny" not in [s.heading for s in _sections()]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python -m pytest tests/test_html_clean.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'femsympqa.html_clean'`

- [ ] **Step 3: Implement**

`src/femsympqa/html_clean.py`:
```python
import re

from bs4 import BeautifulSoup
from pydantic import BaseModel

BOILERPLATE_TAGS = ["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]
MIN_SECTION_CHARS = 40


class Section(BaseModel):
    heading: str
    text: str


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_sections(html: str) -> list[Section]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(BOILERPLATE_TAGS):
        tag.decompose()

    root = soup.find("main") or soup.body or soup
    sections: list[Section] = []
    current_heading: str | None = None
    current_parts: list[str] = []

    def flush() -> None:
        nonlocal current_heading, current_parts
        if current_heading is not None:
            text = _clean(" ".join(current_parts))
            if len(text) >= MIN_SECTION_CHARS:
                sections.append(Section(heading=current_heading, text=text))
        current_parts = []

    for el in root.find_all(["h1", "h2", "h3", "p", "li"]):
        if el.name in ("h1", "h2", "h3"):
            flush()
            current_heading = _clean(el.get_text())
        else:
            current_parts.append(el.get_text())
    flush()
    return sections
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python -m pytest tests/test_html_clean.py -v`
Expected: 4 PASS

- [ ] **Step 5: Sanity-check against a real snapshot**

Run: `.\.venv\Scripts\python -c "from pathlib import Path; from femsympqa.html_clean import extract_sections; s = extract_sections(Path('data/raw/nhs-pcos.html').read_text(encoding='utf-8')); print(len(s)); [print('-', x.heading) for x in s[:10]]"`
Expected: a non-zero section count and plausible headings (e.g. "Symptoms of polycystic ovary syndrome"). If a real page yields 0 sections, inspect its structure and adjust `extract_sections` (e.g. include `article` as root fallback) while keeping the fixture tests green.

- [ ] **Step 6: Commit**

```powershell
git add src/femsympqa/html_clean.py tests/test_html_clean.py tests/fixtures
git commit -m "feat: HTML section extractor with boilerplate removal"
```

---

### Task 5: Stage 2 — LLM extraction, cross-source merge, human review gate

**Files:**
- Create: `src/femsympqa/extraction.py`, `scripts/02_extract.py`, `prompts/extract_v1.md`
- Test: `tests/test_extraction.py`

**Interfaces:**
- Consumes: `html_clean.extract_sections`, `sources.SOURCES`, `risk_mapping.map_urgency_to_risk/merge_risk`, `schemas.BaseEntry/Evidence`.
- Produces:
  - `extraction.RawEntry(condition: str, source_url: str, symptom: str, canonical_query: str, recommended_action: str, urgency_quote: str)` (pydantic model)
  - `extraction.ExtractionResult(entries: list[ExtractedSymptom])` where `ExtractedSymptom(symptom, canonical_query, recommended_action, urgency_quote)` — the structured-output schema for the API call
  - `extraction.build_extract_prompt(condition: str, url: str, sections: list[Section], template: str) -> str`
  - `extraction.merge_entries(raw: list[RawEntry], condition_slug: str) -> list[BaseEntry]` — groups by normalized symptom, assigns `base_id = f"{condition_slug}-s{NN:02d}"`, merges evidence, applies higher-tier-wins with `conflict_note`
  - Running `scripts/02_extract.py` yields `data/base/extract_cache/{slug}.json` (raw per-source results), `data/base/base_entries_draft.json`, `data/base/evidence_passages.json`
  - **Human review gate:** Mazen edits the draft and saves `data/base/base_entries.json`; `scripts/02_extract.py --validate-frozen` validates it.

- [ ] **Step 1: Write the prompt template**

`prompts/extract_v1.md`:
```markdown
<!-- prompt_id: extract_v1 -->
You are extracting structured clinical information for FemSympQA, a research
dataset on female reproductive health, from a trusted clinical web page.

Condition: {condition}
Source URL: {url}

Cleaned page sections follow, each as "## heading" then text:

{sections}

For EACH distinct symptom explicitly described in the text, produce:
- symptom: short clinical name of the symptom (e.g. "irregular periods")
- canonical_query: one first-person, patient-style English sentence or two, as a
  patient might describe this symptom when seeking help (no medical jargon,
  do not name the condition unless the text says patients commonly do)
- recommended_action: the action this page recommends for this symptom,
  paraphrased faithfully (e.g. "See a GP if symptoms affect daily life")
- urgency_quote: the VERBATIM sentence from the sections above that best
  expresses the urgency/action for this symptom. Copy it exactly.

Rules:
- Only symptoms explicitly present in the text. Do not invent or generalize.
- One entry per distinct symptom; do not merge unrelated symptoms.
- urgency_quote must be copied verbatim from the provided sections.
```

- [ ] **Step 2: Write failing tests** (merge + prompt logic only — no API calls in tests)

`tests/test_extraction.py`:
```python
from femsympqa.extraction import RawEntry, build_extract_prompt, merge_entries
from femsympqa.html_clean import Section
from femsympqa.schemas import RiskLevel


def raw(symptom="Irregular periods", url="https://nhs.uk/x",
        quote="See a GP if this persists", action="See a GP"):
    return RawEntry(condition="PCOS", source_url=url, symptom=symptom,
                    canonical_query="My periods are all over the place.",
                    recommended_action=action, urgency_quote=quote)


def test_build_extract_prompt_substitutes_fields():
    template = "Condition: {condition}\nURL: {url}\n{sections}"
    prompt = build_extract_prompt(
        "PCOS", "https://nhs.uk/x",
        [Section(heading="Symptoms", text="Tiredness and acne.")], template)
    assert "Condition: PCOS" in prompt
    assert "## Symptoms" in prompt
    assert "Tiredness and acne." in prompt


def test_merge_same_symptom_across_sources_merges_evidence():
    entries = merge_entries(
        [raw(url="https://nhs.uk/x"), raw(symptom="irregular  PERIODS", url="https://cdc.gov/y")],
        "pcos")
    assert len(entries) == 1
    assert {e.source_url for e in entries[0].evidence} == {"https://nhs.uk/x", "https://cdc.gov/y"}
    assert entries[0].base_id == "pcos-s01"
    assert entries[0].conflict_note is None


def test_merge_conflict_takes_higher_tier_and_records_note():
    entries = merge_entries(
        [raw(quote="This is common and usually nothing to worry about", action="Self-care"),
         raw(url="https://cdc.gov/y", quote="Ask for an urgent GP appointment",
             action="Urgent GP appointment")],
        "pcos")
    assert entries[0].risk_level is RiskLevel.HIGH
    assert "disagree" in entries[0].conflict_note
    # recommended_action comes from the highest-risk source
    assert entries[0].recommended_action == "Urgent GP appointment"


def test_distinct_symptoms_stay_separate_with_sequential_ids():
    entries = merge_entries([raw(), raw(symptom="Acne")], "pcos")
    assert len(entries) == 2
    assert sorted(e.base_id for e in entries) == ["pcos-s01", "pcos-s02"]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.\.venv\Scripts\python -m pytest tests/test_extraction.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'femsympqa.extraction'`

- [ ] **Step 4: Implement the module**

`src/femsympqa/extraction.py`:
```python
import re
from functools import reduce

from pydantic import BaseModel

from femsympqa.html_clean import Section
from femsympqa.risk_mapping import map_urgency_to_risk, merge_risk
from femsympqa.schemas import BaseEntry, Evidence


class ExtractedSymptom(BaseModel):
    symptom: str
    canonical_query: str
    recommended_action: str
    urgency_quote: str


class ExtractionResult(BaseModel):
    entries: list[ExtractedSymptom]


class RawEntry(BaseModel):
    condition: str
    source_url: str
    symptom: str
    canonical_query: str
    recommended_action: str
    urgency_quote: str


def build_extract_prompt(condition: str, url: str, sections: list[Section], template: str) -> str:
    rendered = "\n\n".join(f"## {s.heading}\n{s.text}" for s in sections)
    return template.format(condition=condition, url=url, sections=rendered)


def _normalize_symptom(symptom: str) -> str:
    return re.sub(r"\s+", " ", symptom.strip().lower())


def merge_entries(raw: list[RawEntry], condition_slug: str) -> list[BaseEntry]:
    grouped: dict[str, list[RawEntry]] = {}
    for item in raw:
        grouped.setdefault(_normalize_symptom(item.symptom), []).append(item)

    entries: list[BaseEntry] = []
    for idx, key in enumerate(sorted(grouped), start=1):
        items = grouped[key]
        risks = [map_urgency_to_risk(f"{i.urgency_quote} {i.recommended_action}") for i in items]
        final_risk = reduce(merge_risk, risks)
        conflict = None
        if len({r.value for r in risks}) > 1:
            tiers = ", ".join(sorted({r.value for r in risks}))
            conflict = f"sources disagree on urgency ({tiers}); higher tier applied"
        # action from the (first) highest-risk source
        best = max(zip(risks, items), key=lambda pair: ["low", "moderate", "high"].index(pair[0].value))
        entries.append(BaseEntry(
            base_id=f"{condition_slug}-s{idx:02d}",
            condition=items[0].condition,
            symptom=items[0].symptom,
            canonical_query=items[0].canonical_query,
            risk_level=final_risk,
            recommended_action=best[1].recommended_action,
            evidence=[Evidence(source_url=i.source_url, quote=i.urgency_quote) for i in items],
            conflict_note=conflict,
        ))
    return entries
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.\.venv\Scripts\python -m pytest tests/test_extraction.py -v`
Expected: 4 PASS

- [ ] **Step 6: Write the stage script**

`scripts/02_extract.py`:
```python
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
```

- [ ] **Step 7: Run extraction for real** (requires `.env` with `ANTHROPIC_API_KEY` — ask Mazen to create it from `.env.example` if missing)

Run: `.\.venv\Scripts\python scripts/02_extract.py`
Expected: five `call` lines, then `wrote N draft entries` where N is roughly 40–60. If N is far outside that range, inspect the draft — too few usually means `extract_sections` missed content on a real page (fix Task 4 root fallback); too many usually means the LLM split one symptom into near-duplicates (they will merge at review).

- [ ] **Step 8: HUMAN REVIEW GATE (Mazen, ~1 hour, not automated)**

1. Open `data/base/base_entries_draft.json`.
2. For each entry check against the source page: symptom real? query natural? risk tier correct per the mapping rule? quote verbatim? Merge near-duplicate symptoms, fix wording, delete hallucinations.
3. Save the corrected file as `data/base/base_entries.json`.
4. Validate: `.\.venv\Scripts\python scripts/02_extract.py --validate-frozen` → expect `OK: N frozen base entries` with per-condition counts.

- [ ] **Step 9: Commit**

```powershell
git add prompts/extract_v1.md src/femsympqa/extraction.py scripts/02_extract.py tests/test_extraction.py data/base
git commit -m "feat: stage 2 LLM extraction with cross-source merge and human-reviewed base entries"
```

---

### Task 6: Stage 2b — compound-symptom entries

**Files:**
- Create: `src/femsympqa/compound.py`, `scripts/02b_compound.py`, `prompts/compound_v1.md`
- Test: `tests/test_compound.py`

**Interfaces:**
- Consumes: frozen `data/base/base_entries.json` (list of `BaseEntry`), `risk_mapping.merge_risk`.
- Produces:
  - `compound.sample_combinations(entries: list[BaseEntry], per_condition: int, seed: int) -> list[list[BaseEntry]]` — deterministic 2–3-entry combinations within one condition
  - `compound.build_compound_entry(components: list[BaseEntry], base_id: str, query: str) -> BaseEntry` — symptom = `" + "` join; risk = max of component risks; action = action of the highest-risk component; evidence = deduplicated union; `conflict_note=None`
  - Running `scripts/02b_compound.py` yields `data/base/base_entries_compound_draft.json`; human review freezes `data/base/base_entries_compound.json`; `--validate-frozen` checks it. Compound `base_id`s use suffix `c`: `pcos-c01`, `pcos-c02`, …

- [ ] **Step 1: Write failing tests**

`tests/test_compound.py`:
```python
from femsympqa.compound import build_compound_entry, sample_combinations
from femsympqa.schemas import BaseEntry, Evidence, RiskLevel


def entry(base_id, symptom, risk, action, url):
    return BaseEntry(
        base_id=base_id, condition="PCOS", symptom=symptom,
        canonical_query=f"I have {symptom}.", risk_level=risk,
        recommended_action=action,
        evidence=[Evidence(source_url=url, quote=f"quote about {symptom}")],
    )


ENTRIES = [
    entry("pcos-s01", "irregular periods", "low", "Self-care", "https://a"),
    entry("pcos-s02", "excess hair", "moderate", "See a GP", "https://b"),
    entry("pcos-s03", "severe pelvic pain", "high", "Urgent GP appointment", "https://a"),
    entry("pcos-s04", "acne", "low", "Self-care", "https://c"),
]


def test_sample_combinations_is_deterministic_and_within_condition():
    combos1 = sample_combinations(ENTRIES, per_condition=3, seed=42)
    combos2 = sample_combinations(ENTRIES, per_condition=3, seed=42)
    assert [[e.base_id for e in c] for c in combos1] == [[e.base_id for e in c] for c in combos2]
    assert len(combos1) == 3
    for combo in combos1:
        assert 2 <= len(combo) <= 3
        assert len({e.base_id for e in combo}) == len(combo)  # no repeats


def test_build_compound_entry_labels():
    compound = build_compound_entry(
        [ENTRIES[0], ENTRIES[2]], base_id="pcos-c01",
        query="My periods are irregular and I have really bad pelvic pain.")
    assert compound.risk_level is RiskLevel.HIGH
    assert compound.recommended_action == "Urgent GP appointment"
    assert compound.symptom == "irregular periods + severe pelvic pain"
    assert {e.source_url for e in compound.evidence} == {"https://a"}  # deduped union
    assert compound.canonical_query.startswith("My periods")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python -m pytest tests/test_compound.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'femsympqa.compound'`

- [ ] **Step 3: Implement**

`src/femsympqa/compound.py`:
```python
import random
from functools import reduce

from femsympqa.risk_mapping import merge_risk
from femsympqa.schemas import BaseEntry, Evidence

_RISK_ORDER = ["low", "moderate", "high"]


def sample_combinations(entries: list[BaseEntry], per_condition: int, seed: int) -> list[list[BaseEntry]]:
    rng = random.Random(seed)
    combos: list[list[BaseEntry]] = []
    seen: set[tuple[str, ...]] = set()
    attempts = 0
    while len(combos) < per_condition and attempts < per_condition * 20:
        attempts += 1
        size = rng.choice([2, 2, 3])  # bias toward pairs
        if len(entries) < size:
            break
        combo = sorted(rng.sample(entries, size), key=lambda e: e.base_id)
        key = tuple(e.base_id for e in combo)
        if key in seen:
            continue
        seen.add(key)
        combos.append(combo)
    return combos


def build_compound_entry(components: list[BaseEntry], base_id: str, query: str) -> BaseEntry:
    risk = reduce(merge_risk, (c.risk_level for c in components))
    highest = max(components, key=lambda c: _RISK_ORDER.index(c.risk_level.value))
    evidence: list[Evidence] = []
    seen: set[tuple[str, str]] = set()
    for component in components:
        for ev in component.evidence:
            key = (ev.source_url, ev.quote)
            if key not in seen:
                seen.add(key)
                evidence.append(ev)
    return BaseEntry(
        base_id=base_id,
        condition=components[0].condition,
        symptom=" + ".join(c.symptom for c in components),
        canonical_query=query,
        risk_level=risk,
        recommended_action=highest.recommended_action,
        evidence=evidence,
        conflict_note=None,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python -m pytest tests/test_compound.py -v`
Expected: 2 PASS

- [ ] **Step 5: Write prompt and stage script**

`prompts/compound_v1.md`:
```markdown
<!-- prompt_id: compound_v1 -->
You are writing one realistic patient query for FemSympQA, a research dataset
on female reproductive health.

A patient is experiencing ALL of these symptoms at the same time:
{symptom_list}

Write ONE first-person English query (1–3 sentences) in which the patient
naturally describes all of these symptoms together, as if messaging a health
service. Plain everyday language, no medical jargon, do not name any condition,
do not add symptoms beyond those listed.
```

`scripts/02b_compound.py`:
```python
"""Stage 2b: generate compound-symptom base entries from the frozen entries.

Combinations and labels are deterministic (rule-based); only the canonical
query text comes from the LLM. Idempotent via per-combination caching.
"""
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

from femsympqa.compound import build_compound_entry, sample_combinations
from femsympqa.io_utils import read_json, write_json
from femsympqa.schemas import BaseEntry
from femsympqa.sources import CONDITIONS

MODEL = "claude-opus-4-8"
SEED = 42
PER_CONDITION = 7  # 7 x 4 conditions = 28 compound entries
PROMPT_PATH = Path("prompts/compound_v1.md")
FROZEN_SINGLE = Path("data/base/base_entries.json")
CACHE_DIR = Path("data/base/compound_cache")
DRAFT_PATH = Path("data/base/base_entries_compound_draft.json")
FROZEN_PATH = Path("data/base/base_entries_compound.json")


class CompoundQuery(BaseModel):
    query: str


def main() -> None:
    if "--validate-frozen" in sys.argv:
        entries = [BaseEntry.model_validate(r) for r in read_json(FROZEN_PATH)]
        print(f"OK: {len(entries)} frozen compound entries")
        return

    load_dotenv()
    client = anthropic.Anthropic()
    template = PROMPT_PATH.read_text(encoding="utf-8")
    singles = [BaseEntry.model_validate(r) for r in read_json(FROZEN_SINGLE)]

    drafts: list[BaseEntry] = []
    for slug, condition in CONDITIONS.items():
        condition_entries = [e for e in singles if e.condition == condition]
        combos = sample_combinations(condition_entries, PER_CONDITION, SEED)
        for i, combo in enumerate(combos, start=1):
            base_id = f"{slug}-c{i:02d}"
            cache_file = CACHE_DIR / f"{base_id}.json"
            if cache_file.exists():
                query = read_json(cache_file)["query"]
                print(f"skip  {base_id} (cached)")
            else:
                symptom_list = "\n".join(f"- {e.symptom}" for e in combo)
                response = client.messages.parse(
                    model=MODEL,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": template.format(symptom_list=symptom_list)}],
                    output_format=CompoundQuery,
                )
                query = response.parsed_output.query
                write_json(cache_file, {"query": query})
                print(f"call  {base_id}")
            drafts.append(build_compound_entry(combo, base_id, query))

    write_json(DRAFT_PATH, [d.model_dump() for d in drafts])
    print(f"wrote {len(drafts)} compound drafts -> {DRAFT_PATH}")
    print(f"NEXT: human-review, save as {FROZEN_PATH}, then --validate-frozen")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run for real, then human review gate**

Run: `.\.venv\Scripts\python scripts/02b_compound.py`
Expected: ~28 `call`/`skip` lines, `wrote 28 compound drafts`.
Then **Mazen reviews** the draft (queries natural? all symptoms mentioned? ~15 min), saves as `data/base/base_entries_compound.json`, and runs `.\.venv\Scripts\python scripts/02b_compound.py --validate-frozen` → `OK: 28 frozen compound entries`.

- [ ] **Step 7: Commit**

```powershell
git add prompts/compound_v1.md src/femsympqa/compound.py scripts/02b_compound.py tests/test_compound.py data/base
git commit -m "feat: stage 2b compound-symptom entries with rule-derived labels"
```

---

### Task 7: Stage 3 — paraphrase generation (Batch API) + embedding QC

**Files:**
- Create: `src/femsympqa/paraphrase.py`, `scripts/03_paraphrase.py`, `prompts/paraphrase_v1.md`
- Test: `tests/test_paraphrase.py`

**Interfaces:**
- Consumes: frozen base entries (single + compound).
- Produces:
  - `paraphrase.REGISTERS = ["worried", "casual", "brief", "detailed", "colloquial"]`
  - `paraphrase.N_PARAPHRASES = 18`
  - `paraphrase.PARAPHRASE_OUTPUT_SCHEMA: dict` — raw JSON schema for `output_config.format`
  - `paraphrase.build_batch_request(entry: BaseEntry, template: str) -> dict` — `{"custom_id": entry.base_id, "params": {...}}` with model `claude-opus-4-8`, `max_tokens=4000`, NO sampling params
  - `paraphrase.cosine(a: list[float], b: list[float]) -> float`
  - `paraphrase.qc_filter(canonical_vec, candidates, dup_threshold=0.95, drift_threshold=0.60) -> tuple[list, list]` where `candidates` is `list[tuple[text, register, vec]]`; returns `(kept, flagged)`; `flagged` items are dicts with `text`, `register`, `reason` (`"drift" | "dup_canonical" | "dup_paraphrase"`), `similarity`
  - Running `scripts/03_paraphrase.py submit` creates the batch and writes `data/expanded/batch_id.txt`; `scripts/03_paraphrase.py collect` polls, saves `data/expanded/raw_batch_results.jsonl`, runs QC, writes `data/expanded/paraphrases_en.jsonl` (rows: `{base_id, variant, text, register, generated_at, prompt_id}` with `variant` = `p01`, `p02`, … per base_id) and flags to `data/errors/paraphrase_flags.jsonl`.

- [ ] **Step 1: Write the prompt template**

`prompts/paraphrase_v1.md`:
```markdown
<!-- prompt_id: paraphrase_v1 -->
You are generating diverse patient-style paraphrases for FemSympQA, a research
dataset on female reproductive health.

Canonical patient query:
"{canonical_query}"

Symptom(s) described: {symptom}

Produce exactly {n} distinct English paraphrases of this query, distributed
roughly evenly across these registers:
- worried: anxious tone, seeking reassurance
- casual: relaxed, conversational
- brief: one short sentence, minimal words
- detailed: adds realistic everyday context (duration, impact on daily life)
- colloquial: informal everyday phrasing and idioms

Rules:
- First person, as a patient describing her own experience.
- Preserve the meaning exactly: same symptom(s), same implied severity.
- Do NOT name any medical condition or diagnosis.
- Do NOT add symptoms that are not in the canonical query.
- Vary sentence structure and vocabulary substantially between paraphrases.
```

- [ ] **Step 2: Write failing tests** (no API, no embedding model — vectors injected by hand)

`tests/test_paraphrase.py`:
```python
from femsympqa.paraphrase import (
    N_PARAPHRASES,
    REGISTERS,
    build_batch_request,
    cosine,
    qc_filter,
)
from femsympqa.schemas import BaseEntry, Evidence


ENTRY = BaseEntry(
    base_id="pcos-s01", condition="PCOS", symptom="irregular periods",
    canonical_query="My periods are all over the place.", risk_level="moderate",
    recommended_action="See a GP.",
    evidence=[Evidence(source_url="https://a", quote="see a GP")],
)


def test_build_batch_request_shape():
    req = build_batch_request(ENTRY, template="{canonical_query}|{symptom}|{n}")
    assert req["custom_id"] == "pcos-s01"
    params = req["params"]
    assert params["model"] == "claude-opus-4-8"
    assert "temperature" not in params and "top_p" not in params and "top_k" not in params
    assert str(N_PARAPHRASES) in params["messages"][0]["content"]
    schema = params["output_config"]["format"]["schema"]
    assert schema["properties"]["paraphrases"]["items"]["properties"]["register"]["enum"] == REGISTERS


def test_cosine():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert abs(cosine([1.0, 0.0], [0.0, 1.0])) < 1e-9


def test_qc_filter_drops_dups_and_flags_drift():
    canonical = [1.0, 0.0]
    candidates = [
        ("good paraphrase", "casual", [0.8, 0.6]),          # cos≈0.8 -> keep
        ("near copy of canonical", "brief", [0.999, 0.01]),  # cos≈1.0 -> dup_canonical
        ("drifted meaning", "worried", [0.1, 0.995]),        # cos≈0.1 -> drift
        ("dup of kept one", "detailed", [0.8, 0.6]),         # ≈ kept[0] -> dup_paraphrase
    ]
    kept, flagged = qc_filter(canonical, candidates)
    assert [k[0] for k in kept] == ["good paraphrase"]
    assert sorted(f["reason"] for f in flagged) == ["drift", "dup_canonical", "dup_paraphrase"]
    assert all("similarity" in f for f in flagged)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.\.venv\Scripts\python -m pytest tests/test_paraphrase.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'femsympqa.paraphrase'`

- [ ] **Step 4: Implement the module**

`src/femsympqa/paraphrase.py`:
```python
import math

from femsympqa.schemas import BaseEntry

MODEL = "claude-opus-4-8"
PROMPT_ID = "paraphrase_v1"
N_PARAPHRASES = 18
REGISTERS = ["worried", "casual", "brief", "detailed", "colloquial"]

PARAPHRASE_OUTPUT_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "paraphrases": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "register": {"type": "string", "enum": REGISTERS},
                    },
                    "required": ["text", "register"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["paraphrases"],
        "additionalProperties": False,
    },
}


def build_batch_request(entry: BaseEntry, template: str) -> dict:
    prompt = template.format(
        canonical_query=entry.canonical_query, symptom=entry.symptom, n=N_PARAPHRASES
    )
    return {
        "custom_id": entry.base_id,
        "params": {
            "model": MODEL,
            "max_tokens": 4000,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"format": PARAPHRASE_OUTPUT_SCHEMA},
        },
    }


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def qc_filter(canonical_vec, candidates, dup_threshold=0.95, drift_threshold=0.60):
    kept: list[tuple[str, str, list[float]]] = []
    flagged: list[dict] = []

    def flag(text, register, reason, sim):
        flagged.append({"text": text, "register": register, "reason": reason,
                        "similarity": round(sim, 4)})

    for text, register, vec in candidates:
        sim_canonical = cosine(vec, canonical_vec)
        if sim_canonical < drift_threshold:
            flag(text, register, "drift", sim_canonical)
            continue
        if sim_canonical >= dup_threshold:
            flag(text, register, "dup_canonical", sim_canonical)
            continue
        dup_sim = next((cosine(vec, kv) for _, _, kv in kept if cosine(vec, kv) >= dup_threshold), None)
        if dup_sim is not None:
            flag(text, register, "dup_paraphrase", dup_sim)
            continue
        kept.append((text, register, vec))
    return kept, flagged
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.\.venv\Scripts\python -m pytest tests/test_paraphrase.py -v`
Expected: 3 PASS

- [ ] **Step 6: Write the stage script**

`scripts/03_paraphrase.py`:
```python
"""Stage 3: paraphrase generation via the Message Batches API + embedding QC.

Usage:
  python scripts/03_paraphrase.py submit   # create the batch (once)
  python scripts/03_paraphrase.py collect  # poll, save raw results, QC, write output

Raw results are saved to disk BEFORE parsing so a crash never loses paid output.
"""
import sys
import time
from datetime import date
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from femsympqa.io_utils import append_jsonl, read_json, read_jsonl, write_jsonl
from femsympqa.paraphrase import PROMPT_ID, build_batch_request, qc_filter
from femsympqa.schemas import BaseEntry

PROMPT_PATH = Path("prompts/paraphrase_v1.md")
FROZEN_SINGLE = Path("data/base/base_entries.json")
FROZEN_COMPOUND = Path("data/base/base_entries_compound.json")
BATCH_ID_PATH = Path("data/expanded/batch_id.txt")
RAW_RESULTS = Path("data/expanded/raw_batch_results.jsonl")
OUTPUT = Path("data/expanded/paraphrases_en.jsonl")
FLAGS = Path("data/errors/paraphrase_flags.jsonl")


def load_entries() -> list[BaseEntry]:
    rows = read_json(FROZEN_SINGLE) + read_json(FROZEN_COMPOUND)
    return [BaseEntry.model_validate(r) for r in rows]


def submit(client: anthropic.Anthropic) -> None:
    if BATCH_ID_PATH.exists():
        print(f"batch already submitted: {BATCH_ID_PATH.read_text(encoding='utf-8').strip()}")
        return
    template = PROMPT_PATH.read_text(encoding="utf-8")
    requests_ = [build_batch_request(e, template) for e in load_entries()]
    batch = client.messages.batches.create(requests=requests_)
    BATCH_ID_PATH.parent.mkdir(parents=True, exist_ok=True)
    BATCH_ID_PATH.write_text(batch.id, encoding="utf-8")
    print(f"submitted batch {batch.id} with {len(requests_)} requests")


def collect(client: anthropic.Anthropic) -> None:
    batch_id = BATCH_ID_PATH.read_text(encoding="utf-8").strip()

    if not RAW_RESULTS.exists():
        while True:
            batch = client.messages.batches.retrieve(batch_id)
            print(f"status: {batch.processing_status}")
            if batch.processing_status == "ended":
                break
            time.sleep(30)
        raw_rows = []
        for result in client.messages.batches.results(batch_id):
            row = {"custom_id": result.custom_id, "type": result.result.type}
            if result.result.type == "succeeded":
                msg = result.result.message
                row["text"] = next(b.text for b in msg.content if b.type == "text")
            raw_rows.append(row)
        write_jsonl(RAW_RESULTS, raw_rows)  # save raw BEFORE parsing
        print(f"saved {len(raw_rows)} raw results -> {RAW_RESULTS}")

    import json

    from sentence_transformers import SentenceTransformer

    entries = {e.base_id: e for e in load_entries()}
    raw_rows = read_jsonl(RAW_RESULTS)
    done_ids = {r["base_id"] for r in read_jsonl(OUTPUT)} if OUTPUT.exists() else set()
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    today = date.today().isoformat()

    out_rows, n_flagged = [], 0
    for row in raw_rows:
        base_id = row["custom_id"]
        if base_id in done_ids:
            continue
        if row["type"] != "succeeded":
            append_jsonl(FLAGS, [{"base_id": base_id, "reason": f"batch_{row['type']}"}])
            n_flagged += 1
            continue
        paraphrases = json.loads(row["text"])["paraphrases"]
        entry = entries[base_id]
        texts = [entry.canonical_query] + [p["text"] for p in paraphrases]
        vecs = embedder.encode(texts, normalize_embeddings=True).tolist()
        candidates = [(p["text"], p["register"], v) for p, v in zip(paraphrases, vecs[1:])]
        kept, flagged = qc_filter(vecs[0], candidates)
        for f in flagged:
            f["base_id"] = base_id
        if flagged:
            append_jsonl(FLAGS, flagged)
            n_flagged += len(flagged)
        for i, (text, register, _) in enumerate(kept, start=1):
            out_rows.append({
                "base_id": base_id, "variant": f"p{i:02d}", "text": text,
                "register": register, "generated_at": today, "prompt_id": PROMPT_ID,
            })

    append_jsonl(OUTPUT, out_rows)
    print(f"wrote {len(out_rows)} paraphrases -> {OUTPUT} ({n_flagged} flagged)")


def main() -> None:
    load_dotenv()
    client = anthropic.Anthropic()
    if "submit" in sys.argv:
        submit(client)
    elif "collect" in sys.argv:
        collect(client)
    else:
        print("usage: 03_paraphrase.py submit|collect")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run for real**

```powershell
.\.venv\Scripts\python scripts/03_paraphrase.py submit
.\.venv\Scripts\python scripts/03_paraphrase.py collect
```
Expected: `submitted batch msgbatch_... with ~70-90 requests`; collect polls (batches usually finish well within an hour), saves raw results, then `wrote ~1100-1500 paraphrases`. Spot-read 10 random rows of `paraphrases_en.jsonl` — they should read like real patients, varied in tone. Check `data/errors/paraphrase_flags.jsonl` size is a small fraction of the total.

- [ ] **Step 8: Commit**

```powershell
git add prompts/paraphrase_v1.md src/femsympqa/paraphrase.py scripts/03_paraphrase.py tests/test_paraphrase.py data/expanded data/errors
git commit -m "feat: stage 3 batch paraphrase generation with embedding QC filter"
```

---

### Task 8: Stage 4 — NLLB translation

**Files:**
- Create: `src/femsympqa/translate.py`, `scripts/04_translate.py`
- Test: `tests/test_translate.py`

**Interfaces:**
- Consumes: frozen base entries (canonical queries) + `data/expanded/paraphrases_en.jsonl`.
- Produces:
  - `translate.LANG_CODES = {"en": "eng_Latn", "fr": "fra_Latn", "ar": "arb_Arab"}`
  - `translate.split_sentences(text: str) -> list[str]`
  - `translate.NllbTranslator` with `.translate(text: str, src: str, tgt: str) -> str` (lazy model load in `__init__`)
  - `translate.build_translation_rows(en_rows: list[dict], targets: list[str], translate_fn, generated_at: str) -> list[dict]` — `en_rows` have `{base_id, variant, text}` (`variant` = `can` or `pNN`); output rows: `{base_id, variant, lang, text, generated_at, prompt_id: "nllb-beam4-v1"}` — one per (row, target language). `translate_fn(text, src, tgt)` is injected so tests need no model.
  - Running `scripts/04_translate.py` yields `data/expanded/translations.jsonl`, idempotent on `(base_id, variant, lang)`.

- [ ] **Step 1: Write failing tests**

`tests/test_translate.py`:
```python
from femsympqa.translate import build_translation_rows, split_sentences


def test_split_sentences_english_and_arabic_punctuation():
    assert split_sentences("First one. Second one!") == ["First one.", "Second one!"]
    assert split_sentences("جملة أولى؟ جملة ثانية.") == ["جملة أولى؟", "جملة ثانية."]


def test_split_sentences_no_terminator_returns_whole_text():
    assert split_sentences("no terminator here") == ["no terminator here"]


def test_build_translation_rows():
    en_rows = [
        {"base_id": "pcos-s01", "variant": "can", "text": "My periods are irregular."},
        {"base_id": "pcos-s01", "variant": "p01", "text": "Periods all over the place."},
    ]

    def fake_translate(text, src, tgt):
        return f"[{tgt}] {text}"

    rows = build_translation_rows(en_rows, ["fr", "ar"], fake_translate, "2026-07-05")
    assert len(rows) == 4
    fr_can = next(r for r in rows if r["variant"] == "can" and r["lang"] == "fr")
    assert fr_can["text"] == "[fr] My periods are irregular."
    assert fr_can["base_id"] == "pcos-s01"
    assert fr_can["prompt_id"] == "nllb-beam4-v1"
    assert fr_can["generated_at"] == "2026-07-05"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python -m pytest tests/test_translate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'femsympqa.translate'`

- [ ] **Step 3: Implement**

`src/femsympqa/translate.py`:
```python
import re
from typing import Callable

LANG_CODES = {"en": "eng_Latn", "fr": "fra_Latn", "ar": "arb_Arab"}
PROMPT_ID = "nllb-beam4-v1"
MODEL_NAME = "facebook/nllb-200-distilled-600M"

_SENT_RE = re.compile(r"(?<=[.!?؟])\s+")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_RE.split(text.strip()) if p.strip()]
    return parts or [text.strip()]


class NllbTranslator:
    def __init__(self, model_name: str = MODEL_NAME):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def translate(self, text: str, src: str, tgt: str) -> str:
        self.tokenizer.src_lang = LANG_CODES[src]
        out = []
        for sentence in split_sentences(text):
            inputs = self.tokenizer(sentence, return_tensors="pt")
            tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(LANG_CODES[tgt]),
                num_beams=4,
                do_sample=False,
                max_new_tokens=256,
            )
            out.append(self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0])
        return " ".join(out)


def build_translation_rows(
    en_rows: list[dict],
    targets: list[str],
    translate_fn: Callable[[str, str, str], str],
    generated_at: str,
) -> list[dict]:
    rows = []
    for en_row in en_rows:
        for tgt in targets:
            rows.append({
                "base_id": en_row["base_id"],
                "variant": en_row["variant"],
                "lang": tgt,
                "text": translate_fn(en_row["text"], "en", tgt),
                "generated_at": generated_at,
                "prompt_id": PROMPT_ID,
            })
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python -m pytest tests/test_translate.py -v`
Expected: 3 PASS

- [ ] **Step 5: Write the stage script**

`scripts/04_translate.py`:
```python
"""Stage 4: translate all EN variants to FR and AR with NLLB-200 (CPU batch job).

Idempotent on (base_id, variant, lang) — safe to interrupt and re-run.
Progress is appended after every 50 rows.
"""
from datetime import date
from pathlib import Path

from femsympqa.io_utils import append_jsonl, read_json, read_jsonl
from femsympqa.schemas import BaseEntry
from femsympqa.translate import NllbTranslator, build_translation_rows

FROZEN_SINGLE = Path("data/base/base_entries.json")
FROZEN_COMPOUND = Path("data/base/base_entries_compound.json")
PARAPHRASES = Path("data/expanded/paraphrases_en.jsonl")
OUTPUT = Path("data/expanded/translations.jsonl")
TARGETS = ["fr", "ar"]
CHUNK = 50


def load_en_rows() -> list[dict]:
    rows = []
    for r in read_json(FROZEN_SINGLE) + read_json(FROZEN_COMPOUND):
        entry = BaseEntry.model_validate(r)
        rows.append({"base_id": entry.base_id, "variant": "can", "text": entry.canonical_query})
    for p in read_jsonl(PARAPHRASES):
        rows.append({"base_id": p["base_id"], "variant": p["variant"], "text": p["text"]})
    return rows


def main() -> None:
    en_rows = load_en_rows()
    done = set()
    if OUTPUT.exists():
        done = {(r["base_id"], r["variant"], r["lang"]) for r in read_jsonl(OUTPUT)}

    pending = [r for r in en_rows
               if any((r["base_id"], r["variant"], t) not in done for t in TARGETS)]
    print(f"{len(en_rows)} EN rows, {len(pending)} pending translation")
    if not pending:
        return

    translator = NllbTranslator()
    today = date.today().isoformat()
    for i in range(0, len(pending), CHUNK):
        chunk = pending[i:i + CHUNK]
        rows = build_translation_rows(chunk, TARGETS, translator.translate, today)
        rows = [r for r in rows if (r["base_id"], r["variant"], r["lang"]) not in done]
        append_jsonl(OUTPUT, rows)
        print(f"progress: {min(i + CHUNK, len(pending))}/{len(pending)} EN rows translated")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run for real** (long-running CPU job — hours; run in background, safe to interrupt/resume)

Run: `.\.venv\Scripts\python scripts/04_translate.py`
First run downloads the 2.5 GB NLLB checkpoint. Expected: progress lines until done; `data/expanded/translations.jsonl` ends with `2 × (EN row count)` rows. Spot-check: a few FR rows read as French, AR rows as Arabic script.

- [ ] **Step 7: Commit**

```powershell
git add src/femsympqa/translate.py scripts/04_translate.py tests/test_translate.py data/expanded
git commit -m "feat: stage 4 NLLB EN->FR/AR translation with resumable batch run"
```

---

### Task 9: Stage 4b — back-translation QC

**Files:**
- Create: `scripts/04b_backtranslate_qc.py`
- Test: (reuses tested components — `NllbTranslator`, `cosine`; the sampling function gets a unit test) `tests/test_backtranslate_qc.py`, `src/femsympqa/qc_sampling.py`

**Interfaces:**
- Consumes: `data/expanded/translations.jsonl`, the EN rows (same loader as Task 8), `translate.NllbTranslator`, `paraphrase.cosine`.
- Produces:
  - `qc_sampling.sample_fraction(rows: list[dict], fraction: float, seed: int) -> list[dict]` — deterministic sample of ~`fraction` of rows
  - Running `scripts/04b_backtranslate_qc.py` back-translates a 10% sample to EN, embeds original vs back-translation (all-MiniLM-L6-v2), flags cosine < 0.70 to `data/errors/backtranslation_flags.jsonl`, prints summary.

- [ ] **Step 1: Write failing test**

`tests/test_backtranslate_qc.py`:
```python
from femsympqa.qc_sampling import sample_fraction


def test_sample_fraction_deterministic_and_sized():
    rows = [{"i": i} for i in range(200)]
    s1 = sample_fraction(rows, 0.10, seed=7)
    s2 = sample_fraction(rows, 0.10, seed=7)
    assert s1 == s2
    assert len(s1) == 20
    assert all(r in rows for r in s1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python -m pytest tests/test_backtranslate_qc.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'femsympqa.qc_sampling'`

- [ ] **Step 3: Implement**

`src/femsympqa/qc_sampling.py`:
```python
import random


def sample_fraction(rows: list[dict], fraction: float, seed: int) -> list[dict]:
    k = max(1, round(len(rows) * fraction))
    return random.Random(seed).sample(rows, k)
```

`scripts/04b_backtranslate_qc.py`:
```python
"""Stage 4b: back-translation QC on a 10% sample of translations.

FR/AR -> EN via NLLB, then cosine(original EN, back-translated EN) with
all-MiniLM-L6-v2. Similarity < 0.70 is flagged for manual review.
"""
from pathlib import Path

from sentence_transformers import SentenceTransformer

from femsympqa.io_utils import read_jsonl, write_jsonl
from femsympqa.paraphrase import cosine
from femsympqa.qc_sampling import sample_fraction
from femsympqa.translate import NllbTranslator

TRANSLATIONS = Path("data/expanded/translations.jsonl")
FLAGS = Path("data/errors/backtranslation_flags.jsonl")
THRESHOLD = 0.70
FRACTION = 0.10
SEED = 7

import sys
sys.path.insert(0, "scripts")
from importlib import import_module
load_en_rows = import_module("04_translate").load_en_rows  # reuse the EN loader


def main() -> None:
    en_by_key = {(r["base_id"], r["variant"]): r["text"] for r in load_en_rows()}
    sample = sample_fraction(read_jsonl(TRANSLATIONS), FRACTION, SEED)
    print(f"back-translating {len(sample)} sampled rows")

    translator = NllbTranslator()
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    flags, sims = [], []
    for row in sample:
        original = en_by_key[(row["base_id"], row["variant"])]
        back = translator.translate(row["text"], row["lang"], "en")
        vec_orig, vec_back = embedder.encode([original, back], normalize_embeddings=True).tolist()
        sim = cosine(vec_orig, vec_back)
        sims.append(sim)
        if sim < THRESHOLD:
            flags.append({"base_id": row["base_id"], "variant": row["variant"],
                          "lang": row["lang"], "original": original,
                          "back_translation": back, "similarity": round(sim, 4)})

    write_jsonl(FLAGS, flags)
    print(f"mean similarity: {sum(sims) / len(sims):.3f}; "
          f"flagged {len(flags)}/{len(sample)} -> {FLAGS}")


if __name__ == "__main__":
    main()
```

Note: `04_translate.py` starts with a digit, so it is imported via `import_module` after adding `scripts` to `sys.path` — do not rename the script (stage numbering is a spec convention).

- [ ] **Step 4: Run test, then run the QC for real**

Run: `.\.venv\Scripts\python -m pytest tests/test_backtranslate_qc.py -v` → 1 PASS
Run: `.\.venv\Scripts\python scripts/04b_backtranslate_qc.py`
Expected: mean similarity roughly ≥ 0.80; a small flag file. Record the mean similarity figure — it goes in the thesis. **Manual QC (Mazen):** read ~50 AR and ~50 FR rows from `translations.jsonl`; note any systematic problems in `data/errors/manual_review_notes.md`.

- [ ] **Step 5: Commit**

```powershell
git add src/femsympqa/qc_sampling.py scripts/04b_backtranslate_qc.py tests/test_backtranslate_qc.py data/errors
git commit -m "feat: stage 4b back-translation QC with deterministic sampling"
```

---

### Task 10: Stage 5 — assemble, validate, stats

**Files:**
- Create: `src/femsympqa/assemble.py`, `scripts/05_assemble.py`
- Test: `tests/test_assemble.py`, `tests/fixtures/golden_femsympqa.jsonl`

**Interfaces:**
- Consumes: frozen base entries (single + compound), `paraphrases_en.jsonl`, `translations.jsonl`, all schemas.
- Produces:
  - `assemble.record_id(base_id: str, variant: str, lang: str) -> str` → `fsq-{base_id}-{variant}-{lang}`
  - `assemble.assemble_records(entries: list[BaseEntry], paraphrases: list[dict], translations: list[dict], canonical_generated_at: str) -> list[FinalRecord]` — raises `ValueError` on any integrity violation (unknown `base_id`, translation with no EN parent); drops exact-duplicate query text within `(base_id, lang)`
  - `assemble.build_stats(records: list[FinalRecord]) -> str` — markdown with a condition × lang count table, a risk × lang count table, and totals
  - Running `scripts/05_assemble.py` yields `data/final/femsympqa.jsonl` and `data/final/stats.md`.

- [ ] **Step 1: Write failing tests**

`tests/test_assemble.py`:
```python
import pytest

from femsympqa.assemble import assemble_records, build_stats, record_id
from femsympqa.schemas import BaseEntry, Evidence

ENTRY = BaseEntry(
    base_id="pcos-s01", condition="PCOS", symptom="irregular periods",
    canonical_query="My periods are all over the place.", risk_level="moderate",
    recommended_action="See a GP.",
    evidence=[Evidence(source_url="https://a", quote="see a GP")],
)

PARAPHRASES = [{"base_id": "pcos-s01", "variant": "p01", "text": "Cycle is chaotic lately.",
                "register": "casual", "generated_at": "2026-07-05", "prompt_id": "paraphrase_v1"}]

TRANSLATIONS = [
    {"base_id": "pcos-s01", "variant": "can", "lang": "fr", "text": "Mes règles sont irrégulières.",
     "generated_at": "2026-07-06", "prompt_id": "nllb-beam4-v1"},
    {"base_id": "pcos-s01", "variant": "p01", "lang": "ar", "text": "دورتي غير منتظمة.",
     "generated_at": "2026-07-06", "prompt_id": "nllb-beam4-v1"},
]


def test_record_id_format():
    assert record_id("pcos-s01", "p07", "ar") == "fsq-pcos-s01-p07-ar"


def test_assemble_produces_expected_records():
    records = assemble_records([ENTRY], PARAPHRASES, TRANSLATIONS, "2026-07-04")
    by_id = {r.id: r for r in records}
    assert set(by_id) == {
        "fsq-pcos-s01-can-en", "fsq-pcos-s01-p01-en",
        "fsq-pcos-s01-can-fr", "fsq-pcos-s01-p01-ar",
    }
    canonical = by_id["fsq-pcos-s01-can-en"]
    assert canonical.variant_type == "canonical" and canonical.parent_id is None
    fr = by_id["fsq-pcos-s01-can-fr"]
    assert fr.variant_type == "translation"
    assert fr.parent_id == "fsq-pcos-s01-can-en"
    assert fr.risk_level == ENTRY.risk_level  # labels inherited
    para = by_id["fsq-pcos-s01-p01-en"]
    assert para.variant_type == "paraphrase" and para.parent_id is None


def test_assemble_rejects_unknown_base_id():
    bad = [dict(PARAPHRASES[0], base_id="nope-s99")]
    with pytest.raises(ValueError, match="nope-s99"):
        assemble_records([ENTRY], bad, [], "2026-07-04")


def test_assemble_rejects_orphan_translation():
    orphan = [dict(TRANSLATIONS[1], variant="p77")]
    with pytest.raises(ValueError, match="p77"):
        assemble_records([ENTRY], PARAPHRASES, orphan, "2026-07-04")


def test_assemble_drops_exact_duplicate_text_same_lang():
    dup = PARAPHRASES + [dict(PARAPHRASES[0], variant="p02")]
    records = assemble_records([ENTRY], dup, [], "2026-07-04")
    en_texts = [r.query for r in records if r.lang == "en"]
    assert len(en_texts) == len(set(en_texts))


def test_build_stats_contains_counts():
    records = assemble_records([ENTRY], PARAPHRASES, TRANSLATIONS, "2026-07-04")
    stats = build_stats(records)
    assert "PCOS" in stats and "Total records: 4" in stats
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python -m pytest tests/test_assemble.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'femsympqa.assemble'`

- [ ] **Step 3: Implement**

`src/femsympqa/assemble.py`:
```python
import pandas as pd

from femsympqa.schemas import BaseEntry, FinalRecord, Provenance

LLM_MODEL = "claude-opus-4-8"
NLLB_MODEL = "facebook/nllb-200-distilled-600M"


def record_id(base_id: str, variant: str, lang: str) -> str:
    return f"fsq-{base_id}-{variant}-{lang}"


def assemble_records(
    entries: list[BaseEntry],
    paraphrases: list[dict],
    translations: list[dict],
    canonical_generated_at: str,
) -> list[FinalRecord]:
    by_base = {e.base_id: e for e in entries}
    records: list[FinalRecord] = []
    en_variants: set[tuple[str, str]] = set()

    def make(entry: BaseEntry, variant: str, lang: str, variant_type: str,
             query: str, parent_id: str | None, provenance: Provenance) -> FinalRecord:
        return FinalRecord(
            id=record_id(entry.base_id, variant, lang), base_id=entry.base_id,
            parent_id=parent_id, lang=lang, variant_type=variant_type, query=query,
            condition=entry.condition, risk_level=entry.risk_level,
            recommended_action=entry.recommended_action, evidence=entry.evidence,
            provenance=provenance,
        )

    for entry in entries:
        records.append(make(
            entry, "can", "en", "canonical", entry.canonical_query, None,
            Provenance(generated_by=f"human-reviewed extraction ({LLM_MODEL} draft)",
                       generated_at=canonical_generated_at, prompt_id="extract_v1")))
        en_variants.add((entry.base_id, "can"))

    for p in paraphrases:
        if p["base_id"] not in by_base:
            raise ValueError(f"paraphrase references unknown base_id: {p['base_id']}")
        entry = by_base[p["base_id"]]
        records.append(make(
            entry, p["variant"], "en", "paraphrase", p["text"], None,
            Provenance(generated_by=LLM_MODEL, generated_at=p["generated_at"],
                       prompt_id=p["prompt_id"])))
        en_variants.add((p["base_id"], p["variant"]))

    for t in translations:
        if t["base_id"] not in by_base:
            raise ValueError(f"translation references unknown base_id: {t['base_id']}")
        if (t["base_id"], t["variant"]) not in en_variants:
            raise ValueError(
                f"translation has no EN parent: {t['base_id']}/{t['variant']}")
        entry = by_base[t["base_id"]]
        records.append(make(
            entry, t["variant"], t["lang"], "translation", t["text"],
            record_id(t["base_id"], t["variant"], "en"),
            Provenance(generated_by=NLLB_MODEL, generated_at=t["generated_at"],
                       prompt_id=t["prompt_id"])))

    # drop exact duplicate query text within (base_id, lang), keeping first
    seen: set[tuple[str, str, str]] = set()
    deduped: list[FinalRecord] = []
    for rec in records:
        key = (rec.base_id, rec.lang, rec.query.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rec)
    return deduped


def build_stats(records: list[FinalRecord]) -> str:
    df = pd.DataFrame([{
        "condition": r.condition, "lang": r.lang,
        "risk": r.risk_level.value, "variant_type": r.variant_type,
    } for r in records])
    lines = ["# FemSympQA dataset statistics", ""]
    lines += [f"Total records: {len(df)}", ""]
    lines += ["## Records by condition and language", "",
              df.pivot_table(index="condition", columns="lang", aggfunc="size",
                             fill_value=0).to_markdown(), ""]
    lines += ["## Records by risk level and language", "",
              df.pivot_table(index="risk", columns="lang", aggfunc="size",
                             fill_value=0).to_markdown(), ""]
    lines += ["## Records by variant type", "",
              df["variant_type"].value_counts().to_markdown(), ""]
    return "\n".join(lines)
```

Note: `to_markdown()` uses `tabulate`, which is already in the Task 1 dependency list.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python -m pytest tests/test_assemble.py -v`
Expected: 6 PASS

- [ ] **Step 5: Golden-file test**

Generate once, eyeball it, then freeze:
```powershell
.\.venv\Scripts\python -c "from tests.test_assemble import ENTRY, PARAPHRASES, TRANSLATIONS; from femsympqa.assemble import assemble_records; from femsympqa.io_utils import write_jsonl; write_jsonl('tests/fixtures/golden_femsympqa.jsonl', [r.model_dump() for r in assemble_records([ENTRY], PARAPHRASES, TRANSLATIONS, '2026-07-04')])"
```
Manually read `tests/fixtures/golden_femsympqa.jsonl` (4 records, correct ids/parents/labels), then append to `tests/test_assemble.py`:
```python
def test_golden_file():
    from femsympqa.io_utils import read_jsonl
    expected = read_jsonl("tests/fixtures/golden_femsympqa.jsonl")
    actual = [r.model_dump() for r in
              assemble_records([ENTRY], PARAPHRASES, TRANSLATIONS, "2026-07-04")]
    assert actual == expected
```
Run: `.\.venv\Scripts\python -m pytest tests/test_assemble.py -v` → 7 PASS

- [ ] **Step 6: Write the stage script and run it**

`scripts/05_assemble.py`:
```python
"""Stage 5: assemble the final FemSympQA dataset with validation and stats."""
from pathlib import Path

from femsympqa.assemble import assemble_records, build_stats
from femsympqa.io_utils import read_json, read_jsonl, write_jsonl
from femsympqa.schemas import BaseEntry

FROZEN_SINGLE = Path("data/base/base_entries.json")
FROZEN_COMPOUND = Path("data/base/base_entries_compound.json")
PARAPHRASES = Path("data/expanded/paraphrases_en.jsonl")
TRANSLATIONS = Path("data/expanded/translations.jsonl")
OUT_DATA = Path("data/final/femsympqa.jsonl")
OUT_STATS = Path("data/final/stats.md")


def main() -> None:
    entries = [BaseEntry.model_validate(r)
               for r in read_json(FROZEN_SINGLE) + read_json(FROZEN_COMPOUND)]
    # canonical entries were frozen at review time; use the extraction draft's
    # commit date — recorded here once, at assembly
    records = assemble_records(entries, read_jsonl(PARAPHRASES),
                               read_jsonl(TRANSLATIONS), canonical_generated_at="2026-07-05")
    write_jsonl(OUT_DATA, [r.model_dump() for r in records])
    OUT_STATS.write_text(build_stats(records), encoding="utf-8")
    print(f"wrote {len(records)} records -> {OUT_DATA}")
    print(f"wrote stats -> {OUT_STATS}")


if __name__ == "__main__":
    main()
```

Run: `.\.venv\Scripts\python scripts/05_assemble.py`
Expected: `wrote N records` with N in roughly 3,000–4,500. If N is below 3,000, the levers are: raise `PER_CONDITION` in `02b_compound.py` (then re-run 2b→3→4 for new entries only — idempotency makes this cheap) or check how many paraphrases QC dropped. Read `data/final/stats.md` — counts should be roughly balanced across en/fr/ar.

- [ ] **Step 7: Commit**

```powershell
git add pyproject.toml src/femsympqa/assemble.py scripts/05_assemble.py tests/test_assemble.py tests/fixtures data/final
git commit -m "feat: stage 5 assembly with integrity validation, dedupe, and stats"
```

---

### Task 11: README, full verification, GitHub push

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: everything.
- Produces: documented, verified, pushed repository.

- [ ] **Step 1: Complete the README pipeline section**

Replace the `## Pipeline run order` placeholder in `README.md` with:
```markdown
## Pipeline run order

| # | Command | Output | Notes |
|---|---------|--------|-------|
| 1 | `python scripts/01_scrape.py` | `data/raw/*.html` | idempotent snapshots |
| 2 | `python scripts/02_extract.py` | `data/base/base_entries_draft.json`, `evidence_passages.json` | needs `.env` |
| — | **human review** → save `data/base/base_entries.json`, then `02_extract.py --validate-frozen` | | |
| 2b | `python scripts/02b_compound.py` | `base_entries_compound_draft.json` | then review + freeze |
| 3 | `python scripts/03_paraphrase.py submit` / `collect` | `data/expanded/paraphrases_en.jsonl` | Batch API |
| 4 | `python scripts/04_translate.py` | `data/expanded/translations.jsonl` | CPU, hours, resumable |
| 4b | `python scripts/04b_backtranslate_qc.py` | `data/errors/backtranslation_flags.jsonl` | QC sample |
| 5 | `python scripts/05_assemble.py` | `data/final/femsympqa.jsonl`, `stats.md` | final dataset |

All stages are idempotent; flagged items land in `data/errors/`.
Tests: `python -m pytest` (no network or API key needed).
```

- [ ] **Step 2: Run the full test suite**

Run: `.\.venv\Scripts\python -m pytest -v`
Expected: all tests pass (roughly 27), zero skips, no network use.

- [ ] **Step 3: Final artifact sanity check**

```powershell
.\.venv\Scripts\python -c "from femsympqa.io_utils import read_jsonl; from femsympqa.schemas import FinalRecord; rows = read_jsonl('data/final/femsympqa.jsonl'); recs = [FinalRecord.model_validate(r) for r in rows]; print(len(recs), 'records validate');"
```
Expected: `N records validate` with no exception.

- [ ] **Step 4: Commit and push to GitHub** (Mazen: create an empty repo `HerHealthGPT` on github.com first)

```powershell
git add README.md
git commit -m "docs: pipeline run order and verification instructions"
git remote add origin https://github.com/<mazen-account>/HerHealthGPT.git
git push -u origin master
```

---

## Plan self-review notes

- **Spec coverage:** scrape (T3), extract + review gate + evidence passages (T5), compound entries (T6), paraphrase + Batch API + QC (T7), NLLB translation (T8), back-translation QC + manual review (T9), assemble + integrity + dedupe + stats (T10), idempotency and `data/errors/` throughout, README/repo/GitHub (T1, T11), pytest with fixtures and golden file (T2–T10). Cross-source conflict rule tested in T5.
- **Known deviations from spec (intentional, cosmetic):** record IDs are structured (`fsq-{base_id}-{variant}-{lang}`) instead of sequential (`femsympqa-0421`) for traceability; paraphrase QC runs in stage 3 (before translation, so duplicates aren't needlessly translated) with the final exact-dup check in stage 5.
- **Type consistency check:** `RiskLevel`/`Evidence`/`BaseEntry`/`FinalRecord`/`Provenance` defined once in T2 and imported everywhere; paraphrase row shape `{base_id, variant, text, register, generated_at, prompt_id}` consumed identically in T8/T10; translation row shape `{base_id, variant, lang, text, generated_at, prompt_id}` consumed in T10.
