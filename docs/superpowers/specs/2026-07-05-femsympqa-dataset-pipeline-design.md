# FemSympQA Dataset Pipeline — Design

**Date:** 2026-07-05
**Project:** HerHealthGPT (Queen's University Master's grad project, supervisor Dr. Rahatara Ferdousi)
**Sub-project:** 1 of 4 (dataset pipeline → RAG prototype → risk classifier → evaluation harness)
**Status:** Approved by Mazen 2026-07-05

## Goal

Build a reproducible pipeline that turns five clinical web pages into **FemSympQA**: a
multilingual (English, French, Arabic) dataset mapping patient-style symptom queries to
conditions, risk levels, and recommended actions. This dataset is the foundation the
later RAG prototype, risk classifier, and evaluation harness all consume.

## Sources

| Condition | Source |
|---|---|
| PCOS | https://www.nhs.uk/conditions/polycystic-ovary-syndrome-pcos/symptoms/ |
| PCOS | https://www.cdc.gov/diabetes/basics/pcos.html |
| Heavy periods (menorrhagia) | https://www.nhs.uk/conditions/heavy-periods/ |
| Infertility (female) | https://www.nhs.uk/conditions/infertility/ |
| Endometriosis | https://www.nichd.nih.gov/health/topics/endometri/conditioninfo |

## Key decisions (with rationale)

1. **3-tier risk taxonomy derived from source urgency language.** Every label traces to
   the sources' own wording, which is defensible in the thesis:
   - `low` — self-care advice, "common and usually nothing to worry about"
   - `moderate` — "see a GP", non-urgent medical review
   - `high` — "urgent GP appointment", "call 999 / go to A&E", red-flag symptoms
2. **Hybrid generation tooling.** NLLB-200 (`facebook/nllb-200-distilled-600M`) for
   EN→FR/AR translation, exactly as named in the project brief (free, reproducible,
   good FR/AR quality). An LLM API (Anthropic, model pinned to `claude-opus-4-8`) for
   paraphrasing, where the brief's T5/PEGASUS would produce stiff, low-diversity text —
   patient-style vocabulary is objective #2 of the project, so quality matters most here.
   Framed in the thesis as LLM-assisted data augmentation with logged prompts.
3. **Local Python repo with staged scripts** (not Colab notebooks, not one monolithic
   build command). Each stage is independently re-runnable; intermediate artifacts are
   committed; no GPU is needed anywhere (NLLB-600M runs on CPU as a batch job).
4. **Human review gate on extraction.** LLM-extracted base entries are a draft until
   Mazen reviews and freezes them — gives the thesis a "human-verified against clinical
   sources" claim at the cost of about an hour of review.
5. **Scale target: ~3,000–4,500 final records** via 15–20 paraphrases per base entry
   plus compound-symptom queries, across 3 languages.

## Data model

### Base entries — `data/base/base_entries.json`

One per distinct symptom–condition pair (est. 40–60 single-symptom entries, plus
~20–30 compound-symptom entries combining 2–3 co-occurring symptoms of the same
condition). Fields:

| Field | Description |
|---|---|
| `base_id` | Stable ID, e.g. `pcos-s03` |
| `condition` | One of: PCOS, heavy periods, infertility (female), endometriosis |
| `symptom` | Clinical symptom name(s) |
| `canonical_query` | Canonical English patient-style query |
| `risk_level` | `low` / `moderate` / `high` per the mapping rule above |
| `recommended_action` | Action derived from source text |
| `evidence` | List of `{source_url, quote}` — the exact urgency/symptom passages |
| `conflict_note` | Present when sources disagreed on urgency (higher tier wins) |

### Final records — `data/final/femsympqa.jsonl`

One JSON object per line, one per query variant. Per base entry: 1 canonical +
15–20 paraphrases = 16–21 English variants, each translated to FR and AR → ~48–63
records per base entry. Across 60–90 base entries this yields the ~3,000–4,500 target.

| Field | Description |
|---|---|
| `id` | e.g. `femsympqa-0421` |
| `base_id` | Link back to the base entry |
| `parent_id` | For translations: the `id` of the English variant it was translated from; `null` for English records |
| `lang` | `en` / `fr` / `ar` |
| `variant_type` | `canonical` / `paraphrase` / `translation` |
| `query` | The patient-style query text |
| `condition`, `risk_level`, `recommended_action` | Inherited labels from base entry |
| `evidence` | Inherited from base entry |
| `provenance` | `{generated_by, generated_at, prompt_id}` — model + prompt version |

### Secondary artifact

The extraction stage also emits condition-level **evidence passages**
(`data/base/evidence_passages.json`) — the retrieval corpus the future RAG sub-project
will index. No extra work; it falls out of extraction.

## Pipeline stages

Five standalone scripts; each reads the previous stage's output file. All stages are
idempotent (skip already-processed IDs) so re-runs never re-pay for completed API work.

| Stage | Script | Input → Output | Tooling |
|---|---|---|---|
| 1. Scrape | `01_scrape.py` | URLs → `data/raw/*.html` snapshots with fetch dates | requests |
| 2. Extract | `02_extract.py` | snapshots → `base_entries_draft.json`; **human review** freezes `base_entries.json` | BeautifulSoup + Claude API |
| 3. Paraphrase | `03_paraphrase.py` | base entries → `data/expanded/paraphrases_en.jsonl` | Claude **Batch API**, `claude-opus-4-8` pinned, prompts versioned in `prompts/` |
| 4. Translate | `04_translate.py` | EN variants → `data/expanded/translations.jsonl` (FR + AR) | NLLB-200-distilled-600M, CPU, sentence-split, deterministic beam search |
| 5. Assemble | `05_assemble.py` | everything → `data/final/femsympqa.jsonl` + `stats.md` | pydantic validation, embedding dedupe |

Paraphrase prompts explicitly vary register (worried / casual / brief / detailed /
colloquial) to get diversity, not near-copies. Estimated total API cost: **under $15**
(Batch API is 50% off standard pricing; the workload is a few thousand short generations).

## Quality control

- **Cross-source validation:** entries appearing in multiple sources list all of them;
  urgency disagreements take the more conservative tier and are recorded in
  `conflict_note`.
- **Paraphrase QC:** each paraphrase embedded (Sentence-BERT) and compared to its
  canonical query — too similar → dropped as duplicate; too dissimilar → flagged as
  meaning drift to `data/errors/`. Yields a reportable semantic-fidelity figure.
- **Translation QC:** (1) automatic back-translation on a ~10% sample with similarity
  flagging; (2) manual review of ~50 records per language by Mazen.
- **Label integrity:** every final record must trace to a frozen, human-reviewed base
  entry; assembly fails otherwise.

## Error handling

- Every stage validates its input file against pydantic schemas at load — malformed
  upstream data fails loudly at start, not silently mid-run.
- Anthropic SDK auto-retries transient API errors; Batch API results are written raw
  to disk before any processing, so a crash never loses paid output.
- Failed or flagged items land in `data/errors/` with reasons, re-runnable in isolation.

## Testing

pytest, no network required:
- HTML parsers tested against checked-in snapshot fixtures.
- Risk-mapping rule tested against a table of urgency phrases.
- Schema validation, dedupe logic, and assembly tested on small synthetic cases,
  including a golden-file test for the final assembly step.

## Repository layout

```
HerHealthGPT/
├── HerHealthGPT-RF3.pdf         # project brief
├── README.md                    # setup + pipeline run order
├── pyproject.toml               # requests, beautifulsoup4, pydantic, anthropic,
│                                #   transformers, torch (CPU), sentence-transformers,
│                                #   pandas, pytest
├── .env.example                 # ANTHROPIC_API_KEY placeholder (.env gitignored)
├── prompts/                     # extract_v1.md, paraphrase_v1.md, ...
├── src/femsympqa/               # pydantic schemas, risk mapping, shared utils
├── scripts/                     # 01_scrape.py … 05_assemble.py
├── data/
│   ├── raw/                     # HTML snapshots (committed)
│   ├── base/                    # draft + frozen base entries (committed)
│   ├── expanded/                # paraphrases + translations (committed)
│   ├── final/                   # femsympqa.jsonl + stats.md (committed)
│   └── errors/                  # flagged/failed items
├── tests/                       # pytest + fixtures
└── docs/superpowers/specs/      # this document
```

All generated artifacts are committed so the supervisor can inspect every intermediate
step without running anything or holding an API key.

## Environment prerequisites (one-time)

- Install Python 3.12 on the Windows machine (currently absent — only the Microsoft
  Store stub exists), create a venv.
- `git init` + GitHub repository (required by the brief).
- Anthropic API key in `.env` (console.anthropic.com).

## Out of scope (later sub-projects)

- The HerHealthGPT RAG prototype (retrieval, agent reasoning).
- The risk-prediction classifier (scikit-learn/XGBoost).
- The evaluation harness (baselines, F1, BERTScore, ROUGE, cross-language consistency).
- Real patient data, clinical trials, model training (out of scope for the whole
  project per the brief).

## Deliverables

1. `data/final/femsympqa.jsonl` (~3,000–4,500 records)
2. Human-verified `data/base/base_entries.json`
3. `data/base/evidence_passages.json` (future RAG corpus)
4. `data/final/stats.md` (counts per condition × language × risk — thesis table)
5. Versioned prompts under `prompts/`
6. Tested, documented pipeline code
