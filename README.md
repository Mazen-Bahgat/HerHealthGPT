# HerHealthGPT-LU (LUHME 2026)

**Current project:** *HerHealthGPT-LU — A Multilingual Benchmark for Evaluating LLM
Understanding of Women's-Health Symptom Communication (Menstrual, PCOS, and Fertility)*,
submitted to LUHME 2026 (deadline 15 July 2026).
Design spec (source of truth): `docs/superpowers/specs/2026-07-06-herhealthgpt-lu-design.md`.
Team: Mazen, Hana, Hassan, Mariam. Supervisor: Dr. Rahatara Ferdousi.

The project pivoted from an earlier direction (a symptom→risk-level chatbot/RAG system,
see "Earlier work" below) to a language-*understanding* benchmark: how well LLMs interpret
clinical, layperson, indirect, ambiguous, and emotionally-expressed descriptions of
women's-health symptoms across English, French, and Arabic, and whether multilingual
fine-tuning reduces misunderstanding.

## Setup

1. Python 3.12, then: `python -m venv .venv`
2. `.\.venv\Scripts\python -m pip install -e .[dev]`
3. Copy `.env.example` to `.env` and add your OpenAI API key (needed for
   `scripts/regenerate_style_variants_and_gold.py`; not needed for the rest).

## HerHealthGPT-LU pipeline (current)

The frozen 90-seed English benchmark nucleus (`HerHealthGPT-LU_seed/seeds_en_v1.csv`)
and its build pipeline (`HerHealthGPT-LU_seed/build_seed.py`) already exist — see
`HerHealthGPT-LU_seed/README.md` for that deliverable's own docs. Root-level `scripts/`
holds the pipeline stages built on top of it:

| # | Script | Output | Notes |
|---|--------|--------|-------|
| 1 | `scripts/scrape_grounding_sources.py` | `HerHealthGPT-LU_seed/grounding_sources/` | NHS/CDC/NICHD evidence pages; 5/6 fetched, CDC common-concerns.html is bot-blocked (needs manual fetch) |
| 2 | `scripts/complete_gold_labels.py` | `seeds_en_v1.csv` (gold fields), `gold_label_completion_report.md` | **Done, deterministic, no API key needed.** Fills gold_risk_level/gold_action/evidence_quote/source_url/requires_clarification (previously blank). 61/90 seeds grounded. |
| 3 | `scripts/merge_manual_style_variants.py` | `seeds_en_v1.csv` (style_text), `regeneration_report.md` | **Done.** Merges 450 Claude-authored style variants (`HerHealthGPT-LU_seed/style_variants_manual.json`) — the OpenAI-based alternative below hit a billing quota error before producing output, so Claude wrote all 450 variants directly instead. 100% distinct-string ratio, zero canonical collisions. Human spot-check of the report still required before freezing. |
| 3b | `scripts/regenerate_style_variants_and_gold.py` | same outputs as above | Alternative/second-pass path via OpenAI (`gpt-5.5`) if `OPENAI_API_KEY` billing gets resolved later. Not required — step 3 already produced verified-good data. |
| 4 | `scripts/build_ft_corpus.py` | `HerHealthGPT-LU_seed/ft_corpus_v1.jsonl`, `leakage_log.csv` | Silver fine-tuning corpus, dual leakage key vs the frozen seeds. Already run: 2,700 pairs (900/category). |
| 5 | `scripts/build_translator_handoff.py` | `HerHealthGPT-LU_seed/translation_handoff/` | AR translated in-house by the team (fluent speakers); FR outsourced to a professional localization agency — see `fr_agency_brief.md`. Already re-run against the corrected style text from step 3. |
| 6 | `scripts/run_inference.py` | one JSONL per model | Scaffold against any OpenAI-compatible endpoint (vLLM/TGI). Needs cluster wiring (`--base-url`) before real runs. |
| 7 | `scripts/evaluate.py` | — | Not yet built — needs inference results first. |

Run `pytest` for the FemSympQA-era test suite (still green, but see below).

## Earlier work: FemSympQA dataset pipeline

Before the pivot, this repo built a different pipeline (symptom→condition/risk/action
dataset, `src/femsympqa/`, `scripts/01_scrape.py` / `02_extract.py`) — see
`docs/superpowers/specs/2026-07-05-femsympqa-dataset-pipeline-design.md`. That work is
superseded by the direction above but kept in place; its scraped NHS/CDC snapshots
(`data/raw/`) and HTML-cleaning module (`src/femsympqa/html_clean.py`) are reused by
`scripts/scrape_grounding_sources.py` above.
