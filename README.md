# HerHealthGPT-LU (LUHME 2026)

**Current project:** *HerHealthGPT-LU — A Multilingual Benchmark for Evaluating LLM
Understanding of Women's-Health Symptom Communication (Menstrual, PCOS, and Fertility)*,
submitted to LUHME 2026 (deadline 15 July 2026).
Current v2 execution spec (source of truth):
`docs/superpowers/specs/2026-07-15-new-benchmark-metrics-and-multilingual-ft-design.md`.
The 2026-07-06 design remains the historical v1 benchmark spec.
Team: Mazen, Hana, Hassan, Mariam. Supervisor: Dr. Rahatara Ferdousi.

The project pivoted from an earlier direction (a symptom→risk-level chatbot/RAG system,
see "Earlier work" below) to a language-*understanding* benchmark: how well LLMs interpret
clinical, layperson, indirect, ambiguous, and emotionally-expressed descriptions of
women's-health symptoms across English, French, and Arabic, and whether multilingual
fine-tuning reduces misunderstanding.

## Setup

1. Python 3.12, then: `python -m venv .venv`
2. `.\.venv\Scripts\python -m pip install -e .[dev]`
3. Copy `.env.example` to `.env` and add your OpenAI API key when regenerating
   model-assisted artifacts (`scripts/regenerate_style_variants_and_gold.py` or
   `scripts/translate_handoff_fr.mjs`). Validation and ingest do not need it.

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
| 5 (v1, historical) | `scripts/build_translator_handoff.py` | `HerHealthGPT-LU_seed/translation_handoff/` | The v1 plan records in-house AR and professional-agency FR. This provenance applies only to the historical v1 handoff. |
| 5 (v2, current) | `scripts/build_translation_handoff_v2.py`, `scripts/translate_handoff_fr.mjs` | `Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/translation_handoff_v2/fr.csv` | French silver FT handoff completed with `gpt-5.6-sol`; automated and AI-assisted QA complete, native-French review pending. Arabic remains separate. |
| 6 | `scripts/run_inference.py` | one JSONL per model | Scaffold against any OpenAI-compatible endpoint (vLLM/TGI). Needs cluster wiring (`--base-url`) before real runs. |
| 7 | `scripts/evaluate.py` | — | Not yet built — needs inference results first. |

### French v2 translation provenance

The current French artifact contains 3,580 rows (2,862 train / 718 validation)
for multilingual fine-tuning, not the 540-row evaluation benchmark. Generation
used OpenAI's Responses API (`gpt-5.6-sol`, low reasoning, Structured Outputs,
`store: false`). Corpus-wide validation found zero blocking errors; 404 unique
review jobs were triaged and 180 rows (5.03%) were reviewed across every style
and topic. Native-French human review is explicitly still pending. See the
[handoff README](Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/translation_handoff_v2/README.md)
and machine-readable provenance files beside `fr.csv`. Verified ingest emits
2,858 train / 718 validation rows after its leakage guard removes four natural
French cross-split duplicates.

Run `pytest` for the FemSympQA-era test suite (still green, but see below).

## Earlier work: FemSympQA dataset pipeline

Before the pivot, this repo built a different pipeline (symptom→condition/risk/action
dataset, `src/femsympqa/`, `scripts/01_scrape.py` / `02_extract.py`) — see
`docs/superpowers/specs/2026-07-05-femsympqa-dataset-pipeline-design.md`. That work is
superseded by the direction above but kept in place; its scraped NHS/CDC snapshots
(`data/raw/`) and HTML-cleaning module (`src/femsympqa/html_clean.py`) are reused by
`scripts/scrape_grounding_sources.py` above.
