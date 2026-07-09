# Decisions log — HerHealthGPT-LU English seed v1

## Translation ownership (decided 2026-07-09)

Arabic: translated in-house by team members who are fluent native/near-native speakers —
not a machine-translation-first-pass + external validation workflow. French: outsourced
to a professional translation and localization agency. See `../scripts/build_translator_handoff.py`
and `translation_handoff/fr_agency_brief.md`. Supersedes the original
GPT/NLLB-first-pass plan in the design spec §2A for both languages.

## Style-variant regeneration (started 2026-07-09)

`generate_styles()` in `build_seed.py` is a fixed-template rewriter, verified to drop
clinical content on real seeds (see `../HerHealthGPT-LU_seed/README.md` §Status for the
menst-001/menst-002 example). `../scripts/regenerate_style_variants_and_gold.py`
regenerates all 5 non-canonical style rows per seed via LLM under a meaning-preservation
rubric, and separately completes `gold_risk_level`/`gold_action`/`requires_clarification`
against the evidence in `grounding_sources/`. Requires `ANTHROPIC_API_KEY` — not yet run.
Until it runs, `seeds_en_v1.csv`'s style_text columns (other than `canonical`) should be
treated as **not yet meaning-preserving** and not sent to translators or reviewers.

## Finalized fine-tuning decision (locked)

Finalized target for training split: **800-1,000 per category** (**2,400-3,000 total**) for **SFT / multilingual adaptation**.  
Recommended concrete default: **900 per category (2,700 total)**.

| Split | Per-category count | Total count | Purpose |
|----------|-------------------:|------------:|---------|
| Benchmark seed v1 (current, frozen nucleus) | 30 | 90 seeds / 540 rows | Benchmark/evaluation lineage only |
| Validation (non-benchmark only) | TBD (small balanced set) | TBD | Tuning-time validation |
| Fine-tuning train (non-benchmark only) | 800-1,000 | 2,400-3,000 | SFT / multilingual adaptation |
| Fine-tuning train (recommended default) | 900 | 2,700 | Default first run |

Leakage exclusion rule (explicit): if a benchmark-linked item has key `source_dataset + source_row_id`, then all rows with that same key are excluded from validation/fine-tuning pulls. Operational reference: `leakage_note.md`.

## Phase-1 Clear-tier yield (post-dedup, before quality filter)

| Category | Clear | Borderline |
|----------|------:|-----------:|
| menstrual | 4000 | (see phase2 CSV) |
| pcos | 1550 | (see phase2 CSV) |
| fertility | 1219 | (see phase2 CSV) |

## Quality filter → eligible Clear pool

Filters remove myth-only FAQs (esp. fertility), male-patient context, texts lacking
attested clinical claim tokens, and extreme length outliers.

| Category | Eligible Clear |
|----------|---------------:|
| menstrual | 2724 |
| pcos | 1393 |
| fertility | 1154 |

## Proposed targets (locked after Phase-1+filter)

| Category | Target | Rationale |
|----------|-------:|-----------|
| menstrual | 30 | Abundant Clear; week-1 freeze ~tens; quality over volume → capped at 30 |
| pcos | 30 | Abundant Clear PCOS/PCOD patient language across HCM/iCliniq/MENST → 30 |
| fertility | 30 | Abundant Clear TTC/infertility pool after filters → 30 (ok) |

**Yield-constrained?** None at locked targets.

Not using the July-6 130-seed / 2-category plan. Three categories × ~30 = ~90 seeds.

## Source-mix policy

Target mix per category (~): up to ~20% MENST Set1 anchors (capped by sparse Clear Set1
eligible rows — training2K Set1 is only 117 total rows), ~25% MENST Set2, ~40% HealthCareMagic
authentic patient phrasing, ~15% iCliniq. Prefer Clear only in main set. Quotas auto-shrink
when a bucket is empty.

**Note:** MENST Set1 Clear eligible yield is single-digit overall after category split; the
achieved mix therefore skews toward HCM + Set2 for authentic symptom language while still
including available Set1 anchors.

## Borderline policy

Borderline candidates retained in `borderline_bucket.csv` with reason; **not promoted**
into `seeds_en_v1` while Clear eligible yield meets targets.

## Licensing / provenance

- HealthCareMagic-100k-Chat-Format-en: Apache-2.0
- MENST: MIT (dataset card)
- ChatDoctor-iCliniq: **license unverified** at time of seed build — usable for benchmark /
  seed construction with caution; do not redistribute as own data without clarifying upstream terms.

## Grounding policy

Evidence-only vs:
- NHS PCOS symptoms
- NHS heavy periods
- NHS infertility
- CDC Common Reproductive Health Concerns
- NICHD endometriosis / infertility

If a claim cannot be mapped safely → `needs_grounding_flag=true`, `gold_condition=NEEDS_GROUNDING`.
Heavy periods → NHS heavy periods; endometriosis/fibroids phrases → CDC common concerns;
PCOS/PCOD → NHS PCOS symptoms; TTC/infertility → NHS infertility.
Irregular/PMS/painful without an exact allowed page match → NEEDS_GROUNDING (do not invent).

## Style policy

6 rows/seed: `canonical` (verbatim patient text, light clean) + 5 register variants.
Variants may only reuse claim phrases attested in the patient text — no symptom invention.
