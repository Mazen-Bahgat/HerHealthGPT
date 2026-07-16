# Decisions log — HerHealthGPT-LU English seed v1

## Translation ownership (decided 2026-07-09)

Arabic: translated in-house by team members who are fluent native/near-native speakers —
not a machine-translation-first-pass + external validation workflow. French: outsourced
to a professional translation and localization agency. See `../scripts/build_translator_handoff.py`
and `translation_handoff/fr_agency_brief.md`. Supersedes the original
GPT/NLLB-first-pass plan in the design spec §2A for both languages.

## Gold-label completion (done 2026-07-09, deterministic)

`../scripts/complete_gold_labels.py` filled in `gold_risk_level`, `gold_action`,
`evidence_quote`, `source_url`, `requires_clarification` for all 90 seeds — previously
blank (`draft_grounding()` only ever set `gold_condition`). No LLM needed: all 5 fetched
evidence pages converge on NHS's "See a GP if:" convention with no urgent/emergency
language anywhere, so `gold_risk_level` is `see-doctor` for every grounded seed and as
the conservative default for `NEEDS_GROUNDING` seeds; `gold_action` quotes the matched
page's actual advice text. `requires_clarification` is a length/vagueness heuristic —
approximate, flagged `needs_human_review=true` on every row. 61/90 seeds grounded;
1 seed (`menst-017`) matched the still-inaccessible CDC page and is labeled accordingly
rather than given a fabricated action. See `gold_label_completion_report.md`.

## Style-variant regeneration (done 2026-07-09, Claude-authored manually)

`generate_styles()` in `build_seed.py` is a fixed-template rewriter, verified to drop
clinical content on real seeds (menst-001/menst-002 — different questions, one about
post-miscarriage timing, one about fertility — both collapsed to the identical templated
`clinical` variant). `../scripts/regenerate_style_variants_and_gold.py` was written to
fix this via the OpenAI API, but the account behind the provided key had no billing/quota
set up (`insufficient_quota` on the first call, before any output). Rather than block on
that, all 450 style rows (90 seeds x 5 styles) were written directly by Claude in this
conversation, one at a time, under the same meaning-preservation rubric the script would
have used, then merged via `../scripts/merge_manual_style_variants.py`. Verified before
merging: **100% distinct-string ratio** across all 450 variants (the old template
produced ~51 distinct strings across 90 seeds), zero variant equals its own canonical
text. `regenerate_style_variants_and_gold.py` remains available as an alternative or
second independent pass if `OPENAI_API_KEY` billing gets resolved later — not required
for the current data. `seeds_en_v1.csv`'s style_text columns are now meaning-preserving,
but **every row is still flagged `needs_human_review=true`** — a team member should spot
check `regeneration_report.md` (old vs new per seed) before the benchmark freezes.
Gold-label columns were not touched by this step (see the gold-label section above).

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
