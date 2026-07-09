# Seed dataset stats summary — HerHealthGPT-LU English v1

## Proposed vs achieved

| Category | Clear raw (post-dedup) | Clear eligible (quality filter) | Proposed target | Achieved | Yield note |
|----------|-----------------------:|--------------------------------:|----------------:|---------:|------------|
| menstrual | 4000 | 2724 | 30 | 30 | ok |
| pcos | 1550 | 1393 | 30 | 30 | ok |
| fertility | 1219 | 1154 | 30 | 30 | ok |

**Total seeds:** 90  
**Total rows (seed × style):** 540 (6 styles: canonical + clinical + layperson + indirect_cultural + ambiguous + emotionally_concerned)

## Confidence tiers (seeds)

| Tier | Count |
|------|------:|
| Clear | 90 |

## Source mix (seeds)

| Source | Count |
|--------|------:|
| HealthCareMagic-100k | 45 |
| MENST_train24K | 29 |
| ChatDoctor-iCliniq | 12 |
| MENST_training2K | 4 |

## Dedup

- Near-duplicate drop records: **678** (see `phase2_dedup_drops.jsonl`)

## Grounding

- Seeds with `needs_grounding_flag=true`: **29**
- Draft mappings only against NHS PCOS symptoms, NHS heavy periods, NHS infertility, CDC reproductive health, NICHD endometriosis/infertility.
- No gold_risk_level / gold_action fields invented.
- NHS PCOS page content verified 2026-07 (site may show PMOS rename; URL retained as specified).

## Notes

- MENST Set 3 excluded; `test.csv` excluded from seed pool (eval holdout).
- Keywords applied only to patient-authored fields (`<human>:` / `input` / `Question`).
- Style variants recycle only claim phrases attested in canonical patient text.

## Update (2026-07-09)

- FT corpus v1 built (`../scripts/build_ft_corpus.py`): 2,700 pairs, 900/category, dual
  leakage key applied (317 rows excluded). See `ft_corpus_stats.md`.
- `gold_risk_level`/`gold_action`/`evidence_quote`/`source_url`/`requires_clarification` are
  now filled in (`../scripts/complete_gold_labels.py`, deterministic, no LLM needed): 61/90
  seeds grounded (down from the 29-ungrounded figure above, which only reflected
  `gold_condition`), 4/90 flagged `requires_clarification=yes`. See
  `gold_label_completion_report.md`. All rows still carry `needs_human_review=true`.
- Style variants regenerated (`../scripts/merge_manual_style_variants.py`, from 450
  Claude-authored variants — the OpenAI-based script hit a billing quota error before any
  output): 100% distinct-string ratio across all 450 variants, zero canonical collisions.
  See `regeneration_report.md`. Still needs a human spot-check before freeze.

## Planning note (does not change measured v1 stats)

- Current v1 benchmark nucleus remains **90 seeds / 540 rows** (30 seeds per category).
- Finalized future fine-tuning target is **800-1,000 per category** (**2,400-3,000 total**) for **SFT / multilingual adaptation**.
- Recommended concrete default for first fine-tuning run is **900 per category (2,700 total)**.
- Leakage exclusion for val/fine-tuning uses `source_dataset + source_row_id` and should follow `leakage_note.md`.
