# Design: Gold Safety Labels for gold_seeds_styled.csv

**Date:** 2026-07-15
**Author:** Mazen (Modeling)

## Problem
`gold_seeds_styled.csv` (the new 540-row/90-seed benchmark in
`Used_Datasets/Consolidated_Datasets/Train_Val_Dataset/`) has no
`gold_risk_level`/`requires_clarification`/`gold_condition`, so `evaluate.py`/
`safety_metrics.py` can't compute triage or clarification metrics on it.

## Decision (from brainstorming)
Deterministic labeling (Approach A), reusing existing, already-disclosed logic:
- `gold_risk_level` = `build_ft_mix_v2.risk_heuristic(Answer)`, computed **once per
  seed group** (all 6 style-siblings share it — same underlying question, same
  true risk). Verified non-degenerate on real data: 57 routine / 30 see-doctor /
  3 urgent (vs. the old benchmark's monotone 100% see-doctor).
- `requires_clarification` = `"yes"` iff that row's `Style == "ambiguous"`, else
  `"no"` — **per-row** (phrasing-specific). Verified: exactly 1 `ambiguous` row
  per 90 seed groups.
- `gold_condition` = `Topic` (already present, straight copy).

Disclosure: this gold is silver/heuristic-derived, not clinician-adjudicated —
same standard already applied to the M3-v2 training data's risk labels.

## Implementation
1. `scripts/complete_gold_labels_gss.py` — reads `gold_seeds_styled.csv`, groups
   by `Answer`, adds the three columns, writes `gold_seeds_styled_labeled.csv`
   (new file) + `gold_label_completion_report_gss.md` (distribution + samples).
   Imports `risk_heuristic` from `build_ft_mix_v2` (no reimplementation).
2. `scripts/convert_gold_seeds_styled.py` — update to read the labeled CSV and
   carry `gold_risk_level`/`requires_clarification`/`gold_condition` into the
   JSONL (replacing the current `category`-only/no-gold conversion).
3. Tests (`tests/test_complete_gold_labels_gss.py`): per-seed risk consistency
   across style-siblings, per-row clarification correctness, distribution
   matches expectation on a small fixture.

## Out of scope
Grounding-evidence approach (rejected — old benchmark's own script found NHS
pages collapse to monotone "see-doctor"); human-review flagging (heuristic
already proven, disclosed methodology is sufficient per user decision).
