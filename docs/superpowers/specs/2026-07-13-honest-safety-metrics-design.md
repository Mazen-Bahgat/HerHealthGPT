# Design: Honest Safety Metrics for the M2-vs-M3 Evaluation

**Date:** 2026-07-13
**Author:** Mazen (Modeling)
**Parent spec:** `docs/superpowers/specs/2026-07-12-m2-m3-english-evaluation-design.md`
(and, transitively, `2026-07-06-herhealthgpt-lu-design.md`)

## 1. Problem

Analysis of the completed M3 inference (540 items) exposed that the two
safety-relevant accuracy metrics in `scripts/evaluate.py` are **degenerate against
skewed gold labels** in the frozen benchmark:

- **Risk/triage:** `gold_risk_level` is `see-doctor` for **all 540 seeds** (100%
  monotone). "Risk accuracy" therefore measures only the propensity to output
  "see-doctor", not balanced triage. M3 scored 1.3% — real signal (systematic
  under-triage: 343 `routine`, 43 `urgent`, 5 `see-doctor` on 391 parseable rows),
  but the accuracy framing invites misreading.
- **Clarification:** `requires_clarification` is `no` for **516/540** (95.6%).
  M3's 90.3% "clarification accuracy" is majority-class inflation that **masks 0/24
  recall** on the cases that actually require clarification (confusion:
  gold-yes→pred-no = 20 of 20 parseable; 18 false alarms on gold-no).

The benchmark is frozen (lineage-controlled), so the fix is **not** re-labeling; it
is reporting metrics that remain honest under skew.

## 2. Decision (from brainstorming)

**Approach A:** a new, self-contained `scripts/safety_metrics.py` — pure functions +
CLI over the inference JSONLs — rather than extending `evaluate.py` (keeps the
honest-reporting concern in one focused, testable file; `evaluate.py` and
`compare_models.py` remain unchanged). Benchmark gold-label diversification is a
separately-scoped follow-up if ever desired; this spec deliberately excludes it.

## 3. Inputs (already exist)

- `HerHealthGPT-LU_seed/inference/M3_en.jsonl` — 540 records, complete.
- `HerHealthGPT-LU_seed/inference/M2_en_full.jsonl` — 540 records once the in-flight
  M2 re-run finishes (the module must also run on a single file so M3 analysis does
  not wait).
- Reused helpers: `run_inference.{normalize_risk, normalize_category, parse_bool}`;
  `evaluate.score_record` (for the canonical parse_ok / correctness booleans).

## 4. Component: `scripts/safety_metrics.py`

Pure functions (unit-testable, CPU-only, Windows `.venv`):

- `confusion_matrix(records, gold_key, pred_key, labels) -> dict[gold][pred] -> int`
  — counts over **parse_ok rows only**; rows whose normalized value falls outside
  `labels` are bucketed under `"other"` so nothing is silently dropped.
- `per_class_recall(cm) -> dict[label, float|None]` and
  `per_class_precision(cm) -> dict[label, float|None]` — `None` when a class has no
  gold (recall) / no predictions (precision), rendered as `-`.
- `under_triage_rate(records) -> float` — the safety-framed replacement for "risk
  accuracy" given monotone gold: among parseable rows with gold `see-doctor`, the
  fraction predicting the strictly less urgent `routine` (ordinal scale
  routine < see-doctor < urgent). Over-triage (`urgent`) is reported alongside but
  is a distinct number, not summed with under-triage.
- `clarification_stats(records) -> dict` — the 2×2 confusion plus **recall on
  gold=yes** (the honest headline; currently 0/24 for M3), specificity on gold=no,
  and false-alarm count.
- `majority_baseline(records, gold_key) -> float` — accuracy of always predicting
  the majority gold class; printed beside every skew-affected metric so inflation
  is visible (e.g. "clarification accuracy 0.903 vs majority baseline 0.956").
- `analyze(records) -> dict` — assembles all of the above plus `parse_ok_rate` and
  category per-class recall/precision (category gold is balanced 180/180/180, so
  its plain accuracy is fine — per-class detail is still informative).
- `render_report(analyses: dict[label, dict]) -> str` — markdown. Works with one
  label (M3 alone) or several; with ≥2 labels adds Δ(last−first) columns.
  Foregrounds, in order: parse_ok rate, under-triage rate, clarification recall
  (gold=yes), category per-class recall — each with its majority baseline where
  applicable.

CLI:
```
python scripts/safety_metrics.py \
  --predictions LABEL=PATH [--predictions LABEL=PATH ...] \
  --out-md HerHealthGPT-LU_seed/evaluation/safety_M2_vs_M3.md \
  --out-json HerHealthGPT-LU_seed/evaluation/safety_M2_vs_M3.json
```
(repeatable `--predictions M2=...jsonl M3=...jsonl`; one label is valid.)

## 4b. Extended metric coverage (user-mandated additions)

The full metric list to cover, and where each lives:

| Metric | How | Where |
|---|---|---|
| Misunderstanding rate | Deterministic: share of items misinterpreting the concern — `1 − category_accuracy` on parseable rows, plus a **strict** variant counting parse failures as misunderstandings | `safety_metrics.analyze` |
| Unsafe-response rate | Two-track: self-reported flag (deterministic, caveated as model-generated) **and** judge-scored unsafe flag | `safety_metrics` + `judge_metrics` |
| Clarification | Recall on gold=yes + 2×2 + majority baseline (§4) | `safety_metrics` |
| Cross-language consistency | Reuse `evaluate.cross_language_consistency` (n=0 on EN-only; populated when FR/AR land) | imported into `safety_metrics.analyze` |
| Cross-style consistency | Reuse `evaluate.cross_style_consistency` (same answer across 6 styles of one seed) | imported into `safety_metrics.analyze` |
| McNemar's test | Paired per-item M2-vs-M3 significance on `category_correct`, `risk_correct`, `clarification_correct`, `parse_ok` — items paired by `item_id`; report discordant counts b/c and continuity-corrected χ² p-value | `safety_metrics.mcnemar` |
| Bootstrap CIs | Percentile CIs (default 10,000 resamples, fixed seed) for every headline metric and for each M3−M2 delta | `safety_metrics.bootstrap_ci` |
| Cultural sensitivity | **LLM-as-judge**, 1–5 rubric on `response_text` | `scripts/judge_metrics.py` |
| Helpfulness & clarity | **LLM-as-judge**, 1–5 rubric each | `scripts/judge_metrics.py` |

**`scripts/judge_metrics.py`** — separate module because it needs an external judge
endpoint: scores each response on cultural sensitivity / helpfulness / clarity
(1–5) + unsafe (yes/no) via any OpenAI-compatible chat endpoint (DRY: reuses
`run_inference.call_endpoint`), with an injectable call function so unit tests mock
it. Judge choice is a flag (`--base-url/--model`); results carry the judge id.
Caveat recorded in output: judge-based scores are model opinions, not ground truth;
using the same base model family as judge is methodologically weak — prefer an
external judge (e.g. GPT endpoint when billing allows).

## 5. Data flow

```
inference/M2_en_full.jsonl ─┐
                            ├─► scripts/safety_metrics.py ─► evaluation/safety_M2_vs_M3.{md,json}
inference/M3_en.jsonl ──────┘
```

No GPU; runs while (or after) inference. `evaluate.py`/`compare_models.py` outputs
are unaffected and still produced; the paper cites both, with the safety report as
the primary framing for risk/clarification.

## 6. Error handling

- Records failing `evaluate.score_record`'s parse_ok are **excluded from confusion
  counts** but reported via `parse_ok_rate` (format failure is itself a first-class
  finding — M3: 72.4%).
- Unknown normalized labels bucket to `"other"` (visible, not dropped).
- Empty/missing prediction files: a clear CLI error naming the path.

## 7. Testing

TDD, pytest on the Windows `.venv`, hand-built records:
1. `confusion_matrix` counts + `"other"` bucketing.
2. `under_triage_rate` on a mixed routine/urgent/see-doctor fixture.
3. `clarification_stats` reproducing a 0-recall minority case.
4. `majority_baseline` on a 95/5 skew.
5. `render_report` with two labels: Δ column present; `None` rendered as `-`.

Verification: run on the real `M3_en.jsonl` (expect under-triage ≈ 343/391 ≈ 0.877,
clarification recall 0/20 parseable) and, when M2 lands, both files → the
`safety_M2_vs_M3.md` report.

## 8. Out of scope (explicit)

- Re-labeling / diversifying benchmark gold (`gold_risk_level`,
  `requires_clarification`) — frozen lineage; separate future sub-project.
- M1/M4 models — parent-spec deferral unchanged. (LLM-as-judge, bootstrap CIs and
  McNemar's are now IN scope per §4b — un-deferred by user decision 2026-07-13.)
- Modifying `evaluate.py` or `compare_models.py`.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Two reports (evaluate/compare vs safety) could disagree confusingly | Both derive parse_ok from the same `evaluate.score_record`; safety report states it supersedes accuracy framing for risk/clarification |
| M2 file name/path drift (re-run wrote `M2_en_full.jsonl`) | CLI takes explicit LABEL=PATH pairs; no hardcoded filenames |
| "other"-bucket noise if models emit unmapped labels | Bucketed visibly in the confusion matrix rather than dropped |
