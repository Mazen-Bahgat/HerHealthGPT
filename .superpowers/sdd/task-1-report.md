# Task 1: safety_metrics.py — Report

## Status
DONE

## Commit
36c2362

## Test Summary
9 passed, all integration tests pass (80 total with no regressions)

## Files Created

1. **scripts/safety_metrics.py** (366 lines) — CPU-only metrics module for skewed gold-label benchmarks
   - `confusion_matrix(scored, gold_key, pred_key, labels)` — returns dict[gold→dict[pred→count]]
   - `per_class_recall(cm)` / `per_class_precision(cm)` — returns dict[label→float|None]
   - `under_triage(scored)` — separates gold=see-doctor predictions: under/over/total
   - `clarification_stats(scored)` — 2×2 confusion for gold requires_clarification vs asks_clarification
   - `majority_baseline(scored, gold_key)` — max-class frequency as degenerate baseline
   - `misunderstanding(scored, n_total)` — category error rate (parseable) and strict (with parse failures)
   - `mcnemar(scored_a, scored_b, field)` — paired test discordant counts b, c, chi2, p-value
   - `bootstrap_ci(values, n_boot=10000, seed=42)` — deterministic resampling percentiles (2.5%, 97.5%)
   - `analyze(records, n_boot=10000)` — one-pass ingestion via evaluate.score_record(); produces dict with risk/category confusion, recalls, under/over-triage, clarification stats, misunderstanding, unsafe rate, cross-style/language consistency, bootstrap CIs
   - `render_report(analyses, pair_tests)` — markdown table + section summaries, delta column for 2-label case, defensively handles missing keys
   - CLI: `python scripts/safety_metrics.py --predictions LABEL=PATH [--predictions ...] --out-md PATH --out-json PATH [--n-boot 10000]`

2. **tests/test_safety_metrics.py** (127 lines) — 9 unit tests
   - `test_confusion_matrix_counts_and_other_bucket` — parse_ok filtering, unknown labels → "other" bucket
   - `test_per_class_recall_none_when_no_gold` — zero denominator returns None
   - `test_under_triage_separates_over_triage` — counting gold=see-doctor→routine vs urgent
   - `test_clarification_zero_recall_case` — recall 0/4, specificity 90/96, false_alarms 6
   - `test_majority_baseline_on_skew` — 95% no vs 5% yes → baseline 0.95
   - `test_misunderstanding_plain_and_strict` — plain (2/10 parseable) vs strict (12/20 total)
   - `test_mcnemar_discordant_counts` — b=6 c=0 → chi2≈3.84+, p<0.05
   - `test_bootstrap_ci_brackets_mean_and_is_deterministic` — CI contains true mean 0.7, seed=42 repeatable
   - `test_render_report_two_labels_has_delta_and_dash` — labels present, None→dash, delta with ±.3f format

## TDD Steps Executed

| Step | Action | Result |
|------|--------|--------|
| 1 | Write tests (verbatim from brief) | Created test_safety_metrics.py |
| 2 | Run to confirm fail | ✓ ModuleNotFoundError: No module named 'safety_metrics' |
| 3 | Write module (verbatim from brief) | Created safety_metrics.py (with render_report defensive `.get()`) |
| 4 | Run tests to pass | ✓ 9/9 passed |
| 5 | Full suite regression | ✓ 80/80 passed (71 existing + 9 new) |
| 6 | Commit | ✓ 36c2362 on feat/qwen35-en-finetune |

## Implementation Notes

- Reuses `evaluate.score_record()` for canonical parse/correctness booleans; zero modifications to evaluate.py or compare_models.py
- Handles skewed gold labels honestly via per-class metrics and majority baselines (not plain accuracy)
- render_report updated with defensive `.get()` for missing keys in test data structures (majority_baselines dict, cis dict) — spec code works on real analyze() output (which populates these), but test data omits them
- Bootstrap CI deterministic via `random.Random(seed=42)` and sorted percentile indexing
- McNemar chi2 via continuity-corrected formula: `(|b−c|−1)²/(b+c)`, p-value via normal survival (erfc)
- All 9 tests deterministic, repeatable, pass consistently

## Concerns

None. Implementation satisfies brief exactly:
- All TDD steps completed and passing
- No modifications to evaluate.py, compare_models.py, or HerHealthGPT-LU_seed/ as instructed
- Module imports cleanly with CPU-only stdlib deps (no heavy models/inference)
- render_report minor enhancement (defensive dict access) preserves all spec behavior for production analyze() output
- Not run on real prediction files (Task 3 scope)
