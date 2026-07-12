# Design: M2-vs-M3 Evaluation on the English Benchmark (local generation)

**Date:** 2026-07-12
**Author:** Mazen (Modeling)
**Parent spec:** `docs/superpowers/specs/2026-07-06-herhealthgpt-lu-design.md`
**Depends on:** the M3 English adapter produced by
`docs/superpowers/specs/2026-07-12-qwen35-english-qlora-finetune-design.md`
(`models/qwen3.5-9b-herhealth-en-lora/`).

## 1. Purpose & scope

Produce the first real model comparison the paper needs: **does fine-tuning reduce
misunderstanding?** — by scoring **M2 (Qwen3.5-9B base)** vs **M3 (Qwen3.5-9B + our
English LoRA)** on the frozen 540-seed English benchmark, using the metrics
`scripts/evaluate.py` already computes, and emitting a side-by-side comparison with the
M3−M2 delta.

**In scope**
- A local (no-server) inference runner that generates structured predictions for a
  model (base, or base+adapter) over the benchmark.
- Running it for M2 and M3 → per-model inference JSONL.
- Scoring each with the existing `evaluate.py` → per-model summary.
- A comparison step → one M2-vs-M3 table (overall + per-style + per-category) with deltas.

**Out of scope (deferred to a follow-up)**
- M1 (LLaMA-3-8B-Instruct) and M4 (Menstrual-LLaMA-8B) — this iteration is M2 vs M3.
- LLM-as-judge scoring, per-class confusion matrices, and 3-seed mean±std significance.
  The parent spec wants these eventually; the M2-vs-M3 delta on the existing
  deterministic metrics is the fast first result.
- Multilingual (FR/AR) — translations still pending.
- vLLM / any served endpoint — this iteration is local generation only.

## 2. Inputs (already exist)

- **Benchmark:** `HerHealthGPT-LU_seed/seeds_en_v1.jsonl` — 540 rows (90 seeds × 6
  styles) with gold labels (`category`, `gold_risk_level`, `requires_clarification`, …).
- **M2:** `Qwen/Qwen3.5-9B` — cached locally (19 GB, `~/.cache/huggingface`, Ubuntu/sw2).
- **M3:** `models/qwen3.5-9b-herhealth-en-lora/` — the LoRA adapter (116 MB) from the
  fine-tune (train_loss 1.31 / eval_loss 1.08).
- **Prompt + parsing:** `scripts/run_inference.py` already defines
  `FIXED_PROMPT_TEMPLATE`, `parse_model_content`, and `build_output_record`.
- **Scorer:** `scripts/evaluate.py` (exists, 7 tests) computes per-model metrics.

## 3. Confirmed decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Inference engine | **Local generation** (transformers/Unsloth), no server — headless-reliable, reuses the training env, loads M3's adapter natively |
| Model scope (this iteration) | **M2 vs M3** (both use the cached Qwen3.5-9B; no new downloads) |
| Decoding | **Greedy (do_sample=False, temperature 0)** — deterministic primary results per parent spec §4 |
| Thinking mode | **OFF** at eval (matches training/benchmark protocol) |
| Metrics | The deterministic set `evaluate.py` already computes (see §6) |

## 4. Architecture / data flow

```
seeds_en_v1.jsonl (540)
  └─► scripts/run_local_inference.py   (GPU, Ubuntu ft-train-venv, HF_HUB_OFFLINE=1)
        ├─ M2 = Qwen3.5-9B base           → HerHealthGPT-LU_seed/inference/M2_en.jsonl
        └─ M3 = base + LoRA adapter        → HerHealthGPT-LU_seed/inference/M3_en.jsonl
  └─► scripts/evaluate.py   (CPU)          → evaluation/M2_en_summary.json, M3_en_summary.json (+ scored CSVs)
  └─► scripts/compare_models.py  (CPU)     → evaluation/M2_vs_M3_en.md + .json
```

## 5. Components

### 5a. `scripts/run_local_inference.py` (new — the main gap)

Loads one model locally via Unsloth `FastLanguageModel` and generates structured
predictions over the benchmark, writing JSONL in the **exact schema `evaluate.py`
consumes** (so the scorer is unchanged).

- **DRY:** import `FIXED_PROMPT_TEMPLATE`, `parse_model_content`, `build_output_record`
  from `run_inference.py`. Same prompt, same parsing, same output fields.
- **Fair comparison:** M2 and M3 receive the *identical* benchmark prompt (structured
  JSON, thinking off), greedy decoding. M3 gets the eval prompt too — whether the
  fine-tune preserved instruction/JSON-following is part of what we measure
  (`parse_ok_rate` captures format failures).
- **M3 = base + adapter natively:** `--adapter <dir>` loads the LoRA over the base
  (no merge). `HF_HUB_OFFLINE=1` + local snapshot base path (same fix training used).
- **CLI:** `--model <base id/path>` `--adapter <dir|none>` `--benchmark <jsonl>`
  `--output <jsonl>` `--label M2|M3` `--limit N` `--max-new-tokens 512`.
- **Resumable:** skip `item_id`s already present in the output (mirror
  `run_inference.py`'s resume behavior).
- **Supersedes `scripts/run_local_raw_baseline.py`**, which used the wrong
  `AutoModelForImageTextToText` class and an incomplete output schema. This one clean
  runner handles both base and adapter; the old script is removed.

### 5b. `scripts/evaluate.py` (exists, unchanged)

Run once per model file → summary JSON + scored CSV. No change expected; if a field
name from the new runner doesn't line up, fix the runner to match `evaluate.py`, not
the reverse.

### 5c. `scripts/compare_models.py` (new)

Reads ≥2 per-model summary JSONs (M2, M3) → a comparison **markdown table + JSON**:
each metric side-by-side with **Δ(M3−M2)**, at three levels — overall, per-style
(clinical/layperson/indirect/ambiguous/emotional/canonical), per-category
(menstrual/pcos/fertility). Pure function over the summary dicts; unit-testable.

## 6. Metrics (already in `evaluate.py`)

Per model, overall and broken down by style and category:
`parse_ok_rate`, `prediction_coverage`, `category_accuracy`,
`risk_accuracy` (triage), `clarification_accuracy`, `self_reported_unsafe_rate`
(carries the "model-generated flag, not independently validated" caveat). The
comparison surfaces the M3−M2 delta on each.

## 7. Environments

- **`run_local_inference.py`** — Ubuntu `ft-train-venv` (GPU), `HF_HUB_OFFLINE=1`
  `HF_HUB_DISABLE_XET=1`, base loaded from the local snapshot path.
- **`evaluate.py` / `compare_models.py`** — Windows `.venv` (CPU; `run_inference` is
  importable there and `requests` is installed).

## 8. Testing / verification

1. **Unit (Windows `.venv`, no GPU):** `compare_models.py` table math + delta signs on
   hand-built summary dicts; a record-shape check that the runner's output matches the
   keys `evaluate.py` requires.
2. **Smoke (GPU):** `run_local_inference.py --limit 4` for M2 and M3 → 4 valid records
   each, parse_ok.
3. **Verify:** full 540-seed run for M2 and M3, then `evaluate.py` + `compare_models.py`
   produce the comparison with a plausible, non-degenerate M3−M2 delta and high
   `parse_ok_rate` for both.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Fine-tuned M3 stops emitting valid JSON (format drift) | `parse_ok_rate` is a first-class metric; low M3 coverage is itself a reportable finding |
| Serial local generation over 540 items is slow | Greedy + `--max-new-tokens 512`; resumable output so a stall doesn't lose progress; acceptable for a one-off eval |
| Runner output schema drifts from `evaluate.py` expectations | Reuse `run_inference.build_output_record`; unit-test the record shape |
| Adapter base-path resolution triggers HF prefetch stall | `HF_HUB_OFFLINE=1` + local snapshot base path (proven in training) |
