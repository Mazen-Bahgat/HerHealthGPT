# Design: M3-v2 — Data-Mix Fine-Tune (Experiment B)

**Date:** 2026-07-13
**Author:** Mazen (Modeling)
**Parent specs:** `2026-07-12-qwen35-english-qlora-finetune-design.md` (M3 recipe),
`2026-07-13-honest-safety-metrics-design.md` (measurement), and transitively
`2026-07-06-herhealthgpt-lu-design.md`.

## 1. Problem

The M2-vs-M3 evaluation (all McNemar p < 1e-11, reparsed pairing) showed the English
silver-SFT **degraded safety behaviors while leaving comprehension unchanged**:

| metric | M2 | M3 | cause (diagnosed) |
|---|---|---|---|
| parse_ok | 0.998 | 0.915 | corpus is 100% conversational; zero structured-output examples |
| under-triage | 0.456 | 0.862 | corpus answers are reassuring patient education → reassurance bias |
| clarification recall | 0.208 | 0.000 | corpus always answers, never asks |
| cross-style consistency (risk) | 0.433 | 0.167 | single clean register; no input-side style variation |
| misunderstanding (parseable) | 0.390 | 0.393 | — unchanged; knowledge intact |

Every failure is data-composition, not capacity — so M3-v2 changes the data mix, not
the architecture.

## 2. Decisions (from brainstorming)

- **Experiment B** (data-mix v2), not just hyperparameter backoff.
- **Synthesis: rule-based + Claude-authored** (no GPU distillation, no external API):
  JSON examples by deterministic reformat; clarification + style examples
  Claude-authored (same precedent as the repo's 450 Claude-authored style variants).
- Training: parent recipe with **2 epochs, lr 1e-4** (halved), seed 3407.
- **Behavioral gate before benchmark contact** — eval-loss is proven insufficient
  (M3's eval_loss was excellent while behavior degraded).

## 3. Data-mix v2 — `scripts/build_ft_mix_v2.py` → `data/ft/en_v2/`

All components derive ONLY from `HerHealthGPT-LU_seed/ft_corpus_v1.jsonl` (already
dual-key leakage-cleaned against the frozen 540-seed benchmark). No seed text
anywhere. Output schema identical to v1 (`{"messages": [...], "category": ...}`) so
`train_qlora.py` runs unchanged.

| Component | ≈Size | Construction |
|---|---|---|
| Chat pairs | 2,565 | v1 train split unchanged |
| JSON-task examples | 900 (300/category) | user turn = `run_inference.FIXED_PROMPT_TEMPLATE` with a corpus question; assistant = the exact 8-key JSON: `predicted_category` = corpus `category`; `interpreted_symptom` = first sentence of answer (≤120 chars); `predicted_risk` = **heuristic**: urgent-words (emergency, immediately, ER, 911, severe bleeding) → `urgent`, consult-language (doctor, clinician, gynecologist, provider, medical attention, checkup) → `see-doctor`, else `routine`; `asks_clarification` = false; `clarifying_question` = ""; `unsafe_response` = false; `response_text` = corpus answer trimmed ≤ 500 chars |
| Clarification examples | 270 (90/category) | Claude-authored: vague/underspecified variants of corpus questions → 180 JSON-format replies with `asks_clarification=true` + a specific `clarifying_question`, and 90 chat-register clarifying replies |
| Style-augmented pairs | 600 (200/category) | Claude-authored rewrites of corpus **questions** into layperson / indirect / emotional registers; answers unchanged |

Total ≈ 4,335 train. Val: v1's 135 unchanged (loss monitoring) **plus** a
**behavioral probe set** `data/ft/en_v2/probe.jsonl` (~90 items, built from val-split
questions + 15 held-out clarification items, formatted as eval-style JSON prompts) —
never used in training.

Determinism: JSON reformat is pure-rule (seeded sampling of which rows); the
Claude-authored files are committed as static JSON (like `style_variants_manual.json`)
so the build is reproducible from the repo.

## 4. Training

`train_qlora.py` unchanged: QLoRA 4-bit NF4, LoRA r=16, warmup 0.03, max-grad-norm
0.3, max-seq 2048, paged AdamW 8-bit, cosine, bf16; **--epochs 2 --lr 1e-4**, seed
3407; `--data-dir data/ft/en_v2`; output `models/qwen3.5-9b-herhealth-en-lora-v2`.
Env: Ubuntu `ft-train-venv`, `HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1`, local snapshot
path (per `docs/ENVIRONMENTS.md`).

## 5. Behavioral gate (pre-benchmark)

Run `run_local_inference.py` with the v2 adapter on `probe.jsonl` (~90 items), score
with `safety_metrics.py` (single label). **Pass thresholds:** parse_ok ≥ 0.95;
under-triage ≤ 0.50 on probe consult-gold items; clarification recall > 0 on the 15
probe clarification items. Fail → one budgeted iteration (drop to 1 epoch or
rebalance mix), then re-gate. Only a passing adapter touches the benchmark.

## 6. Final evaluation

Full 540-seed run (M3-v2) → `safety_metrics.py` **three-way M2 / M3 / M3-v2**
(`--expected-count 540`; multi-label supported) + `compare_models.py`. Deliverable:
the before/after arc — *naive silver-SFT harms safety behaviors; targeted data
composition recovers them* (or doesn't — reportable either way).

## 7. Testing

TDD on `build_ft_mix_v2.py` (Windows `.venv`): JSON examples parse and are
schema-valid against `run_inference.validate_prediction_object`; category balance;
risk-heuristic mapping cases; leakage guard (no seed answer-hash collisions —
reuse the v1 dual-key check); probe set disjoint from train; determinism (fixed
seed). GPU steps validated by the behavioral gate itself.

## 8. Paper disclosures

1. JSON-task examples share the evaluation *format* (public fixed prompt) with zero
   benchmark content — format-retention training data, disclosed.
2. Heuristic risk labels are silver, rule-based, disclosed.
3. Clarification/style examples are Claude-authored (existing repo precedent),
   disclosed; committed as static reproducible files.

## 9. Out of scope

M1/M4 models; multilingual; judge metrics; DPO/preference training; batched
inference. Benchmark and gold labels remain frozen.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Format training overfits the eval prompt | Content fully disjoint; disclosed; parse_ok gate would also catch degenerate memorization on probe items it has never seen |
| Heuristic risk labels noisy | Silver by design; consult-language dominance matches the benchmark's clinical grounding; disclosed |
| Mix still fails behaviorally | Gate catches it pre-benchmark; one iteration budgeted; a second failure is itself a reportable finding |
| Clarification examples teach over-asking | Specificity tracked on probe (false alarms) and in final three-way report |
