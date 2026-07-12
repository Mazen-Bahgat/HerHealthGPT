# Design: First English QLoRA Fine-Tune of Qwen3.5-9B (M3), Local

**Date:** 2026-07-12
**Author:** Mazen (Modeling)
**Parent spec (source of truth):** `docs/superpowers/specs/2026-07-06-herhealthgpt-lu-design.md`
**Deliverable:** M3 — `Qwen3.5-9B + LoRA on corpus 2B` (English), produced locally.

## 1. Purpose & scope

Operationalize the fine-tuning decisions already locked in the parent spec by producing
**M3's English adapter** on local hardware, and — as a first-class byproduct — a
**validated, reproducible local training pipeline** the team can rerun for additional
seeds and (later) the multilingual corpus.

This sub-project does not re-open any modeling decision from the parent spec (model,
method, recipe, leakage rules). It fills in the parent spec's explicit TBD: *how and
where the fine-tune actually runs*, now that the target environment is a **local
Windows workstation with an RTX 5000 Ada (32 GB), via WSL2**, rather than the
originally-assumed Linux HPC cluster.

**In scope**
- WSL2 + Unsloth environment setup (reproducible, documented).
- Formatting the existing English corpus into the Qwen chat template.
- QLoRA training of Qwen3.5-9B with the parent spec's recipe.
- Adapter output + optional merged-16bit export.
- Sanity-generation check and a unit test on the data step.

**Out of scope (downstream or pending elsewhere)**
- Multilingual FR/AR fine-tuning — translations are still pending (FR outsourced, AR
  in-house), so this first run is **English-only**.
- The evaluation / metrics pipeline (`scripts/evaluate.py`, not yet built).
- Serving via vLLM / TGI (`scripts/run_inference.py`, downstream cluster wiring).
- The 3-seed mean ± std reporting — this sub-project delivers **seed 1**; the pipeline
  is built so seeds 2–3 are a rerun with a different `--seed`.

## 2. Inputs (already exist)

- **Corpus:** `HerHealthGPT-LU_seed/ft_corpus_v1.jsonl` — 2,700 pairs, balanced
  900/category (menstrual, pcos, fertility), alpaca `instruction`/`input`/`output`
  schema, plus `category`/`source_dataset`/`source_row_id`.
- **Leakage status:** already dual-key-cleaned (`source_dataset + source_row_id` and
  `seed_answer_hash`) against the 90 frozen benchmark seeds. No benchmark text appears
  in the corpus, so no additional leakage filtering is needed here.

## 3. Confirmed decisions (from brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Compute | Local RTX 5000 Ada, 32 GB, WSL2 Ubuntu (already running, GPU passes through) | Available now, no queue, 32 GB is ample for 9B QLoRA |
| Framework | **Unsloth** (in WSL2) | Purpose-built for single-GPU QLoRA; fastest/lowest-VRAM path. Diverges from parent spec's Llama-Factory — see §9 paper note |
| Model | **Qwen3.5-9B directly** | User's target model; attempt it first |
| Fallback | Qwen2.5-7B-Instruct, one `--model` flag change | Guaranteed Unsloth support; protects the July 15 deadline if GatedDeltaNet is unsupported |
| Code layout | 3 modular scripts under `scripts/` | Matches repo convention; each independently runnable/testable |

## 4. Environment (WSL2 Ubuntu)

Documented in a new `docs/wsl_finetune_setup.md`.

- Create an isolated env: `uv venv --python 3.11` (the WSL system Python is **3.14**,
  which torch/Unsloth do not support yet — this is the one non-obvious gotcha).
- Install: `torch` (CUDA 12.4 wheels; RTX 5000 Ada is sm_89, supported), `unsloth`,
  a **recent** `transformers` (required for Qwen3.5's hybrid GatedDeltaNet), plus
  `trl`, `peft`, `bitsandbytes`, `datasets`, `accelerate`.
- No system `nvcc` is required: torch/bitsandbytes ship CUDA runtime binaries, and
  Unsloth does not require a locally-built flash-attn.
- Verify step: `nvidia-smi` inside WSL already reports the 32 GB card (confirmed).

## 5. Data prep — `scripts/prepare_ft_data.py`

Deterministic, CPU-only, unit-testable.

- Read `HerHealthGPT-LU_seed/ft_corpus_v1.jsonl`.
- Map each row to the **Qwen chat template**, **thinking-mode OFF** (train template
  matches the eval template so train/eval are consistent):
  - `system` = `instruction`
  - `user` = `input`
  - `assistant` = `output`
- **Train on responses only** — prompt tokens masked so loss is computed on the
  assistant answer only (standard instruction tuning; Unsloth `train_on_responses_only`).
- **Validation split:** balanced 5% (135 rows = 45/category), selected with a fixed
  seed, for eval-loss monitoring only. (Benchmark seeds are already excluded from the
  corpus, so this split carries no benchmark-leakage risk.)
- Output: `data/ft/en/train.jsonl` and `data/ft/en/val.jsonl` (gitignored if large;
  reproducible from the committed corpus + fixed seed).

## 6. Training — `scripts/train_qlora.py` (Unsloth)

Recipe carried over verbatim from the parent spec (MenstLLaMA-derived), adjusted only
where noted.

- **Quantization:** QLoRA 4-bit NF4, double-quant.
- **LoRA:** r=16, alpha=16, dropout=0, on attention + MLP projections (Unsloth default
  target modules).
- **Optimizer/schedule:** paged AdamW 8-bit, lr 2e-4, warmup ratio 0.03, max grad norm
  0.3, cosine schedule, **bf16** (Ada supports bf16).
- **Sequence length:** 2048.
- **Epochs:** start at **3** (parent-spec ceiling is 5; "adjust down" since this is
  silver data and we monitor eval loss). Revisit to 5 only if eval loss is still
  clearly decreasing and generations underfit.
- **Effective batch:** ~16 (e.g. per-device 2 × grad-accum 8); room to grow on 32 GB.
- **Seed:** fixed and recorded (this run = seed 1).
- **`--model`** flag: defaults to Qwen3.5-9B; Qwen2.5-7B-Instruct fallback is the same
  flag with a different value. No other change needed to fall back.
- **Reproducibility artifact:** emit `run_config.json` capturing every hyperparameter,
  the resolved model id, the git SHA, and the seed.

### Go/no-go gate (protects the deadline)

Before the full run, execute a `--max-steps 10` smoke run. It confirms the architecture
**loads, steps, and saves** end-to-end. This is the parent spec's Day-1 go/no-go
realized locally:

- **Go:** Qwen3.5-9B loads and steps → proceed to the full 3-epoch run.
- **No-go:** if Qwen3.5-9B will not load/train in Unsloth within a bounded debugging
  window (~2–3 h), switch `--model` to Qwen2.5-7B-Instruct and proceed. Do **not**
  deep-debug the architecture past that window.

## 7. Outputs & sanity check — `scripts/sanity_generate.py`

- **Adapter:** `models/qwen3.5-9b-herhealth-en-lora/` (gitignored — large).
- **Optional:** merged-16bit export for downstream vLLM serving.
- **Sanity generations:** ~8 prompts (a few held-out val items + a few benchmark-style
  phrasings across the three categories). Eyeball for: coherent, on-topic answers that
  recommend seeing a clinician where appropriate and avoid unsafe directive advice.

## 8. Testing / verification

1. **Unit (pytest, no GPU):** `prepare_ft_data` produces the expected chat structure,
   correct response-only masking boundary, and a balanced 45/category val split — fits
   the existing test suite.
2. **Smoke (GPU):** the 10-step run in §6 saves an adapter without error.
3. **Verify:** sanity generations in §7 are coherent, on-topic, and safety-appropriate
   before M3 (English) is called done.

## 9. Paper / reproducibility note

Record in the method section that the local runs used **Unsloth** (not Llama-Factory as
the parent spec stated); the QLoRA **recipe is unchanged**. Note the local single-GPU
setup (RTX 5000 Ada, 32 GB), fixed seeds, and — if the fallback triggered — that M3's
base is Qwen2.5-7B-Instruct rather than Qwen3.5-9B. Fold this into the parent spec's
open-questions/decisions when it becomes the paper.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Qwen3.5-9B GatedDeltaNet unsupported by installed Unsloth/transformers | 10-step go/no-go gate; bounded debug window; one-flag fallback to Qwen2.5-7B-Instruct |
| WSL Python 3.14 breaks torch/Unsloth install | Pin Python 3.11 via `uv` (documented in setup doc) |
| Overfitting on silver data at 5 epochs | Start at 3 epochs; monitor eval loss on the 5% val split |
| Adapter/model artifacts bloat the repo | `models/` and large `data/ft/` outputs gitignored; reproducible from committed corpus + `run_config.json` |
| Train/eval template mismatch (thinking mode) | Format with thinking-mode OFF to match the eval-time template |
