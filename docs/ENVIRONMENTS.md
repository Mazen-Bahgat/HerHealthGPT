# Environments & Status — HerHealthGPT-LU

Pick-up-later reference for the compute environments and where things stand.
Last updated: 2026-07-12.

## TL;DR

- **To fine-tune (M3):** use the **`Ubuntu`** distro, user `sw2`, venv
  **`/home/sw2/ft-train-venv`**. Smoke gate passed; the full 3-epoch run is what
  produces the adapter.
- vLLM serving (for the M2 baseline eval) is **parked** — see below. Not needed to
  fine-tune.
- The environments are **intentionally different** (different jobs, conflicting deps).
  They are NOT meant to be identical. What must match for valid results is the **chat
  template + thinking-mode-off** (enforced) and the **model id**, not library versions.

## The environments

| Env | WSL distro / user | Python | Purpose | Status |
|---|---|---|---|---|
| `.venv` (repo root) | Windows (native) | 3.12 | Data pipeline + `pytest` (CPU only) | ✅ works |
| **`/home/sw2/ft-train-venv`** | **Ubuntu / sw2** | 3.11 | **Unsloth QLoRA training (M3)** | ✅ works (smoke passed) |
| `vllm-qwen/.venv` (`/home/sw2/vllm-qwen`) | Ubuntu / sw2 | 3.12 | vLLM serving (M2 baseline eval) | ⏸️ parked (headless blocker) |
| `.venv-ft` (`/mnt/d/.venv-ft`) | HerHealthUbuntu / root | 3.11 | (abandoned training attempt) | ❌ orphaned — safe to delete |

**Why they differ (and must):** Unsloth (training) and vLLM (serving) pin different,
incompatible torch + CUDA-kernel builds — they cannot coexist in one venv. This is
standard: training env ≠ inference env. The Windows `.venv` is CPU-only for data/tests.

**Why training is on Ubuntu, not HerHealthUbuntu:** the committed setup doc originally
targeted `HerHealthUbuntu`, but that distro is **unfit for training** — it has no C
compiler (Triton needs one to JIT its kernels) and its `apt` is broken
(`archive.ubuntu.com` returns 403, so `build-essential` can't be installed). The
`Ubuntu` distro already has `gcc`/`cc`/`g++`, the 19 GB model cached, and working
network — so training moved there. `HerHealthUbuntu`'s `.venv-ft` is now orphaned.

## Fine-tuning (M3) — how to run

Env: `Ubuntu` / sw2 / `/home/sw2/ft-train-venv`. Data prepped at
`data/ft/en/{train,val}.jsonl` (2565 / 135, Qwen chat messages, thinking-off).

Two env quirks this machine needs, both baked into the commands below:
- `HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1` — Unsloth's snapshot prefetcher otherwise
  stalls re-verifying against HF (unauthenticated rate-limit). Offline + xet-off makes
  it load straight from the local cache.
- `--model <local snapshot path>` — pass the cached snapshot dir, not the repo id, so
  no download is attempted. (The model is complete in `~/.cache/huggingface`.)

```powershell
# 10-step smoke gate (go/no-go) — PASSED (loss 2.17->2.02, eval 1.79, 116MB adapter).
wsl.exe -d Ubuntu -- bash -c 'cd /mnt/d/Grad-Project/HerHealthGPT && HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 /home/sw2/ft-train-venv/bin/python scripts/train_qlora.py --model /home/sw2/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a --max-steps 10 --output models/_smoke'

# Full 3-epoch run (seed 1) — produces the M3 adapter.
wsl.exe -d Ubuntu -- bash -c 'cd /mnt/d/Grad-Project/HerHealthGPT && HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 /home/sw2/ft-train-venv/bin/python scripts/train_qlora.py --model /home/sw2/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a --epochs 3 --output models/qwen3.5-9b-herhealth-en-lora'
```

Recipe: QLoRA 4-bit NF4, LoRA r=16, lr 2e-4, warmup 0.03, max-grad-norm 0.3,
max-seq 2048, paged AdamW 8-bit, cosine, bf16, seed 3407. Unsloth confirms native
Qwen3.5 support ("Fast Qwen3_5 patching"), so the GatedDeltaNet architecture risk is
resolved — no Qwen2.5 fallback needed. Outputs (gitignored): adapter +
`run_config.json` under `models/…`.

## vLLM baseline (M2) — parked, for later

Serves the untouched Qwen3.5-9B so `scripts/run_inference.py` can produce M2 results.
The model is cached for sw2 (`~/.cache/huggingface`, 19 GB). The pipeline is validated
(a 2-item smoke through vLLM produced clean, thinking-off JSON).

**Leaner run command (frees ~11 GB vs the old 90% reservation) — run in an interactive
Ubuntu terminal** (the interactive shell has the CUDA build env a headless service lacks):

```bash
cd ~/vllm-qwen
.venv/bin/vllm serve Qwen/Qwen3.5-9B --port 8000 \
  --max-model-len 8192 --gpu-memory-utilization 0.65 --reasoning-parser qwen3
```

Then, once `/health` is 200:

```powershell
wsl.exe -d Ubuntu -- bash -c "cd /mnt/d/Grad-Project/HerHealthGPT && /home/sw2/vllm-qwen/.venv/bin/python scripts/run_inference.py --base-url http://localhost:8000/v1 --model Qwen/Qwen3.5-9B --model-label M2 --benchmark HerHealthGPT-LU_seed/seeds_en_v1.csv --output HerHealthGPT-LU_seed/inference/M2_en.jsonl"
```

**Known headless blocker:** as a systemd/background service the CUDA env is stripped, so
startup fails in a cascade (`libcudart.so.13` → `nvcc` → `ninja` → FlashInfer's sampling
kernel JIT hitting missing `curand.h` / incompatible toolkit headers). Running from the
interactive terminal is the reliable path until the CUDA dev toolchain is completed for
headless use.

## Notes / gotchas

- **Only one big GPU job at a time**: a bf16 vLLM server (~21 GB) and a QLoRA fine-tune
  can't comfortably share 32 GB. Stop vLLM before the full fine-tune, or vice-versa.
- **`--gpu-memory-utilization`**: vLLM always uses paged attention; the value only sets
  how much VRAM it reserves up front. 0.65 ≈ 21 GB. Lower needs fp8 weight quantization.
- **HF cache is per-distro**: each distro has its own `~/.cache/huggingface`. Training
  (Ubuntu/sw2) reuses the same 19 GB cache vLLM downloaded.
- **This branch has a parallel committer** — commits appear on `feat/qwen35-en-finetune`
  from other sessions. Pull before large edits.
- WSL system Python is 3.14 (unsupported by torch/Unsloth) — that's why the training
  venv pins 3.11.
