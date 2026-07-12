# Environments & Status — HerHealthGPT-LU

Pick-up-later reference for the compute environments and where things stand.
Last updated: 2026-07-12.

## TL;DR

- **To fine-tune (M3):** use `.venv-ft` on WSL distro **`HerHealthUbuntu`** as **root**. It works.
- vLLM serving (for the M2 baseline eval) is **parked** — it won't start headless (see below). Not needed to fine-tune.
- The three environments are **intentionally different** (different jobs, conflicting deps). They are NOT meant to be identical. What must match for valid results is the **chat template + thinking-mode-off** (enforced) and the **model id**, not library versions.

## The three environments

| Env | WSL distro / user | Python | Purpose | Status |
|---|---|---|---|---|
| `.venv` (repo root) | Windows (native) | 3.12 | Data pipeline + `pytest` (CPU only) | ✅ works |
| **`.venv-ft`** | **HerHealthUbuntu / root** | 3.11 | **Unsloth QLoRA training (M3)** | ✅ works |
| `vllm-qwen/.venv` (`/home/sw2/vllm-qwen`) | Ubuntu / sw2 | 3.12 | vLLM serving (M2 baseline eval) | ⏸️ parked (headless blocker) |

**Why they differ (and must):** Unsloth (training) and vLLM (serving) pin different,
incompatible torch + CUDA-kernel builds — they cannot coexist in one venv. This is
standard: training env ≠ inference env. The Windows `.venv` is CPU-only for data/tests.

**Known inconsistency (deferred):** the two GPU jobs currently live on two different
distros (`HerHealthUbuntu` for training, `Ubuntu` for serving); there is also an unused
`Ubuntu-24.04`. Consolidating both GPU venvs onto one distro is optional cleanup for
later — it does not block fine-tuning.

## Fine-tuning (M3) — how to run

Env: `HerHealthUbuntu` / root / `.venv-ft`. Data already prepped at
`data/ft/en/{train,val}.jsonl` (2565 / 135, Qwen chat messages, thinking-off).

```powershell
# 10-step smoke gate (go/no-go). Downloads the 4-bit model to the distro HF cache.
wsl.exe -d HerHealthUbuntu -u root -- bash -c "cd /mnt/d/Grad-Project/HerHealthGPT && .venv-ft/bin/python scripts/train_qlora.py --max-steps 10 --output models/_smoke"

# Full 3-epoch run (seed 1) once the smoke gate passes.
wsl.exe -d HerHealthUbuntu -u root -- bash -c "cd /mnt/d/Grad-Project/HerHealthGPT && .venv-ft/bin/python scripts/train_qlora.py --epochs 3 --output models/qwen3.5-9b-herhealth-en-lora"
```

Recipe (from the plan): QLoRA 4-bit NF4, LoRA r=16, lr 2e-4, warmup 0.03,
max-grad-norm 0.3, max-seq 2048, paged AdamW 8-bit, cosine, bf16, seed 3407.
Outputs (gitignored): adapter + `run_config.json` under `models/…`.

## vLLM baseline (M2) — parked, for later

Serves the untouched Qwen3.5-9B so `scripts/run_inference.py` can produce M2 results.
The model is cached for sw2 (`~/.cache/huggingface`, 19 GB). The pipeline is validated
(a 2-item smoke through vLLM produced clean, thinking-off JSON).

**Leaner run command (frees ~11 GB vs the old 90% reservation) — run in an interactive
Ubuntu terminal** (the interactive shell has the CUDA build env a service lacks):

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
kernel JIT hitting missing `curand.h` / incompatible toolkit headers). Fixes tried:
`LD_LIBRARY_PATH` (nvidia libs), `CUDA_HOME`+`.venv/bin` on PATH, `VLLM_USE_FLASHINFER_SAMPLER=0`.
Running from the interactive terminal (above) is the reliable path until the CUDA dev
toolchain is completed for headless use.

## Notes / gotchas

- **`--gpu-memory-utilization`**: vLLM always uses paged attention; the value only sets
  how much VRAM it reserves up front. 0.65 ≈ 21 GB (18 GB bf16 weights + KV), frees ~11 GB.
  Lower needs fp8 weight quantization (changes numerics).
- **Only one big GPU job at a time**: a bf16 vLLM server (~21 GB) and a QLoRA fine-tune
  can't comfortably share 32 GB. Stop vLLM before the full fine-tune, or vice-versa.
- **HF cache is per-distro**: `HerHealthUbuntu` (training) and `Ubuntu` (serving) do not
  share `~/.cache/huggingface`. The training env downloads the 4-bit model once on first
  smoke run.
- WSL system Python is 3.14 (unsupported by torch/Unsloth) — that's why `.venv-ft` pins 3.11.
