# WSL2 fine-tuning environment (local RTX 5000 Ada, 32 GB)

Repeatable direct-package setup for the M3 QLoRA fine-tune. Runs inside WSL2 Ubuntu; the
Windows `.venv` is CPU-only and unchanged. Training deps are intentionally NOT
in pyproject.toml — install them here.

Env: `/mnt/d/Grad-Project/HerHealthGPT/.venv-ft` (Python 3.11 via uv).
Why 3.11: this Ubuntu 24.04 distro currently reports system Python 3.12.3, while
the fine-tuning interface requires the explicitly tested Python 3.11 stack. The
original plan also guarded against newer WSL images exposing Python 3.14, which
is unsupported by the selected torch/Unsloth stack. Do not use the system
interpreter for these commands.

The commands below use the installed distro name `HerHealthUbuntu` and run as
root. The versions shown were resolved and verified on 2026-07-12. Current
Unsloth requires a newer PyTorch stack than the original CUDA 12.4 proposal, so
the direct packages below pin the verified CUDA 12.8-compatible versions. This
is not a full environment lock: transitive dependencies and the remote installer
may change, so a future resolution can differ.

**Qwen3.5 status: provisional.** Task 1 proves generic PyTorch CUDA access and
an Unsloth import only. It does not prove that Qwen3.5's FLA/GatedDeltaNet path
uses GPU acceleration. Task 3's 10-step training smoke is the binding go/no-go.

## 1. Confirm GPU passthrough in WSL

Run from Windows PowerShell:

```powershell
wsl.exe -d HerHealthUbuntu -u root -- bash -lc "nvidia-smi --query-gpu=name,memory.total --format=csv | tr -d '\0'"
```

Expected output includes:

```text
NVIDIA RTX 5000 Ada Generation, 32760 MiB
```

## 2. Install uv and create the isolated Python 3.11 venv

Install the verified `uv` version if it is not already available, print and
check its version, then create the environment with the verified Python patch:

```powershell
wsl.exe -d HerHealthUbuntu -u root -- bash -lc "set -euo pipefail; export PATH=/root/.local/bin:`$PATH; command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/0.11.28/install.sh -o /tmp/uv-install.sh && sh /tmp/uv-install.sh; }; uv --version | grep '^uv 0\.11\.28'; cd /mnt/d/Grad-Project/HerHealthGPT && uv venv --seed --python 3.11.15 .venv-ft"
```

Expected output includes `uv 0.11.28`, `Using CPython 3.11.15`, and
`Creating virtual environment ... .venv-ft`. If `.venv-ft` already exists,
remove it only when an intentional clean rebuild is required.

## 3. Install the verified Unsloth training stack

```powershell
wsl.exe -d HerHealthUbuntu -u root -- bash -lc "set -euo pipefail; cd /mnt/d/Grad-Project/HerHealthGPT && .venv-ft/bin/python -m pip install 'pip==26.1.2' && .venv-ft/bin/pip install 'torch==2.10.0' 'unsloth==2026.7.2' 'transformers==5.5.0' 'trl==0.24.0' 'peft==0.19.1' 'bitsandbytes==0.49.2' 'datasets==4.3.0' 'accelerate==1.14.0'"
```

Expected: installation completes without error. The verified resolver result is
PyTorch `2.10.0+cu128` with CUDA runtime `12.8`, plus the exact package versions
listed above. The RTX 5000 Ada (`sm_89`) is supported by this stack.

## 4. Verify PyTorch sees the GPU and Unsloth imports

```powershell
wsl.exe -d HerHealthUbuntu -u root -- bash -lc "cd /mnt/d/Grad-Project/HerHealthGPT && .venv-ft/bin/python -c 'import torch, unsloth; print(\"cuda\", torch.cuda.is_available(), torch.cuda.get_device_name(0)); print(\"torch\", torch.__version__, \"runtime\", torch.version.cuda); print(\"unsloth\", unsloth.__version__)'"
```

Expected output includes the successful lines below, followed by the known
warnings shown afterward:

```text
cuda True NVIDIA RTX 5000 Ada Generation
torch 2.10.0+cu128 runtime 12.8
unsloth 2026.7.2
```

Known warning from the verified environment (emitted twice):

```text
Triton is not supported on current platform, roll back to CPU.
```

Generic CUDA remains available: PyTorch detected the RTX 5000 Ada and a CUDA
tensor allocation succeeded. The warning means FLA/Triton acceleration is
unverified, so this environment is not yet a Qwen3.5 go. The binding go/no-go is
Task 3's 10-step smoke:

```powershell
wsl.exe -d HerHealthUbuntu -u root -- bash -lc "cd /mnt/d/Grad-Project/HerHealthGPT && .venv-ft/bin/python scripts/train_qlora.py --max-steps 10 --output models/_smoke"
```

Only a completed run that saves the adapter establishes GO. If it fails on the
Qwen3.5 architecture or FLA/GatedDeltaNet execution and is not resolved within
the Task 3 debug window, rerun that gate with
`--model Qwen/Qwen2.5-7B-Instruct` as the fallback.
