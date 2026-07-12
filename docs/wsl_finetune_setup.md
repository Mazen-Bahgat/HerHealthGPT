# WSL2 fine-tuning environment (local RTX 5000 Ada, 32 GB)

Reproducible setup for the M3 QLoRA fine-tune. Runs inside WSL2 Ubuntu; the
Windows `.venv` is CPU-only and unchanged. Training deps are intentionally NOT
in pyproject.toml — install them here.

Env: `/mnt/d/Grad-Project/HerHealthGPT/.venv-ft` (Python 3.11 via uv).
Why 3.11: the project fine-tuning stack is pinned to a supported Python release
rather than the WSL system Python.

The commands below use the installed distro name `HerHealthUbuntu` and run as
root. The versions shown were resolved and verified on 2026-07-12. Current
Unsloth requires a newer PyTorch stack than the original CUDA 12.4 proposal, so
the reproducible install pins the verified CUDA 12.8-compatible versions.

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

Install `uv` if it is not already available, then create the environment:

```powershell
wsl.exe -d HerHealthUbuntu -u root -- bash -lc "set -euo pipefail; command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv-install.sh && sh /tmp/uv-install.sh; }; export PATH=/root/.local/bin:$PATH; cd /mnt/d/Grad-Project/HerHealthGPT && uv venv --seed --python 3.11 .venv-ft"
```

Expected output includes `Using CPython 3.11` and
`Creating virtual environment ... .venv-ft`. If `.venv-ft` already exists,
remove it only when an intentional clean rebuild is required.

## 3. Install the verified Unsloth training stack

```powershell
wsl.exe -d HerHealthUbuntu -u root -- bash -lc "set -euo pipefail; cd /mnt/d/Grad-Project/HerHealthGPT && .venv-ft/bin/python -m pip install --upgrade pip && .venv-ft/bin/pip install 'torch==2.10.0' 'unsloth==2026.7.2' 'transformers==5.5.0' 'trl==0.24.0' 'peft==0.19.1' 'bitsandbytes==0.49.2' 'datasets==4.3.0' 'accelerate==1.14.0'"
```

Expected: installation completes without error. The verified resolver result is
PyTorch `2.10.0+cu128` with CUDA runtime `12.8`, plus the exact package versions
listed above. The RTX 5000 Ada (`sm_89`) is supported by this stack.

## 4. Verify PyTorch sees the GPU and Unsloth imports

```powershell
wsl.exe -d HerHealthUbuntu -u root -- bash -lc "cd /mnt/d/Grad-Project/HerHealthGPT && .venv-ft/bin/python -c 'import torch, unsloth; print(\"cuda\", torch.cuda.is_available(), torch.cuda.get_device_name(0)); print(\"torch\", torch.__version__, \"runtime\", torch.version.cuda); print(\"unsloth\", unsloth.__version__)'"
```

Expected output includes:

```text
cuda True NVIDIA RTX 5000 Ada Generation
torch 2.10.0+cu128 runtime 12.8
unsloth 2026.7.2
```

If `import unsloth` fails on a Qwen3.5 architecture error, that is the go/no-go signal for the Task 3 fallback.
