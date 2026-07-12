# Task 1 report: WSL2 + Unsloth environment

Date: 2026-07-12

Status: implementation and verification completed; commit recorded below after
the commit is created.

## Environment and decisions

- WSL distro: `HerHealthUbuntu`, invoked as root with
  `wsl.exe -d HerHealthUbuntu -u root -- ...`.
- GPU: `NVIDIA RTX 5000 Ada Generation, 32760 MiB`.
- `uv`: installed from the official Astral installer; version `0.11.28`.
- Python: uv-managed CPython `3.11.15` in
  `/mnt/d/Grad-Project/HerHealthGPT/.venv-ft`.
- The pre-existing `.venv-ft` was unusable because its interpreter home pointed
  at `/home/sw2/.local/share/uv/...`, which does not exist in the new root WSL
  environment. It was removed after resolving and checking the exact path, then
  recreated cleanly.
- Installing the brief's initial `torch==2.5.*` CUDA 12.4 wheel succeeded, but
  the current `unsloth==2026.7.2` resolver upgraded it to its compatible current
  stack. The setup doc therefore pins the verified final stack instead of
  leaving a misleading intermediate pin.

## Commands and observed outputs

### GPU passthrough

```powershell
wsl.exe -d HerHealthUbuntu -u root -- bash -lc "nvidia-smi --query-gpu=name,memory.total --format=csv | tr -d '\0'"
```

```text
name, memory.total [MiB]
NVIDIA RTX 5000 Ada Generation, 32760 MiB
```

### uv installation and venv creation

```powershell
wsl.exe -d HerHealthUbuntu -u root -- bash -lc "curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv-install.sh; sh /tmp/uv-install.sh; export PATH=/root/.local/bin:$PATH; cd /mnt/d/Grad-Project/HerHealthGPT; uv venv --seed --python 3.11 .venv-ft"
```

```text
uv 0.11.28 (x86_64-unknown-linux-gnu)
Using CPython 3.11.15
Creating virtual environment with seed packages at: .venv-ft
```

uv warned that the repository application declares `requires-python >=3.12`.
This isolated training environment deliberately uses Python 3.11 per the task
interface and does not install the application package itself.

### Dependency installation

The requested sequence was run: pip upgrade, `torch==2.5.*` from the cu124
index, then Unsloth and training dependencies. The first stage installed
`torch-2.5.1+cu124`. The second stage resolved the current compatible final
stack and completed successfully after approximately 22 minutes on mounted
NTFS.

Final direct versions:

```text
pip 26.1.2
torch 2.10.0+cu128
unsloth 2026.7.2
transformers 5.5.0
trl 0.24.0
peft 0.19.1
bitsandbytes 0.49.2
datasets 4.3.0
accelerate 1.14.0
```

The reproducible command documented for future runs pins those final versions:

```powershell
wsl.exe -d HerHealthUbuntu -u root -- bash -lc "set -euo pipefail; cd /mnt/d/Grad-Project/HerHealthGPT && .venv-ft/bin/python -m pip install --upgrade pip && .venv-ft/bin/pip install 'torch==2.10.0' 'unsloth==2026.7.2' 'transformers==5.5.0' 'trl==0.24.0' 'peft==0.19.1' 'bitsandbytes==0.49.2' 'datasets==4.3.0' 'accelerate==1.14.0'"
```

### Import and CUDA verification

The verification imported all direct dependencies, queried CUDA, and allocated
a CUDA tensor. Key output:

```text
python 3.11.15
torch 2.10.0+cu128 cuda_runtime 12.8
unsloth 2026.7.2
transformers 5.5.0
trl 0.24.0
peft 0.19.1
bitsandbytes 0.49.2
datasets 4.3.0
accelerate 1.14.0
cuda True NVIDIA RTX 5000 Ada Generation
cuda_tensor 1.0
```

## Concern for later model verification

During `import unsloth`, vendored FLA emitted two warnings:

```text
Triton is not supported on current platform, roll back to CPU.
```

The overall import exited zero, PyTorch reported CUDA available, and an actual
CUDA tensor allocation succeeded. The warning appears limited to the vendored
FLA device probe, but Task 3 should explicitly verify the Qwen3.5 model load and
training kernels before treating the accelerated GatedDeltaNet path as viable.
