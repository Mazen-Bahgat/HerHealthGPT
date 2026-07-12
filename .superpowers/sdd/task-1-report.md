# Task 1 report: WSL2 + Unsloth environment

Date: 2026-07-12

Status: initial implementation and verification committed as `c8f5617`; review
corrections are recorded in the follow-up section below.

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

The repeatable direct-package command documented for future runs pins those
final versions and pip itself. It is intentionally not described as a full
environment lock because transitive resolution may change:

```powershell
wsl.exe -d HerHealthUbuntu -u root -- bash -lc "set -euo pipefail; cd /mnt/d/Grad-Project/HerHealthGPT && .venv-ft/bin/python -m pip install 'pip==26.1.2' && .venv-ft/bin/pip install 'torch==2.10.0' 'unsloth==2026.7.2' 'transformers==5.5.0' 'trl==0.24.0' 'peft==0.19.1' 'bitsandbytes==0.49.2' 'datasets==4.3.0' 'accelerate==1.14.0'"
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
CUDA tensor allocation succeeded. That proves generic CUDA, not accelerated
FLA/GatedDeltaNet usability. The environment remains provisional for Qwen3.5;
Task 3's `--max-steps 10` training smoke is the binding go/no-go.

## Review corrections and non-install verification

The setup doc now:

- labels the procedure as a repeatable direct-package setup rather than a fully
  reproducible locked environment;
- exports `/root/.local/bin` before checking for `uv`, checks `uv 0.11.28`, and
  requests Python `3.11.15` explicitly;
- records this Ubuntu 24.04 distro's actual system Python (`3.12.3`) and explains
  why the isolated tested 3.11 interpreter is used, while retaining the plan's
  Python 3.14 compatibility warning for newer images;
- documents the actual FLA/Triton CPU-fallback warning and marks Qwen3.5 GPU
  acceleration provisional; and
- identifies Task 3's 10-step training smoke as the binding go/no-go without
  adding an expensive model test to Task 1.

Non-install WSL verification (the Python payload also asserted the displayed
versions, device name, and CUDA tensor value):

```powershell
wsl.exe -d HerHealthUbuntu -u root -- bash -lc "set -euo pipefail; /root/.local/bin/uv --version | grep '^uv 0\.11\.28'; python3 --version; cd /mnt/d/Grad-Project/HerHealthGPT; .venv-ft/bin/python -c 'import sys, torch, unsloth; assert sys.version_info[:3] == (3, 11, 15); assert torch.__version__ == \"2.10.0+cu128\"; assert unsloth.__version__ == \"2026.7.2\"; assert torch.cuda.is_available(); assert torch.cuda.get_device_name(0) == \"NVIDIA RTX 5000 Ada Generation\"; assert torch.ones(1, device=\"cuda\").item() == 1.0; print(\"python\", sys.version.split()[0]); print(\"torch\", torch.__version__, \"runtime\", torch.version.cuda); print(\"unsloth\", unsloth.__version__); print(\"cuda\", torch.cuda.is_available(), torch.cuda.get_device_name(0)); print(\"cuda_tensor ok\")' 2>/tmp/task1-review-stderr.log; grep -c 'Triton is not supported on current platform, roll back to CPU.' /tmp/task1-review-stderr.log"
```

```text
uv 0.11.28 (x86_64-unknown-linux-gnu)
Python 3.12.3
python 3.11.15
torch 2.10.0+cu128 runtime 12.8
unsloth 2026.7.2
cuda True NVIDIA RTX 5000 Ada Generation
cuda_tensor ok
2
```

Documentation contract verification:

```powershell
$doc = Get-Content -Raw 'docs\wsl_finetune_setup.md'; # assert nine required disclosures and reject the old overbroad heading
```

```text
doc_contract PASS (9 required disclosures; no overbroad heading)
git diff --check: exit 0
```
