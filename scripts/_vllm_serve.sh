#!/bin/bash
# Local helper (not committed): serve Qwen3.5-9B via vLLM with the full CUDA env
# a headless systemd service needs. Qwen3.5's FLA/GatedDeltaNet path JIT-compiles
# kernels at init, so CUDA_HOME/nvcc + the nvidia libs must be on the paths.
cd /home/sw2/vllm-qwen
export HOME=/home/sw2
NVDIR=/home/sw2/vllm-qwen/.venv/lib/python3.12/site-packages/nvidia
export CUDA_HOME="$NVDIR/cu13"
export PATH="/home/sw2/vllm-qwen/.venv/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$(ls -d $NVDIR/*/lib | paste -sd:):/usr/lib/wsl/lib"
# Qwen3.5 + this vLLM build tries to JIT-compile FlashInfer's sampling kernels at
# init, which fails here (pip-only CUDA lacks curand.h / matching toolkit headers).
# Disable the FlashInfer sampler so vLLM uses its native PyTorch sampler instead.
export VLLM_USE_FLASHINFER_SAMPLER=0
exec .venv/bin/vllm serve Qwen/Qwen3.5-9B \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.65 \
  --reasoning-parser qwen3
