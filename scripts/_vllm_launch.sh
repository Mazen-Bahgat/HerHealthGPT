#!/bin/bash
# Local helper (not committed): (re)launch vLLM as a systemd service, wait healthy.
systemctl reset-failed vllm-qwen 2>/dev/null
systemd-run --unit=vllm-qwen --collect -p User=sw2 -p Group=sw2 \
  /mnt/d/Grad-Project/HerHealthGPT/scripts/_vllm_serve.sh
for i in $(seq 1 40); do
  curl -s -m 5 http://localhost:8000/v1/models > /tmp/vc.txt
  if grep -q Qwen /tmp/vc.txt; then echo "HEALTHY after ~$((i*8))s"; break; fi
  sleep 8
done
echo "active=$(systemctl is-active vllm-qwen)"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader
