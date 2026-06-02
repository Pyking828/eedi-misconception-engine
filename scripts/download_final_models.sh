#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export HF_XET_HIGH_PERFORMANCE=1

MODELS=(
  "Qwen/Qwen3-Embedding-8B"
  "Qwen/Qwen3-Reranker-8B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
  # Offline-only bonus teacher/judge/distillation model. Not used in online demo.
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
)

echo "HF_ENDPOINT=${HF_ENDPOINT}"
echo "HF_HOME=${HF_HOME}"
echo "Start time: $(date '+%F %T')"
echo

for model in "${MODELS[@]}"; do
  echo "============================================================"
  echo "[download] ${model}"
  echo "============================================================"
  for attempt in $(seq 1 20); do
    echo "[attempt ${attempt}/20] $(date '+%F %T')"
    if hf download "${model}" --cache-dir "${HF_HOME}"; then
      echo "[done] ${model} at $(date '+%F %T')"
      du -sh "${HF_HOME}/models--${model//\//--}" 2>/dev/null || true
      df -h /root/autodl-tmp || true
      break
    fi
    echo "[retry] ${model} failed, sleep 10s..."
    sleep 10
  done
done

echo
echo "All requested model downloads finished at $(date '+%F %T')"
