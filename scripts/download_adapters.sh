#!/usr/bin/env bash
# Download LoRA adapters and FAISS index from Hugging Face Hub.
set -euo pipefail

HF_DATASET="${HF_DATASET:-Pyking828/eedi-misconception-engine-assets}"
STAGING="${STAGING:-/tmp/eedi_assets}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUTS="${OUTPUTS:-$ROOT/outputs}"
EEDI_DATA="${EEDI_DATA:-$(dirname "$ROOT")/eedi-data}"

echo "[download] dataset=$HF_DATASET -> $STAGING"
huggingface-cli download "$HF_DATASET" \
  --repo-type dataset \
  --local-dir "$STAGING"

mkdir -p "$OUTPUTS/retriever/lora_best_8b" \
  "$OUTPUTS/reranker/manual_lora/manual_lora_fold0_n31464_bs4_hn8_len768" \
  "$OUTPUTS/reranker/manual_lora/manual_lora_fold0_n45448_bs4_hn12_len768" \
  "$OUTPUTS/reranker/listwise_lora_r1-14b" \
  "$EEDI_DATA"

cp -f "$STAGING/retriever/"* "$OUTPUTS/retriever/lora_best_8b/"
cp -f "$STAGING/reranker_best31k/"* "$OUTPUTS/reranker/manual_lora/manual_lora_fold0_n31464_bs4_hn8_len768/"
cp -f "$STAGING/reranker_hn12/"* "$OUTPUTS/reranker/manual_lora/manual_lora_fold0_n45448_bs4_hn12_len768/"
cp -f "$STAGING/listwise_r1_14b/"* "$OUTPUTS/reranker/listwise_lora_r1-14b/"
cp -f "$STAGING/index/faiss_index.bin" "$EEDI_DATA/"
cp -f "$STAGING/index/misconception_mapping.csv" "$EEDI_DATA/"
cp -f "$STAGING/index/misc_embs.npy" "$OUTPUTS/retriever/"
cp -f "$STAGING/index/misc_ids.json" "$OUTPUTS/retriever/"

echo "[download] done."
echo "  retriever LoRA -> $OUTPUTS/retriever/lora_best_8b"
echo "  FAISS index    -> $EEDI_DATA/faiss_index.bin"
echo "Next: download base models (Qwen3-Embedding-8B, Qwen3-Reranker-8B) into HF_HOME, then:"
echo "  uvicorn service.app:app --host 0.0.0.0 --port 6006"
