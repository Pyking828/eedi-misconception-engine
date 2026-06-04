# Training Pipeline

Reproduce the full Eedi Misconception Engine training stack on your own GPU. Base models are **not** hosted on Hugging Face; download them locally or into `hf_cache/`.

## Prerequisites

- Python 3.10+
- CUDA GPU (recommended ≥ 48 GB for the 8B stages; 14B listwise / GRPO needs more VRAM or gradient checkpointing)
- Kaggle data: [Eedi - Mining Misconceptions in Mathematics](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/data) → extract to `../eedi-data/`

```bash
pip install -e ".[dev]"
export HF_HOME=../hf_cache
export EEDI_DATA=../eedi-data
```

## Pipeline (in order)

| Step | Script | Output |
|------|--------|--------|
| 0 | `scripts/prepare_data.py` | folds, cleaned CSVs (HF-mirror data prep) |
| 0b | `scripts/eda.py` | optional EDA |
| 1 | `scripts/retriever_baseline.py` | retriever zero-shot baseline + LoRA fine-tune |
| 2 | `scripts/synth_data.py` | synthetic MCQs (R1 teacher + self-judge) |
| 3 | `scripts/retriever_multistage.py` | multistage retriever LoRA (synth pretrain → real finetune) |
| 4 | `scripts/build_index.py` | `faiss_index.bin`, `misc_embs.npy`, `misc_ids.json` |
| 5 | `scripts/reranker_zeroshot_eval.py` | reranker zero-shot metrics |
| 6 | `scripts/reranker_pointwise_train.py` | pointwise LoRA (yes/no logit; 31k & hn12) |
| 7 | `scripts/generate_cot.py` | CoT rationales (injected into rerankers) |
| 8 | `scripts/rerank_scores.py` | per-candidate reranker score files |
| 9 | `scripts/score_scaling.py` | retrieval×rerank score-fusion sweep |
| 10 | `scripts/listwise_sft.py` | R1-14B listwise SFT LoRA (option-logit) |
| 11 | `scripts/listwise_rerank.py` | listwise rerank + eval |
| 12 | `scripts/grpo_listwise.py` | GRPO RL on the listwise reranker |
| 13 | `scripts/reranker_ensemble.py` | weighted ensemble → final MAP@25 |
| 14 | `scripts/unseen_eval.py` | seen vs. unseen misconception breakdown |
| — | `scripts/smoke_service.py` | API smoke test (no Kaggle data needed) |
| — | `scripts/spaces_precompute.py` | precompute the HF Spaces CPU-demo assets |
| — | `scripts/trackio_log.py` | log the MAP@25 progression to a Trackio dashboard |
| — | `scripts/demo_share.py` | launch the full-pipeline Gradio UI locally |

Configure paths in `configs/base.yaml` (`paths.data`, `paths.hf_cache`, `paths.outputs`).

> The exploratory / superseded first-pass scripts (early vLLM synth, early TRL-GRPO, the
> generic reranker trainer) were moved out of the repo to keep the mainline unambiguous.

## Inference assets (skip training)

Download the pre-built LoRA adapters + FAISS index from Hugging Face:

```bash
bash scripts/download_adapters.sh
```

Dataset: [Pyking828/eedi-misconception-engine-assets](https://huggingface.co/datasets/Pyking828/eedi-misconception-engine-assets)

## Expected metrics (fold-0 val)

| Stage | MAP@25 | Recall@25 |
|-------|--------|-----------|
| Retriever 8B zero-shot | 0.2248 | 0.6535 |
| Retriever 8B LoRA (real) | 0.4289 | 0.9416 |
| Pointwise reranker (31k) | 0.5700 | 0.9211 |
| + baseline-pool rescoring | 0.5807 | 0.9611 |
| 3-model ensemble | **0.5974** | 0.9611 |
