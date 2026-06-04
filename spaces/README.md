---
title: Eedi Math Misconception Engine
emoji: 🎓
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.9.1"
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

# 🎓 Eedi Math Misconception Diagnosis — Demo

Given a math question, the correct answer and a student's wrong answer, this engine
retrieves the most likely **misconception** behind the error (Kaggle *Eedi - Mining
Misconceptions in Mathematics*).

This **Space is the lightweight CPU demo**: `bge-m3` + FAISS retrieval over the 2587 real
misconceptions (embeddings are precomputed and shipped, so cold start is fast).

## Full pipeline (local GPU)

The complete system (in the GitHub repo) is a retrieve → rerank → listwise → CoT cascade:

- **Retriever:** Qwen3-Embedding-8B + LoRA (contrastive learning)
- **Pointwise reranker:** Qwen3-Reranker-8B + LoRA (yes/no logit)
- **Ensemble + listwise:** two rerankers + R1-14B listwise (option-logit SFT)
- **Reasoning:** DeepSeek-R1-Distill-Qwen-14B CoT explanation (offline 32B teacher/judge)
- **Serving:** FastAPI (async + SSE) · Gradio UI · MCP Server

**Final offline result:** fold0 CV **MAP@25 = 0.597** (≈ Kaggle private-LB top-3).

**GitHub (full code + reproduction):** [Pyking828/eedi-misconception-engine](https://github.com/Pyking828/eedi-misconception-engine)

**Pre-trained assets (LoRA + FAISS index):** [Pyking828/eedi-misconception-engine-assets](https://huggingface.co/datasets/Pyking828/eedi-misconception-engine-assets)
