---
title: Eedi Math Misconception Engine
emoji: 🎓
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
---

# Eedi 数学错因检索 Demo

CPU 版轻量演示，使用 bge-m3 + FAISS 实现零样本错因检索。

完整版（本地 GPU）包含：
- Qwen3-Embedding-8B LoRA 微调召回器
- Qwen3-Reranker-8B 粗排
- DeepSeek-R1-Distill-Qwen-14B + GRPO 精排/推理
- DeepSeek-R1-Distill-Qwen-32B 离线 teacher / judge / 蒸馏增强
- CoT 推理 SubAgent
- FastAPI SSE 流式服务
- MCP Server 接入

GitHub：https://github.com/YOUR_GITHUB/eedi-misconception-engine
