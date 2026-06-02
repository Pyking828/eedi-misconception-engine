# Eedi 数学错因诊断中控系统

> **Misconception Mining Engine** — 企业级"召回→粗排→精排→推理 SubAgent"级联检索系统
> 基于 Kaggle Eedi 竞赛（数学错因挖掘，MAP@25）× vivo 蓝心小v 中控3.0架构风格

[![CI](https://github.com/YOUR_GITHUB/eedi-misconception-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_GITHUB/eedi-misconception-engine/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 项目背景与定位

本项目基于 Kaggle **Eedi - Mining Misconceptions in Mathematics** 竞赛（1446 支队伍，$55,000 奖金），将其重构为一个**企业级中控风格的 AI 系统**：

- **任务**：给定数学选择题 + 学生的错误选项，从 2587 条错因描述库中检索最匹配的错因（MAP@25）
- **工程目标**：不只是冲榜，而是打造可落地、可面试讲解的全链路 AI 系统

### 岗位 JD → 项目模块映射（面试讲解用）

| JD 条目 | 对应模块 |
|---------|---------|
| 工具检索召回、排序模型优化 | `retriever/`（召回）+ `reranker/`（粗排/精排）三级级联 |
| 智能路由，对 query 精准化调度 | `router/`：学科分类 + 成本感知升级 |
| 主流程架构设计、prompt 调优管理 | `orchestrator.py` + `prompts/`（版本化 Jinja 模板 + A/B）|
| 上游感知记忆 + 下游 mcp/subagent | `memory/`（SQLite）+ `mcp_server/` + `reasoner/`（CoT）|
| asyncio/fastapi/函数注解 | `service/app.py`（全异步 + pydantic v2 + SSE）|
| 30B内大模型 SFT + 强化学习（qwen）| LoRA SFT（召回/重排）+ **GRPO** 精排 |
| 30B 内大模型做 Agent | DeepSeek-R1-Distill-Qwen-14B 线上推理/精排 + DeepSeek-R1-Distill-Qwen-32B 离线 teacher/judge/蒸馏增强 |

---

## 系统架构

```
用户 Query（题目 + 正答 + 错答）
         │
         ▼
┌─────────────────────────────┐
│  智能路由 Router              │  学科分类 + 置信度 → 成本感知升级
│  + 感知记忆 Memory            │  SQLite 缓存命中 → 直接返回
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  召回 Retriever              │  Qwen3-Embedding-8B + LoRA
│  FAISS CPU IndexFlatIP      │  InfoNCE loss + 难负例挖掘
│  top-50 候选                 │  MAP@25 / Recall@25 / nDCG@25
└─────────────┬───────────────┘
              │ [置信度低才升级]
              ▼
┌─────────────────────────────┐
│  粗排 Pointwise              │  Qwen3-Reranker-8B + LoRA
│  top-50 → top-10            │  cross-encoder 打分头
└─────────────┬───────────────┘
              │
      ┌───────┴────────┐
      │                │
      ▼                ▼
┌──────────┐    ┌───────────────────┐
│ CoT      │    │ 精排 Listwise      │  DeepSeek-R1-Distill-Qwen-14B + LoRA/GRPO
│ Reasoner │───▶│ top-10 → top-5    │  reward = nDCG@5 增益
│ SubAgent │    └───────────────────┘
└──────────┘
              │
              ▼
       MAP@25 候选 + 推理解释
              │
    ┌─────────┴──────────┐
    │    服务层            │
    │  FastAPI (SSE)      │
    │  Gradio Demo UI     │
    │  MCP Server         │
    └────────────────────┘
```

---

## 环境要求

| 组件 | 最低 | 推荐（本项目） |
|------|------|--------------|
| GPU | RTX 3090 24GB | RTX PRO 6000 Blackwell 96GB |
| CUDA | 11.8+ | 12.8 |
| Python | 3.10+ | 3.12.3 |
| 磁盘 | 100GB | 500GB（数据盘）|

---

## 快速上手

### 1. 克隆并安装

```bash
git clone https://github.com/YOUR_GITHUB/eedi-misconception-engine.git
cd eedi-misconception-engine

# 设置环境变量（重定向所有缓存到数据盘）
export HF_HOME=/root/autodl-tmp/hf_cache
export EEDI_DATA=/root/autodl-tmp/eedi-data

pip install -e .
```

### 2. 下载数据

```bash
# 需要 Kaggle 账号并在比赛页接受规则
# https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics
kaggle competitions download -c eedi-mining-misconceptions-in-mathematics -p $EEDI_DATA
unzip $EEDI_DATA/eedi-mining-misconceptions-in-mathematics.zip -d $EEDI_DATA
```

### 3. 数据预处理 & EDA

```bash
python scripts/00_eda.py
```

### 4. 构建召回基线（零样本）

```bash
python scripts/01_retriever_baseline.py --fold 0
```

### 5. LoRA 微调召回器

```bash
python scripts/01_retriever_baseline.py --fold 0 --train
python scripts/05_build_index.py --model Qwen/Qwen3-Embedding-8B --adapter-path outputs/retriever/lora_best_8b
```

### 6. 合成数据生成（可选，需要大模型）

```bash
python scripts/02_synth_data.py --teacher deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --n 5
```

### 7. 重排训练

```bash
python scripts/03_reranker_train.py --stage both
```

### 8. GRPO 强化学习

```bash
python scripts/04_grpo_train.py --reward ndcg_gain
```

### 9. 启动服务

```bash
# FastAPI + Gradio（端口 6006）
make serve
# 或：
uvicorn service.app:app --host 0.0.0.0 --port 6006
```

访问：
- **Gradio UI**：`http://localhost:6006/ui`
- **API 文档**：`http://localhost:6006/docs`
- **健康检查**：`http://localhost:6006/health`

---

## 离线指标（消融实验）

| 阶段 | 方法 | Recall@25 | MAP@25 | nDCG@25 |
|------|------|-----------|--------|---------|
| 快速基线 | Qwen3-Embedding-0.6B 零样本/LoRA | 0.8146(Fold0) | 0.3172(Fold0) | 0.4280(Fold0) |
| 最终主线 | Qwen3-Embedding-8B zero-shot | 0.6535(5折均值) | 0.2248(5折均值) | 0.3194(5折均值) |
| 最终主线 | Qwen3-Embedding-8B LoRA | 0.8822(Fold0) | 0.4012(Fold0) | 0.5102(Fold0) |
| 候选池 | Qwen3-Embedding-8B LoRA top-50 | **0.9700(Recall@50)** | - | - |
| 阶段1 | + LoRA + InfoNCE 微调 | - | - | - |
| 阶段2 | + 合成数据（DeepSeek-R1-Distill-Qwen-14B 教师） | - | - | - |
| 阶段3 | + 粗排 + 精排（SFT）| - | - | - |
| 阶段4 | + GRPO 强化学习 | - | - | - |

> 指标在实验执行后更新（见 `note.md`）

---

## MCP 接入配置

在 Cursor 或 Claude Desktop 中添加（`.cursor/mcp.json`）：

```json
{
  "mcpServers": {
    "eedi-misconception-engine": {
      "command": "python",
      "args": ["/path/to/eedi-misconception-engine/mcp_server/server.py"],
      "env": {
        "HF_HOME": "/root/autodl-tmp/hf_cache"
      }
    }
  }
}
```

可用工具：
- `diagnose_misconception`：完整诊断（召回→重排→CoT）
- `search_misconceptions`：纯向量检索
- `get_misconception_detail`：查询错因详情

---

## API 参考

```bash
# 错因诊断
curl -X POST http://localhost:6006/diagnose \
  -H "Content-Type: application/json" \
  -d '{
    "question_text": "Simplify: 5 × 4 + 6 ÷ 2",
    "correct_answer": "23",
    "wrong_answer": "13",
    "subject_name": "Number",
    "top_k": 10
  }'

# SSE 流式
curl -X POST http://localhost:6006/diagnose/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"question_text": "...", "correct_answer": "...", "wrong_answer": "..."}'
```

---

## 项目结构

```
eedi-misconception-engine/
├── src/eedi/
│   ├── data/           # 数据加载 / 长表转换 / 5折CV
│   ├── retriever/      # 召回器（Qwen3-Embedding + FAISS + LoRA）
│   ├── reranker/       # 粗排(Pointwise) + 精排(Listwise) + GRPO
│   ├── reasoner/       # CoT 推理 SubAgent
│   ├── router/         # 智能路由（学科分类 + 成本感知）
│   ├── synth/          # 合成数据生成（vLLM + LLM-as-judge）
│   ├── memory/         # 感知记忆（SQLite 缓存/反馈/难负例）
│   └── orchestrator.py # 中控主流程（异步串联）
├── service/
│   └── app.py          # FastAPI 异步服务 + Gradio UI
├── mcp_server/
│   └── server.py       # MCP stdio 服务
├── eval/
│   └── evaluator.py    # MAP@25 / Recall@K / nDCG@K
├── scripts/            # 各阶段训练脚本
├── prompts/            # 版本化 Jinja 模板
├── configs/            # OmegaConf/Hydra YAML 配置
├── tests/              # pytest 单元测试
├── spaces/             # HuggingFace Spaces（免费 CPU demo）
├── docs/               # 技术复盘 + JD面试稿
├── note.md             # 端到端操作日志（面试复盘用）
├── Makefile
└── pyproject.toml
```

---

## 技术复盘（经验）

详见 [`note.md`](note.md) 和 [`docs/`](docs/) 目录。

关键技术沉淀：
1. **sm_120 Blackwell 兼容性**：flash-attn 不支持 sm_120，改用 PyTorch SDPA，性能无损
2. **InfoNCE vs MNRL**：InfoNCE（τ=0.02）比 MNRL 在本任务提升 MAP@25 约 0.02
3. **未见错因分数缩放**：对训练集出现过的 misconception 乘 0.4，显著提升对新错因的召回（复刻3rd place）
4. **GRPO reward 选择**：nDCG@5 连续奖励比 top1_hit 0/1 奖励梯度更丰富，训练更稳

---

## 参考资料

- [Kaggle 比赛主页](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics)
- [1st Place 解法（Raja Biswas）](https://github.com/rbiswasfc/eedi-mining-misconceptions)
- [5th Place 解法](https://github.com/ebinan92/Eedi-5th-solution)
- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)

---

## License

MIT License — 见 [LICENSE](LICENSE)
