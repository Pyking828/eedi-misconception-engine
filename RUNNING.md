# 当前运行状态

更新时间：**2026-06-02 14:45**

## 当前阶段

- 当前总阶段：**阶段1 / 最终召回器完成，等待进入阶段3**
- 当前关键节点：**`Qwen/Qwen3-Embedding-8B` FAISS 索引与 candidate_pool 构建完成；等待 `Qwen3-Reranker-8B` 下载完成后进入重排阶段**
- 当前执行方式：后台脚本 `scripts/download_final_models.sh`
- 是否需要你手动点 Build：**不需要**
- 是否建议关闭 `plan.md`：可以关闭，后续看本文件即可

## 下载队列

按顺序执行，前一个完成后自动进入下一个：

1. `Qwen/Qwen3-Embedding-8B`：已完成（2026-06-02 14:20:49）
2. `Qwen/Qwen3-Reranker-8B`：正在下载
3. `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`：等待
4. `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`：等待，仅做离线 teacher / judge / 蒸馏增强项

## 当前进度

| 项目 | 状态 |
|---|---|
| 数据准备 | 已完成，`eedi-data` 已就绪 |
| 快速 baseline | 已完成，`Qwen3-Embedding-0.6B + LoRA` Fold0 MAP@25=0.3172 |
| 最终模型下载 | 进行中，当前在下载 `Qwen3-Reranker-8B` |
| 8B zero-shot | 已完成，5折平均 MAP@25=0.2248，Recall@25=0.6535 |
| 8B LoRA 微调 | 已完成，Fold0 MAP@25=0.4012，Recall@25=0.8822 |
| 8B FAISS 索引 | 已完成，`/root/autodl-tmp/eedi-data/faiss_index.bin` |
| 8B candidate_pool | 已完成，4370 条 query × top-50，Recall@50=0.9700 |
| 当前模型缓存 | `Qwen3-Embedding-8B` 已完整缓存约 15GB；`Qwen3-Reranker-8B` 已缓存约 13GB |
| 当前未完成下载块 | `Qwen3-Embedding-8B` 0 个 incomplete；`Qwen3-Reranker-8B` 5 个 incomplete，合计约 7.65GB |
| 数据盘 | 500GB 总量，已用约 37GB，剩余约 464GB |

## 不再每分钟更新

本文件不会每分钟自动刷新。后续我只会在以下关键节点更新：

- 当前模型下载完成
- 下载进入下一个模型
- 四个最终模型全部下载完成
- 开始/完成 8B 召回训练
- 开始/完成 8B reranker 重排
- 开始/完成 32B 合成数据与 judge
- 开始/完成 14B 精排 / GRPO
- 服务 demo 跑通
- 推 GitHub 前检查点

## 当前后台任务

正在运行：

```text
scripts/download_final_models.sh
└── hf download Qwen/Qwen3-Reranker-8B
```

刚完成：

```text
python scripts/01_retriever_baseline.py --model Qwen/Qwen3-Embedding-8B --fold 0 --train --epochs 2 --batch-size 16
```

刚完成：

```text
python scripts/05_build_index.py --model Qwen/Qwen3-Embedding-8B --adapter-path outputs/retriever/lora_best_8b --top-k 50
```

## 我接下来准备执行的计划

### 阶段1：最终召回器（当前阶段）

目标：把快速 baseline 的 `Qwen3-Embedding-0.6B` 升级为最终主线 `Qwen3-Embedding-8B`。

执行顺序：

1. 等待 `Qwen3-Embedding-8B` 下载完成。
2. 运行 8B zero-shot 召回评测，记录 5 折平均 `MAP@25 / Recall@25 / Recall@10 / nDCG@25`。
3. 对 `Qwen3-Embedding-8B` 做 LoRA + MultipleNegativesRankingLoss 微调。
4. 复跑 fold0 和全量候选池评测，和 0.6B baseline 对比。
5. 构建最终 FAISS 索引与 `candidate_pool.json`，作为后续 reranker 训练输入。
6. 更新 `note.md`、`RUNNING.md`、README 指标表。

当前已知 baseline：

| 模型 | Fold0 MAP@25 | Fold0 Recall@25 | 说明 |
|---|---:|---:|---|
| `Qwen3-Embedding-0.6B` zero-shot | 0.2038 | 0.5984 | 快速基线 |
| `Qwen3-Embedding-0.6B` LoRA | 0.3172 | 0.8146 | 已完成，作为对照 |
| `Qwen3-Embedding-8B` zero-shot | 0.2371 | 0.6602 | 已完成，5折平均 MAP@25=0.2248 |
| `Qwen3-Embedding-8B` LoRA | **0.4012** | **0.8822** | 已完成，当前最强召回器 |

当前召回候选池：

| 产物 | 状态 |
|---|---|
| FAISS index | 已完成 |
| `candidate_pool.json` | 已完成 |
| Recall@50 | **0.9700** |

### 阶段2：合成数据与离线 teacher/judge

目标：用 32B teacher 提升未见错因覆盖和 CoT 质量。

执行顺序：

1. 等待 `DeepSeek-R1-Distill-Qwen-32B` 下载完成。
2. 用 32B 生成未见错因的 MCQ 合成数据。
3. 用 32B 作为 LLM-as-judge，对合成样本打 0-10 分，过滤低质量数据。
4. 生成 CoT 错因解释轨迹，作为 14B/8B 的蒸馏数据。
5. 合并真实数据 + 合成数据，重新训练 8B 召回器。
6. 记录“合成数据前后”的指标增益。

注意：

- 32B 只做离线 teacher / judge / distillation。
- 32B 不进入线上服务，避免 demo 成本过高。

### 阶段3：重排级联

目标：构建企业检索系统常见的“召回 -> 粗排 -> 精排”链路。

执行顺序：

1. 等待 `Qwen3-Reranker-8B` 下载完成。
2. 先跑 zero-shot rerank：`candidate_pool top50 -> top10`。
3. 若收益明显，再做 reranker LoRA 微调。
4. 接入 `DeepSeek-R1-Distill-Qwen-14B` 做 listwise 精排：`top10 -> top5`。
5. 加入“未见错因分数缩放”策略，对比是否提升泛化。
6. 输出阶段3消融：只召回 / +粗排 / +精排 / +未见缩放。

### 阶段4：GRPO 强化学习

目标：把“排序指标”直接做成 reward，展示强化学习微调能力。

执行顺序：

1. 用阶段3的 listwise 精排样本构造 GRPO 数据。
2. reward 使用 `nDCG@5` 或 `top1_hit`，优先 `nDCG@5`。
3. 对 `DeepSeek-R1-Distill-Qwen-14B` 做 LoRA + GRPO。
4. 与 SFT-only 精排模型对比。
5. 将 GRPO 训练曲线、reward 设计、指标提升写入 `note.md`。

### 阶段5：中控、路由、记忆、Prompt 管理

目标：把算法链路包装成岗位 JD 中的“中控3.0”工程系统。

执行顺序：

1. 完善 `orchestrator.py`：路由 -> 召回 -> 粗排 -> 精排 -> 推理解释。
2. 完善智能路由：按召回置信度决定是否升级到 rerank / CoT。
3. 完善 SQLite 记忆：缓存、会话历史、用户反馈、难负例池。
4. 完善 `prompts/`：版本化模板、A/B 配置、调优记录。
5. 写清楚 JD 映射：召回、排序、智能路由、prompt 管理、memory、subagent。

### 阶段6：服务化与 MCP

目标：做成可演示、可调用、符合工程岗位要求的系统。

执行顺序：

1. 启动 FastAPI：`/diagnose`、`/diagnose/stream`、`/search`、`/feedback`。
2. 启动 Gradio UI：输入题目、正确答案、错误答案，返回错因候选和解释。
3. 启动 MCP Server：暴露 `diagnose_misconception` 和 `search_misconceptions`。
4. 在本 AutoDL 实例上录屏/截图，保存到 README 或 docs。

### 阶段7：持久化部署

目标：AutoDL 实例释放后，项目仍然可以展示。

执行顺序：

1. 轻量 CPU demo 部署到 HuggingFace Spaces。
2. 主项目代码推到 GitHub。
3. 小权重 LoRA / FAISS / demo 所需资源推到 HuggingFace Hub。
4. README 放：架构图、结果表、Demo链接、录屏/GIF、面试讲解入口。

需要用户后续提供：

- HuggingFace write token：发布 Space / 模型资源时需要。
- GitHub 仓库名与 push 凭证：CP5 前我会再找你确认。

### 阶段8：工程化收尾

目标：把项目变成“能投简历”的 GitHub 仓库。

执行顺序：

1. `pytest`、`ruff`、CI 检查。
2. 清理 `.gitignore`，确保不提交 token、大模型权重、超大缓存。
3. 整理 `note.md`：按“问题 -> 原因 -> 解决 -> 收益”完善。
4. 整理 `docs/interview_guide.md`：面试讲解稿。
5. 推 GitHub 前进入 CP5 检查点，请你确认后再推送。

## 你现在不需要做什么

- 不需要点 `plan.md` 的 Build
- 不需要手动下载模型
- 不需要扩容数据盘
- 不需要保持 `plan.md` 打开

## 风险与处理

- 如果 `hf-mirror` 下载中断，下载脚本会自动重试，且支持断点续传。
- 如果数据盘剩余空间低于 120GB，或总占用接近 350GB，我会提醒你扩容。
- 如果后续训练 OOM，会优先降低 batch / 并发，不会改变主模型路线。
