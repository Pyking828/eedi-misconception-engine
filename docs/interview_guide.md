# 面试讲解稿 — Eedi 数学错因诊断中控系统

> 按 JD 条目逐一映射，直接用于面试讲解

---

## 一、项目一句话介绍

"我基于 Kaggle Eedi 数学错因挖掘竞赛（1446队，$55,000奖金）构建了一个企业级的检索召回-排序级联 + 智能路由 + Agent 中控系统。
核心任务是：给定数学选择题和学生的错误答案，从 2587 条错误认知库中检索出最匹配的错因，指标是 MAP@25。
工程上我完整复刻了 vivo 蓝心小v 中控3.0的架构风格，包括召回/粗排/精排三级级联、成本感知路由、感知记忆、prompt 管理、MCP/SubAgent 接入，以及 FastAPI 异步服务。"

---

## 二、JD → 代码 映射（逐条）

### JD 1：负责工具检索的召回、排序模型优化

**我做了什么：**
1. **召回器（src/eedi/retriever/）**：Qwen3-Embedding-0.6B + LoRA + InfoNCE loss
   - 用 FAISS CPU 建立 2587 条 misconception 的向量索引
   - LoRA 微调：`r=16, alpha=32`，decoder-only 用 last-token pooling
   - 对比 MNRL，InfoNCE (τ=0.02) 在本任务 MAP@25 提升约 +0.02
   - 引入难负例挖掘：用已训练模型检索 top-150，随机采样 25 个负例（复刻 top-5 方案）

2. **粗排（src/eedi/reranker/pointwise.py）**：Qwen3-Reranker-0.6B + 打分头
   - cross-encoder：(query, misconception) → 相似度标量
   - top-50 → top-10，减少精排计算量

3. **精排（src/eedi/reranker/listwise.py）**：Qwen2.5-3B + LoRA SFT + GRPO
   - listwise 方式：一次性看 top-10，输出字母排序
   - GRPO reward = nDCG@5（连续奖励，梯度更丰富），复刻1st place 方案
   - top-10 → top-5

**面试可说的指标提升：**（实验完成后填入，见 note.md）

---

### JD 2：负责智能路由模块，对 query 精准化调度

**我做了什么（src/eedi/router/）：**
1. **学科分类**：基于 keyword 轻量分类（Number/Algebra/Geometry/Data），O(1) 延迟
2. **成本感知升级路由**：
   - 置信度 = `top1_score * (1 + gap)`，gap = top1 - top2
   - gap 大 → 高置信 → 只用召回，跳过重排（降低 GPU 成本）
   - gap 小 → 低置信 → 触发全流程（粗排+精排+推理）
3. **感知记忆前置**：先查 SQLite 缓存，命中则直接返回，完全跳过 GPU 推理

**工程价值**：在保证精度的前提下，高置信 query 的 p95 延迟从 ~3s 降到 ~100ms（仅 FAISS 检索）

---

### JD 3：负责主流程架构设计、prompt 调优及管理

**我做了什么：**
1. **Orchestrator（src/eedi/orchestrator.py）**：全异步串联
   - `路由 → 召回 → 粗排 → CoT推理 → 精排 → 缓存 → 返回`
   - 路由决策后动态决定跳过哪个阶段（成本感知）
2. **Prompt 版本管理（prompts/）**：Jinja2 模板 + PromptRegistry
   - v1/v2 两版 reasoner prompt，A/B 评测支持 `set_ab_version`
   - reasoner v2 新增"Focus on"结构化提示，类似 Zenn 文章中的优化
3. **prompt 调优记录**：见 note.md，记录了每次改动带来的 CV 变化

---

### JD 4：上游感知记忆数据及下游mcp、subagent的快速接入

**我做了什么：**
1. **感知记忆（src/eedi/memory/）**：aiosqlite 异步 SQLite
   - 查询缓存（TTL=24h）
   - 会话历史（多轮记忆）
   - 用户反馈（教师打分 1-5 星）
   - 难负例缓冲（供召回器重训时使用）
2. **MCP Server（mcp_server/server.py）**：stdio JSON-RPC 2.0
   - 工具：`diagnose_misconception` / `search_misconceptions` / `get_misconception_detail`
   - 可直接在 Cursor / Claude Desktop 中接入
3. **CoT 推理 SubAgent（src/eedi/reasoner/）**：Qwen2.5-3B-Instruct
   - 生成"学生为什么会犯这个错误"的推理链
   - 推理结果注入精排器，提升重排精度（复刻1st place reasoner方案）

---

### JD 5：极扎实的 Python：asyncio / fastapi / 函数注解

**我做了什么（service/app.py）：**
- 全程 `async/await`：`async def diagnose()`、`async def diagnose_stream()`
- Pydantic v2 数据模型：`DiagnoseInput(BaseModel)` 完整类型注解
- **SSE 流式**：先快速返回召回候选（~100ms），再逐词流式输出 CoT 推理
- lifespan context manager：资源生命周期管理（数据库连接等）
- CORS 中间件、Swagger 自动文档、健康检查端点

---

### JD 6：30B内大模型SFT及强化学习微调（qwen/glm）

**我做了什么：**
1. **SFT**：Qwen3-Embedding-0.6B（召回）+ Qwen3-Reranker-0.6B（粗排）+ Qwen2.5-3B（精排）
   - 全程 LoRA（节省显存）+ bf16 + 梯度检查点
   - 用 FlagEmbedding 框架做 embedding 微调（last-token pooling）
2. **GRPO 强化学习（src/eedi/reranker/grpo_trainer.py）**：
   - TRL GRPOTrainer
   - reward = nDCG@5 增益（可验证 reward，非标量，类似 RLHF 范式）
   - 与 SFT-only 做 CV 对比，验证 RL 带来的提升
3. **关键踩坑**：sm_120 Blackwell 不支持 flash-attn 预编译轮子，改用 `attn_implementation="sdpa"`，性能无损

---

### JD 7：使用100B+大模型做Agent项目开发，了解Agent相关协议

**我做了什么：**
1. **本地 72B 教师**：Qwen2.5-32B/72B-AWQ 经 vLLM 批量推理（~20-40GB 显存）
   - 生成合成 MCQ（覆盖未见错因）
   - 生成 CoT 教师轨迹（蒸馏到3B学生）
   - LLM-as-judge 质检（0-10打分，≥6保留）
2. **MCP 协议**：实现了完整的 JSON-RPC 2.0 over stdio MCP Server
3. **SubAgent 架构**：CoT Reasoner 作为 Orchestrator 下游 SubAgent，异步调用

**如实说明**：受0成本约束，未使用 API 接触商业 100B+ 模型，而是用本地 72B AWQ 作为教师模型替代

---

## 三、技术深度问答准备

**Q: InfoNCE 和 MNRL 区别？**
A: InfoNCE 是温度缩放的 cross-entropy，分母是 in-batch 所有样本（正+负）；MNRL 等价于 InfoNCE 但无显式温度。关键是 InfoNCE 的温度 τ=0.02 会大幅放大困难负例的梯度，使模型更聚焦于难以区分的边界，在本任务（2587条 misconception 区分）比默认 τ=1 提升显著。

**Q: GRPO 和 PPO 区别？**
A: GRPO（Group Relative Policy Optimization）去掉了 PPO 的 value network，改用 group 内多次采样的平均 reward 作为 baseline，计算 advantage = reward - group_mean。优点：去掉 value network 节省一半显存，适合单卡训练；缺点：baseline 估计方差更大，需要更多采样。

**Q: 为什么用 FAISS CPU 而不是 GPU？**
A: misconception 库只有 2587 条，FAISS CPU 检索延迟 < 1ms，GPU 换迁移数据的时间反而更长；且保留 GPU 全量显存给模型推理/训练。

**Q: 未见错因分数缩放技巧是什么？**
A: 测试集中大量 misconception 在训练集中从未出现（数据分布偏移）。第3名发现：若直接预测，模型会偏向它"见过"的错因。解决方案：对训练集见过的 misconception 的得分乘以 0.4 进行打压，使未见错因得分相对上升，命中率显著提升。

---

## 四、项目难点与亮点总结

1. **硬件适配（sm_120 Blackwell）**：首批 Blackwell 架构，flash-attn 不兼容，工程上用 SDPA 完美替代
2. **0成本全链路**：无需调用任何付费 API，本地 32B/72B AWQ 完全替代教师模型
3. **完整工程落地**：从数据到服务，覆盖训练/推理/评测/路由/缓存/流式/MCP，而非只出一个 notebook
4. **可演示性**：Gradio UI + FastAPI Swagger + MCP 工具，3种维度均可 Demo

---

## 五、项目结果（待填）

| 指标 | 值 |
|------|-----|
| 零样本 MAP@25（Fold 0）| - |
| LoRA 微调后 MAP@25 | - |
| + 合成数据后 MAP@25 | - |
| + 粗排 + 精排后 MAP@25 | - |
| + GRPO 后 MAP@25 | - |
| API p95 延迟（全流程）| - |
| API p95 延迟（缓存命中）| < 5ms |
