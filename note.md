# Eedi 错因检索中控系统 — 端到端操作日志

> 用途：全程记录执行细节，供面试复盘使用。格式：**问题 → 原因 → 解决 → 收益**

---

## 硬件环境（基线）

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| 显存 | 97887 MiB (~96GB) 空闲 97251 MiB |
| CUDA Driver | 580.82.09 / CUDA 13.0 |
| 架构 | sm_120 (Blackwell) |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| 根分区 | 30GB 总量，441MB 已用（务必保持<20GB以内） |
| 数据盘 | /root/autodl-tmp 500GB，当前几乎空闲 |

---

## 资源护栏配置

```bash
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache/hub
export PIP_CACHE_DIR=/root/autodl-tmp/pip_cache
export TMPDIR=/root/autodl-tmp/tmp
```

> 所有模型权重、数据集、pip 包缓存一律落在数据盘，禁止写根分区

## 运行状态可视化约定

- `plan.md`：只作为设计规格和阶段计划，不再用于实时运行状态展示。
- `RUNNING.md`：中文运行状态面板，只在阶段或关键节点变化时更新。
- `RUNNING_STATUS.md`：旧英文状态文件，已停止自动刷新并指向 `RUNNING.md`。
- 不建议点击 `plan.md` 的 Build，避免启动重复下载/训练进程抢占网络、磁盘和 GPU。

---

## 阶段 0：环境搭建与数据准备

### 0.1 时间戳

开始时间：（执行时自动填入）

---

### 0.2 工具安装

#### 安装 uv（包管理）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 安装 kaggle CLI

```bash
pip install kaggle -q --cache-dir /root/autodl-tmp/pip_cache
```

#### 安装 huggingface-cli（hf）

```bash
pip install huggingface_hub[cli] -q --cache-dir /root/autodl-tmp/pip_cache
```

---

### 0.3 sm_120 Blackwell 兼容性冒烟测试（实测）

| 组件 | 版本 | 状态 | 备注 |
|------|------|------|------|
| torch CUDA bf16 matmul | 2.8.0+cu128 | ✅ | sm_120 原生支持 |
| torch SDPA (FlashAttention 替代) | - | ✅ | flash_sdp_enabled=True，无需装 flash-attn |
| faiss-cpu | 1.14.2 | ✅ | 2587向量 IndexFlatIP <1ms |
| FlagEmbedding | 1.4.0 | ✅ | FlagModel/FlagReranker OK |
| peft / trl / accelerate | 0.19.1 / 1.5.1 / 1.13.0 | ✅ | LoRA/GRPO/SFT 均可用 |
| bitsandbytes | 0.49.2 | ✅ | 4bit QLoRA 配置正常 |
| transformers | 5.9.0 | ✅ | - |
| vLLM | 0.22.0 | ✅ | 阶段2合成数据用 |

**关键结论（面试必说）**：
- Blackwell sm_120 是 2024 最新架构，但 torch 2.8+cu128 已原生支持
- `flash_sdp_enabled=True`：PyTorch 内置 SDPA 在 sm_120 上自动走 FlashAttention 路径，**无需单独编译 flash-attn**（避免了 sm_120 轮子不兼容的大坑）
- 统一用 `attn_implementation="sdpa"` 加载所有模型，零兼容问题

---

### 0.4 Kaggle 数据下载 —— 重大网络踩坑（面试高频考点）

**问题**：`kaggle competitions download` 静默卡死（90秒0字节，退出码124）

**定位过程**（逐步排查，体现工程能力）：
1. `kaggle competitions list` 正常 → 认证OK，token配置正确（新版 KGAT_ token 用 `~/.kaggle/access_token`）
2. `kaggle competitions download` 卡死 → 用 `curl -v` 直连 Kaggle API 抓包
3. 发现：Kaggle API 返回 `HTTP 302` 重定向到 `storage.googleapis.com`
4. 连接 `storage.googleapis.com`（173.194.x.x + IPv6）全部超时

**根因**：AutoDL 国内网络**无法访问 Google Cloud Storage**（Kaggle 数据实际存储地被墙），而非 kaggle 配置问题

**解决**：改用 **HuggingFace 镜像** `cdtmc/eedi-ir`（Eedi 比赛的完整 IR 格式镜像）
- 同时发现 huggingface.co 直连也超时，但 `hf-mirror.com` 国内镜像可用
- 设置 `export HF_ENDPOINT=https://hf-mirror.com`（写入 .bashrc，影响后续所有模型下载）

**收益**：绕过墙，2.5秒完成数据准备；同时为后续 Qwen/bge 模型下载铺平道路

**数据源映射**（cdtmc/eedi-ir → 项目 schema）：
- corpus (2587, [id_, text]) → misconception_mapping.csv
- queries (4370, [fold, id_, text]) → 题目（text含 "Subject | Construct | Question"，正则解析）
- qrels (4370, [fold, qid, mid]) → query→misconception 金标

---

### 0.5 EDA 关键结论（实测）

| 指标 | 值 |
|------|-----|
| 错因总数 | **2587** |
| 标注的 query-distractor 对 | **4370** |
| 训练中出现的错因 | **1604** |
| **未见错因** | **983 (38.0%)** ⭐ 核心难点 |
| 学科(SubjectName)种类 | 细粒度上千类（如 Linear Equations/BIDMAS/Area）|

**核心洞察（面试必说）**：
- 38% 错因在训练集从未出现 → 纯监督学习上限低，必须解决泛化
- → 解法1：合成数据覆盖未见错因（阶段2）
- → 解法2：未见错因分数缩放（阶段3，复刻 3rd place）
- → 解法3：强语义 embedding（Qwen3）保证未见错因也能召回

**样例验证**（IR text 正则解析正确）：
- QID `0_D`: Subject=BIDMAS, Q="3×2+4-5 括号放哪使结果=13", Wrong="Does not need brackets" → 错因[1672]"Confuses the order of operations"

---

### 0.6 5 折 CV 构建方式

- GroupKFold by QuestionId，避免同题 distractor 跨折泄露
- 每折 train≈3496 / val≈874
- 每折验证集含 ~185-193 个"训练中未见"错因 → 专测泛化能力
- 保存至：`folds.parquet`

---

### 0.7 磁盘使用记录

| 时间点 | /root 已用 | /root/autodl-tmp 已用 |
|--------|-----------|----------------------|
| 阶段0开始 | 441MB | ~72KB |
| 阶段0结束（装包后）| 1.7GB | 5.8GB（主要pip缓存）|

**阶段0小结**：环境全绿、数据就绪（4370样本/2587错因）、骨架50文件、22测试通过、git已提交

---

## 阶段 1：召回器（Retriever）

引擎：sentence-transformers（稳健，原生支持 Qwen3-Embedding 的 last-token pooling + query/document 指令）

### 1.0 模型路线调整：0.6B 作为 baseline，最终主线升级到 30B 内最优组合

**背景**：用户指出 `Qwen3-Embedding-0.6B` 对简历主项目偏小。经重新评估，决定保留 0.6B 作为快速 baseline 和 HF Spaces 轻量 demo，最终主线升级为 30B 内更强模型组合。

| 模块 | 快速 baseline | 最终主线 | 原因 |
|------|---------------|----------|------|
| 召回 Retriever | Qwen3-Embedding-0.6B | **Qwen3-Embedding-8B** | Qwen3 embedding 系列 8B 为 30B 内最强检索模型，dim=4096，长上下文 32K，instruction-aware |
| 粗排 Reranker | 无 / 0.6B | **Qwen3-Reranker-8B** | 官方 reranker 8B，专门为排序任务训练，适合 top-50→top-10 |
| 合成数据/CoT教师 | 早期候选：Qwen2.5-3B / R1-1.5B（已废弃） | **DeepSeek-R1-Distill-Qwen-32B 离线 teacher/judge + 14B student** | 32B 提升合成/质检质量，14B 负责线上推理和蒸馏目标 |
| 精排/GRPO | 早期候选：Qwen2.5-3B（已废弃） | **R1-Distill-Qwen-14B LoRA/GRPO** | 让精排模型具备显式数学推理能力，更贴合错因诊断 |

**最终主线说明**：
- 0.6B 实验已经证明工程链路有效：LoRA 56 秒即可让 MAP@25 +55.6%
- 之后用 8B/14B 复跑时，重点展示“从快速 baseline → 强模型主线”的工程迭代能力
- 所有主模型均 ≤30B，完全命中 JD 中“30B 内大模型 SFT/RL 微调”要求
- 96GB 显存足够支撑：8B embedding LoRA、8B reranker LoRA、14B teacher 推理/LoRA；在线服务采用懒加载/分阶段加载，避免显存长期堆满

### 零样本基线（5折平均）

| 指标 | 值 |
|------|-----|
| MAP@25 | 0.2053 |
| Recall@25 | 0.6087 |
| Recall@10 | 0.4499 |
| nDCG@25 | 0.2939 |

各折：Fold0 MAP@25=0.2038 / F1=0.1958 / F2=0.2154 / F3=0.2143 / F4=0.1972

### LoRA 微调结果（Fold 0，MultipleNegativesRankingLoss = in-batch InfoNCE）

| 指标 | 零样本 | LoRA微调 | 绝对增益 | 相对增益 |
|------|--------|---------|---------|---------|
| **MAP@25** | 0.2038 | **0.3172** | +0.1134 | **+55.6%** |
| Recall@25 | 0.5984 | **0.8146** | +0.2162 | +36.1% |
| Recall@10 | 0.4439 | **0.6407** | +0.1968 | +44.3% |
| nDCG@25 | 0.2907 | **0.4280** | +0.1373 | +47.2% |

**超参**：LoRA r=16/α=32，target=q/k/v/o_proj，lr=2e-4，2 epochs，batch=32，bf16
**训练耗时**：仅 **56 秒**（0.6B + LoRA，96GB 显卡）；训练吞吐 125 样本/秒
**train_loss**：0.863 → 0.39（收敛良好）

**关键经验（面试）**：
- 召回是天花板：Recall@25 决定了重排能达到的上限。微调把 Recall@25 从 0.60→0.81，给后续重排留出充足空间
- MNRL（in-batch 对比学习）在 batch=32 时每个正例有 31 个 in-batch 负例，无需显式难负例即可大幅提升
- 0.6B 小模型 + LoRA 即可获得 +55% 增益，验证了"小模型+强工程"路线的有效性（契合效率赛道思路）

### 1.1 30B 内最终主线模型下载计划

用户进一步要求：时间不是问题，主项目不要显得模型过小，因此把 0.6B 降级为 baseline，最终主线改为 30B 内强模型组合。

| 模型 | 用途 | 参数量 | 预计权重体积 | 选择理由 |
|------|------|--------|--------------|----------|
| `Qwen/Qwen3-Embedding-8B` | 最终召回 | 8B | ~15-17GB | Qwen3 embedding 系列最大版本，dim=4096，官方检索强模型 |
| `Qwen/Qwen3-Reranker-8B` | 最终粗排 | 8B | ~15-17GB | 官方 reranker 最大版本，专门优化排序 |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | CoT/精排/GRPO主模型 | 14B | ~28-30GB | 30B 内数学推理性价比最高，MATH-500 pass@1 93.9 |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | **离线 teacher/judge/蒸馏增强项** | 32B | ~60-65GB | 只做高质量合成数据、LLM-as-judge、CoT蒸馏，不作为线上主模型 |

**总下载量预估**：约 120-130GB。当前 `/root/autodl-tmp` 剩余约 487GB，空间充足，暂不需要扩容。

**预计耗时**：
- hf-mirror 大文件速度波动明显，按 2-5MB/s + 重连估计：**8-18 小时**
- 使用脚本 `scripts/download_final_models.sh` 顺序下载、断点续传、最多20次自动重试
- 下载顺序：8B embedding → 8B reranker → R1-14B → R1-32B，先保证召回/重排主线可跑，32B 作为增强项放最后

**为什么加入 32B 但不把它做线上主模型**：
- 32B 对简历加分点是“teacher-student / judge / distillation”，不是线上推理
- 14B 已经可做主推理模型，32B 只负责提高合成数据和质检质量
- 这样既体现大模型工程能力，又保持 demo 可落地、可复现、可部署
- 72B 边际收益低：不在 30B 内，下载/推理/复现成本高，且仍不满足 100B+，不作为默认路线

### 1.2 资源与GPU使用策略（按用户要求：安全前提下榨干算力）

**数据盘**：
- 当前 `/root/autodl-tmp`：500GB，总使用约 14GB，剩余约 487GB
- 预计新增：8B embedding ~16GB + 8B reranker ~16GB + 14B ~30GB + 32B ~65GB ≈ 127GB
- 保留训练 checkpoint / LoRA / 合成数据后预计仍 < 250GB
- **扩容阈值**：如果 `/root/autodl-tmp` 可用空间低于 120GB 或总占用接近 350GB，提醒用户扩容

**GPU**：
- 训练阶段优先 bf16 + gradient checkpointing + LoRA；先用显存探测自动调大 batch
- 8B embedding/reranker LoRA：目标显存利用 70-85GB
- 14B listwise/GRPO：目标显存利用 80-90GB
- 32B teacher/judge 离线推理：vLLM `gpu_memory_utilization=0.90`，短上下文批量生成，逼近 90GB 但保留安全余量
- 出现 OOM 时优先降：batch_size → max_seq_len → 并发数；不牺牲模型主线

### 1.3 项目一致性同步检查（2026-06-02 14:20）

已同步以下文件，确保主线一致：

- `configs/base.yaml`：8B embedding / 8B reranker / 14B online reasoner-listwise / 32B offline teacher-judge。
- `README.md`：更新 JD 映射、训练命令、指标表说明。
- `RUNNING.md`：中文运行状态与后续执行计划。
- `docs/interview_guide.md`：面试稿从 0.6B/3B 路线升级为 8B/14B/32B 路线。
- `spaces/README.md`、`spaces/app.py`：公开 CPU demo 文案同步为轻量 demo + 完整 GPU 主线说明。
- `src/eedi/retriever/*`、`src/eedi/reranker/*`、`src/eedi/reasoner/*`、`src/eedi/synth/*`：注释和默认模型同步。
- `scripts/01_retriever_baseline.py`：8B LoRA adapter 保存到 `outputs/retriever/lora_best_8b`，避免和 0.6B baseline adapter 混淆。

当前下载状态：

- 正在下载：`Qwen/Qwen3-Embedding-8B`
- 已缓存：约 16GB
- 未完成块：2 个 incomplete，合计约 8.78GB
- 数据盘：500GB 总量，已用约 26GB，剩余约 475GB

### 1.4 关键节点：Qwen3-Embedding-8B 下载完成（2026-06-02 14:20:49）

**事件**：最终主线召回模型 `Qwen/Qwen3-Embedding-8B` 下载完成。

**结果**：

- 缓存目录：`/root/autodl-tmp/hf_cache/models--Qwen--Qwen3-Embedding-8B`
- 缓存大小：约 15GB
- incomplete 文件：0 个
- 下载脚本自动进入下一个模型：`Qwen/Qwen3-Reranker-8B`

**当前状态（2026-06-02 14:31）**：

| 项目 | 状态 |
|------|------|
| `Qwen3-Embedding-8B` | 已完成 |
| `Qwen3-Reranker-8B` | 正在下载，缓存约 13GB，5 个 incomplete，合计约 7.65GB |
| 数据盘 | 500GB，总用量约 37GB，剩余约 464GB |

**后续动作**：

1. 立即启动 `Qwen3-Embedding-8B` zero-shot 召回评测（可与 reranker 下载并行，因为下载不占 GPU）。
2. zero-shot 完成后更新 `RUNNING.md` / `note.md` / README 指标表。
3. 若 zero-shot 正常，再启动 8B LoRA 微调；训练时逐步调大 batch，安全前提下尽量吃满 GPU。

### 1.5 关键节点：Qwen3-Embedding-8B zero-shot 评测完成（2026-06-02 14:33）

**命令**：

```bash
python scripts/01_retriever_baseline.py --model Qwen/Qwen3-Embedding-8B --all-folds
```

**耗时**：约 78 秒（模型加载约 31 秒，5 折评测约 47 秒）

**5 折结果**：

| Fold | MAP@25 | Recall@25 | Recall@10 | nDCG@25 |
|------|--------|-----------|-----------|---------|
| 0 | 0.2371 | 0.6602 | 0.5229 | 0.3309 |
| 1 | 0.2079 | 0.6590 | 0.4817 | 0.3075 |
| 2 | 0.2312 | 0.6373 | 0.4828 | 0.3211 |
| 3 | 0.2299 | 0.6602 | 0.4977 | 0.3248 |
| 4 | 0.2179 | 0.6510 | 0.4748 | 0.3127 |
| **平均** | **0.2248** | **0.6535** | **0.4920** | **0.3194** |

**与 0.6B zero-shot 对比**：

| 模型 | MAP@25 | Recall@25 | Recall@10 | nDCG@25 |
|------|--------|-----------|-----------|---------|
| Qwen3-Embedding-0.6B zero-shot | 0.2053 | 0.6087 | 0.4499 | 0.2939 |
| Qwen3-Embedding-8B zero-shot | **0.2248** | **0.6535** | **0.4920** | **0.3194** |
| 增益 | +0.0195 (+9.5%) | +0.0448 (+7.4%) | +0.0421 (+9.4%) | +0.0255 (+8.7%) |

**结论**：

- 8B zero-shot 比 0.6B zero-shot 稳定提升，但仍低于 0.6B LoRA（Fold0 MAP@25=0.3172）。
- 说明本任务对 domain adaptation 非常敏感，**微调比单纯增大模型更关键**。
- 下一步：启动 8B LoRA + MNRL 微调。训练参数先用 batch=16、2 epochs、bf16、gradient checkpointing；若显存余量大，再提高 batch。

### 1.6 关键节点：Qwen3-Embedding-8B LoRA 微调完成（2026-06-02 14:41）

**命令**：

```bash
python scripts/01_retriever_baseline.py \
  --model Qwen/Qwen3-Embedding-8B \
  --fold 0 \
  --train \
  --epochs 2 \
  --batch-size 16
```

**训练配置**：

- LoRA：r=16，alpha=32，target=q/k/v/o_proj
- loss：MultipleNegativesRankingLoss（in-batch InfoNCE）
- dtype：bf16
- max_seq_length：512（避免默认 32K 上下文导致显存浪费）
- gradient checkpointing：已开启
- batch：16
- epochs：2

**训练耗时**：343 秒（约 5.7 分钟）

**训练曲线**：

| 位置 | loss |
|------|------|
| epoch 0.23 | 0.5795 |
| epoch 0.46 | 0.2786 |
| epoch 0.92 | 0.2343 |
| epoch 1.38 | 0.1558 |
| epoch 1.84 | 0.1977 |
| final train_loss | **0.2506** |

**Fold0 指标**：

| 模型 | MAP@25 | Recall@25 | Recall@10 | nDCG@25 |
|------|--------|-----------|-----------|---------|
| Qwen3-Embedding-0.6B zero-shot | 0.2038 | 0.5984 | 0.4439 | 0.2907 |
| Qwen3-Embedding-0.6B LoRA | 0.3172 | 0.8146 | 0.6407 | 0.4280 |
| Qwen3-Embedding-8B zero-shot | 0.2371 | 0.6602 | 0.5229 | 0.3309 |
| **Qwen3-Embedding-8B LoRA** | **0.4012** | **0.8822** | **0.7483** | **0.5102** |

**增益总结**：

- 8B LoRA 相比 8B zero-shot：MAP@25 +0.1641（+69.2%），Recall@25 +0.2220（+33.6%）
- 8B LoRA 相比 0.6B LoRA：MAP@25 +0.0840（+26.5%），Recall@25 +0.0676（+8.3%）
- 8B LoRA 相比 0.6B zero-shot：MAP@25 几乎翻倍（0.2038 → 0.4012）

**结论（面试重点）**：

- 模型规模有帮助，但必须结合任务域 LoRA 微调；8B zero-shot 只小幅优于 0.6B zero-shot，但 8B LoRA 显著领先。
- 召回器当前达到 Recall@25=0.8822，已经为后续 reranker 留出较高天花板。
- 下一步构建最终 FAISS 索引与 `candidate_pool.json`，供 8B reranker 和 14B/32B 阶段使用。

### 1.7 关键节点：最终 FAISS 索引与 candidate_pool 构建完成（2026-06-02 14:44）

**命令**：

```bash
python scripts/05_build_index.py \
  --model Qwen/Qwen3-Embedding-8B \
  --adapter-path outputs/retriever/lora_best_8b \
  --top-k 50
```

**耗时**：约 75 秒

**产物**：

- FAISS 索引：`/root/autodl-tmp/eedi-data/faiss_index.bin`
- misconception 向量：`outputs/retriever/misc_embs.npy`
- FAISS 行号映射：`outputs/retriever/misc_ids.json`
- 重排训练候选池：`outputs/retriever/candidate_pool.json`
- 候选分数：`outputs/retriever/candidate_scores.json`

**结果**：

| 指标 | 值 |
|------|-----|
| query 数 | 4370 |
| 每条 query 候选数 | top-50 |
| candidate_pool Recall@50 | **0.9700** |

**结论**：

- Recall@50=0.9700 意味着 97% 的样本金标错因已进入重排候选池。
- 这为后续 `Qwen3-Reranker-8B` 粗排和 `DeepSeek-R1-Distill-Qwen-14B` listwise 精排提供了很高上限。
- 阶段1（最终召回器）核心链路已完成：zero-shot → LoRA → FAISS → candidate_pool。

---

## 阶段 2：合成数据

（待填入）

---

## 阶段 3：重排级联

（待填入）

---

## 阶段 4：强化学习 GRPO

（待填入）

---

## 阶段 5：路由 + 中控 + Prompt + 记忆

（待填入）

---

## 阶段 6：服务化

（待填入）

---

## 阶段 7：部署

（待填入）

---

## 阶段 8：工程化与文档

（待填入）

---

## 踩坑汇总（面试重点）

| # | 问题 | 原因 | 解决 | 收益 |
|---|------|------|------|------|
| 1 | kaggle download 卡死 | GCS(storage.googleapis.com)被墙 | 改用 hf-mirror 的 cdtmc/eedi-ir | 数据2.5s就绪 |
| 2 | huggingface.co 超时 | 直连被墙 | export HF_ENDPOINT=hf-mirror.com | 模型可下载 |
| 3 | 模型下载卡在1.1G | 残留进程锁+连接停滞 | 杀进程清锁，hf download CLI 断点续传 | 下载完成 |
| 4 | bitsandbytes 加载失败 | 缺 libnvJitLink.so.13(CUDA13库,torch用cu128) | 不影响 bf16 LoRA（96GB无需量化），QLoRA 需要时再修复 | 不阻塞 |

### bitsandbytes 修复方案（如需 QLoRA 时）
```bash
# 方案1：装 CUDA13 的 nvjitlink
pip install nvidia-nvjitlink-cu13
# 方案2：软链到现有 cu12 版本
# 当前用 bf16 LoRA 无需 bnb，故暂不处理

---

## 指标汇总（消融）

| 阶段 | 方法 | Recall@25 | MAP@25 | nDCG@25 |
|------|------|-----------|--------|---------|
| 基线 | 零样本 | - | - | - |
| 阶段1 | LoRA微调召回 | - | - | - |
| 阶段2 | +合成数据 | - | - | - |
| 阶段3 | +重排级联 | - | - | - |
| 阶段4 | +GRPO | - | - | - |
