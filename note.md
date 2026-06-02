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

### 0.3 sm_120 Blackwell 兼容性冒烟测试

| 组件 | 状态 | 备注 |
|------|------|------|
| torch CUDA | - | - |
| torch SDPA (FlashAttention 替代) | - | - |
| faiss-cpu | - | - |
| vLLM | - | - |
| FlagEmbedding | - | - |
| peft/trl/accelerate | - | - |
| bitsandbytes | - | - |

> 填写：✅ 正常 / ⚠️ 降级 / ❌ 不可用 + 具体报错与解决

---

### 0.4 Kaggle 数据下载

- 需要用户提供：`kaggle.json`（API token）
- 需要接受比赛规则：https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/rules
- 下载命令：`kaggle competitions download -c eedi-mining-misconceptions-in-mathematics -p /root/autodl-tmp/eedi-data`

数据文件：
- `train.csv`：1869 道题
- `misconception_mapping.csv`：2587 条错因
- `test.csv` / `sample_submission.csv`

---

### 0.5 EDA 关键结论

（待填入）

- 训练集题目数：
- 错因总数：
- 训练集覆盖的错因数（有标注）：
- 未见错因数（仅在 misconception_mapping 中）：
- 每道题有 A/B/C/D 四个选项，一个正确答案，1-3 个 distractor
- 数据集长表展开后（每个 distractor 一行）：约 N 行

---

### 0.6 5 折 CV 构建方式

- 以 QuestionId 为单位分层，避免数据泄露
- 覆盖了训练集中所有 MisconceptionId（部分折可能未见）
- 折信息保存至：`data/folds.parquet`

---

### 0.7 磁盘使用记录

| 时间点 | /root 已用 | /root/autodl-tmp 已用 |
|--------|-----------|----------------------|
| 阶段0开始 | 441MB | ~72KB |
| （后续填入） | | |

---

## 阶段 1：召回器（Retriever）

（待填入）

### 基线结果

| 模型 | Recall@25 | MAP@25 | 耗时 |
|------|-----------|--------|------|
| Qwen3-Embedding-0.6B（零样本） | - | - | - |
| bge-m3（零样本） | - | - | - |

### 微调后结果

| 模型 | 方法 | Recall@25 | MAP@25 | 增益 | 耗时 |
|------|------|-----------|--------|------|------|
| Qwen3-Embedding-0.6B | LoRA+InfoNCE | - | - | - | - |

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
| 1 | | | | |

---

## 指标汇总（消融）

| 阶段 | 方法 | Recall@25 | MAP@25 | nDCG@25 |
|------|------|-----------|--------|---------|
| 基线 | 零样本 | - | - | - |
| 阶段1 | LoRA微调召回 | - | - | - |
| 阶段2 | +合成数据 | - | - | - |
| 阶段3 | +重排级联 | - | - | - |
| 阶段4 | +GRPO | - | - | - |
