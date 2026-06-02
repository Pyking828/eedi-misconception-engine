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
| vLLM | 安装中 | ⏳ | 阶段2合成数据用 |

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
