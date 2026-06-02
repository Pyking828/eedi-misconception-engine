"""
召回器（Bi-Encoder）模块。
架构：Qwen3-Embedding-8B（最终主线）/ Qwen3-Embedding-0.6B（快速基线）+ LoRA + InfoNCE / MNRL
向量库：FAISS CPU IndexFlatIP（2587 条 misconception，L2 归一化后等价余弦）
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

# ────────────────────────────────────────────────────────────
# 1. 编码工具函数
# ────────────────────────────────────────────────────────────

QUERY_PROMPT = (
    "Identify the mathematical misconception that best explains "
    "why a student gave this incorrect answer:\n"
)


def last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Decoder-only 模型用最后一个有效 token 的隐层作为 embedding（同 FlagEmbedding 方案）。"""
    seq_lens = attention_mask.sum(dim=1) - 1
    batch_size = hidden_states.size(0)
    return hidden_states[torch.arange(batch_size, device=hidden_states.device), seq_lens]


@torch.no_grad()
def encode_texts(
    texts: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int = 512,
    batch_size: int = 64,
    device: str = "cuda",
    normalize: bool = True,
    show_progress: bool = False,
) -> np.ndarray:
    """批量编码文本，返回 (N, D) float32 numpy 数组。"""
    model.eval()
    all_embs: list[np.ndarray] = []
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Encoding")

    for start in iterator:
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            max_length=max_seq_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        out = model(**enc, output_hidden_states=False)
        # 优先使用 last_hidden_state
        if hasattr(out, "last_hidden_state"):
            emb = last_token_pool(out.last_hidden_state, enc["attention_mask"])
        else:
            emb = out[0][:, -1, :]
        if normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        all_embs.append(emb.cpu().float().numpy())

    return np.vstack(all_embs)


# ────────────────────────────────────────────────────────────
# 2. InfoNCE 损失
# ────────────────────────────────────────────────────────────

class InfoNCELoss(nn.Module):
    """
    In-batch 对比学习损失（温度缩放余弦相似度 cross-entropy）。
    等价于 MultipleNegativesRankingLoss，但温度可调。
    """

    def __init__(self, temperature: float = 0.02) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query_embs: torch.Tensor,   # (B, D)
        pos_embs: torch.Tensor,     # (B, D)
        neg_embs: Optional[torch.Tensor] = None,  # (B*K, D) 可选
    ) -> torch.Tensor:
        # 归一化
        q = F.normalize(query_embs, p=2, dim=-1)
        p = F.normalize(pos_embs, p=2, dim=-1)

        if neg_embs is not None:
            n = F.normalize(neg_embs, p=2, dim=-1)
            # candidates = [正例, 负例...]
            candidates = torch.cat([p, n], dim=0)  # (B + B*K, D)
        else:
            candidates = p  # 仅 in-batch negative

        logits = torch.matmul(q, candidates.T) / self.temperature  # (B, B or B+B*K)
        labels = torch.arange(len(q), device=q.device)
        return F.cross_entropy(logits, labels)


# ────────────────────────────────────────────────────────────
# 3. FAISS 索引构建
# ────────────────────────────────────────────────────────────

def build_faiss_index(
    embeddings: np.ndarray,
    save_path: Optional[str | Path] = None,
) -> faiss.IndexFlatIP:
    """
    构建 L2 归一化后的内积索引（等价余弦相似度）。
    embeddings 应已 L2 归一化。
    """
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype(np.float32))
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(save_path))
    return index


def load_faiss_index(path: str | Path) -> faiss.IndexFlatIP:
    return faiss.read_index(str(path))


# ────────────────────────────────────────────────────────────
# 4. EediRetriever：推理封装
# ────────────────────────────────────────────────────────────

class EediRetriever:
    """
    推理时使用：给定 query 文本，返回 top-k MisconceptionId 列表。

    示例：
        retriever = EediRetriever.from_pretrained(
            model_name="Qwen/Qwen3-Embedding-8B",
            adapter_path="outputs/retriever/lora",
            index_path="data/faiss_index.bin",
            misc_ids=[0, 1, 2, ...],
        )
        results = retriever.retrieve(query_text, top_k=50)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        index: faiss.IndexFlatIP,
        misc_ids: list[int],
        misc_texts: dict[int, str],
        max_seq_len: int = 512,
        device: str = "cuda",
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.index = index
        self.misc_ids = misc_ids          # FAISS 行 i → MisconceptionId
        self.misc_texts = misc_texts      # MisconceptionId → text
        self.max_seq_len = max_seq_len
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        index_path: str | Path,
        misc_ids: list[int],
        misc_texts: dict[int, str],
        adapter_path: Optional[str | Path] = None,
        max_seq_len: int = 512,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
    ) -> "EediRetriever":
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",  # Blackwell 安全方案
            cache_dir=cache_dir,
        )
        if adapter_path is not None and Path(adapter_path).exists():
            model = PeftModel.from_pretrained(model, str(adapter_path))
            model = model.merge_and_unload()
        index = load_faiss_index(index_path)
        return cls(model, tokenizer, index, misc_ids, misc_texts, max_seq_len, device)

    def retrieve(
        self,
        query: str,
        top_k: int = 50,
        with_scores: bool = False,
    ) -> list[int] | tuple[list[int], list[float]]:
        """返回 top-k MisconceptionId（可含分数）。"""
        query_text = QUERY_PROMPT + query
        emb = encode_texts(
            [query_text],
            self.model,
            self.tokenizer,
            max_seq_len=self.max_seq_len,
            batch_size=1,
            device=self.device,
        )  # (1, D)
        D, I = self.index.search(emb, top_k)
        ids = [self.misc_ids[i] for i in I[0]]
        if with_scores:
            return ids, D[0].tolist()
        return ids

    def batch_retrieve(
        self,
        queries: list[str],
        top_k: int = 50,
        batch_size: int = 32,
    ) -> list[list[int]]:
        """批量召回，返回每条 query 的 top-k id 列表。"""
        query_texts = [QUERY_PROMPT + q for q in queries]
        embs = encode_texts(
            query_texts,
            self.model,
            self.tokenizer,
            max_seq_len=self.max_seq_len,
            batch_size=batch_size,
            device=self.device,
            show_progress=True,
        )
        D, I = self.index.search(embs, top_k)
        return [[self.misc_ids[j] for j in row] for row in I]


# ────────────────────────────────────────────────────────────
# 5. RetrieverTrainer：LoRA 微调
# ────────────────────────────────────────────────────────────

class RetrieverTrainer:
    """
    LoRA + InfoNCE 微调召回器。

    特点：
    - 用 last-token pooling（decoder-only friendly）
    - InfoNCE loss，温度 0.02（复刻第1名 & top-5 方案）
    - 难负例挖掘在 EediDataset 层做，此处接收已挖掘的 DataLoader
    - 保存 LoRA adapter（非完整权重，节省磁盘）
    """

    def __init__(
        self,
        model_name: str,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        temperature: float = 0.02,
        cache_dir: Optional[str] = None,
        output_dir: str = "outputs/retriever",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temperature = temperature

        # 加载 base model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            cache_dir=cache_dir,
        )

        # 注入 LoRA
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        self.model = get_peft_model(base_model, lora_cfg)
        self.model.print_trainable_parameters()

        self.loss_fn = InfoNCELoss(temperature=temperature)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 3,
        lr: float = 2e-4,
        warmup_steps: int = 100,
        eval_every_n_steps: int = 200,
        misc_embeddings: Optional[np.ndarray] = None,
        misc_ids: Optional[list[int]] = None,
    ) -> list[dict]:
        """训练并返回每个 epoch 的 loss / 指标历史。"""
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        total_steps = len(train_loader) * num_epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.1)

        history = []
        global_step = 0
        best_map = 0.0

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            t0 = time.time()

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                # batch 结构由 EmbedCollator 决定
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                n_q = batch["n_queries"]
                n_p = batch["n_positives"]

                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                embs = last_token_pool(out.last_hidden_state, attention_mask)

                q_embs = embs[:n_q]
                p_embs = embs[n_q : n_q + n_p]
                neg_embs = embs[n_q + n_p :] if embs.size(0) > n_q + n_p else None

                loss = self.loss_fn(q_embs, p_embs, neg_embs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                global_step += 1

            avg_loss = epoch_loss / len(train_loader)
            elapsed = time.time() - t0
            record = {"epoch": epoch + 1, "loss": avg_loss, "elapsed_s": elapsed}

            # 可选：每 epoch 末做 FAISS 检索评测
            if val_loader is not None and misc_embeddings is not None:
                map25 = self._quick_eval(val_loader, misc_embeddings, misc_ids or [])
                record["MAP@25"] = map25
                if map25 > best_map:
                    best_map = map25
                    self.save_adapter("best")
                print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, MAP@25={map25:.4f}, time={elapsed:.0f}s")
            else:
                print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, time={elapsed:.0f}s")

            history.append(record)

        self.save_adapter("final")
        return history

    @torch.no_grad()
    def _quick_eval(
        self,
        val_loader: DataLoader,
        misc_embeddings: np.ndarray,
        misc_ids: list[int],
        top_k: int = 25,
    ) -> float:
        from eval.evaluator import EediEvaluator

        self.model.eval()
        index = build_faiss_index(misc_embeddings)
        evaluator = EediEvaluator(k=top_k)

        for batch in tqdm(val_loader, desc="Eval", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            q_embs = last_token_pool(out.last_hidden_state, attention_mask)
            q_embs = F.normalize(q_embs, p=2, dim=-1).cpu().float().numpy()

            D, I = index.search(q_embs, top_k)
            pos_ids = batch.get("pos_ids", [])
            for i, pos_id in enumerate(pos_ids):
                predicted = [misc_ids[j] for j in I[i]]
                evaluator.update(predicted, int(pos_id))

        metrics = evaluator.compute()
        self.model.train()
        return metrics.get(f"MAP@{top_k}", 0.0)

    def save_adapter(self, suffix: str = "final") -> None:
        save_path = self.output_dir / f"lora_{suffix}"
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        print(f"[Retriever] LoRA adapter saved → {save_path}")
