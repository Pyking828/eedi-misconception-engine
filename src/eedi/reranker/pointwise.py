"""
粗排器（Pointwise Cross-Encoder）。
模型：Qwen3-Reranker-0.6B / bge-reranker-v2-m3
输入：(query, candidate_misconception) → 相似度标量
训练：LoRA + BCE / MSE loss
推理：对召回的 top-50 逐一打分 → 取 top-10
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedModel
from peft import LoraConfig, TaskType, get_peft_model, PeftModel


class PointwiseScorer(nn.Module):
    """在 backbone 最后 token 上接一个标量打分头（复刻 MilchstraB/Eedi 方案）。"""

    def __init__(self, backbone: PreTrainedModel, hidden_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.score_head = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # 取最后有效 token 的 hidden state
        seq_lens = attention_mask.sum(dim=1) - 1
        last_hidden = out.last_hidden_state[
            torch.arange(input_ids.size(0), device=input_ids.device), seq_lens
        ]
        return self.score_head(last_hidden).squeeze(-1)  # (B,)


class PointwiseReranker:
    """
    推理封装：对 candidate list 逐一打分，返回排序后的 ids。

    示例：
        reranker = PointwiseReranker.from_pretrained(
            model_name="Qwen/Qwen3-Reranker-0.6B",
            adapter_path="outputs/reranker/pointwise/lora_best",
        )
        ranked_ids = reranker.rerank(query, candidate_ids, misc_texts, top_k=10)
    """

    def __init__(
        self,
        model: PointwiseScorer,
        tokenizer,
        max_seq_len: int = 1024,
        device: str = "cuda",
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        adapter_path: Optional[str | Path] = None,
        max_seq_len: int = 1024,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
    ) -> "PointwiseReranker":
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        backbone = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            cache_dir=cache_dir,
        )
        if adapter_path is not None and Path(adapter_path).exists():
            backbone = PeftModel.from_pretrained(backbone, str(adapter_path))
            backbone = backbone.merge_and_unload()
        hidden_size = backbone.config.hidden_size
        model = PointwiseScorer(backbone, hidden_size)
        return cls(model, tokenizer, max_seq_len, device)

    @torch.no_grad()
    def rerank(
        self,
        query: str,
        candidate_ids: list[int],
        misc_texts: dict[int, str],
        top_k: int = 10,
        batch_size: int = 16,
    ) -> list[int]:
        """返回按相似度降序排列的 top-k MisconceptionId。"""
        self.model.eval()
        queries = [query] * len(candidate_ids)
        candidates = [misc_texts.get(cid, "") for cid in candidate_ids]

        all_scores: list[float] = []
        for i in range(0, len(candidates), batch_size):
            q_batch = queries[i : i + batch_size]
            c_batch = candidates[i : i + batch_size]
            enc = self.tokenizer(
                q_batch, c_batch,
                max_length=self.max_seq_len,
                padding=True,
                truncation="only_second",
                return_tensors="pt",
            ).to(self.device)
            scores = self.model(**enc)
            all_scores.extend(scores.cpu().float().tolist())

        ranked = sorted(zip(candidate_ids, all_scores), key=lambda x: -x[1])
        return [cid for cid, _ in ranked[:top_k]]

    def score_unseen_adjustment(
        self,
        candidate_ids: list[int],
        scores: list[float],
        seen_ids: set[int],
        scale_seen: float = 0.4,
    ) -> list[float]:
        """
        复刻 3rd place 技巧：对训练集出现过的 misconception 打压分数。
        认为测试集更可能是未见错因。
        """
        return [s * (scale_seen if cid in seen_ids else 1.0) for cid, s in zip(candidate_ids, scores)]


class PointwiseTrainer:
    """LoRA 微调 pointwise reranker。"""

    def __init__(
        self,
        model_name: str,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        cache_dir: Optional[str] = None,
        output_dir: str = "outputs/reranker/pointwise",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        backbone = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            cache_dir=cache_dir,
        )
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        lora_backbone = get_peft_model(backbone, lora_cfg)
        hidden_size = backbone.config.hidden_size
        self.model = PointwiseScorer(lora_backbone, hidden_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 3,
        lr: float = 1e-4,
    ) -> list[dict]:
        from torch.optim import AdamW

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        criterion = nn.BCEWithLogitsLoss()
        history = []

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Pointwise Epoch {epoch+1}"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                scores = self.model(input_ids, attention_mask)
                loss = criterion(scores, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            record = {"epoch": epoch + 1, "loss": avg_loss}
            history.append(record)
            print(f"  Pointwise Epoch {epoch+1}: loss={avg_loss:.4f}")

        self.model.backbone.save_pretrained(str(self.output_dir / "lora_final"))
        self.tokenizer.save_pretrained(str(self.output_dir / "lora_final"))
        return history
