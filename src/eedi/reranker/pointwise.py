"""
Pointwise cross-encoder reranker.

Model: Qwen3-Reranker-8B (main) / 0.6B (fast baseline)
Input: (query, candidate_misconception) → scalar score
Training: LoRA + BCE / MSE
Inference: score top-50 from retrieval → take top-10
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedModel


class PointwiseScorer(nn.Module):
    """Scalar score head on last token hidden state (MilchstraB/Eedi style)."""

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
        # Last valid token hidden state
        seq_lens = attention_mask.sum(dim=1) - 1
        last_hidden = out.last_hidden_state[
            torch.arange(input_ids.size(0), device=input_ids.device), seq_lens
        ]
        return self.score_head(last_hidden).squeeze(-1)  # (B,)


class PointwiseReranker:
    """
    Score each candidate; return ids sorted by score.

    Example:
        reranker = PointwiseReranker.from_pretrained(
            model_name="Qwen/Qwen3-Reranker-8B",
            adapter_path="outputs/reranker/pointwise/lora_best_8b",
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
        adapter_path: str | Path | None = None,
        max_seq_len: int = 1024,
        device: str = "cuda",
        cache_dir: str | None = None,
    ) -> PointwiseReranker:
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
        """Return top-k MisconceptionIds by descending score."""
        self.model.eval()
        queries = [query] * len(candidate_ids)
        candidates = [misc_texts.get(cid, "") for cid in candidate_ids]

        all_scores: list[float] = []
        for i in range(0, len(candidates), batch_size):
            q_batch = queries[i : i + batch_size]
            c_batch = candidates[i : i + batch_size]
            enc = self.tokenizer(
                q_batch,
                c_batch,
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
        Down-weight misconceptions seen in training (3rd-place unseen trick).
        Test set is more likely to contain unseen misconceptions.
        """
        return [
            s * (scale_seen if cid in seen_ids else 1.0) for cid, s in zip(candidate_ids, scores)
        ]


class LogitReranker:
    """Production pointwise reranker (yes/no last-token logit; scripts 08/11).

    Qwen3-Reranker uses CausalLM yes/no logits at the final position (not a score head).
    Matches training that reached fold0 MAP@25=0.5700.

    Example:
        rr = LogitReranker.from_pretrained(
            "Qwen/Qwen3-Reranker-8B",
            adapter_path="outputs/reranker/manual_lora/manual_lora_fold0_n31464_bs4_hn8_len768")
        ranked = rr.rerank(query, candidate_ids, misc_texts, top_k=10)
    """

    INSTRUCTION = (
        "Given a mathematics question, the correct answer, and a student's incorrect answer, "
        "judge whether the document is the misconception that best explains why the student made the error."
    )
    PREFIX = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
        'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, model, tokenizer, max_length: int = 768, device: str = "cuda") -> None:
        import torch

        self.model = model
        self.tok = tokenizer
        self.max_length = max_length
        self.device = device
        self.yes_id = tokenizer.convert_tokens_to_ids("yes")
        self.no_id = tokenizer.convert_tokens_to_ids("no")
        self._prefix = tokenizer.encode(self.PREFIX, add_special_tokens=False)
        self._suffix = tokenizer.encode(self.SUFFIX, add_special_tokens=False)
        self._torch = torch

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        adapter_path: str | Path | None = None,
        base_model_path: str | None = None,
        max_length: int = 768,
        device: str = "cuda",
        cache_dir: str | None = None,
    ) -> LogitReranker:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok_src = str(adapter_path) if adapter_path and Path(adapter_path).exists() else model_name
        tokenizer = AutoTokenizer.from_pretrained(tok_src, padding_side="left", cache_dir=cache_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path or model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            cache_dir=cache_dir,
        )
        if adapter_path is not None and Path(adapter_path).exists():
            model = PeftModel.from_pretrained(model, str(adapter_path)).merge_and_unload()
        model = model.to(device).eval()
        return cls(model, tokenizer, max_length, device)

    def _fmt(self, query: str, doc: str) -> str:
        return f"<Instruct>: {self.INSTRUCTION}\n<Query>: {query}\n<Document>: {doc}"

    @torch.no_grad()
    def rerank_with_scores(
        self,
        query: str,
        candidate_ids: list[int],
        misc_texts: dict[int, str],
        top_k: int = 10,
        batch_size: int = 16,
    ) -> tuple[list[int], list[float]]:
        texts = [self._fmt(query, misc_texts.get(cid, "")) for cid in candidate_ids]
        max_pair = self.max_length - len(self._prefix) - len(self._suffix)
        scores: list[float] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            t = self.tok(
                chunk,
                padding=False,
                truncation="longest_first",
                max_length=max_pair,
                add_special_tokens=False,
                return_attention_mask=False,
            )
            ids = [self._prefix + x + self._suffix for x in t["input_ids"]]
            enc = self.tok.pad({"input_ids": ids}, padding=True, return_tensors="pt").to(
                self.device
            )
            logits = self.model(**enc).logits[:, -1, :]
            pair = self._torch.stack([logits[:, self.no_id], logits[:, self.yes_id]], dim=1)
            probs = self._torch.softmax(pair, dim=1)[:, 1]
            scores.extend(float(x) for x in probs.cpu())
        ranked = sorted(zip(candidate_ids, scores), key=lambda x: -x[1])
        ids_sorted = [c for c, _ in ranked][:top_k]
        scores_sorted = [s for _, s in ranked][:top_k]
        return ids_sorted, scores_sorted

    def rerank(
        self,
        query: str,
        candidate_ids: list[int],
        misc_texts: dict[int, str],
        top_k: int = 10,
        batch_size: int = 16,
    ) -> list[int]:
        ids_sorted, _ = self.rerank_with_scores(query, candidate_ids, misc_texts, top_k, batch_size)
        return ids_sorted


class PointwiseTrainer:
    """LoRA fine-tune pointwise reranker."""

    def __init__(
        self,
        model_name: str,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        cache_dir: str | None = None,
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
        val_loader: DataLoader | None = None,
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
            for batch in tqdm(train_loader, desc=f"Pointwise Epoch {epoch + 1}"):
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
            print(f"  Pointwise Epoch {epoch + 1}: loss={avg_loss:.4f}")

        self.model.backbone.save_pretrained(str(self.output_dir / "lora_final"))
        self.tokenizer.save_pretrained(str(self.output_dir / "lora_final"))
        return history
