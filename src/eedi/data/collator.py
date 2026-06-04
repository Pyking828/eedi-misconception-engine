"""Data collators for retriever and reranker training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class EmbedCollator:
    """
    Collator for bi-encoder (retriever) training.
    Packs query + positives + negatives into a flat batch
    that FlagEmbedding / sentence-transformers trainer expects.
    """

    tokenizer: PreTrainedTokenizerBase
    max_seq_len: int = 512
    query_prompt: str = (
        "Identify the mathematical misconception that best explains "
        "why a student gave this incorrect answer:\n"
    )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        queries = [self.query_prompt + f["query"] for f in features]
        positives = [f["pos"] for f in features]

        # Flatten all negatives
        all_texts = queries + positives
        for f in features:
            all_texts.extend(f.get("negs", []))

        enc = self.tokenizer(
            all_texts,
            max_length=self.max_seq_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        n_q = len(queries)
        n_p = len(positives)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "n_queries": n_q,
            "n_positives": n_p,
        }


@dataclass
class RerankCollator:
    """
    Collator for cross-encoder (pointwise reranker) training.
    Formats as: [CLS] query [SEP] candidate [SEP]
    """

    tokenizer: PreTrainedTokenizerBase
    max_seq_len: int = 1024

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        queries = [f["query"] for f in features]
        candidates = [f["candidate"] for f in features]
        labels = torch.tensor([f["label"] for f in features], dtype=torch.float32)

        enc = self.tokenizer(
            queries,
            candidates,
            max_length=self.max_seq_len,
            padding=True,
            truncation="only_second",
            return_tensors="pt",
        )
        return {**enc, "labels": labels}
