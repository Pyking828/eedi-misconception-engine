"""
Sentence-transformers retrieval engine (production default).

Why ST instead of raw AutoModel:
- Qwen3-Embedding needs last-token pooling, L2 norm, and query/document prompts
- ST supports all of the above and PEFT LoRA training
- Hand-rolled code in retriever.py remains as a teaching reference

Main: Qwen3-Embedding-8B (embed_dim=4096, prompts query/document)
Baseline: Qwen3-Embedding-0.6B (embed_dim=1024)
"""

from __future__ import annotations

import os
from pathlib import Path

import faiss
import numpy as np

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def load_st_model(
    model_name: str,
    adapter_path: str | Path | None = None,
    device: str = "cuda",
    cache_dir: str = "/root/autodl-tmp/hf_cache",
):
    """Load SentenceTransformer (optionally merge LoRA adapter)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        model_name,
        device=device,
        cache_folder=cache_dir,
        model_kwargs={"torch_dtype": "bfloat16", "attn_implementation": "sdpa"},
    )
    if adapter_path is not None and Path(adapter_path).exists():
        # ST exposes backbone at model[0].auto_model
        from peft import PeftModel

        backbone = model[0].auto_model
        merged = PeftModel.from_pretrained(backbone, str(adapter_path)).merge_and_unload()
        model[0].auto_model = merged
    return model


def st_encode(
    model,
    texts: list[str],
    prompt_name: str | None = None,
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """Encode texts; return L2-normalized float32 matrix."""
    prompts = getattr(model, "prompts", {}) or {}
    use_prompt = prompt_name if prompt_name in prompts else None
    embs = model.encode(
        texts,
        prompt_name=use_prompt,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return embs.astype(np.float32)


def build_faiss_index(
    embeddings: np.ndarray, save_path: str | Path | None = None
) -> faiss.IndexFlatIP:
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype(np.float32))
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(save_path))
    return index


class STRetriever:
    """
    Production retriever used by the service layer.

    Example:
        r = STRetriever.from_pretrained("Qwen/Qwen3-Embedding-8B",
                index_path="...", misc_ids=[...], misc_texts={...},
                adapter_path="outputs/retriever/lora_best_8b")
        ids, scores = r.retrieve("query text", top_k=50)
    """

    def __init__(
        self, model, index: faiss.IndexFlatIP, misc_ids: list[int], misc_texts: dict[int, str]
    ) -> None:
        self.model = model
        self.index = index
        self.misc_ids = misc_ids
        self.misc_texts = misc_texts

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        index_path: str | Path,
        misc_ids: list[int],
        misc_texts: dict[int, str],
        adapter_path: str | Path | None = None,
        device: str = "cuda",
        cache_dir: str = "/root/autodl-tmp/hf_cache",
    ) -> STRetriever:
        model = load_st_model(model_name, adapter_path, device, cache_dir)
        index = faiss.read_index(str(index_path))
        return cls(model, index, misc_ids, misc_texts)

    def retrieve(self, query: str, top_k: int = 50, with_scores: bool = False):
        emb = st_encode(self.model, [query], prompt_name="query", batch_size=1)
        scores, idxs = self.index.search(emb, top_k)
        ids = [self.misc_ids[i] for i in idxs[0]]
        if with_scores:
            return ids, scores[0].tolist()
        return ids

    def batch_retrieve(
        self, queries: list[str], top_k: int = 50, batch_size: int = 64
    ) -> list[list[int]]:
        embs = st_encode(
            self.model, queries, prompt_name="query", batch_size=batch_size, show_progress=True
        )
        scores, idxs = self.index.search(embs, top_k)
        return [[self.misc_ids[j] for j in row] for row in idxs]
