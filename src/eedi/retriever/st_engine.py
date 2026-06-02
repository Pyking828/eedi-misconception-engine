"""
基于 sentence-transformers 的召回引擎（生产主引擎，稳健）。

为什么用 sentence-transformers 而非手写 AutoModel：
- Qwen3-Embedding 需要正确的 last-token pooling + L2 归一化 + query/document 指令前缀
- ST 原生支持上述全部，且无缝集成 PEFT LoRA 训练
- 手写实现（src/eedi/retriever/retriever.py 的 InfoNCELoss/last_token_pool）保留作为
  "理解内部原理" 的教学参考

Qwen3-Embedding-0.6B：embed_dim=1024，prompts={'query','document'}
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def load_st_model(
    model_name: str,
    adapter_path: Optional[str | Path] = None,
    device: str = "cuda",
    cache_dir: str = "/root/autodl-tmp/hf_cache",
):
    """加载 SentenceTransformer（可选合并 LoRA adapter）。"""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        model_name,
        device=device,
        cache_folder=cache_dir,
        model_kwargs={"torch_dtype": "bfloat16", "attn_implementation": "sdpa"},
    )
    if adapter_path is not None and Path(adapter_path).exists():
        # ST 会把 transformer backbone 暴露在 model[0].auto_model
        from peft import PeftModel
        backbone = model[0].auto_model
        merged = PeftModel.from_pretrained(backbone, str(adapter_path)).merge_and_unload()
        model[0].auto_model = merged
    return model


def st_encode(
    model,
    texts: list[str],
    prompt_name: Optional[str] = None,
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """编码文本，返回 L2 归一化的 float32 矩阵。"""
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


def build_faiss_index(embeddings: np.ndarray, save_path: Optional[str | Path] = None) -> faiss.IndexFlatIP:
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype(np.float32))
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(save_path))
    return index


class STRetriever:
    """
    生产推理用召回器（service 层使用）。

    示例：
        r = STRetriever.from_pretrained("Qwen/Qwen3-Embedding-0.6B",
                index_path="...", misc_ids=[...], misc_texts={...},
                adapter_path="outputs/retriever/lora_best")
        ids, scores = r.retrieve("query text", top_k=50)
    """

    def __init__(self, model, index: faiss.IndexFlatIP, misc_ids: list[int], misc_texts: dict[int, str]) -> None:
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
        adapter_path: Optional[str | Path] = None,
        device: str = "cuda",
        cache_dir: str = "/root/autodl-tmp/hf_cache",
    ) -> "STRetriever":
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

    def batch_retrieve(self, queries: list[str], top_k: int = 50, batch_size: int = 64) -> list[list[int]]:
        embs = st_encode(self.model, queries, prompt_name="query", batch_size=batch_size, show_progress=True)
        scores, idxs = self.index.search(embs, top_k)
        return [[self.misc_ids[j] for j in row] for row in idxs]
