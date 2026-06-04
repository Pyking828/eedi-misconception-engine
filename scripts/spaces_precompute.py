"""Precompute bge-m3 embeddings of the 2587 misconceptions for the HF Spaces CPU demo.

Shipping a small .npy (2587 x 1024 f32 ~10MB) makes the free-tier Space cold-start
instant and robust (no 1-2 min encode of the bank on first request). The Space then
only encodes the single user query at runtime.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")

import numpy as np
from sentence_transformers import SentenceTransformer

SP = Path("/root/autodl-tmp/eedi-misconception-engine/spaces")
texts = SP.joinpath("misconceptions.txt").read_text().strip().split("\n")
print(f"misconceptions: {len(texts)}")

model = SentenceTransformer("BAAI/bge-m3", device="cuda", cache_folder=os.environ["HF_HOME"])
embs = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True).astype(
    np.float32
)
np.save(SP / "misc_embs_bge_m3.npy", embs)
print(f"saved {SP / 'misc_embs_bge_m3.npy'} shape={embs.shape}")
