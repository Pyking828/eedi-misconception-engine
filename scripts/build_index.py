"""
Build FAISS index + candidate pool (sentence-transformers).

Candidate pool feeds stage 3/4 rerankers and GRPO training.

Usage:
  python scripts/build_index.py --model Qwen/Qwen3-Embedding-8B --adapter-path outputs/retriever/lora_best_8b --top-k 50
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from eval.evaluator import EediEvaluator
from rich.console import Console
from src.eedi.retriever.st_engine import build_faiss_index, load_st_model, st_encode

console = Console()
DATA_DIR = Path(os.environ.get("EEDI_DATA", "/root/autodl-tmp/eedi-data"))
HF_CACHE = os.environ.get("HF_HOME", "/root/autodl-tmp/hf_cache")
OUTPUT_DIR = Path("/root/autodl-tmp/eedi-misconception-engine/outputs/retriever")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--adapter-path", default=str(OUTPUT_DIR / "lora_best_8b"))
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    console.rule("[bold blue]构建 FAISS 索引 + 候选池")
    adapter = args.adapter_path if Path(args.adapter_path).exists() else None
    console.print(f"[cyan]模型: {args.model}  adapter: {adapter or '无(零样本)'}")

    model = load_st_model(args.model, adapter_path=adapter, cache_dir=HF_CACHE)

    misc_df = pl.read_csv(str(DATA_DIR / "misconception_mapping.csv"))
    misc_texts = {
        int(r["MisconceptionId"]): r["MisconceptionName"] for r in misc_df.iter_rows(named=True)
    }
    misc_ids = sorted(misc_texts.keys())
    misc_text_list = [misc_texts[mid] for mid in misc_ids]

    # Encode misconception bank + build index
    console.print(f"[cyan]编码 {len(misc_ids)} 条 misconception...")
    misc_embs = st_encode(
        model, misc_text_list, prompt_name="document", batch_size=128, show_progress=True
    )
    index_path = DATA_DIR / "faiss_index.bin"
    build_faiss_index(misc_embs, save_path=str(index_path))
    np.save(str(OUTPUT_DIR / "misc_embs.npy"), misc_embs)
    # Save misc_ids order (FAISS row → MisconceptionId)
    with open(OUTPUT_DIR / "misc_ids.json", "w") as f:
        json.dump(misc_ids, f)
    console.print(f"[green]✓ FAISS 索引: {index_path}")

    # Retrieve all queries → candidate pool
    long_df = pl.read_parquet(str(DATA_DIR / "folds.parquet"))
    labeled = long_df.filter(pl.col("MisconceptionId") >= 0)
    qa_keys = labeled["QuestionId_Answer"].to_list()
    query_texts = labeled["AllText"].to_list()
    true_ids = labeled["MisconceptionId"].to_list()

    console.print(f"[cyan]编码 {len(query_texts)} 条 query 并召回 top-{args.top_k}...")
    import faiss

    index = faiss.read_index(str(index_path))
    q_embs = st_encode(model, query_texts, prompt_name="query", batch_size=64, show_progress=True)
    scores, idxs = index.search(q_embs, args.top_k)

    candidate_pool = {qa: [misc_ids[j] for j in idxs[i]] for i, qa in enumerate(qa_keys)}
    candidate_scores = {qa: scores[i].tolist() for i, qa in enumerate(qa_keys)}
    with open(OUTPUT_DIR / "candidate_pool.json", "w") as f:
        json.dump(candidate_pool, f)
    with open(OUTPUT_DIR / "candidate_scores.json", "w") as f:
        json.dump(candidate_scores, f)
    console.print(f"[green]✓ 候选池: {len(candidate_pool)} 条 query × top-{args.top_k}")

    # Eval pool recall (with seen/unseen split)
    seen_ids = set(json.load(open(DATA_DIR / "seen_misc_ids.json")))
    ev_all = EediEvaluator(k=args.top_k)
    ev_seen = EediEvaluator(k=args.top_k)
    ev_unseen = EediEvaluator(k=args.top_k)
    for i, (qa, tid) in enumerate(zip(qa_keys, true_ids)):
        pred = [misc_ids[j] for j in idxs[i]]
        ev_all.update(pred, int(tid))
        (ev_seen if tid in seen_ids else ev_unseen).update(pred, int(tid))
    console.print(
        f"[bold]候选池 Recall@{args.top_k}: 全部={ev_all.compute()[f'Recall@{args.top_k}']:.4f} "
        f"已见={ev_seen.compute().get(f'Recall@{args.top_k}', 0):.4f} "
        f"未见={ev_unseen.compute().get(f'Recall@{args.top_k}', 0):.4f}"
    )

    console.rule("[bold green]索引构建完成")


if __name__ == "__main__":
    main()
