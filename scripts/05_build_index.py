"""
构建 FAISS 索引并保存候选池（供后续重排使用）。
运行：python scripts/05_build_index.py [--adapter-path path/to/lora]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from rich.console import Console

from src.eedi.retriever.retriever import EediRetriever, build_faiss_index, encode_texts, QUERY_PROMPT
from eval.evaluator import EediEvaluator

console = Console()
DATA_DIR = Path(os.environ.get("EEDI_DATA", "/root/autodl-tmp/eedi-data"))
HF_CACHE = os.environ.get("HF_HOME", "/root/autodl-tmp/hf_cache")
OUTPUT_DIR = Path("/root/autodl-tmp/eedi-misconception-engine/outputs/retriever")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--adapter-path", default=None, help="LoRA adapter 路径")
    parser.add_argument("--top-k", type=int, default=50, help="每条 query 保存多少候选")
    args = parser.parse_args()

    console.rule("[bold blue]构建 FAISS 索引 & 候选池")

    from transformers import AutoModel, AutoTokenizer
    from peft import PeftModel

    misc_df = pl.read_csv(str(DATA_DIR / "misconception_mapping.csv"))
    misc_texts = {int(r["MisconceptionId"]): r["MisconceptionName"] for r in misc_df.iter_rows(named=True)}
    misc_ids = list(misc_texts.keys())
    misc_text_list = [misc_texts[mid] for mid in misc_ids]

    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=HF_CACHE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", cache_dir=HF_CACHE
    )
    if args.adapter_path and Path(args.adapter_path).exists():
        console.print(f"[cyan]加载 LoRA adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()
    model = model.cuda()

    # 编码 misconception 库
    console.print(f"[cyan]编码 {len(misc_ids)} 条 misconception...")
    misc_embs = encode_texts(misc_text_list, model, tokenizer, max_seq_len=256, batch_size=128, device="cuda", show_progress=True)

    # 构建并保存 FAISS 索引
    index_path = DATA_DIR / "faiss_index.bin"
    build_faiss_index(misc_embs, save_path=str(index_path))
    console.print(f"[green]✓ FAISS 索引保存: {index_path}")

    # 保存 misc_embs（用于训练时快速 eval）
    np.save(str(OUTPUT_DIR / "misc_embs.npy"), misc_embs)

    # 构建候选池（对训练集所有 query 做检索，供重排训练使用）
    long_df = pl.read_parquet(str(DATA_DIR / "folds.parquet"))
    queries_info = [
        (row["QuestionId_Answer"], QUERY_PROMPT + row["AllText"])
        for row in long_df.filter(pl.col("MisconceptionId") >= 0).iter_rows(named=True)
    ]
    qa_keys = [info[0] for info in queries_info]
    query_texts = [info[1] for info in queries_info]

    console.print(f"[cyan]编码 {len(query_texts)} 条 query 并检索候选...")
    import faiss
    index = faiss.read_index(str(index_path))
    query_embs = encode_texts(query_texts, model, tokenizer, max_seq_len=512, batch_size=32, device="cuda", show_progress=True)
    D, I = index.search(query_embs, args.top_k)

    candidate_pool = {
        qa_key: [misc_ids[j] for j in I[i]]
        for i, qa_key in enumerate(qa_keys)
    }
    with open(OUTPUT_DIR / "candidate_pool.json", "w") as f:
        json.dump(candidate_pool, f)
    console.print(f"[green]✓ 候选池保存: {OUTPUT_DIR}/candidate_pool.json  ({len(candidate_pool)} 条 query)")

    # 快速评测一下召回效果
    console.print("\n[cyan]快速评测召回效果（全部样本）...")
    evaluator = EediEvaluator(k=25)
    true_ids = long_df.filter(pl.col("MisconceptionId") >= 0)["MisconceptionId"].to_list()
    for i, (qa_key, true_id) in enumerate(zip(qa_keys, true_ids)):
        evaluator.update([misc_ids[j] for j in I[i]], int(true_id))
    metrics = evaluator.compute()
    console.print(f"[green]训练集全量 MAP@25={metrics['MAP@25']:.4f}  Recall@25={metrics['Recall@25']:.4f}")

    console.rule("[bold green]索引构建完成")


if __name__ == "__main__":
    main()
