"""
阶段1：召回器基线评测 + LoRA 微调
运行：python scripts/01_retriever_baseline.py [--fold 0] [--train]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from rich.console import Console
from torch.utils.data import DataLoader

from src.eedi.data.dataset import EediDataset, load_folds
from src.eedi.data.collator import EmbedCollator
from src.eedi.retriever.retriever import (
    EediRetriever, RetrieverTrainer, build_faiss_index, encode_texts, QUERY_PROMPT
)
from eval.evaluator import EediEvaluator, evaluate_pipeline

console = Console()

DATA_DIR = Path(os.environ.get("EEDI_DATA", "/root/autodl-tmp/eedi-data"))
MODELS_DIR = Path(os.environ.get("EEDI_MODELS", "/root/autodl-tmp/models"))
HF_CACHE = os.environ.get("HF_HOME", "/root/autodl-tmp/hf_cache")
OUTPUT_DIR = Path("/root/autodl-tmp/eedi-misconception-engine/outputs/retriever")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_zero_shot_baseline(model_name: str, fold: int = 0) -> dict:
    """零样本基线：直接用预训练 embedding 做 FAISS 检索。"""
    from transformers import AutoModel, AutoTokenizer

    console.print(f"\n[cyan]零样本基线：{model_name}  Fold={fold}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa", cache_dir=HF_CACHE
    ).cuda()

    long_df = pl.read_parquet(str(DATA_DIR / "folds.parquet"))
    misc_df = pl.read_csv(str(DATA_DIR / "misconception_mapping.csv"))
    misc_texts = {
        int(r["MisconceptionId"]): r["MisconceptionName"]
        for r in misc_df.iter_rows(named=True)
    }
    misc_ids = list(misc_texts.keys())
    misc_text_list = [misc_texts[mid] for mid in misc_ids]

    # 编码所有 misconception
    console.print(f"  编码 {len(misc_ids)} 条 misconception...")
    t0 = time.time()
    misc_embs = encode_texts(
        misc_text_list, model, tokenizer, max_seq_len=256, batch_size=128,
        device="cuda", show_progress=True
    )
    console.print(f"  编码耗时: {time.time()-t0:.1f}s  显存: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    index = build_faiss_index(misc_embs)

    # 验证集
    val_df = long_df.filter(pl.col("fold") == fold).filter(pl.col("MisconceptionId") >= 0)
    queries = [QUERY_PROMPT + row["AllText"] for row in val_df.iter_rows(named=True)]
    true_ids = val_df["MisconceptionId"].to_list()

    console.print(f"  评测 {len(queries)} 条验证样本...")
    query_embs = encode_texts(queries, model, tokenizer, max_seq_len=512, batch_size=32, device="cuda")
    D, I = index.search(query_embs, 25)

    evaluator = EediEvaluator(k=25)
    for i, true_id in enumerate(true_ids):
        predicted = [misc_ids[j] for j in I[i]]
        evaluator.update(predicted, int(true_id))

    metrics = evaluator.compute()
    console.print(f"  [green]MAP@25={metrics['MAP@25']:.4f}  Recall@25={metrics['Recall@25']:.4f}")

    # 释放显存
    del model
    torch.cuda.empty_cache()
    return {**metrics, "model": model_name, "fold": fold, "mode": "zero-shot"}


def run_lora_training(
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    fold: int = 0,
    num_epochs: int = 3,
    batch_size: int = 8,
) -> dict:
    """LoRA + InfoNCE 微调召回器。"""
    console.print(f"\n[cyan]LoRA 微调：{model_name}  Fold={fold}")
    long_df = pl.read_parquet(str(DATA_DIR / "folds.parquet"))
    misc_df = pl.read_csv(str(DATA_DIR / "misconception_mapping.csv"))

    train_dataset = EediDataset(long_df, misc_df, mode="retriever", fold=fold, split="train")
    val_dataset = EediDataset(long_df, misc_df, mode="retriever", fold=fold, split="val")
    console.print(f"  训练集: {len(train_dataset)}  验证集: {len(val_dataset)}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    collator = EmbedCollator(tokenizer=tokenizer, max_seq_len=512)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    trainer = RetrieverTrainer(
        model_name=model_name,
        lora_r=16,
        lora_alpha=32,
        temperature=0.02,
        cache_dir=HF_CACHE,
        output_dir=str(OUTPUT_DIR),
    )

    # 预先编码 misconception 库（供 eval 用）
    misc_texts = {int(r["MisconceptionId"]): r["MisconceptionName"] for r in misc_df.iter_rows(named=True)}
    misc_ids = list(misc_texts.keys())
    misc_text_list = [misc_texts[mid] for mid in misc_ids]
    misc_embs = encode_texts(
        misc_text_list, trainer.model.base_model if hasattr(trainer.model, "base_model") else trainer.model,
        trainer.tokenizer, max_seq_len=256, batch_size=128, device="cuda"
    )

    history = trainer.train(
        train_loader, val_loader,
        num_epochs=num_epochs,
        lr=2e-4,
        misc_embeddings=misc_embs,
        misc_ids=misc_ids,
    )
    return {"history": history, "model": model_name, "fold": fold, "mode": "lora"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--train", action="store_true", help="运行 LoRA 微调（否则只跑基线）")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    console.rule("[bold blue]阶段1：召回器基线 & 微调")

    results = {}

    # 零样本基线
    baseline = run_zero_shot_baseline(args.model, fold=args.fold)
    results["baseline"] = baseline

    if args.train:
        lora_result = run_lora_training(args.model, fold=args.fold, num_epochs=args.epochs)
        results["lora"] = lora_result

    # 保存结果
    result_path = OUTPUT_DIR / "results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[green]结果已保存: {result_path}")


if __name__ == "__main__":
    main()
