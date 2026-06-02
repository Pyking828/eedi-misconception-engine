"""
阶段1：召回器零样本基线 + LoRA 微调（sentence-transformers 引擎）

用法：
  python scripts/01_retriever_baseline.py --fold 0 --model Qwen/Qwen3-Embedding-8B         # 最终主线零样本
  python scripts/01_retriever_baseline.py --fold 0 --train --model Qwen/Qwen3-Embedding-8B # 最终主线 LoRA
  python scripts/01_retriever_baseline.py --all-folds              # 全5折零样本
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from rich.console import Console

from src.eedi.retriever.st_engine import load_st_model, st_encode, build_faiss_index
from eval.evaluator import EediEvaluator

console = Console()
DATA_DIR = Path(os.environ.get("EEDI_DATA", "/root/autodl-tmp/eedi-data"))
HF_CACHE = os.environ.get("HF_HOME", "/root/autodl-tmp/hf_cache")
OUTPUT_DIR = Path("/root/autodl-tmp/eedi-misconception-engine/outputs/retriever")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    long_df = pl.read_parquet(str(DATA_DIR / "folds.parquet"))
    misc_df = pl.read_csv(str(DATA_DIR / "misconception_mapping.csv"))
    misc_texts = {int(r["MisconceptionId"]): r["MisconceptionName"] for r in misc_df.iter_rows(named=True)}
    misc_ids = sorted(misc_texts.keys())
    return long_df, misc_texts, misc_ids


def evaluate_fold(model, long_df, misc_texts, misc_ids, fold, misc_embs=None, ks=(25, 10)):
    """对指定 fold 的验证集评测召回。"""
    misc_text_list = [misc_texts[mid] for mid in misc_ids]
    if misc_embs is None:
        misc_embs = st_encode(model, misc_text_list, prompt_name="document", batch_size=128, show_progress=True)
    index = build_faiss_index(misc_embs)

    val = long_df.filter(pl.col("fold") == fold).filter(pl.col("MisconceptionId") >= 0)
    queries = val["AllText"].to_list()
    true_ids = val["MisconceptionId"].to_list()

    q_embs = st_encode(model, queries, prompt_name="query", batch_size=64, show_progress=True)
    scores, idxs = index.search(q_embs, max(ks))

    evaluators = {k: EediEvaluator(k=k) for k in ks}
    for i, true_id in enumerate(true_ids):
        predicted = [misc_ids[j] for j in idxs[i]]
        for k in ks:
            evaluators[k].update(predicted, int(true_id))

    result = {}
    for k in ks:
        m = evaluators[k].compute()
        result[f"MAP@{k}"] = round(m[f"MAP@{k}"], 4)
        result[f"Recall@{k}"] = round(m[f"Recall@{k}"], 4)
        result[f"nDCG@{k}"] = round(m[f"nDCG@{k}"], 4)
    result["n_val"] = len(true_ids)
    return result, misc_embs


def run_zero_shot(model_name, fold, all_folds=False):
    console.print(f"\n[cyan]零样本基线: {model_name}")
    t0 = time.time()
    model = load_st_model(model_name, cache_dir=HF_CACHE)
    console.print(f"  模型加载: {time.time()-t0:.1f}s")

    long_df, misc_texts, misc_ids = load_data()

    folds = range(5) if all_folds else [fold]
    misc_embs = None
    all_results = {}
    for f in folds:
        res, misc_embs = evaluate_fold(model, long_df, misc_texts, misc_ids, f, misc_embs=misc_embs)
        all_results[f"fold{f}"] = res
        console.print(f"  [green]Fold {f}: MAP@25={res['MAP@25']}  Recall@25={res['Recall@25']}  Recall@10={res['Recall@10']}  nDCG@25={res['nDCG@25']}")

    if all_folds:
        avg = {k: round(np.mean([all_results[f"fold{f}"][k] for f in folds]), 4)
               for k in ["MAP@25", "Recall@25", "Recall@10", "nDCG@25"]}
        all_results["avg"] = avg
        console.print(f"  [bold green]平均: {avg}")

    return all_results


def run_lora_train(model_name, fold, epochs, batch_size, lr):
    """LoRA + MultipleNegativesRankingLoss 微调（sentence-transformers Trainer）。"""
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
    from sentence_transformers.training_args import BatchSamplers
    from datasets import Dataset
    from peft import LoraConfig
    import torch

    console.print(f"\n[cyan]LoRA 微调: {model_name}  fold={fold}")
    long_df, misc_texts, misc_ids = load_data()

    # 构建 (anchor=query, positive=misconception) 训练对
    train = long_df.filter(pl.col("fold") != fold).filter(pl.col("MisconceptionId") >= 0)
    anchors = train["AllText"].to_list()
    positives = [misc_texts[mid] for mid in train["MisconceptionId"].to_list()]
    train_ds = Dataset.from_dict({"anchor": anchors, "positive": positives})
    console.print(f"  训练对: {len(train_ds)}")

    model = load_st_model(model_name, cache_dir=HF_CACHE)
    # Eedi query 较短，限制 max_seq_length 可以显著降低激活显存，避免默认 32K 上下文误伤。
    model.max_seq_length = 512

    # 注入 LoRA
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="FEATURE_EXTRACTION",
    )
    model[0].auto_model.add_adapter(lora_cfg)
    # 安全前提下尽量提高 batch；gradient checkpointing 用计算换显存，适合 8B/14B LoRA。
    if hasattr(model[0].auto_model, "gradient_checkpointing_enable"):
        model[0].auto_model.gradient_checkpointing_enable()
    if hasattr(model[0].auto_model.config, "use_cache"):
        model[0].auto_model.config.use_cache = False
    console.print("  LoRA adapter 已注入")

    # MNRL = in-batch InfoNCE 对比损失
    loss = losses.MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(OUTPUT_DIR / "lora_train"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.05,
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # 同 batch 不重复正例
        logging_steps=50,
        save_strategy="no",
        report_to=[],
        dataloader_drop_last=True,
    )

    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train_ds, loss=loss)
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    console.print(f"  训练耗时: {train_time:.0f}s")

    # 保存 LoRA adapter。0.6B baseline 与 8B final mainline 分开，避免后续索引脚本读错。
    if "Qwen3-Embedding-8B" in model_name:
        adapter_dir = OUTPUT_DIR / "lora_best_8b"
    elif "Qwen3-Embedding-0.6B" in model_name:
        adapter_dir = OUTPUT_DIR / "lora_best_0_6b"
    else:
        adapter_dir = OUTPUT_DIR / "lora_best"
    model[0].auto_model.save_pretrained(str(adapter_dir))
    console.print(f"  [green]✓ adapter 保存: {adapter_dir}")

    # 微调后评测
    res, _ = evaluate_fold(model, long_df, misc_texts, misc_ids, fold)
    console.print(f"  [bold green]微调后 Fold {fold}: {res}")
    return {"train_time_s": train_time, "metrics": res}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    console.rule("[bold blue]阶段1：召回器")
    results = {"model": args.model, "fold": args.fold}

    if args.train:
        results["lora"] = run_lora_train(args.model, args.fold, args.epochs, args.batch_size, args.lr)
    else:
        results["zero_shot"] = run_zero_shot(args.model, args.fold, args.all_folds)

    out_path = OUTPUT_DIR / ("results_train.json" if args.train else "results_baseline.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    console.print(f"\n[green]结果保存: {out_path}")
    console.rule("[bold green]阶段1 完成")


if __name__ == "__main__":
    main()
