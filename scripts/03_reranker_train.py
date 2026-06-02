"""
阶段3：粗排 + 精排训练
运行：python scripts/03_reranker_train.py [--fold 0] [--stage pointwise|listwise|both]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path

import polars as pl
import torch
from rich.console import Console

from eval.evaluator import EediEvaluator

console = Console()
DATA_DIR = Path(os.environ.get("EEDI_DATA", "/root/autodl-tmp/eedi-data"))
HF_CACHE = os.environ.get("HF_HOME", "/root/autodl-tmp/hf_cache")
OUTPUT_DIR = Path("/root/autodl-tmp/eedi-misconception-engine/outputs")


def build_reranker_dataset(
    long_df: pl.DataFrame,
    misc_df: pl.DataFrame,
    candidate_pool: dict[str, list[int]],
    fold: int = 0,
    split: str = "train",
    n_neg: int = 9,
):
    """构建 pointwise 重排数据集（正例1:负例n）。"""
    from datasets import Dataset
    import random

    if split == "train":
        df = long_df.filter(pl.col("fold") != fold)
    else:
        df = long_df.filter(pl.col("fold") == fold)
    df = df.filter(pl.col("MisconceptionId") >= 0)

    misc_texts = {int(r["MisconceptionId"]): r["MisconceptionName"] for r in misc_df.iter_rows(named=True)}
    rng = random.Random(42)

    records = []
    for row in df.iter_rows(named=True):
        qa_key = row["QuestionId_Answer"]
        true_id = row["MisconceptionId"]
        query = row["AllText"]
        cands = candidate_pool.get(qa_key, [])
        if not cands:
            continue
        # 正例
        records.append({"query": query, "candidate": misc_texts.get(true_id, ""), "label": 1.0})
        # 负例
        neg_pool = [c for c in cands if c != true_id]
        for neg_id in rng.sample(neg_pool, min(n_neg, len(neg_pool))):
            records.append({"query": query, "candidate": misc_texts.get(neg_id, ""), "label": 0.0})

    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--stage", choices=["pointwise", "listwise", "both"], default="both")
    args = parser.parse_args()

    console.rule("[bold blue]阶段3：重排器训练")

    long_df = pl.read_parquet(str(DATA_DIR / "folds.parquet"))
    misc_df = pl.read_csv(str(DATA_DIR / "misconception_mapping.csv"))

    # 加载召回器的候选池（需要先跑 01_retriever_baseline.py 并保存候选）
    candidate_pool_path = OUTPUT_DIR / "retriever" / "candidate_pool.json"
    if candidate_pool_path.exists():
        with open(candidate_pool_path) as f:
            candidate_pool = json.load(f)
        console.print(f"[cyan]加载召回候选池：{len(candidate_pool)} 条 query")
    else:
        console.print("[yellow]警告：candidate_pool.json 不存在，使用随机候选代替")
        misc_ids = misc_df["MisconceptionId"].to_list()
        import random
        rng = random.Random(42)
        candidate_pool = {
            row["QuestionId_Answer"]: rng.sample(misc_ids, min(50, len(misc_ids)))
            for row in long_df.filter(pl.col("MisconceptionId") >= 0).iter_rows(named=True)
        }

    if args.stage in ("pointwise", "both"):
        console.print("\n[cyan]Pointwise 粗排训练...")
        from src.eedi.reranker.pointwise import PointwiseTrainer
        from torch.utils.data import DataLoader
        from src.eedi.data.collator import RerankCollator
        from transformers import AutoTokenizer

        model_name = "Qwen/Qwen3-Reranker-0.6B"
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_ds = build_reranker_dataset(long_df, misc_df, candidate_pool, fold=args.fold, split="train")
        collator = RerankCollator(tokenizer=tokenizer, max_seq_len=512)
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collator)

        trainer = PointwiseTrainer(
            model_name=model_name,
            cache_dir=HF_CACHE,
            output_dir=str(OUTPUT_DIR / "reranker" / "pointwise"),
        )
        history = trainer.train(train_loader, num_epochs=3)
        console.print(f"[green]✓ Pointwise 训练完成，最终 loss={history[-1]['loss']:.4f}")

    if args.stage in ("listwise", "both"):
        console.print("\n[cyan]Listwise 精排 SFT 训练...")
        from src.eedi.reranker.listwise import ListwiseTrainer
        from trl import SFTConfig, SFTTrainer

        model_name = "Qwen/Qwen2.5-3B-Instruct"
        misc_texts = {int(r["MisconceptionId"]): r["MisconceptionName"] for r in misc_df.iter_rows(named=True)}

        trainer = ListwiseTrainer(
            model_name=model_name,
            cache_dir=HF_CACHE,
            output_dir=str(OUTPUT_DIR / "reranker" / "listwise"),
        )

        # 构建 SFT 样本
        train_df = long_df.filter(pl.col("fold") != args.fold).filter(pl.col("MisconceptionId") >= 0)
        sft_records = []
        for row in train_df.iter_rows(named=True):
            qa_key = row["QuestionId_Answer"]
            cands = candidate_pool.get(qa_key, [])
            if not cands or row["MisconceptionId"] not in cands:
                continue
            example = trainer.build_sft_example(
                query=row["AllText"],
                candidate_ids=cands[:10],
                misc_texts=misc_texts,
                true_id=row["MisconceptionId"],
            )
            sft_records.append(example)

        from datasets import Dataset as HFDataset
        sft_dataset = HFDataset.from_list(sft_records)
        console.print(f"  SFT 样本数：{len(sft_records)}")

        sft_config = SFTConfig(
            output_dir=str(OUTPUT_DIR / "reranker" / "listwise"),
            num_train_epochs=2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=5e-5,
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=10,
            save_steps=200,
            report_to="none",
        )
        sft_trainer = SFTTrainer(
            model=trainer.model,
            processing_class=trainer.tokenizer,
            args=sft_config,
            train_dataset=sft_dataset,
        )
        sft_trainer.train()
        trainer.save_adapter("sft_final")
        console.print("[green]✓ Listwise SFT 训练完成")

    console.rule("[bold green]阶段3 完成")


if __name__ == "__main__":
    main()
