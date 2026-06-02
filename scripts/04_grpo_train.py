"""
阶段4：GRPO 强化学习微调精排器
运行：python scripts/04_grpo_train.py [--fold 0] [--reward ndcg_gain|top1_hit]

关键点（面试重点）：
- reward = nDCG@5 增益（连续奖励，比 top1_hit 梯度信号更丰富）
- 用 TRL GRPOTrainer，prompt=listwise 输入，completion=排序字母串
- 与 SFT-only 做 CV 对比，验证 RL 带来的实际提升
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path

import polars as pl
from rich.console import Console

from eval.evaluator import EediEvaluator

console = Console()
DATA_DIR = Path(os.environ.get("EEDI_DATA", "/root/autodl-tmp/eedi-data"))
HF_CACHE = os.environ.get("HF_HOME", "/root/autodl-tmp/hf_cache")
OUTPUT_DIR = Path("/root/autodl-tmp/eedi-misconception-engine/outputs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--reward", choices=["ndcg_gain", "top1_hit"], default="ndcg_gain")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--sft-model-path", default=None,
                        help="SFT 训练后的 checkpoint（默认：Qwen2.5-3B-Instruct 原版）")
    args = parser.parse_args()

    console.rule("[bold blue]阶段4：GRPO 强化学习")
    console.print(f"[cyan]reward={args.reward}  episodes={args.episodes}")

    long_df = pl.read_parquet(str(DATA_DIR / "folds.parquet"))
    misc_df = pl.read_csv(str(DATA_DIR / "misconception_mapping.csv"))
    misc_texts = {int(r["MisconceptionId"]): r["MisconceptionName"] for r in misc_df.iter_rows(named=True)}

    candidate_pool_path = OUTPUT_DIR / "retriever" / "candidate_pool.json"
    with open(candidate_pool_path) as f:
        candidate_pool = json.load(f)

    # CoT 缓存（若已生成则注入）
    cot_cache_path = DATA_DIR / "cot_cache.json"
    cot_cache = {}
    if cot_cache_path.exists():
        with open(cot_cache_path) as f:
            cot_cache = json.load(f)
        console.print(f"[cyan]加载 CoT 缓存：{len(cot_cache)} 条")

    from src.eedi.reranker.grpo_trainer import prepare_grpo_dataset, run_grpo_training

    train_dataset = prepare_grpo_dataset(
        long_df, misc_texts, candidate_pool,
        cot_cache=cot_cache, n_candidates=10,
        split="train", fold=args.fold,
    )
    console.print(f"[cyan]GRPO 训练集：{len(train_dataset)} 条")

    model_path = args.sft_model_path or "Qwen/Qwen2.5-3B-Instruct"
    grpo_output = str(OUTPUT_DIR / "reranker" / "grpo")

    run_grpo_training(
        model_path=model_path,
        train_dataset=train_dataset,
        output_dir=grpo_output,
        reward_type=args.reward,
        num_epochs=1,
        lr=1e-5,
        per_device_batch_size=2,
        gradient_accumulation_steps=8,
        cache_dir=HF_CACHE,
    )

    console.print(f"\n[green]✓ GRPO 训练完成，模型保存至 {grpo_output}")
    console.rule("[bold green]阶段4 完成")


if __name__ == "__main__":
    main()
