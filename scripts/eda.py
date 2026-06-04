"""
Stage 0: EDA, preprocessing, and 5-fold CV.
Run: python scripts/eda.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import polars as pl
from rich.console import Console
from rich.table import Table
from src.eedi.data.dataset import build_cv_folds, build_long_table, load_raw_data

console = Console()

DATA_DIR = Path(os.environ.get("EEDI_DATA", "/root/autodl-tmp/eedi-data"))
OUTPUT_DIR = DATA_DIR


def main():
    console.rule("[bold blue]Eedi EDA & 数据预处理")

    # ── Load raw data ──
    console.print(f"[cyan]加载数据: {DATA_DIR}")
    train_df, misc_df, test_df = load_raw_data(DATA_DIR)

    t = Table(title="原始数据概览")
    t.add_column("文件", style="cyan")
    t.add_column("行数", style="green")
    t.add_column("列数", style="green")
    t.add_row("train.csv", str(len(train_df)), str(len(train_df.columns)))
    t.add_row("misconception_mapping.csv", str(len(misc_df)), str(len(misc_df.columns)))
    t.add_row("test.csv", str(len(test_df)), str(len(test_df.columns)))
    console.print(t)

    console.print("\ntrain.csv 列名:", train_df.columns)

    # ── Wide → long table ──
    console.print("\n[cyan]构建长表（宽表→每 distractor 一行）")
    long_df = build_long_table(train_df, misc_df)
    console.print(f"长表行数: {len(long_df)}（有标注的 distractor）")
    console.print(f"长表列名: {long_df.columns}")

    # ── EDA stats ──
    n_total_misc = len(misc_df)
    n_train_misc = long_df["MisconceptionId"].n_unique()
    n_unseen_misc = n_total_misc - n_train_misc

    console.print("\n[bold]错因统计：")
    console.print(f"  错因总数（misconception_mapping）: {n_total_misc}")
    console.print(f"  训练集中出现过的错因数: {n_train_misc}")
    console.print(
        f"  未见错因数（仅在 mapping 中）:  {n_unseen_misc} ({n_unseen_misc / n_total_misc * 100:.1f}%)"
    )

    # Distractors per question distribution
    distractor_counts = (
        long_df.group_by("QuestionId")
        .agg(pl.len().alias("n_distractors"))["n_distractors"]
        .value_counts()
        .sort("n_distractors")
    )
    console.print("\n[bold]每题 distractor 数量分布：")
    console.print(distractor_counts)

    # Subject distribution
    subject_dist = (
        long_df.group_by("SubjectName").agg(pl.len().alias("count")).sort("count", descending=True)
    )
    console.print("\n[bold]学科分布：")
    console.print(subject_dist)

    # ── 5-fold CV ──
    console.print("\n[cyan]构建 5 折 CV（GroupKFold by QuestionId）")
    long_df = build_cv_folds(long_df, n_folds=5, seed=42, save_path=OUTPUT_DIR / "folds.parquet")
    console.print(f"  已保存至: {OUTPUT_DIR}/folds.parquet")

    for fold in range(5):
        n_train = len(long_df.filter(pl.col("fold") != fold))
        n_val = len(long_df.filter(pl.col("fold") == fold))
        # Unseen misconceptions in val fold
        train_misc = set(long_df.filter(pl.col("fold") != fold)["MisconceptionId"].to_list())
        val_misc = set(long_df.filter(pl.col("fold") == fold)["MisconceptionId"].to_list())
        n_unseen_in_val = len(val_misc - train_misc)
        console.print(
            f"  Fold {fold}: train={n_train}, val={n_val}, val中未见错因={n_unseen_in_val}"
        )

    # ── Save long table ──
    long_df.write_parquet(str(OUTPUT_DIR / "long_table.parquet"))
    console.print(f"\n[green]✓ 长表已保存: {OUTPUT_DIR}/long_table.parquet")

    # ── seen_misc_ids for 3rd-place score scaling ──
    seen_ids = long_df["MisconceptionId"].unique().to_list()
    import json

    with open(OUTPUT_DIR / "seen_misc_ids.json", "w") as f:
        json.dump(seen_ids, f)
    console.print(f"[green]✓ seen_misc_ids 已保存: {len(seen_ids)} 条")

    console.rule("[bold green]EDA 完成")


if __name__ == "__main__":
    main()
