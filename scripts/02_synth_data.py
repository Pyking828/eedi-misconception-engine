"""
阶段2：合成数据生成（vLLM + DeepSeek-R1-Distill-Qwen-32B 离线 teacher/judge）
运行：python scripts/02_synth_data.py [--teacher deepseek-ai/DeepSeek-R1-Distill-Qwen-32B] [--n 5]

耗时预估：
  - 下载 32B bf16 模型：~60-65GB，约数小时到十数小时（取决于 hf-mirror 带宽）
  - 生成 2587*5=12935 条：在 96GB 显卡上约 2-4h
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path

import polars as pl
from rich.console import Console

from src.eedi.synth.synth import SynthDataGenerator, MisconceptionExpander, merge_real_and_synth

console = Console()

DATA_DIR = Path(os.environ.get("EEDI_DATA", "/root/autodl-tmp/eedi-data"))
HF_CACHE = os.environ.get("HF_HOME", "/root/autodl-tmp/hf_cache")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--student-reasoner", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                        help="后续蒸馏目标模型；本脚本先生成32B教师数据")
    parser.add_argument("--n", type=int, default=5, help="每条错因生成几道题")
    parser.add_argument("--judge-threshold", type=float, default=6.0)
    parser.add_argument("--gpu-util", type=float, default=0.80)
    parser.add_argument("--expand-misconceptions", action="store_true", help="同时扩写错因描述")
    args = parser.parse_args()

    console.rule("[bold blue]阶段2：合成数据生成")
    console.print(f"[cyan]教师模型：{args.teacher}")
    console.print(f"[cyan]每条错因生成：{args.n} 道题")

    misc_df = pl.read_csv(str(DATA_DIR / "misconception_mapping.csv"))
    long_df = pl.read_parquet(str(DATA_DIR / "folds.parquet"))

    # 找出训练集中「未见」的 misconception（重点覆盖）
    seen_ids = set(long_df.filter(pl.col("MisconceptionId") >= 0)["MisconceptionId"].to_list())
    all_misc = {int(r["MisconceptionId"]): r["MisconceptionName"] for r in misc_df.iter_rows(named=True)}
    unseen_misc = {mid: name for mid, name in all_misc.items() if mid not in seen_ids}
    console.print(f"  未见错因数（优先生成）：{len(unseen_misc)}/{len(all_misc)}")

    # 构建参考样例
    seen_rows = long_df.filter(pl.col("MisconceptionId") >= 0)
    reference_examples: dict[str, list[dict]] = {}
    for row in seen_rows.iter_rows(named=True):
        name = row["MisconceptionName"]
        if name not in reference_examples:
            reference_examples[name] = []
        if len(reference_examples[name]) < 3:
            reference_examples[name].append(
                {
                    "ConstructName": row["ConstructName"],
                    "SubjectName": row["SubjectName"],
                    "QuestionText": row["QuestionText"],
                    "CorrectAnswerText": row["CorrectAnswerText"],
                    "WrongAnswerText": row["WrongAnswerText"],
                }
            )

    # 初始化生成器（下载模型）
    console.print("\n[cyan]初始化 vLLM 生成器（可能需要下载模型...）")
    generator = SynthDataGenerator(
        model_name=args.teacher,
        gpu_memory_utilization=args.gpu_util,
        cache_dir=HF_CACHE,
    )

    # 优先生成未见错因，再补充已见错因
    target_misconceptions = list(unseen_misc.values()) + [
        name for mid, name in all_misc.items() if mid in seen_ids
    ]
    target_misconceptions = target_misconceptions[:len(all_misc)]  # 全量

    console.print(f"\n[cyan]生成 MCQ（{len(target_misconceptions)} 条错因 × {args.n} 题/条）")
    raw_records = generator.generate_mcqs(
        misconceptions=target_misconceptions,
        reference_examples=reference_examples,
        n_per_misconception=args.n,
        output_path=DATA_DIR / "synth_raw.jsonl",
    )
    console.print(f"  生成原始样本：{len(raw_records)} 条")

    # LLM-as-Judge 质检
    console.print(f"\n[cyan]LLM-as-Judge 质检（阈值={args.judge_threshold}）")
    filtered_records = generator.llm_judge(raw_records, threshold=args.judge_threshold)
    console.print(f"  质检通过：{len(filtered_records)}/{len(raw_records)} ({len(filtered_records)/max(1,len(raw_records))*100:.1f}%)")

    # 保存过滤后的合成数据
    synth_path = DATA_DIR / "synth_filtered.jsonl"
    with open(synth_path, "w") as f:
        for r in filtered_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    console.print(f"  [green]✓ 合成数据保存至: {synth_path}")

    # 合并真实+合成数据
    merged = merge_real_and_synth(long_df, filtered_records, misc_df)
    merged.write_parquet(str(DATA_DIR / "long_table_with_synth.parquet"))
    console.print(f"  [green]✓ 合并数据保存（真实 {len(long_df)} + 合成 {len(filtered_records)} = {len(merged)} 行）")

    # 可选：扩写错因描述
    if args.expand_misconceptions:
        console.print("\n[cyan]扩写 misconception 描述...")
        expander = MisconceptionExpander(model_name=args.teacher, gpu_memory_utilization=args.gpu_util, cache_dir=HF_CACHE)
        misc_names = [v for v in all_misc.values()]
        expanded = expander.expand(misc_names)
        with open(DATA_DIR / "misconception_expanded.json", "w") as f:
            json.dump(expanded, f, ensure_ascii=False, indent=2)
        console.print(f"  [green]✓ 扩写完成，{len(expanded)} 条，保存至 misconception_expanded.json")

    console.rule("[bold green]阶段2 完成")


if __name__ == "__main__":
    main()
