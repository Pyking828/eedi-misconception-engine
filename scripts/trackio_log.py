"""Log fold0 CV MAP@25 progression to Trackio for dashboard visualization.

Records retrieve → rerank → ensemble → listwise/GRPO → retriever fix climb (0.40→0.5974).
Optional --space-id syncs to a permanent HF Space dashboard.

Usage:
  python scripts/trackio_log.py
  python scripts/trackio_log.py --space-id user/eedi-trackio
"""

from __future__ import annotations

import argparse

import trackio

# fold0 CV MAP@25 milestones (same eval protocol across stages)
PROGRESSION = [
    ("retriever_8B_zeroshot", 0.2248, 0.6535),
    ("retriever_8B_LoRA(real)", 0.4289, 0.9416),
    ("reranker_pointwise_6k", 0.4707, 0.9153),
    ("reranker_pointwise_24k", 0.5244, 0.9142),
    ("reranker_pointwise_31k", 0.5700, 0.9211),
    ("reranker_31k_baselinepool", 0.5807, 0.9611),
    ("reranker_hn12_baselinepool", 0.5842, 0.9588),
    ("ensemble_2model", 0.5950, 0.9611),
    ("ensemble_3model_final", 0.5974, 0.9611),
]
# SFT vs RL comparison (top-10 rerank strategy, separate run)
SFT_RL = [
    ("listwise_zeroshot", 0.3254),
    ("listwise_SFT_8B", 0.5700),
    ("listwise_SFT_R1-14B", 0.5717),
    ("GRPO_v1_hit", 0.5700),
    ("GRPO_v2_ndcg", 0.5741),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space-id", default="", help="非空则同步到该 HF Space 看板（需已登录 hf）")
    args = ap.parse_args()
    kw = {"space_id": args.space_id} if args.space_id else {}

    # Run 1: main pipeline progression
    trackio.init(
        project="eedi-misconception-engine",
        name="pipeline-progression",
        config={"metric": "MAP@25", "fold": 0, "kaggle_1st": 0.63881},
        **kw,
    )
    for step, (name, mapk, r25) in enumerate(PROGRESSION):
        trackio.log({"MAP@25": mapk, "Recall@25": r25, "stage": name}, step=step)
    trackio.finish()

    # Run 2: SFT vs RL (listwise strategy)
    trackio.init(
        project="eedi-misconception-engine",
        name="sft-vs-rl",
        config={"note": "top-10 重排策略；GRPO 两种 reward 均不超 SFT"},
        **kw,
    )
    for step, (name, mapk) in enumerate(SFT_RL):
        trackio.log({"MAP@25": mapk, "method": name}, step=step)
    trackio.finish()

    print(
        f"已记录到 Trackio 项目 'eedi-misconception-engine'（最终最优 MAP@25={PROGRESSION[-1][1]}）"
    )
    print("本地查看：trackio show  | 同步 Space：加 --space-id user/eedi-trackio")


if __name__ == "__main__":
    main()
