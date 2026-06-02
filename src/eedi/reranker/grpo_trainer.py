"""
GRPO 强化学习精排。
Reward：
  - top1_hit: 金标 misconception 是否排在第 1 位 (0 or 1)
  - ndcg_gain: nDCG@5 相对于随机基线的增益（连续，可微分代理）
使用 TRL GRPOTrainer，prompt = listwise 输入，response = 排序字母串。
"""
from __future__ import annotations

import re
from typing import Optional

from eval.evaluator import ndcg_at_k
from src.eedi.reranker.listwise import ALPHABET, parse_listwise_output


# ─────────────────────────────────────────────
# Reward 函数（供 TRL GRPOTrainer 使用）
# ─────────────────────────────────────────────

def make_ranking_reward(reward_type: str = "ndcg_gain"):
    """
    返回 reward_fn(completions, true_ids, candidate_ids_list, **kwargs) -> list[float]
    TRL GRPOTrainer 传入 completions（模型输出文本列表）和 kwargs 中的自定义字段。
    """

    def reward_fn(
        completions: list[str],
        true_ids: list[int],
        candidate_ids_list: list[list[int]],
        **kwargs,
    ) -> list[float]:
        rewards = []
        for completion, true_id, cand_ids in zip(completions, true_ids, candidate_ids_list):
            n = len(cand_ids)
            ranked_indices = parse_listwise_output(completion, n)
            ranked_ids = [cand_ids[i] for i in ranked_indices]

            if reward_type == "top1_hit":
                r = 1.0 if ranked_ids and ranked_ids[0] == true_id else 0.0
            elif reward_type == "ndcg_gain":
                r = ndcg_at_k(ranked_ids, true_id, k=5)
            else:
                raise ValueError(f"Unknown reward_type: {reward_type}")

            rewards.append(r)
        return rewards

    return reward_fn


# ─────────────────────────────────────────────
# GRPO 训练入口（dataset 准备 + GRPOTrainer 配置）
# ─────────────────────────────────────────────

def prepare_grpo_dataset(
    long_df,
    misc_texts: dict[int, str],
    candidate_pool: dict[str, list[int]],
    cot_cache: Optional[dict[str, str]] = None,
    n_candidates: int = 10,
    split: str = "train",
    fold: int = 0,
) -> "datasets.Dataset":  # type: ignore[name-defined]
    """
    将 EediDataset 转为 GRPOTrainer 需要的格式：
      - prompt: listwise 输入文本
      - true_id: int（ground truth misconception id）
      - candidate_ids: list[int]（召回候选）
    """
    from datasets import Dataset
    from src.eedi.reranker.listwise import build_listwise_prompt, LISTWISE_SYSTEM

    import polars as pl

    if split == "train":
        df = long_df.filter(pl.col("fold") != fold)
    else:
        df = long_df.filter(pl.col("fold") == fold)

    df = df.filter(pl.col("MisconceptionId") >= 0)

    records = []
    for row in df.iter_rows(named=True):
        qa_key = row["QuestionId_Answer"]
        true_id = row["MisconceptionId"]
        cand_ids = candidate_pool.get(qa_key, [])
        if not cand_ids or true_id not in cand_ids:
            continue

        query = row["AllText"]
        candidates = [misc_texts.get(cid, "") for cid in cand_ids]
        cot = (cot_cache or {}).get(qa_key)
        prompt_text = build_listwise_prompt(query, candidates, cot)

        messages = [
            {"role": "system", "content": LISTWISE_SYSTEM},
            {"role": "user", "content": prompt_text},
        ]
        records.append(
            {
                "prompt": messages,
                "true_id": true_id,
                "candidate_ids": cand_ids,
                "qa_key": qa_key,
            }
        )

    return Dataset.from_list(records)


def run_grpo_training(
    model_path: str,
    train_dataset,
    output_dir: str = "outputs/reranker/grpo",
    reward_type: str = "ndcg_gain",
    num_epochs: int = 1,
    lr: float = 1e-5,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    lora_r: int = 32,
    lora_alpha: int = 64,
    cache_dir: Optional[str] = None,
) -> None:
    """启动 TRL GRPOTrainer 进行强化学习微调。"""
    import torch
    from trl import GRPOConfig, GRPOTrainer
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        cache_dir=cache_dir,
    )

    peft_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    reward_fn = make_ranking_reward(reward_type)

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        learning_rate=lr,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        bf16=True,
        gradient_checkpointing=True,
        max_new_tokens=64,
        num_generations=4,         # G：每个 prompt 采样 4 个 completion
        logging_steps=10,
        save_steps=100,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=train_dataset,
        peft_config=peft_cfg,
    )

    print("[GRPO] 开始训练...")
    trainer.train()
    trainer.save_model(output_dir)
    print(f"[GRPO] 训练完成，模型保存至 {output_dir}")
