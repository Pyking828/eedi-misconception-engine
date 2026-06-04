"""
GRPO reinforcement learning for listwise reranking.

Rewards:
  - top1_hit: gold misconception at rank 1 (0 or 1)
  - ndcg_gain: nDCG@5 gain vs random baseline (continuous proxy)
Uses TRL GRPOTrainer: prompt = listwise input, response = letter ranking string.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from eval.evaluator import ndcg_at_k

from src.eedi.reranker.listwise import parse_listwise_output

if TYPE_CHECKING:
    import datasets

# ─────────────────────────────────────────────
# Reward functions for TRL GRPOTrainer
# ─────────────────────────────────────────────


def make_ranking_reward(reward_type: str = "ndcg_gain"):
    """
    Returns reward_fn(completions, true_ids, candidate_ids_list, **kwargs) -> list[float].
    TRL passes completions (model outputs) and custom fields in kwargs.
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
# GRPO training entry (dataset prep + GRPOTrainer)
# ─────────────────────────────────────────────


def prepare_grpo_dataset(
    long_df,
    misc_texts: dict[int, str],
    candidate_pool: dict[str, list[int]],
    cot_cache: dict[str, str] | None = None,
    n_candidates: int = 10,
    split: str = "train",
    fold: int = 0,
) -> datasets.Dataset:  # type: ignore[name-defined]
    """
    Build GRPOTrainer dataset rows:
      - prompt: listwise input
      - true_id: gold misconception id
      - candidate_ids: retrieval candidates
    """
    import polars as pl
    from datasets import Dataset

    from src.eedi.reranker.listwise import LISTWISE_SYSTEM, build_listwise_prompt

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
    cache_dir: str | None = None,
) -> None:
    """Run TRL GRPOTrainer RL fine-tuning."""
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

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
        num_generations=4,  # G: 4 completions per prompt
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
