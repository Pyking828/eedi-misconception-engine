"""
精排器（Listwise Reranker）。
模型：Qwen2.5-3B-Instruct + LoRA SFT → GRPO RL
方式：一次性输入 top-N 候选，输出排序字母序列（复刻 5th place 方案）
     或直接输出最匹配 misconception 序号（复刻 1st place listwise 思路）
GRPO reward：nDCG 增益 / top-1 命中率
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

from eval.evaluator import ndcg_at_k


# ─────────────────────────────────────────────
# Prompt 构建工具
# ─────────────────────────────────────────────

LISTWISE_SYSTEM = (
    "You are an expert mathematics educator. "
    "Given a student's incorrect answer and a list of candidate misconceptions, "
    "rank the candidates from most to least likely to explain the error. "
    "Output ONLY a comma-separated list of the candidate letters (e.g. 'A, C, B, D'), "
    "from most relevant to least relevant."
)

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def build_listwise_prompt(
    query: str,
    candidates: list[str],
    cot_rationale: Optional[str] = None,
) -> str:
    cand_str = "\n".join(
        f"{ALPHABET[i]}. {c}" for i, c in enumerate(candidates)
    )
    rationale_block = ""
    if cot_rationale:
        rationale_block = f"\nStudent reasoning analysis:\n{cot_rationale}\n"
    return (
        f"Question context:\n{query}"
        f"{rationale_block}"
        f"\n\nCandidate misconceptions:\n{cand_str}"
        f"\n\nRank the candidates (most to least likely):"
    )


def parse_listwise_output(output: str, n_candidates: int) -> list[int]:
    """解析模型输出的字母排序，返回 0-indexed 位置列表。"""
    # 提取字母
    letters = re.findall(r"[A-Z]", output.upper())
    seen: set[int] = set()
    ranked: list[int] = []
    for letter in letters:
        idx = ord(letter) - ord("A")
        if 0 <= idx < n_candidates and idx not in seen:
            ranked.append(idx)
            seen.add(idx)
    # 补全未出现的候选（顺序不变）
    for i in range(n_candidates):
        if i not in seen:
            ranked.append(i)
    return ranked


# ─────────────────────────────────────────────
# ListwiseReranker：推理
# ─────────────────────────────────────────────

class ListwiseReranker:
    """
    精排：接收 top-10 候选，输出重排后 top-5 id。

    示例：
        reranker = ListwiseReranker.from_pretrained(
            model_name="Qwen/Qwen2.5-3B-Instruct",
            adapter_path="outputs/reranker/listwise/lora_best",
        )
        ranked_ids = reranker.rerank(query, candidate_ids, misc_texts, top_k=5)
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer,
        max_new_tokens: int = 128,
        device: str = "cuda",
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        adapter_path: Optional[str | Path] = None,
        max_new_tokens: int = 128,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
    ) -> "ListwiseReranker":
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            cache_dir=cache_dir,
        )
        if adapter_path is not None and Path(adapter_path).exists():
            model = PeftModel.from_pretrained(model, str(adapter_path))
            model = model.merge_and_unload()
        return cls(model, tokenizer, max_new_tokens, device)

    @torch.no_grad()
    def rerank(
        self,
        query: str,
        candidate_ids: list[int],
        misc_texts: dict[int, str],
        top_k: int = 5,
        cot_rationale: Optional[str] = None,
    ) -> list[int]:
        candidates = [misc_texts.get(cid, "") for cid in candidate_ids]
        prompt = build_listwise_prompt(query, candidates, cot_rationale)

        messages = [
            {"role": "system", "content": LISTWISE_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)

        out = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = self.tokenizer.decode(
            out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True
        )
        ranked_indices = parse_listwise_output(generated, len(candidate_ids))
        ranked_ids = [candidate_ids[i] for i in ranked_indices]
        return ranked_ids[:top_k]


# ─────────────────────────────────────────────
# ListwiseTrainer：SFT（为 GRPO 铺垫）
# ─────────────────────────────────────────────

class ListwiseTrainer:
    """SFT 微调 listwise reranker（先于 GRPO 运行，作为冷启动）。"""

    def __init__(
        self,
        model_name: str,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        cache_dir: Optional[str] = None,
        output_dir: str = "outputs/reranker/listwise",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            cache_dir=cache_dir,
        )
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(model, lora_cfg)
        self.model.print_trainable_parameters()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def build_sft_example(
        self,
        query: str,
        candidate_ids: list[int],
        misc_texts: dict[int, str],
        true_id: int,
        cot_rationale: Optional[str] = None,
    ) -> dict:
        """构建单条 SFT 训练样本（prompt + 正确排序输出）。"""
        candidates = [misc_texts.get(cid, "") for cid in candidate_ids]
        try:
            true_idx = candidate_ids.index(true_id)
            true_letter = ALPHABET[true_idx]
        except ValueError:
            true_letter = "A"  # 金标不在候选中，跳过（训练时过滤）

        prompt = build_listwise_prompt(query, candidates, cot_rationale)
        # 构造：金标排第一，其余按原顺序
        other_letters = [ALPHABET[i] for i in range(len(candidates)) if i != true_idx]
        target_output = ", ".join([true_letter] + other_letters[:4])

        messages = [
            {"role": "system", "content": LISTWISE_SYSTEM},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target_output},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": text, "true_id": true_id, "query": query}

    def save_adapter(self, suffix: str = "final") -> None:
        save_path = self.output_dir / f"lora_{suffix}"
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        print(f"[ListwiseTrainer] saved → {save_path}")
