"""
Listwise reranker.

Model: DeepSeek-R1-Distill-Qwen-14B + LoRA SFT → GRPO RL
Inputs top-N candidates; outputs a letter ranking (5th-place style) or best index (1st-place style).
GRPO rewards: nDCG gain / top-1 hit rate.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────
# Prompt helpers
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
    cot_rationale: str | None = None,
) -> str:
    cand_str = "\n".join(f"{ALPHABET[i]}. {c}" for i, c in enumerate(candidates))
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
    """Parse letter ranking from model output; returns 0-indexed positions."""
    # Extract letters
    letters = re.findall(r"[A-Z]", output.upper())
    seen: set[int] = set()
    ranked: list[int] = []
    for letter in letters:
        idx = ord(letter) - ord("A")
        if 0 <= idx < n_candidates and idx not in seen:
            ranked.append(idx)
            seen.add(idx)
    # Append missing candidates in original order
    for i in range(n_candidates):
        if i not in seen:
            ranked.append(i)
    return ranked


# ─────────────────────────────────────────────
# ListwiseReranker inference
# ─────────────────────────────────────────────


class ListwiseReranker:
    """
    Rerank top-10 candidates to top-5 ids.

    Example:
        reranker = ListwiseReranker.from_pretrained(
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            adapter_path="outputs/reranker/listwise/lora_best_14b",
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
        adapter_path: str | Path | None = None,
        max_new_tokens: int = 128,
        device: str = "cuda",
        cache_dir: str | None = None,
    ) -> ListwiseReranker:
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
        cot_rationale: str | None = None,
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
# ListwiseTrainer: SFT warm-start for GRPO
# ─────────────────────────────────────────────


class ListwiseTrainer:
    """SFT fine-tune listwise reranker (cold start before GRPO)."""

    def __init__(
        self,
        model_name: str,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        cache_dir: str | None = None,
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
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
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
        cot_rationale: str | None = None,
    ) -> dict:
        """Build one SFT example (prompt + gold ranking)."""
        candidates = [misc_texts.get(cid, "") for cid in candidate_ids]
        try:
            true_idx = candidate_ids.index(true_id)
            true_letter = ALPHABET[true_idx]
        except ValueError:
            true_letter = "A"  # gold not in pool; filter at train time

        prompt = build_listwise_prompt(query, candidates, cot_rationale)
        # Gold first, then remaining letters in original order
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
