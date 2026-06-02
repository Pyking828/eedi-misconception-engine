"""
推理 Subagent（CoT Reasoner）。
功能：给定题目 + 学生错误选项，生成"为什么学生会犯这个错误"的推理链。
生成的 CoT 注入给精排器（listwise），提升重排精度（复刻第1名方案）。
同时用于合成数据生成（蒸馏教师轨迹）。

设计：
- 本地模型：Qwen2.5-3B-Instruct（快）
- 结果缓存到 SQLite，避免重复推理
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


REASONER_SYSTEM = (
    "You are an expert mathematics educator. "
    "Analyze why a student gave an incorrect answer to a math question. "
    "Explain the likely reasoning flaw in 3-5 concise sentences. "
    "Focus on the specific mathematical misconception, not general advice."
)

REASONER_USER_TEMPLATE = (
    "Question: {question}\n"
    "Correct Answer: {correct_answer}\n"
    "Student's Incorrect Answer: {wrong_answer}\n\n"
    "Analyze the student's likely reasoning error:"
)


class CoTReasoner:
    """
    生成 CoT 错因推理链，并缓存到 SQLite。

    示例：
        reasoner = CoTReasoner.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
        rationale = reasoner.generate(
            question="What is 5 × 4 + 6 ÷ 2?",
            correct_answer="23",
            wrong_answer="13",
        )
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        device: str = "cuda",
        cache_db: Optional[str] = None,
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self._init_cache(cache_db)

    def _init_cache(self, db_path: Optional[str]) -> None:
        if db_path is None:
            self._conn = None
            return
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cot_cache "
            "(key TEXT PRIMARY KEY, rationale TEXT)"
        )
        self._conn.commit()

    def _cache_key(self, question: str, correct: str, wrong: str) -> str:
        s = f"{question}|{correct}|{wrong}"
        return hashlib.md5(s.encode()).hexdigest()

    def _get_cache(self, key: str) -> Optional[str]:
        if self._conn is None:
            return None
        row = self._conn.execute(
            "SELECT rationale FROM cot_cache WHERE key=?", (key,)
        ).fetchone()
        return row[0] if row else None

    def _set_cache(self, key: str, rationale: str) -> None:
        if self._conn is None:
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO cot_cache (key, rationale) VALUES (?, ?)",
            (key, rationale),
        )
        self._conn.commit()

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        adapter_path: Optional[str | Path] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
        cache_db: Optional[str] = None,
    ) -> "CoTReasoner":
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
        return cls(model, tokenizer, max_new_tokens, temperature, device, cache_db)

    @torch.no_grad()
    def generate(
        self,
        question: str,
        correct_answer: str,
        wrong_answer: str,
        use_cache: bool = True,
    ) -> str:
        """生成单条推理链。"""
        key = self._cache_key(question, correct_answer, wrong_answer)
        if use_cache:
            cached = self._get_cache(key)
            if cached:
                return cached

        user_text = REASONER_USER_TEMPLATE.format(
            question=question,
            correct_answer=correct_answer,
            wrong_answer=wrong_answer,
        )
        messages = [
            {"role": "system", "content": REASONER_SYSTEM},
            {"role": "user", "content": user_text},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        rationale = self.tokenizer.decode(
            out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

        if use_cache:
            self._set_cache(key, rationale)
        return rationale

    def batch_generate(
        self,
        items: list[dict],  # list of {question, correct_answer, wrong_answer, qa_key}
        use_cache: bool = True,
    ) -> dict[str, str]:
        """批量生成，返回 {qa_key: rationale}。"""
        results: dict[str, str] = {}
        for item in items:
            rationale = self.generate(
                item["question"],
                item["correct_answer"],
                item["wrong_answer"],
                use_cache=use_cache,
            )
            results[item["qa_key"]] = rationale
        return results

    async def async_generate(
        self,
        question: str,
        correct_answer: str,
        wrong_answer: str,
        use_cache: bool = True,
    ) -> str:
        """异步包装（在 FastAPI 中使用 run_in_executor）。"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate(question, correct_answer, wrong_answer, use_cache),
        )
