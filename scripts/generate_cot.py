"""Stage 3.5: Generate CoT reasoning traces for (question, correct, wrong) using R1-Distill.

Backend: plain transformers batched generation (robust on Blackwell sm_120; vLLM
FlashInfer is currently incompatible). The generated rationale ("why a student
would pick this wrong answer") is later prepended to reranker / listwise inputs,
following the 1st place solution's biggest lever (CoT-augmented reranking).

Output: JSON {qa_key: rationale} cached at outputs/cot/cot_<model>.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import polars as pl
import torch
from rich.console import Console
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()
DATA_DIR = Path(os.environ.get("EEDI_DATA", "/root/autodl-tmp/eedi-data"))
HF_CACHE = os.environ.get("HF_HOME", "/root/autodl-tmp/hf_cache")
OUTPUT_DIR = Path("/root/autodl-tmp/eedi-misconception-engine/outputs/cot")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIRS = {
    "r1-14b": "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-14B",
    "r1-32b": "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B",
}

PROMPT_TMPL = (
    "A student was asked a maths question and chose a wrong answer.\n"
    "Question: {question}\n"
    "Correct answer: {correct}\n"
    "Student's wrong answer: {wrong}\n\n"
    "In 1-2 sentences, explain the specific mathematical misconception or faulty "
    "reasoning that leads to this wrong answer. Be concise and focus only on the misconception."
)


def local_path(model_key: str) -> str:
    pattern = f"{HF_CACHE}/{MODEL_DIRS[model_key]}/snapshots/*/"
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"model snapshot not found: {pattern}")
    return paths[0]


def clean_rationale(text: str) -> str:
    # R1 emits <think> ... </think> then the answer; keep the post-think content.
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = text.replace("<think>", "").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:600]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="r1-14b", choices=list(MODEL_DIRS))
    parser.add_argument("--limit", type=int, default=0, help="0 = all labeled rows")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    args = parser.parse_args()

    console.rule(f"[bold blue]CoT generation ({args.model}, transformers backend)")
    model_path = local_path(args.model)
    console.print(f"[cyan]model path: {model_path}")

    long_df = pl.read_parquet(DATA_DIR / "folds.parquet").filter(pl.col("MisconceptionId") >= 0)
    rows = long_df.select(
        ["QuestionId_Answer", "QuestionText", "CorrectAnswerText", "WrongAnswerText"]
    ).to_dicts()
    if args.limit:
        rows = rows[: args.limit]
    console.print(f"[cyan]rows to generate: {len(rows)}")

    out_path = OUTPUT_DIR / f"cot_{args.model}.json"
    cache: dict[str, str] = {}
    if out_path.exists():
        cache = json.loads(out_path.read_text())
        console.print(f"[cyan]resuming from cache: {len(cache)} done")

    todo = [r for r in rows if r["QuestionId_Answer"] not in cache]
    console.print(f"[cyan]remaining: {len(todo)}")
    if not todo:
        console.print("[green]all cached, nothing to do")
        return

    tok = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda"
    ).eval()

    t0 = time.time()
    for i in tqdm(range(0, len(todo), args.batch_size), desc="CoT batches"):
        batch = todo[i : i + args.batch_size]
        prompts = []
        for r in batch:
            user = PROMPT_TMPL.format(
                question=(r["QuestionText"] or "")[:600],
                correct=(r["CorrectAnswerText"] or "")[:200],
                wrong=(r["WrongAnswerText"] or "")[:200],
            )
            msgs = [{"role": "user", "content": user}]
            base = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            # Force-close the <think> block so R1-Distill answers concisely without long reasoning.
            prompts.append(base + "<think>\n\n</think>\n\n")
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(
            "cuda"
        )
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=tok.eos_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1] :]
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        for r, t in zip(batch, texts):
            cache[r["QuestionId_Answer"]] = clean_rationale(t)
        if (i // args.batch_size) % 10 == 0:
            out_path.write_text(json.dumps(cache, ensure_ascii=False))

    out_path.write_text(json.dumps(cache, ensure_ascii=False))
    dt = time.time() - t0
    console.print(f"[green]done: {len(cache)} rationales, {dt:.0f}s, saved {out_path}")
    # show a couple samples
    for r in todo[:2]:
        console.print(
            f"[bold]{r['QuestionId_Answer']}[/bold]: {cache[r['QuestionId_Answer']][:200]}"
        )


if __name__ == "__main__":
    main()
