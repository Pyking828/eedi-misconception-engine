"""Stage 3 (listwise): R1-Distill-Qwen-14B reranks the pointwise reranker's top-K.

- Input per val query: top-K candidates from saved pointwise scores
  (outputs/reranker/scores_<tag>_fold{f}_val.json).
- Prompt: question + correct/wrong + optional CoT + lettered candidate list.
- Position-bias mitigation: rank in ORIGINAL and REVERSED candidate order, then
  fuse by averaging rank positions (RankGPT-style robustness).
- Final top-25 = listwise-ordered top-K, then remaining pointwise ids appended.

Backend: transformers (vLLM FlashInfer unusable on sm_120). Zero-shot by default
(no training) — a cheap probe to see whether listwise beats pointwise 0.57.
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
from eval.evaluator import EediEvaluator
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()
DATA_DIR = Path(os.environ.get("EEDI_DATA", "/root/autodl-tmp/eedi-data"))
HF_CACHE = os.environ.get("HF_HOME", "/root/autodl-tmp/hf_cache")
ROOT = Path("/root/autodl-tmp/eedi-misconception-engine")
OUT = ROOT / "outputs/reranker"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def model_path() -> str:
    return glob.glob(f"{HF_CACHE}/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-14B/snapshots/*/")[
        0
    ]


def build_prompt(tok, question: str, correct: str, wrong: str, cands: list[str], cot: str) -> str:
    lst = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(cands))
    cot_block = f"\nAnalysis of the student's error: {cot}\n" if cot else ""
    user = (
        f"A student answered a maths question incorrectly. Rank the candidate misconceptions "
        f"from MOST to LEAST likely to explain the student's specific error.\n\n"
        f"Question: {question}\nCorrect answer: {correct}\nStudent's wrong answer: {wrong}\n"
        f"{cot_block}\nCandidate misconceptions:\n{lst}\n\n"
        f"Output ONLY the candidate letters from most to least likely, comma-separated "
        f"(e.g. {LETTERS[0]}, {LETTERS[2]}, {LETTERS[1]}, ...). Use each letter exactly once."
    )
    msgs = [{"role": "user", "content": user}]
    return (
        tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        + "<think>\n\n</think>\n\n"
    )


def parse_order(text: str, n: int) -> list[int]:
    seen, order = set(), []
    for ch in re.findall(r"[A-Z]", text.upper()):
        idx = ord(ch) - 65
        if 0 <= idx < n and idx not in seen:
            seen.add(idx)
            order.append(idx)
    for i in range(n):
        if i not in seen:
            order.append(i)
    return order


@torch.no_grad()
def gen(model, tok, prompts: list[str], max_new_tokens: int, batch_size: int) -> list[str]:
    outs = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=1536).to(
            "cuda"
        )
        g = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id
        )
        outs.extend(tok.batch_decode(g[:, enc["input_ids"].shape[1] :], skip_special_tokens=True))
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(OUT / "scores_best31k_fold0_val.json"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=10, help="candidates fed to listwise")
    ap.add_argument("--cot", default=str(ROOT / "outputs/cot/cot_r1-14b.json"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--bidir", type=int, default=1, help="1=fuse original+reversed order")
    args = ap.parse_args()

    console.rule(f"[bold blue]Listwise rerank (R1-14B, top{args.top_k}, bidir={args.bidir})")
    scores = json.loads(Path(args.scores).read_text())
    misc_df = pl.read_csv(DATA_DIR / "misconception_mapping.csv")
    misc_texts = {
        int(r["MisconceptionId"]): r["MisconceptionName"] for r in misc_df.iter_rows(named=True)
    }
    long_df = pl.read_parquet(DATA_DIR / "folds.parquet").filter(pl.col("MisconceptionId") >= 0)
    meta = {r["QuestionId_Answer"]: r for r in long_df.iter_rows(named=True)}
    cot_map = json.loads(Path(args.cot).read_text()) if Path(args.cot).exists() else {}

    tok = AutoTokenizer.from_pretrained(model_path(), padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path(), torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda"
    ).eval()

    qa_keys = list(scores.keys())
    # Build prompts (original + optional reversed)
    prompts, index = [], []  # index: (qa_key, "fwd"/"rev", cand_ids_in_prompt_order)
    for qa in qa_keys:
        topk_ids = scores[qa]["ids"][: args.top_k]
        row = meta[qa]
        cot = cot_map.get(qa, "")
        for tag, ids in (("fwd", topk_ids), ("rev", topk_ids[::-1])):
            cands = [misc_texts.get(i, "") for i in ids]
            prompts.append(
                build_prompt(
                    tok,
                    row["QuestionText"],
                    row["CorrectAnswerText"],
                    row["WrongAnswerText"],
                    cands,
                    cot,
                )
            )
            index.append((qa, tag, ids))
            if not args.bidir:
                break

    t0 = time.time()
    gens = gen(model, tok, prompts, args.max_new_tokens, args.batch_size)
    console.print(f"[cyan]generation {time.time() - t0:.0f}s for {len(prompts)} prompts")

    # Fuse rank positions per qa
    rank_acc: dict[str, dict[int, list[float]]] = {qa: {} for qa in qa_keys}
    for (qa, tag, ids), text in zip(index, gens):
        order = parse_order(text, len(ids))  # positions into ids
        for pos, local_idx in enumerate(order):
            mid = ids[local_idx]
            rank_acc[qa].setdefault(mid, []).append(pos)

    evaluator = EediEvaluator(k=25)
    for qa in qa_keys:
        topk_ids = scores[qa]["ids"][: args.top_k]
        rest = scores[qa]["ids"][args.top_k :]
        fused = sorted(
            topk_ids, key=lambda m: sum(rank_acc[qa].get(m, [99])) / len(rank_acc[qa].get(m, [99]))
        )
        final = fused + rest
        evaluator.update(final, scores[qa]["true_id"])

    m = evaluator.compute()
    res = {
        "top_k": args.top_k,
        "bidir": args.bidir,
        "MAP@25": round(m["MAP@25"], 4),
        "Recall@25": round(m["Recall@25"], 4),
        "nDCG@25": round(m["nDCG@25"], 4),
        "n": m["n_samples"],
        "gen_seconds": round(time.time() - t0, 1),
    }
    out_path = OUT / f"listwise_zeroshot_fold{args.fold}_top{args.top_k}.json"
    out_path.write_text(json.dumps(res, indent=2))
    console.print(f"[bold green]{res}")
    console.print(f"[green]saved {out_path}")


if __name__ == "__main__":
    main()
