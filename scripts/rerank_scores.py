"""Stage 3 infra: run the best pointwise reranker and SAVE per-query ranked
candidates + scores. Reusable for: listwise input (top-K), score ensembling,
unseen-misconception score scaling.

Output: outputs/reranker/scores_<tag>_fold{fold}_{split}.json
  { qa_key: {"ids": [...top50 ranked...], "scores": [...], "true_id": int} }
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import polars as pl
import torch
from eval.evaluator import EediEvaluator
from peft import PeftModel
from rich.console import Console
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()
DATA_DIR = Path(os.environ.get("EEDI_DATA", "/root/autodl-tmp/eedi-data"))
HF_CACHE = os.environ.get("HF_HOME", "/root/autodl-tmp/hf_cache")
ROOT = Path("/root/autodl-tmp/eedi-misconception-engine")
RET_OUT = ROOT / "outputs/retriever"
OUT = ROOT / "outputs/reranker"
OUT.mkdir(parents=True, exist_ok=True)

INSTRUCTION = (
    "Given a mathematics question, the correct answer, and a student's incorrect answer, "
    "judge whether the document is the misconception that best explains why the student made the error."
)
PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def base_model_path() -> str:
    return glob.glob(f"{HF_CACHE}/models--Qwen--Qwen3-Reranker-8B/snapshots/*/")[0]


def fmt(query: str, doc: str) -> str:
    return f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {doc}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--adapter", default=str(OUT / "manual_lora/manual_lora_fold0_n31464_bs4_hn8_len768")
    )
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--split", choices=["val", "train", "all"], default="val")
    ap.add_argument("--top-input", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=768)
    ap.add_argument("--tag", default="best31k")
    args = ap.parse_args()

    console.rule(f"[bold blue]Reranker scoring ({args.tag}, fold{args.fold}, {args.split})")
    long_df = pl.read_parquet(DATA_DIR / "folds.parquet").filter(pl.col("MisconceptionId") >= 0)
    misc_df = pl.read_csv(DATA_DIR / "misconception_mapping.csv")
    misc_texts = {
        int(r["MisconceptionId"]): r["MisconceptionName"] for r in misc_df.iter_rows(named=True)
    }
    candidate_pool = {
        k: [int(x) for x in v]
        for k, v in json.loads((RET_OUT / "candidate_pool.json").read_text()).items()
    }

    if args.split == "val":
        df = long_df.filter(pl.col("fold") == args.fold)
    elif args.split == "train":
        df = long_df.filter(pl.col("fold") != args.fold)
    else:
        df = long_df

    tok = AutoTokenizer.from_pretrained(args.adapter, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path(), torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().to("cuda").eval()
    yes_id = tok.convert_tokens_to_ids("yes")
    no_id = tok.convert_tokens_to_ids("no")

    prefix_tokens = tok.encode(PREFIX, add_special_tokens=False)
    suffix_tokens = tok.encode(SUFFIX, add_special_tokens=False)
    max_pair = args.max_length - len(prefix_tokens) - len(suffix_tokens)

    @torch.no_grad()
    def score(texts: list[str]) -> list[float]:
        out_scores: list[float] = []
        for i in range(0, len(texts), args.batch_size):
            chunk = texts[i : i + args.batch_size]
            t = tok(
                chunk,
                padding=False,
                truncation="longest_first",
                max_length=max_pair,
                add_special_tokens=False,
                return_attention_mask=False,
            )
            ids = [prefix_tokens + x + suffix_tokens for x in t["input_ids"]]
            enc = tok.pad({"input_ids": ids}, padding=True, return_tensors="pt").to("cuda")
            logits = model(**enc).logits[:, -1, :]
            pair = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1)
            probs = torch.softmax(pair, dim=1)[:, 1]
            out_scores.extend(float(x) for x in probs.cpu())
        return out_scores

    results = {}
    evaluator = EediEvaluator(k=25)
    for row in tqdm(df.iter_rows(named=True), total=len(df), desc="scoring"):
        qa = row["QuestionId_Answer"]
        true_id = int(row["MisconceptionId"])
        cands = candidate_pool.get(qa, [])[: args.top_input]
        if not cands:
            continue
        texts = [fmt(row["AllText"], misc_texts.get(c, "")) for c in cands]
        sc = score(texts)
        order = sorted(range(len(cands)), key=lambda j: -sc[j])
        ranked = [cands[j] for j in order]
        results[qa] = {
            "ids": ranked,
            "scores": [round(sc[j], 6) for j in order],
            "true_id": true_id,
        }
        evaluator.update(ranked, true_id)

    out_path = OUT / f"scores_{args.tag}_fold{args.fold}_{args.split}.json"
    out_path.write_text(json.dumps(results), encoding="utf-8")
    m = evaluator.compute()
    console.print(
        f"[bold green]{args.split} MAP@25={m['MAP@25']:.4f} Recall@25={m['Recall@25']:.4f} n={m['n_samples']}"
    )
    console.print(f"[green]saved scores: {out_path}")


if __name__ == "__main__":
    main()
