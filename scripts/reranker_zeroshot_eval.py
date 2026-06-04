"""Stage 3: Qwen3-Reranker-8B zero-shot reranking evaluation.

Input:
  - folds.parquet
  - misconception_mapping.csv
  - outputs/retriever/candidate_pool.json

For each validation sample in a fold:
  1. Take top-N candidates from retriever candidate_pool.
  2. Score (query, candidate_misconception) pairs with Qwen3-Reranker-8B.
  3. Rerank candidates and compute MAP@25 / Recall@25 / nDCG@25.

This script uses sentence-transformers CrossEncoder, which is the recommended
interface in the Qwen3-Reranker model card and avoids FlagEmbedding /
Transformers tokenizer compatibility issues.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")

import polars as pl
from eval.evaluator import EediEvaluator
from rich.console import Console
from tqdm import tqdm

console = Console()
DATA_DIR = Path(os.environ.get("EEDI_DATA", "/root/autodl-tmp/eedi-data"))
HF_CACHE = os.environ.get("HF_HOME", "/root/autodl-tmp/hf_cache")
OUTPUT_DIR = Path("/root/autodl-tmp/eedi-misconception-engine/outputs/reranker")
RETRIEVER_OUT = Path("/root/autodl-tmp/eedi-misconception-engine/outputs/retriever")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_reranker(model_name: str, batch_size: int, max_length: int):
    """Load Qwen3-Reranker through sentence-transformers CrossEncoder."""
    import torch
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        model_name,
        device="cuda",
        cache_folder=HF_CACHE,
        trust_remote_code=True,
        max_length=max_length,
        model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"},
    )


def compute_scores(reranker, pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
    """Compute reranker scores in batches."""
    scores: list[float] = []
    for i in tqdm(range(0, len(pairs), batch_size), desc="Reranking batches"):
        batch = pairs[i : i + batch_size]
        # Qwen3-Reranker CrossEncoder returns raw logit differences by default.
        batch_scores = reranker.predict(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores.extend(float(s) for s in batch_scores)
    return scores


def evaluate_fold(
    reranker,
    long_df: pl.DataFrame,
    misc_texts: dict[int, str],
    candidate_pool: dict[str, list[int]],
    fold: int,
    top_input: int = 50,
    top_eval: int = 25,
    batch_size: int = 8,
) -> dict:
    val = long_df.filter(pl.col("fold") == fold).filter(pl.col("MisconceptionId") >= 0)

    pairs: list[tuple[str, str]] = []
    meta: list[tuple[str, int, list[int]]] = []
    for row in val.iter_rows(named=True):
        qa_key = row["QuestionId_Answer"]
        true_id = int(row["MisconceptionId"])
        cands = candidate_pool.get(qa_key, [])[:top_input]
        if not cands:
            continue
        meta.append((qa_key, true_id, cands))
        query = row["AllText"]
        for cid in cands:
            pairs.append((query, misc_texts.get(int(cid), "")))

    scores = compute_scores(reranker, pairs, batch_size=batch_size)

    evaluator = EediEvaluator(k=top_eval)
    ptr = 0
    predictions = {}
    for qa_key, true_id, cands in meta:
        n = len(cands)
        cand_scores = scores[ptr : ptr + n]
        ptr += n
        ranked = [cid for cid, _ in sorted(zip(cands, cand_scores), key=lambda x: -x[1])]
        evaluator.update(ranked, true_id)
        predictions[qa_key] = ranked

    result = evaluator.compute()
    return {
        "fold": fold,
        "n_val": len(meta),
        "MAP@25": round(result["MAP@25"], 4),
        "Recall@25": round(result["Recall@25"], 4),
        "nDCG@25": round(result["nDCG@25"], 4),
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-Reranker-8B")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--top-input", type=int, default=50)
    parser.add_argument("--top-eval", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()

    console.rule("[bold blue]Stage 3: Qwen3-Reranker zero-shot eval")
    console.print(f"[cyan]model={args.model}, fold={args.fold}, top_input={args.top_input}")

    long_df = pl.read_parquet(DATA_DIR / "folds.parquet")
    misc_df = pl.read_csv(DATA_DIR / "misconception_mapping.csv")
    misc_texts = {
        int(r["MisconceptionId"]): r["MisconceptionName"] for r in misc_df.iter_rows(named=True)
    }
    with open(RETRIEVER_OUT / "candidate_pool.json") as f:
        candidate_pool = {k: [int(x) for x in v] for k, v in json.load(f).items()}

    reranker = load_reranker(args.model, args.batch_size, args.max_length)
    result = evaluate_fold(
        reranker=reranker,
        long_df=long_df,
        misc_texts=misc_texts,
        candidate_pool=candidate_pool,
        fold=args.fold,
        top_input=args.top_input,
        top_eval=args.top_eval,
        batch_size=args.batch_size,
    )

    pred_path = OUTPUT_DIR / f"zeroshot_predictions_fold{args.fold}.json"
    with open(pred_path, "w") as f:
        json.dump(result.pop("predictions"), f)

    out_path = OUTPUT_DIR / f"zeroshot_metrics_fold{args.fold}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    console.print(f"[bold green]Fold {args.fold}: {result}")
    console.print(f"[green]saved: {out_path}")
    console.rule("[bold green]done")


if __name__ == "__main__":
    main()
