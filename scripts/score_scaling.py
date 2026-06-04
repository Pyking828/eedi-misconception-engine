"""Unseen-misconception score scaling sweep (3rd-place trick), CPU-only.

Loads saved reranker scores for the val split, down-weights the scores of
misconceptions that ARE present in the training folds (seen), so that unseen
misconceptions rank higher. Sweeps the scaling factor and reports MAP@25.

This is a post-processing lever; reuses outputs/reranker/scores_<tag>_fold{f}_val.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
from eval.evaluator import EediEvaluator
from rich.console import Console

console = Console()
DATA_DIR = Path(os.environ.get("EEDI_DATA", "/root/autodl-tmp/eedi-data"))
OUT = Path("/root/autodl-tmp/eedi-misconception-engine/outputs/reranker")


def seen_ids_for_fold(fold: int) -> set[int]:
    """Misconceptions present in the TRAIN folds (fold != given)."""
    long_df = pl.read_parquet(DATA_DIR / "folds.parquet").filter(pl.col("MisconceptionId") >= 0)
    train = long_df.filter(pl.col("fold") != fold)
    return set(int(x) for x in train["MisconceptionId"].unique().to_list())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(OUT / "scores_best31k_fold0_val.json"))
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    scores = json.loads(Path(args.scores).read_text())
    seen = seen_ids_for_fold(args.fold)
    console.rule("[bold blue]Unseen score-scaling sweep")
    console.print(f"[cyan]val queries={len(scores)}  seen misconceptions in train={len(seen)}")

    # how many val golds are unseen?
    n_unseen_gold = sum(1 for v in scores.values() if v["true_id"] not in seen)
    console.print(f"[cyan]val gold that are UNSEEN in train: {n_unseen_gold}/{len(scores)}")

    def eval_factor(f: float) -> dict:
        ev = EediEvaluator(k=25)
        for v in scores.values():
            ids, sc = v["ids"], v["scores"]
            adj = [(i, s * (f if i in seen else 1.0)) for i, s in zip(ids, sc)]
            ranked = [i for i, _ in sorted(adj, key=lambda x: -x[1])]
            ev.update(ranked, v["true_id"])
        m = ev.compute()
        return m

    base = eval_factor(1.0)
    console.print(
        f"[bold]baseline (f=1.0): MAP@25={base['MAP@25']:.4f} nDCG@25={base['nDCG@25']:.4f}"
    )

    best_f, best_map = 1.0, base["MAP@25"]
    results = {"baseline": base}
    for f in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        m = eval_factor(f)
        results[f"f={f}"] = m
        flag = " <== best" if m["MAP@25"] > best_map else ""
        if m["MAP@25"] > best_map:
            best_map, best_f = m["MAP@25"], f
        console.print(f"  seen×{f}: MAP@25={m['MAP@25']:.4f}  Recall@25={m['Recall@25']:.4f}{flag}")

    console.print(
        f"[bold green]best factor={best_f}  MAP@25={best_map:.4f}  (baseline {base['MAP@25']:.4f}, gain {best_map - base['MAP@25']:+.4f})"
    )
    (OUT / "score_scaling_result.json").write_text(
        json.dumps(
            {
                "best_factor": best_f,
                "best_map": best_map,
                "baseline_map": base["MAP@25"],
                "sweep": {k: v.get("MAP@25") for k, v in results.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
