"""Reranker score ensemble (CPU-only). Combines per-query score files from script 11.

Two fusion modes:
  - weighted: per-query min-max normalize each model's scores, weighted sum, re-rank.
  - rrf: Reciprocal Rank Fusion (scale-robust): score = sum_m w_m / (k_rrf + rank_m).

Sweeps weight grids over the provided score files and reports the best MAP@25 vs the
single best member. Score files already embed true_id (from script 11).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.evaluator import EediEvaluator
from rich.console import Console
from rich.table import Table

console = Console()
OUT = Path("/root/autodl-tmp/eedi-misconception-engine/outputs/reranker")


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def minmax(vals: list[float]) -> list[float]:
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return [0.5] * len(vals)
    return [(v - lo) / (hi - lo) for v in vals]


def eval_single(d: dict) -> float:
    ev = EediEvaluator(k=25)
    for qa, r in d.items():
        ev.update(r["ids"], int(r["true_id"]))
    return ev.compute()["MAP@25"]


def fuse(models: list[dict], weights: list[float], mode: str, k_rrf: int = 60) -> float:
    ev = EediEvaluator(k=25)
    keys = set(models[0].keys())
    for m in models[1:]:
        keys &= set(m.keys())
    for qa in keys:
        agg: dict[int, float] = {}
        for w, m in zip(weights, models):
            r = m[qa]
            ids = r["ids"]
            if mode == "weighted":
                ns = minmax(r["scores"])
                for i, mid in enumerate(ids):
                    agg[mid] = agg.get(mid, 0.0) + w * ns[i]
            else:  # rrf
                for rank, mid in enumerate(ids):
                    agg[mid] = agg.get(mid, 0.0) + w / (k_rrf + rank)
        ranked = sorted(agg, key=lambda x: -agg[x])[:25]
        ev.update(ranked, int(models[0][qa]["true_id"]))
    return ev.compute()["MAP@25"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scores", nargs="+", required=True, help="score json files (script 11 output)"
    )
    ap.add_argument("--mode", choices=["weighted", "rrf", "both"], default="both")
    ap.add_argument("--grid", default="0,0.25,0.5,0.75,1.0", help="weight grid per model")
    args = ap.parse_args()

    models = [load(p) for p in args.scores]
    names = [Path(p).stem.replace("scores_", "").replace("_fold0_val", "") for p in args.scores]
    console.rule("[bold blue]Reranker ensemble")

    tbl = Table(title="single-model MAP@25")
    tbl.add_column("model")
    tbl.add_column("MAP@25", justify="right")
    singles = {}
    for n, d in zip(names, models):
        s = eval_single(d)
        singles[n] = s
        tbl.add_row(n, f"{s:.4f}")
    console.print(tbl)
    best_single = max(singles.values())

    grid = [float(x) for x in args.grid.split(",")]
    modes = ["weighted", "rrf"] if args.mode == "both" else [args.mode]
    best = {"map": 0.0}
    for mode in modes:
        for combo in itertools.product(grid, repeat=len(models)):
            if sum(combo) == 0:
                continue
            m = fuse(models, list(combo), mode)
            if m > best["map"]:
                best = {"map": m, "mode": mode, "weights": list(combo)}
    console.print(
        f"[bold green]BEST ensemble MAP@25={best['map']:.4f} (mode={best['mode']}, weights={dict(zip(names, best['weights']))})"
    )
    gain = best["map"] - best_single
    console.print(
        f"[{'green' if gain > 0 else 'yellow'}]vs best single ({best_single:.4f}): {gain:+.4f}"
    )
    (OUT / "ensemble_result.json").write_text(
        json.dumps(
            {
                "singles": singles,
                "best": best,
                "best_single": best_single,
                "gain_vs_single": round(gain, 4),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    console.print(f"[green]saved {OUT / 'ensemble_result.json'}")


if __name__ == "__main__":
    main()
