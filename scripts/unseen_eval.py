"""Unseen-misconception evaluation — synth/multistage value on held-out misconceptions.

Fold0 val is mostly seen misconceptions, hiding synth/multistage gains (test set is unseen-heavy).
Splits fold0 val into seen vs unseen by whether gold appeared in fold0-train, then evaluates:
  A) three retrievers (real-only baseline / multistage 4694 synth / multistage 15205 synth) Recall/MAP;
  B) best pointwise reranker and ensemble MAP@25 on both splits.
Expect multistage to win on unseen even when overall CV is flat.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")

import polars as pl
from rich.console import Console
from rich.table import Table

ROOT = Path("/root/autodl-tmp/eedi-misconception-engine")
sys.path.insert(0, str(ROOT))
from eval.evaluator import EediEvaluator
from src.eedi.retriever.st_engine import build_faiss_index, load_st_model, st_encode

console = Console()
DATA_DIR = Path("/root/autodl-tmp/eedi-data")
HF_CACHE = os.environ["HF_HOME"]
RET_OUT = ROOT / "outputs/retriever"
RR_OUT = ROOT / "outputs/reranker"


def split_seen_unseen():
    df = pl.read_parquet(DATA_DIR / "folds.parquet").filter(pl.col("MisconceptionId") >= 0)
    train_mis = set(df.filter(pl.col("fold") != 0)["MisconceptionId"].to_list())
    val = df.filter(pl.col("fold") == 0)
    seen_qa, unseen_qa = set(), set()
    for r in val.iter_rows(named=True):
        (seen_qa if int(r["MisconceptionId"]) in train_mis else unseen_qa).add(
            r["QuestionId_Answer"]
        )
    return val, train_mis, seen_qa, unseen_qa


def eval_retriever(adapter, val, misc_texts, seen_qa, unseen_qa):
    ids = sorted(misc_texts.keys())
    model = load_st_model("Qwen/Qwen3-Embedding-8B", adapter_path=adapter, cache_dir=HF_CACHE)
    model.max_seq_length = 512
    membs = st_encode(model, [misc_texts[i] for i in ids], prompt_name="document", batch_size=128)
    index = build_faiss_index(membs)
    q = st_encode(model, val["AllText"].to_list(), prompt_name="query", batch_size=64)
    _, idx = index.search(q, 50)
    del model
    import torch

    torch.cuda.empty_cache()
    out = {}
    for name, qaset in [("seen", seen_qa), ("unseen", unseen_qa), ("all", None)]:
        ev = EediEvaluator(k=25)
        r50 = r25 = n = 0
        for i, row in enumerate(val.iter_rows(named=True)):
            if qaset is not None and row["QuestionId_Answer"] not in qaset:
                continue
            tid = int(row["MisconceptionId"])
            ranked = [ids[j] for j in idx[i]]
            ev.update(ranked[:25], tid)
            r25 += int(tid in ranked[:25])
            r50 += int(tid in ranked[:50])
            n += 1
        m = ev.compute()
        out[name] = {
            "MAP@25": round(m["MAP@25"], 4),
            "R@25": round(r25 / n, 4),
            "R@50": round(r50 / n, 4),
            "n": n,
        }
    return out


def eval_scores_split(scores_path, seen_qa, unseen_qa):
    sc = json.loads(Path(scores_path).read_text())
    out = {}
    for name, qaset in [("seen", seen_qa), ("unseen", unseen_qa), ("all", None)]:
        ev = EediEvaluator(k=25)
        for qa, r in sc.items():
            if qaset is not None and qa not in qaset:
                continue
            ev.update(r["ids"], int(r["true_id"]))
        out[name] = round(ev.compute()["MAP@25"], 4)
    return out


def main():
    console.rule("[bold blue]Unseen-misconception evaluation (seen vs unseen 拆分)")
    val, train_mis, seen_qa, unseen_qa = split_seen_unseen()
    misc_df = pl.read_csv(DATA_DIR / "misconception_mapping.csv")
    misc_texts = {
        int(r["MisconceptionId"]): r["MisconceptionName"] for r in misc_df.iter_rows(named=True)
    }
    console.print(
        f"[cyan]val={len(val)}  seen={len(seen_qa)}  unseen={len(unseen_qa)} ({100*len(unseen_qa)/len(val):.1f}%)"
    )

    results = {
        "split": {"val": len(val), "seen": len(seen_qa), "unseen": len(unseen_qa)},
        "retriever": {},
        "reranker": {},
    }

    retrievers = {
        "baseline(real-only)": RET_OUT / "lora_best_8b",
        "multistage(4694合成)": RET_OUT / "lora_best_8b_multistage",
        "multistage(15205合成)": RET_OUT / "lora_best_8b_multistage_10k",
    }
    tbl = Table(title="召回器 × seen/unseen")
    for c in ["retriever", "subset", "MAP@25", "Recall@25", "Recall@50", "n"]:
        tbl.add_column(c)
    for name, adp in retrievers.items():
        if not Path(adp).exists():
            console.print(f"[yellow]skip {name} (no adapter)")
            continue
        r = eval_retriever(str(adp), val, misc_texts, seen_qa, unseen_qa)
        results["retriever"][name] = r
        for sub in ["all", "seen", "unseen"]:
            tbl.add_row(
                name if sub == "all" else "",
                sub,
                f"{r[sub]['MAP@25']}",
                f"{r[sub]['R@25']}",
                f"{r[sub]['R@50']}",
                str(r[sub]["n"]),
            )
    console.print(tbl)

    # reranker / ensemble splits
    for tag, path in [
        ("pointwise best31k", RR_OUT / "scores_best31k_multistage_pool_fold0_val.json")
    ]:
        if Path(path).exists():
            results["reranker"][tag] = eval_scores_split(path, seen_qa, unseen_qa)
    tbl2 = Table(title="reranker × seen/unseen (MAP@25)")
    for c in ["method", "all", "seen", "unseen"]:
        tbl2.add_column(c)
    for tag, r in results["reranker"].items():
        tbl2.add_row(tag, str(r["all"]), str(r["seen"]), str(r["unseen"]))
    console.print(tbl2)

    (RR_OUT / "unseen_eval.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    console.print(f"[green]saved {RR_OUT / 'unseen_eval.json'}")


if __name__ == "__main__":
    main()
