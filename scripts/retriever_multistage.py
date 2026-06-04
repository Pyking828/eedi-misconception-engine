"""Stage 2 payoff: multi-stage retriever training (synthetic pretrain -> real finetune).

Replicates the top-solutions recipe: first adapt the embedding model on a large
synthetic MCQ set (covering unseen/rare misconceptions), then finetune on the real
Eedi training fold. Evaluates fold0 val MAP@25/Recall and compares to the real-only
8B LoRA baseline (0.4012 / 0.8822).

Engine: sentence-transformers + PEFT LoRA + MultipleNegativesRankingLoss (same as script 01).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")

import polars as pl
from eval.evaluator import EediEvaluator
from rich.console import Console
from src.eedi.retriever.st_engine import build_faiss_index, load_st_model, st_encode

console = Console()
DATA_DIR = Path(os.environ.get("EEDI_DATA", "/root/autodl-tmp/eedi-data"))
HF_CACHE = os.environ.get("HF_HOME", "/root/autodl-tmp/hf_cache")
OUT = Path("/root/autodl-tmp/eedi-misconception-engine/outputs/retriever")
OUT.mkdir(parents=True, exist_ok=True)


def all_text(subj, cons, q, corr, wrong):
    return (
        f"Subject: {subj}\nTopic: {cons}\nQuestion: {q}\n"
        f"Correct Answer: {corr}\nIncorrect Answer: {wrong}"
    )


def load_misc():
    misc_df = pl.read_csv(DATA_DIR / "misconception_mapping.csv")
    return {
        int(r["MisconceptionId"]): r["MisconceptionName"] for r in misc_df.iter_rows(named=True)
    }


def real_pairs(long_df, misc_texts, fold):
    tr = long_df.filter(pl.col("fold") != fold).filter(pl.col("MisconceptionId") >= 0)
    return (
        [r["AllText"] for r in tr.iter_rows(named=True)],
        [misc_texts[int(r["MisconceptionId"])] for r in tr.iter_rows(named=True)],
    )


def synth_pairs(synth_path, misc_texts):
    anchors, positives = [], []
    if not Path(synth_path).exists():
        return anchors, positives
    for line in Path(synth_path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        mid = int(d.get("MisconceptionId", -1))
        if mid not in misc_texts:
            continue
        anchors.append(
            all_text(
                d.get("SubjectName", ""),
                d.get("ConstructName", ""),
                d.get("QuestionText", ""),
                d.get("CorrectAnswerText", ""),
                d.get("WrongAnswerText", ""),
            )
        )
        positives.append(misc_texts[mid])
    return anchors, positives


def train_stage(model, anchors, positives, epochs, lr, batch_size, tag):
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        losses,
    )
    from sentence_transformers.training_args import BatchSamplers

    ds = Dataset.from_dict({"anchor": anchors, "positive": positives})
    loss = losses.MultipleNegativesRankingLoss(model)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(OUT / f"ms_{tag}"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.05,
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
        dataloader_drop_last=True,
    )
    tr = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss)
    t0 = time.time()
    tr.train()
    return time.time() - t0


def evaluate(model, long_df, misc_texts, fold):
    ids = sorted(misc_texts.keys())
    embs = st_encode(
        model,
        [misc_texts[i] for i in ids],
        prompt_name="document",
        batch_size=128,
        show_progress=True,
    )
    index = build_faiss_index(embs)
    val = long_df.filter(pl.col("fold") == fold).filter(pl.col("MisconceptionId") >= 0)
    q = st_encode(
        model, val["AllText"].to_list(), prompt_name="query", batch_size=64, show_progress=True
    )
    _, idx = index.search(q, 25)
    ev = EediEvaluator(k=25)
    for i, tid in enumerate(val["MisconceptionId"].to_list()):
        ev.update([ids[j] for j in idx[i]], int(tid))
    return ev.compute()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    ap.add_argument(
        "--synth",
        default=str(
            Path(
                "/root/autodl-tmp/eedi-misconception-engine/outputs/synth/synth_r1-32b_filtered.jsonl"
            )
        ),
    )
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument(
        "--out-tag", default="", help="非空则 adapter/结果另存为带 tag 的路径，避免覆盖生产权重"
    )
    ap.add_argument("--pretrain-epochs", type=int, default=1)
    ap.add_argument("--finetune-epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    console.rule("[bold blue]Multi-stage retriever (synth pretrain -> real finetune)")
    long_df = pl.read_parquet(DATA_DIR / "folds.parquet")
    misc_texts = load_misc()

    sa, sp = synth_pairs(args.synth, misc_texts)
    ra, rp = real_pairs(long_df, misc_texts, args.fold)
    console.print(f"[cyan]synth pairs={len(sa)}  real pairs={len(ra)}")

    from peft import LoraConfig

    model = load_st_model(args.model, cache_dir=HF_CACHE)
    model.max_seq_length = 512
    model[0].auto_model.add_adapter(
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
    )
    if hasattr(model[0].auto_model, "gradient_checkpointing_enable"):
        model[0].auto_model.gradient_checkpointing_enable()
        model[0].auto_model.config.use_cache = False

    results = {}
    if sa:
        console.print("[cyan]Stage 1: pretrain on synthetic (+real mixed)")
        t1 = train_stage(
            model, sa + ra, sp + rp, args.pretrain_epochs, 2e-4, args.batch_size, "stage1_synth"
        )
        results["stage1_seconds"] = round(t1, 1)
        results["after_pretrain"] = {
            k: round(v, 4)
            for k, v in evaluate(model, long_df, misc_texts, args.fold).items()
            if "@" in k
        }
        console.print(f"[green]after pretrain: {results['after_pretrain']}")
    else:
        console.print("[yellow]no synthetic data found; skipping stage 1")

    console.print("[cyan]Stage 2: finetune on real only")
    t2 = train_stage(model, ra, rp, args.finetune_epochs, 1e-4, args.batch_size, "stage2_real")
    results["stage2_seconds"] = round(t2, 1)
    results["final"] = {
        k: round(v, 4)
        for k, v in evaluate(model, long_df, misc_texts, args.fold).items()
        if "@" in k
    }
    console.print(f"[bold green]final (multi-stage): {results['final']}")

    suffix = f"_{args.out_tag}" if args.out_tag else ""
    adapter = OUT / f"lora_best_8b_multistage{suffix}"
    model[0].auto_model.save_pretrained(str(adapter))
    (OUT / f"multistage_results{suffix}.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )
    console.print(f"[green]saved adapter {adapter} + results")
    console.print("[bold]baseline real-only fold0: MAP@25=0.4012 Recall@25=0.8822 (compare above)")


if __name__ == "__main__":
    main()
