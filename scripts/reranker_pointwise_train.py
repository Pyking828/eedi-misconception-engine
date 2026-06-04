"""Manual Qwen3-Reranker-8B LoRA training with yes/no logits.

This bypasses CrossEncoderTrainer/FlagEmbedding compatibility issues by using
the Qwen3-Reranker model-card inference formulation directly:
  score = P("yes" | query, document, instruction)

Training objective:
  labels: 1 for gold misconception, 0 for hard negatives
  loss: cross entropy over [no_logit, yes_logit]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")

import polars as pl
import torch
import torch.nn.functional as F
from eval.evaluator import EediEvaluator
from peft import LoraConfig, TaskType, get_peft_model
from rich.console import Console
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

console = Console()
DATA_DIR = Path(os.environ.get("EEDI_DATA", "/root/autodl-tmp/eedi-data"))
HF_CACHE = os.environ.get("HF_HOME", "/root/autodl-tmp/hf_cache")
ROOT = Path("/root/autodl-tmp/eedi-misconception-engine")
RETRIEVER_OUT = ROOT / "outputs/retriever"
OUTPUT_DIR = ROOT / "outputs/reranker/manual_lora"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INSTRUCTION = (
    "Given a mathematics question, the correct answer, and a student's incorrect answer, "
    "judge whether the document is the misconception that best explains why the student made the error."
)

PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class PairDataset(Dataset):
    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def load_data():
    long_df = pl.read_parquet(DATA_DIR / "folds.parquet")
    misc_df = pl.read_csv(DATA_DIR / "misconception_mapping.csv")
    misc_texts = {
        int(r["MisconceptionId"]): r["MisconceptionName"] for r in misc_df.iter_rows(named=True)
    }
    with open(RETRIEVER_OUT / "candidate_pool.json") as f:
        candidate_pool = {k: [int(x) for x in v] for k, v in json.load(f).items()}
    return long_df, misc_texts, candidate_pool


def format_instruction(query: str, doc: str) -> str:
    return f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {doc}"


def _augment_query(all_text: str, qa_key: str, cot_map: dict[str, str] | None) -> str:
    """Optionally append the R1-generated misconception reasoning (CoT) to the query."""
    if not cot_map:
        return all_text
    cot = cot_map.get(qa_key, "")
    if not cot:
        return all_text
    return f"{all_text}\n\nStudent's likely reasoning/misconception: {cot}"


def build_examples(
    long_df: pl.DataFrame,
    misc_texts: dict[int, str],
    candidate_pool: dict[str, list[int]],
    fold: int,
    hard_neg_per_pos: int,
    max_train_samples: int | None,
    cot_map: dict[str, str] | None = None,
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)
    train_df = long_df.filter(pl.col("fold") != fold).filter(pl.col("MisconceptionId") >= 0)
    examples: list[dict] = []

    for row in train_df.iter_rows(named=True):
        qa_key = row["QuestionId_Answer"]
        query = _augment_query(row["AllText"], qa_key, cot_map)
        true_id = int(row["MisconceptionId"])
        true_text = misc_texts.get(true_id, "")
        if not true_text:
            continue
        examples.append({"query": query, "doc": true_text, "label": 1})
        negs = [cid for cid in candidate_pool.get(qa_key, []) if cid != true_id]
        pool = negs[: min(30, len(negs))]
        if len(pool) > hard_neg_per_pos:
            negs = rng.sample(pool, hard_neg_per_pos)
        else:
            negs = pool
        for neg_id in negs:
            neg_text = misc_texts.get(int(neg_id), "")
            if neg_text:
                examples.append({"query": query, "doc": neg_text, "label": 0})

    rng.shuffle(examples)
    if max_train_samples:
        examples = examples[:max_train_samples]
    return examples


def make_collate(tokenizer, max_length: int):
    prefix_tokens = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(SUFFIX, add_special_tokens=False)
    max_pair_len = max_length - len(prefix_tokens) - len(suffix_tokens)

    def collate(batch: list[dict]) -> dict:
        texts = [format_instruction(x["query"], x["doc"]) for x in batch]
        tokenized = tokenizer(
            texts,
            padding=False,
            truncation="longest_first",
            max_length=max_pair_len,
            return_attention_mask=False,
            add_special_tokens=False,
        )
        input_ids = [prefix_tokens + ids + suffix_tokens for ids in tokenized["input_ids"]]
        enc = tokenizer.pad(
            {"input_ids": input_ids}, padding=True, return_tensors="pt", max_length=max_length
        )
        labels = torch.tensor([int(x["label"]) for x in batch], dtype=torch.long)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }

    return collate


def compute_loss(model, batch, yes_id: int, no_id: int, device: str) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    last_logits = out.logits[:, -1, :]
    yes = last_logits[:, yes_id]
    no = last_logits[:, no_id]
    binary_logits = torch.stack([no, yes], dim=1)
    return F.cross_entropy(binary_logits, labels)


@torch.no_grad()
def score_pairs(
    model,
    tokenizer,
    pairs: list[tuple[str, str]],
    batch_size: int,
    max_length: int,
    yes_id: int,
    no_id: int,
) -> list[float]:
    device = "cuda"
    model.eval()
    collate = make_collate(tokenizer, max_length)
    scores: list[float] = []
    for i in tqdm(range(0, len(pairs), batch_size), desc="Score pairs"):
        batch_items = [{"query": q, "doc": d, "label": 0} for q, d in pairs[i : i + batch_size]]
        batch = collate(batch_items)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        last = out.logits[:, -1, :]
        logits = torch.stack([last[:, no_id], last[:, yes_id]], dim=1)
        probs = torch.softmax(logits, dim=1)[:, 1]
        scores.extend(float(x) for x in probs.cpu())
    return scores


def evaluate(
    model,
    tokenizer,
    long_df,
    misc_texts,
    candidate_pool,
    fold: int,
    max_length: int,
    batch_size: int,
    cot_map: dict[str, str] | None = None,
) -> dict:
    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")
    val_df = long_df.filter(pl.col("fold") == fold).filter(pl.col("MisconceptionId") >= 0)
    evaluator = EediEvaluator(k=25)

    for row in tqdm(val_df.iter_rows(named=True), total=len(val_df), desc="Eval reranker"):
        qa_key = row["QuestionId_Answer"]
        query = _augment_query(row["AllText"], qa_key, cot_map)
        true_id = int(row["MisconceptionId"])
        cands = candidate_pool.get(qa_key, [])[:50]
        pairs = [(query, misc_texts.get(cid, "")) for cid in cands]
        scores = score_pairs(model, tokenizer, pairs, batch_size, max_length, yes_id, no_id)
        ranked = [cid for cid, _ in sorted(zip(cands, scores), key=lambda x: -x[1])]
        evaluator.update(ranked, true_id)

    metrics = evaluator.compute()
    return {
        "fold": fold,
        "MAP@25": round(metrics["MAP@25"], 4),
        "Recall@25": round(metrics["Recall@25"], 4),
        "nDCG@25": round(metrics["nDCG@25"], 4),
        "n_val": metrics["n_samples"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-Reranker-8B")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--hard-neg-per-pos", type=int, default=8)
    parser.add_argument("--max-train-samples", type=int, default=12000)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--cot",
        default="",
        help="path to CoT json {qa_key: rationale}; enables CoT-augmented query",
    )
    args = parser.parse_args()

    console.rule("[bold blue]Manual Qwen3-Reranker LoRA training")
    cot_map = None
    if args.cot and Path(args.cot).exists():
        cot_map = json.loads(Path(args.cot).read_text())
        console.print(f"[cyan]CoT loaded: {len(cot_map)} rationales -> CoT-augmented reranker")
    long_df, misc_texts, candidate_pool = load_data()
    examples = build_examples(
        long_df,
        misc_texts,
        candidate_pool,
        args.fold,
        args.hard_neg_per_pos,
        args.max_train_samples,
        cot_map=cot_map,
    )
    console.print(f"[cyan]examples={len(examples)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left", cache_dir=HF_CACHE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        cache_dir=HF_CACHE,
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.to("cuda")
    model.print_trainable_parameters()

    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")
    console.print(f"[cyan]yes_id={yes_id}, no_id={no_id}")

    dl = DataLoader(
        PairDataset(examples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate(tokenizer, args.max_length),
    )
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(dl) * args.epochs
    sched = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=max(10, int(total_steps * 0.05)), num_training_steps=total_steps
    )

    # Optional Trackio live logging (skipped silently if unavailable)
    try:
        import trackio

        trackio.init(
            project="eedi-misconception-engine",
            name=f"reranker_fold{args.fold}_n{len(examples)}",
            config={
                "model": args.model,
                "lr": args.lr,
                "epochs": args.epochs,
                "hard_neg_per_pos": args.hard_neg_per_pos,
                "max_length": args.max_length,
            },
        )
        _track = True
    except Exception:
        _track = False

    t0 = time.time()
    global_step = 0
    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(dl, desc=f"Train epoch {epoch + 1}/{args.epochs}")
        running = 0.0
        for step, batch in enumerate(pbar, 1):
            loss = compute_loss(model, batch, yes_id, no_id, "cuda")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            global_step += 1
            running += float(loss.detach().cpu())
            if step % 50 == 0:
                pbar.set_postfix(loss=running / step)
                if _track:
                    trackio.log(
                        {"loss": running / step, "lr": sched.get_last_lr()[0]}, step=global_step
                    )
    train_s = time.time() - t0

    cot_suffix = "_cot" if cot_map else ""
    run_tag = f"fold{args.fold}_n{len(examples)}_bs{args.batch_size}_hn{args.hard_neg_per_pos}_len{args.max_length}{cot_suffix}"
    adapter_dir = OUTPUT_DIR / f"manual_lora_{run_tag}"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    console.print(f"[green]adapter saved: {adapter_dir}")
    console.print(f"[green]training seconds={train_s:.1f}")

    metrics = evaluate(
        model,
        tokenizer,
        long_df,
        misc_texts,
        candidate_pool,
        args.fold,
        args.max_length,
        args.eval_batch_size,
        cot_map=cot_map,
    )
    metrics["train_runtime_s"] = round(train_s, 1)
    metrics["train_examples"] = len(examples)
    out_path = OUTPUT_DIR / f"manual_lora_metrics_{run_tag}.json"
    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[bold green]metrics={metrics}")
    console.print(f"[green]saved: {out_path}")
    if _track:
        trackio.log({k: v for k, v in metrics.items() if "@" in k})
        trackio.finish()
    console.rule("[bold green]done")


if __name__ == "__main__":
    main()
