"""Listwise reranker via option-logit method (the recipe top teams actually used).

NOT generative ranking (that failed zero-shot at 0.3285). Instead: list the top-k
candidates as labelled options (A..J), do a single forward pass, read the logits of
the option-label tokens, and rank candidates by those logits. SFT optimizes a
cross-entropy over the option labels with the gold candidate's label as target.

Acts as a final-stage reranker on top-10 from the best pointwise reranker scores
(reorder top-10, keep 11-25 in original order). CoT (incorrect reason) is injected.

Refs: ebinan92/Eedi-5th (inference_listwise_vllm.py), rbiswasfc/Eedi-1st, zenn mkj.
Usage:
  zero-shot sanity:  python scripts/listwise_sft.py --epochs 0
  SFT:               python scripts/listwise_sft.py --epochs 2 --base Qwen/Qwen3-Reranker-8B
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")

import polars as pl
import torch
import torch.nn.functional as F
from rich.console import Console
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.evaluator import EediEvaluator

console = Console()
DATA_DIR = Path(os.environ.get("EEDI_DATA", "/root/autodl-tmp/eedi-data"))
HF_CACHE = os.environ.get("HF_HOME", "/root/autodl-tmp/hf_cache")
ROOT = Path("/root/autodl-tmp/eedi-misconception-engine")
RET_OUT = ROOT / "outputs/retriever"
RR_OUT = ROOT / "outputs/reranker"
LABELS = list("ABCDEFGHIJ")  # top-10

SYS = (
    "You will be given a math problem, its overview, the correct answer, an "
    "incorrect answer, and the likely incorrect reason. From the misconception "
    "list, choose the single option letter that best matches the misconception "
    "behind the incorrect answer. Answer with only the option letter."
)


def load_misc():
    df = pl.read_csv(DATA_DIR / "misconception_mapping.csv")
    return {int(r["MisconceptionId"]): r["MisconceptionName"] for r in df.iter_rows(named=True)}


def build_prompt(tok, row_text: str, cot: str, cand_texts: list[str]) -> str:
    opts = "\n".join(f"{LABELS[i]}. {t}" for i, t in enumerate(cand_texts))
    user = f"{row_text}\nIncorrectReason: {cot}\n# Misconception List\n{opts}"
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # Close think block: R1/Qwen3 templates leave <think> so last logit targets thinking not letters;
    # closing aligns the answer position with option labels (same as script 11). No-op without think.
    if p.rstrip().endswith("<think>"):
        p = p.rstrip() + "\n\n</think>\n\n"
    return p


def label_token_ids(tok) -> list[int]:
    ids = []
    for c in LABELS:
        enc = tok.encode(c, add_special_tokens=False)
        ids.append(enc[-1])
    return ids


def make_examples(split_df, scores_path, misc_texts, cot_map, top_k):
    scores = json.loads(Path(scores_path).read_text())
    ex = []
    for row in split_df.iter_rows(named=True):
        qa = row["QuestionId_Answer"]
        true_id = int(row["MisconceptionId"])
        rr = scores.get(qa)
        if not rr:
            continue
        cands = rr["ids"][:top_k]
        if len(cands) < top_k:
            continue
        gold_pos = cands.index(true_id) if true_id in cands else -1
        ex.append(
            {
                "qa": qa,
                "text": row["AllText"],
                "cot": cot_map.get(qa, ""),
                "cands": cands,
                "gold_pos": gold_pos,
                "true_id": true_id,
                "full": rr["ids"],
            }
        )
    return ex


def forward_logits(model, tok, prompts, lab_ids, max_len, device):
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(
        device
    )
    out = model(**enc)
    last = enc["attention_mask"].sum(1) - 1
    logits = out.logits[torch.arange(out.logits.size(0)), last]  # [B, vocab]
    return logits[:, lab_ids]  # [B, k]


def evaluate(model, tok, examples, lab_ids, top_k, max_len, device, bs=8, save_path=None):
    model.eval()
    ev = EediEvaluator(k=25)
    saved = {}
    with torch.no_grad():
        for i in tqdm(range(0, len(examples), bs), desc="listwise eval"):
            batch = examples[i : i + bs]
            prompts = [
                build_prompt(tok, e["text"], e["cot"], [MISC[c] for c in e["cands"]]) for e in batch
            ]
            lab_logits = forward_logits(model, tok, prompts, lab_ids, max_len, device).float()
            probs = torch.softmax(lab_logits, dim=1)
            order = lab_logits.argsort(dim=1, descending=True).cpu().tolist()
            for e, od, pr in zip(batch, order, probs.cpu().tolist()):
                reranked = [e["cands"][j] for j in od]
                tail = [c for c in e["full"] if c not in set(reranked)]
                ev.update(reranked + tail, e["true_id"])
                if save_path is not None:
                    saved[e["qa"]] = {
                        "ids": reranked,
                        "scores": [round(pr[j], 6) for j in od],
                        "true_id": e["true_id"],
                    }
    if save_path is not None:
        Path(save_path).write_text(json.dumps(saved), encoding="utf-8")
    return ev.compute()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-Reranker-8B")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-train", type=int, default=8000)
    ap.add_argument("--tag", default="8b", help="run tag for output dir/scores (e.g. r1-14b)")
    ap.add_argument(
        "--eval-adapter", default="", help="非空则只加载该 listwise adapter 评测+存分数（不训练）"
    )
    ap.add_argument("--train-scores", default=str(RR_OUT / "scores_best31k_fold0_train.json"))
    ap.add_argument(
        "--val-scores", default=str(RR_OUT / "scores_best31k_multistage_pool_fold0_val.json")
    )
    args = ap.parse_args()

    global MISC
    console.rule("[bold blue]Listwise SFT (option-logit method)")
    MISC = load_misc()
    long_df = pl.read_parquet(DATA_DIR / "folds.parquet")
    cot_map = {}
    cot_file = ROOT / "outputs/cot/cot_r1-14b.json"
    if cot_file.exists():
        cot_map = json.loads(cot_file.read_text())
        console.print(f"[cyan]CoT loaded: {len(cot_map)}")

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.base, cache_dir=HF_CACHE, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.truncation_side = "left"
    lab_ids = label_token_ids(tok)
    console.print(f"[cyan]label token ids={lab_ids}")

    model = AutoModelForCausalLM.from_pretrained(
        args.base, cache_dir=HF_CACHE, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)

    val_df = long_df.filter(pl.col("fold") == args.fold).filter(pl.col("MisconceptionId") >= 0)
    val_ex = make_examples(val_df, args.val_scores, MISC, cot_map, args.top_k)
    console.print(
        f"[cyan]val examples={len(val_ex)} (gold in top{args.top_k}: {sum(e['gold_pos'] >= 0 for e in val_ex)})"
    )

    if args.eval_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.eval_adapter).merge_and_unload()
        save_path = str(RR_OUT / f"scores_listwise_{args.tag}_fold{args.fold}_val.json")
        m = evaluate(
            model, tok, val_ex, lab_ids, args.top_k, args.max_len, device, save_path=save_path
        )
        console.print(
            f"[bold green]EVAL-ONLY listwise({args.tag}) MAP@25={m['MAP@25']:.4f} R@25={m['Recall@25']:.4f} -> scores saved"
        )
        return

    if args.epochs == 0:
        m = evaluate(model, tok, val_ex, lab_ids, args.top_k, args.max_len, device)
        console.print(
            f"[bold green]ZERO-SHOT listwise(logit) MAP@25={m['MAP@25']:.4f} R@25={m['Recall@25']:.4f}"
        )
        return

    from peft import LoraConfig, get_peft_model

    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.print_trainable_parameters()

    train_df = long_df.filter(pl.col("fold") != args.fold).filter(pl.col("MisconceptionId") >= 0)
    train_ex = [
        e
        for e in make_examples(train_df, args.train_scores, MISC, cot_map, args.top_k)
        if e["gold_pos"] >= 0
    ]
    random.seed(42)
    random.shuffle(train_ex)
    train_ex = train_ex[: args.max_train]
    console.print(f"[cyan]train examples (gold in top{args.top_k})={len(train_ex)}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    lab_t = torch.tensor(lab_ids, device=device)
    for ep in range(args.epochs):
        model.train()
        random.shuffle(train_ex)
        tot = 0.0
        pbar = tqdm(range(0, len(train_ex), args.batch_size), desc=f"epoch {ep + 1}")
        for i in pbar:
            batch = train_ex[i : i + args.batch_size]
            prompts = [
                build_prompt(tok, e["text"], e["cot"], [MISC[c] for c in e["cands"]]) for e in batch
            ]
            tgt = torch.tensor([e["gold_pos"] for e in batch], device=device)
            enc = tok(
                prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_len
            ).to(device)
            out = model(**enc)
            last = enc["attention_mask"].sum(1) - 1
            logits = out.logits[torch.arange(out.logits.size(0)), last][:, lab_t]
            loss = F.cross_entropy(logits.float(), tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad()
            tot += loss.item()
            pbar.set_postfix(loss=f"{tot / (i // args.batch_size + 1):.4f}")
        last_ep = ep + 1 == args.epochs
        save_path = (
            str(RR_OUT / f"scores_listwise_{args.tag}_fold{args.fold}_val.json")
            if last_ep
            else None
        )
        m = evaluate(
            model, tok, val_ex, lab_ids, args.top_k, args.max_len, device, save_path=save_path
        )
        console.print(
            f"[bold green]epoch {ep + 1}: listwise MAP@25={m['MAP@25']:.4f} R@25={m['Recall@25']:.4f}"
        )

    out_dir = RR_OUT / f"listwise_lora_{args.tag}"
    model.save_pretrained(str(out_dir))
    (RR_OUT / f"listwise_result_{args.tag}.json").write_text(
        json.dumps(
            {
                "final": {k: round(v, 4) for k, v in m.items() if "@" in k},
                "baseline_ensemble": 0.5820,
                "base": args.base,
                "top_k": args.top_k,
                "val_scores": save_path,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    console.print(f"[green]saved {out_dir} + listwise_result_{args.tag}.json + val scores")
    console.print("[bold]baseline (pointwise ensemble) MAP@25=0.5820 — compare above")


if __name__ == "__main__":
    main()
