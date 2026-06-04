"""Phase 4: GRPO RL fine-tuning of the listwise reranker (JD hard requirement).

We implement the GRPO objective directly (group-relative advantages + KL to a frozen
reference, no critic) over the listwise option distribution. This is robust on Blackwell
sm_120, where TRL's GRPOTrainer hits a transformers-5.x generate incompatibility
(attention_mask empty in `has_right_padding`). The math is identical to
GRPO specialized to a 1-step bandit with a verifiable reward.

Setup: policy = SFT listwise (script 18) + a fresh trainable LoRA. For each prompt the
policy yields a distribution pi over the 10 option labels (A..J) via the option-logit
method. We sample G options, reward r_i = 1 if the sampled option is the gold candidate
else 0, advantage A_i = r_i - mean(r), policy loss = -E[A_i * log pi(o_i)], plus
beta * KL(pi || pi_ref) where pi_ref is the SFT policy (LoRA disabled). Maximizing reward
pushes the gold misconception to rank-1, which MAP@25 rewards.
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
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import polars as pl
import torch
import torch.nn.functional as F
from rich.console import Console
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("/root/autodl-tmp/eedi-misconception-engine")
sys.path.insert(0, str(ROOT))
from eval.evaluator import EediEvaluator

console = Console()
DATA_DIR = Path(os.environ.get("EEDI_DATA", "/root/autodl-tmp/eedi-data"))
HF_CACHE = os.environ.get("HF_HOME", "/root/autodl-tmp/hf_cache")
RR_OUT = ROOT / "outputs/reranker"
LABELS = list("ABCDEFGHIJ")
SYS = (
    "You will be given a math problem, its overview, the correct answer, an "
    "incorrect answer, and the likely incorrect reason. From the misconception "
    "list, choose the single option letter that best matches the misconception "
    "behind the incorrect answer. Answer with only the option letter."
)


def load_misc():
    df = pl.read_csv(DATA_DIR / "misconception_mapping.csv")
    return {int(r["MisconceptionId"]): r["MisconceptionName"] for r in df.iter_rows(named=True)}


def build_prompt(tok, text, cot, cand_texts):
    opts = "\n".join(f"{LABELS[i]}. {t}" for i, t in enumerate(cand_texts))
    user = f"{text}\nIncorrectReason: {cot}\n# Misconception List\n{opts}"
    return tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def make_rows(split_df, scores_path, misc, cot_map, top_k, require_gold):
    scores = json.loads(Path(scores_path).read_text())
    rows = []
    for r in split_df.iter_rows(named=True):
        qa, tid = r["QuestionId_Answer"], int(r["MisconceptionId"])
        rr = scores.get(qa)
        if not rr:
            continue
        cands = rr["ids"][:top_k]
        if len(cands) < top_k:
            continue
        if require_gold and tid not in cands:
            continue
        rows.append(
            {
                "text": r["AllText"],
                "cot": cot_map.get(qa, ""),
                "cands": cands,
                "gold_pos": cands.index(tid) if tid in cands else -1,
                "true_id": tid,
                "full": rr["ids"],
            }
        )
    return rows


def plackett_luce_sample(logp, g):
    """Plackett-Luce sample g full rankings without replacement.
    Returns (rankings[B,g,K], ranking_logp[B,g]); logp is differentiable, sampling is detached."""
    bsz, k = logp.shape
    lp = logp.unsqueeze(1).expand(bsz, g, k)
    rankings = torch.zeros(bsz, g, k, dtype=torch.long, device=logp.device)
    rlogp = torch.zeros(bsz, g, device=logp.device)
    mask = torch.zeros(bsz, g, k, dtype=torch.bool, device=logp.device)
    for step in range(k):
        step_logp = torch.log_softmax(
            lp.masked_fill(mask, float("-inf")), dim=-1
        )  # renorm over remaining
        choice = torch.multinomial(step_logp.exp().detach().view(bsz * g, k), 1).view(bsz, g)
        rankings[:, :, step] = choice
        rlogp = rlogp + torch.gather(step_logp, 2, choice.unsqueeze(-1)).squeeze(-1)
        mask = mask.scatter(2, choice.unsqueeze(-1), True)
    return rankings, rlogp


def opt_logits(model, tok, prompts, lab_ids, max_len, device):
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(
        device
    )
    last = enc["attention_mask"].sum(1) - 1
    out = model(**enc)
    return out.logits[torch.arange(out.logits.size(0)), last][:, lab_ids]


@torch.no_grad()
def evaluate(model, tok, rows, lab_ids, max_len, device, bs=8):
    model.eval()
    ev = EediEvaluator(k=25)
    for i in tqdm(range(0, len(rows), bs), desc="grpo eval"):
        b = rows[i : i + bs]
        pr = [build_prompt(tok, e["text"], e["cot"], [MISC[c] for c in e["cands"]]) for e in b]
        ll = opt_logits(model, tok, pr, lab_ids, max_len, device).float()
        order = ll.argsort(1, descending=True).cpu().tolist()
        for e, od in zip(b, order):
            rer = [e["cands"][j] for j in od]
            tail = [c for c in e["full"] if c not in set(rer)]
            ev.update(rer + tail, e["true_id"])
    return ev.compute()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-Reranker-8B")
    ap.add_argument("--sft-adapter", default=str(RR_OUT / "listwise_lora"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--batch-prompts", type=int, default=4)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument(
        "--reward",
        choices=["hit", "ndcg"],
        default="ndcg",
        help="hit=二值top-1(v1); ndcg=Plackett-Luce整条排序+nDCG(v2,优化全排序)",
    )
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-prompts", type=int, default=3000)
    args = ap.parse_args()

    global MISC
    console.rule("[bold blue]GRPO listwise (manual group-relative PO, verifiable reward)")
    MISC = load_misc()
    long_df = pl.read_parquet(DATA_DIR / "folds.parquet")
    cot_map = {}
    cf = ROOT / "outputs/cot/cot_r1-14b.json"
    if cf.exists():
        cot_map = json.loads(cf.read_text())

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.base, cache_dir=HF_CACHE, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.truncation_side = "left"
    lab_ids = [tok.encode(c, add_special_tokens=False)[-1] for c in LABELS]

    base = AutoModelForCausalLM.from_pretrained(
        args.base, cache_dir=HF_CACHE, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    from peft import PeftModel

    # policy = base + SFT LoRA (trainable, further-tuned by GRPO)
    model = PeftModel.from_pretrained(base, args.sft_adapter, is_trainable=True)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    # reference = frozen SFT policy (textbook GRPO KL anchor)
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.base, cache_dir=HF_CACHE, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    ref_model = (
        PeftModel.from_pretrained(ref_model, args.sft_adapter).merge_and_unload().to(device).eval()
    )
    for p in ref_model.parameters():
        p.requires_grad_(False)

    train_rows = make_rows(
        long_df.filter(pl.col("fold") != args.fold).filter(pl.col("MisconceptionId") >= 0),
        RR_OUT / "scores_best31k_fold0_train.json",
        MISC,
        cot_map,
        args.top_k,
        True,
    )[: args.max_prompts]
    val_rows = make_rows(
        long_df.filter(pl.col("fold") == args.fold).filter(pl.col("MisconceptionId") >= 0),
        RR_OUT / "scores_best31k_multistage_pool_fold0_val.json",
        MISC,
        cot_map,
        args.top_k,
        False,
    )
    console.print(f"[cyan]train prompts={len(train_rows)} val={len(val_rows)} group={args.group}")

    m0 = evaluate(model, tok, val_rows, lab_ids, args.max_len, device)
    console.print(f"[yellow]pre-GRPO (=SFT) MAP@25={m0['MAP@25']:.4f}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    import random

    random.seed(0)
    step = 0
    best = m0["MAP@25"]
    pbar = tqdm(total=args.steps, desc="GRPO")
    while step < args.steps:
        random.shuffle(train_rows)
        for i in range(0, len(train_rows), args.batch_prompts):
            if step >= args.steps:
                break
            b = train_rows[i : i + args.batch_prompts]
            pr = [build_prompt(tok, e["text"], e["cot"], [MISC[c] for c in e["cands"]]) for e in b]
            model.train()
            logits = opt_logits(model, tok, pr, lab_ids, args.max_len, device).float()  # [B,10]
            logp = F.log_softmax(logits, dim=1)
            with torch.no_grad():
                ref_logits = opt_logits(ref_model, tok, pr, lab_ids, args.max_len, device).float()
                ref_logp = F.log_softmax(ref_logits, dim=1)
            probs = logp.exp()
            gold = torch.tensor([e["gold_pos"] for e in b], device=device)  # [B]
            if args.reward == "ndcg":
                # Plackett-Luce full ranking + nDCG reward (single relevant → 1/log2(rank+2))
                rankings, rlogp = plackett_luce_sample(logp, args.group)  # [B,G,K],[B,G]
                pos = (
                    (rankings == gold.view(-1, 1, 1)).float().argmax(dim=2)
                )  # gold rank in sample [B,G]
                rewards = 1.0 / torch.log2(pos.float() + 2.0)  # nDCG@K (one relevant)
                adv = rewards - rewards.mean(1, keepdim=True)
                pol_loss = -(adv * rlogp).mean()
            else:
                # Binary top-1 reward (v1)
                dist = torch.distributions.Categorical(probs=probs.detach())
                samples = dist.sample((args.group,)).T  # [B,G]
                rewards = (samples == gold.unsqueeze(1)).float()
                adv = rewards - rewards.mean(1, keepdim=True)
                chosen_logp = torch.gather(logp, 1, samples)
                pol_loss = -(adv * chosen_logp).mean()
            kl = (probs * (logp - ref_logp)).sum(1).mean()  # analytic KL over 10 options
            loss = pol_loss + args.beta * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad()
            step += 1
            pbar.update(1)
            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                reward=f"{rewards.mean().item():.3f}",
                kl=f"{kl.item():.3f}",
            )
            if step % 100 == 0:
                m = evaluate(model, tok, val_rows, lab_ids, args.max_len, device)
                console.print(
                    f"[green]step {step}: MAP@25={m['MAP@25']:.4f} (reward={rewards.mean().item():.3f})"
                )
                best = max(best, m["MAP@25"])
                model.train()
    pbar.close()

    mf = evaluate(model, tok, val_rows, lab_ids, args.max_len, device)
    console.print(
        f"[bold green]GRPO final MAP@25={mf['MAP@25']:.4f} R@25={mf['Recall@25']:.4f} (best={max(best, mf['MAP@25']):.4f})"
    )
    model.save_pretrained(str(RR_OUT / "grpo_listwise"))
    (RR_OUT / "grpo_result.json").write_text(
        json.dumps(
            {
                "pre_grpo_sft": round(m0["MAP@25"], 4),
                "final": {k: round(v, 4) for k, v in mf.items() if "@" in k},
                "best": round(max(best, mf["MAP@25"]), 4),
                "sft_only": 0.5700,
                "pointwise_ensemble": 0.5820,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    console.print("[bold]compare: SFT-only=0.5700, pointwise ensemble=0.5820")


if __name__ == "__main__":
    main()
