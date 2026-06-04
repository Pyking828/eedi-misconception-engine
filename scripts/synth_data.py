"""Stage 2: synthetic MCQ generation for unseen/rare misconceptions.

Teacher: DeepSeek-R1-Distill-Qwen-32B (or 14B) via transformers (vLLM unusable on sm_120).
For each target misconception (prioritising those UNSEEN in the train folds), few-shot
from real examples of similar misconceptions, generate a JSON MCQ:
{ConstructName, SubjectName, QuestionText, CorrectAnswerText, WrongAnswerText}.
Then a self-judge pass scores misconception<->wrong-answer alignment 0-10; keep >= threshold.

Output JSONL: outputs/synth/synth_<model>.jsonl  (+ filtered file)
Reusable to retrain the retriever (real + synthetic, multi-stage).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
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
from rich.console import Console
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()
DATA_DIR = Path(os.environ.get("EEDI_DATA", "/root/autodl-tmp/eedi-data"))
HF_CACHE = os.environ.get("HF_HOME", "/root/autodl-tmp/hf_cache")
OUT = Path("/root/autodl-tmp/eedi-misconception-engine/outputs/synth")
OUT.mkdir(parents=True, exist_ok=True)

MODEL_DIRS = {
    "r1-14b": "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-14B",
    "r1-32b": "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B",
}


def mpath(key: str) -> str:
    return glob.glob(f"{HF_CACHE}/{MODEL_DIRS[key]}/snapshots/*/")[0]


GEN_TMPL = (
    "You are a maths teacher creating a diagnostic multiple-choice question that targets a "
    "SPECIFIC student misconception.\n\n"
    'Target misconception: "{mis}"\n\n'
    "Here are reference examples (real questions that diagnose related misconceptions):\n{shots}\n\n"
    "Create ONE NEW middle/high-school maths question where a student holding the target "
    "misconception would pick the wrong answer. Output STRICT JSON on a single line with keys: "
    "ConstructName, SubjectName, QuestionText, CorrectAnswerText, WrongAnswerText. "
    "The WrongAnswerText must be the answer a student with the target misconception would give."
)

JUDGE_TMPL = (
    'Question: {q}\nCorrect answer: {c}\nA student\'s wrong answer: {w}\nSuspected misconception: "{mis}"\n\n'
    "Does this wrong answer plausibly result from that misconception? "
    "Reply with ONLY an integer 0-10 (10 = perfect alignment)."
)


def force_nothink(tok, user: str) -> str:
    return (
        tok.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
        )
        + "<think>\n\n</think>\n\n"
    )


KEYS = {"ConstructName", "SubjectName", "QuestionText", "CorrectAnswerText", "WrongAnswerText"}


def _try_load(s: str) -> dict | None:
    for cand in (s, re.sub(r",\s*([}\]])", r"\1", s)):  # strip trailing commas
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None


def extract_json(text: str) -> dict | None:
    """Robust JSON extract: strip fences → first{..last} → tolerant parse → per-key regex fallback."""
    t = text.strip()
    t = re.sub(r"```(?:json)?", "", t)  # remove ```json fences
    d = None
    i, j = t.find("{"), t.rfind("}")
    if i != -1 and j != -1 and j > i:
        d = _try_load(t[i : j + 1])
    if isinstance(d, dict) and KEYS.issubset(d.keys()):
        if str(d.get("QuestionText", "")).strip() and str(d.get("WrongAnswerText", "")).strip():
            return {k: str(d[k]) for k in KEYS}
    # Fallback: per-key regex if JSON is slightly broken
    out = {}
    for k in KEYS:
        m = re.search(rf'"{k}"\s*:\s*"((?:[^"\\]|\\.)*)"', t)
        if m:
            out[k] = m.group(1).replace('\\"', '"').replace("\\n", " ").strip()
    if KEYS.issubset(out.keys()) and out["QuestionText"].strip() and out["WrongAnswerText"].strip():
        return out
    return None


@torch.no_grad()
def batch_gen(model, tok, prompts, max_new_tokens, batch_size, temperature=0.7):
    outs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="gen"):
        chunk = prompts[i : i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=1400).to(
            "cuda"
        )
        do_sample = temperature > 0
        g = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=0.9 if do_sample else None,
            pad_token_id=tok.eos_token_id,
        )
        outs.extend(tok.batch_decode(g[:, enc["input_ids"].shape[1] :], skip_special_tokens=True))
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="r1-32b", choices=list(MODEL_DIRS))
    ap.add_argument("--per-mis", type=int, default=4, help="MCQs per target misconception")
    ap.add_argument("--limit-mis", type=int, default=0, help="0 = all (prioritise unseen)")
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--judge-threshold", type=int, default=6)
    args = ap.parse_args()

    console.rule(f"[bold blue]Synthetic MCQ generation (teacher={args.model})")
    long_df = pl.read_parquet(DATA_DIR / "folds.parquet")
    misc_df = pl.read_csv(DATA_DIR / "misconception_mapping.csv")
    all_mis = {
        int(r["MisconceptionId"]): r["MisconceptionName"] for r in misc_df.iter_rows(named=True)
    }
    seen = set(
        int(x)
        for x in long_df.filter(pl.col("MisconceptionId") >= 0)["MisconceptionId"]
        .unique()
        .to_list()
    )
    unseen = [mid for mid in all_mis if mid not in seen]
    targets = unseen + [m for m in all_mis if m in seen]  # unseen first
    if args.limit_mis:
        targets = targets[: args.limit_mis]
    console.print(f"[cyan]targets={len(targets)} (unseen={len(unseen)}), per_mis={args.per_mis}")

    # few-shot pool from real labelled rows
    real = long_df.filter(pl.col("MisconceptionId") >= 0)
    shots_pool = [
        f'{{"ConstructName":"{r["ConstructName"]}","SubjectName":"{r["SubjectName"]}",'
        f'"QuestionText":"{str(r["QuestionText"])[:200]}","CorrectAnswerText":"{str(r["CorrectAnswerText"])[:60]}",'
        f'"WrongAnswerText":"{str(r["WrongAnswerText"])[:60]}"}}'
        for r in real.iter_rows(named=True)
    ]
    rng = random.Random(42)

    tok = AutoTokenizer.from_pretrained(mpath(args.model), padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        mpath(args.model), torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda"
    ).eval()

    # build generation prompts
    gen_prompts, gen_meta = [], []
    for mid in targets:
        mis = all_mis[mid]
        for _ in range(args.per_mis):
            shots = "\n".join(rng.sample(shots_pool, min(3, len(shots_pool))))
            gen_prompts.append(force_nothink(tok, GEN_TMPL.format(mis=mis, shots=shots)))
            gen_meta.append(mid)

    t0 = time.time()
    raw = batch_gen(
        model, tok, gen_prompts, max_new_tokens=384, batch_size=args.batch_size, temperature=0.8
    )
    recs = []
    for mid, text in zip(gen_meta, raw):
        d = extract_json(text)
        if d:
            d["MisconceptionId"] = mid
            d["MisconceptionName"] = all_mis[mid]
            recs.append(d)
    raw_path = OUT / f"synth_{args.model}_raw.jsonl"
    raw_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8"
    )
    console.print(
        f"[cyan]parsed {len(recs)}/{len(gen_prompts)} valid MCQs; gen {time.time() - t0:.0f}s"
    )

    # judge pass
    judge_prompts = [
        force_nothink(
            tok,
            JUDGE_TMPL.format(
                q=r["QuestionText"][:300],
                c=r["CorrectAnswerText"][:80],
                w=r["WrongAnswerText"][:80],
                mis=r["MisconceptionName"],
            ),
        )
        for r in recs
    ]
    jraw = batch_gen(
        model, tok, judge_prompts, max_new_tokens=8, batch_size=args.batch_size, temperature=0.0
    )
    kept = []
    for r, jt in zip(recs, jraw):
        m = re.search(r"\d+", jt)
        score = int(m.group(0)) if m else 0
        r["judge_score"] = min(score, 10)
        if r["judge_score"] >= args.judge_threshold:
            kept.append(r)
    filt_path = OUT / f"synth_{args.model}_filtered.jsonl"
    filt_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in kept), encoding="utf-8"
    )
    console.print(
        f"[bold green]kept {len(kept)}/{len(recs)} (judge>={args.judge_threshold}); saved {filt_path}"
    )
    console.print(f"[green]total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
