"""
Stage 0: data prep via HuggingFace mirror (cdtmc/eedi-ir).

Background:
- Some CN networks cannot reach storage.googleapis.com (Kaggle GCS)
- Kaggle API lists but download hangs on GCS redirect
- Fix: HF mirror cdtmc/eedi-ir (full IR format for the Eedi competition)

cdtmc/eedi-ir layout:
- corpus: (id_, text) = 2587 misconceptions
- queries: (fold, id_, text) = 4370 queries
- qrels: (fold, qid, mid) = query → gold misconception

Outputs project schema:
- misconception_mapping.csv
- long_table.parquet (with fold)
- folds.parquet
- seen_misc_ids.json
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import polars as pl
from rich.console import Console
from sklearn.model_selection import GroupKFold

console = Console()

DATA_DIR = Path(os.environ.get("EEDI_DATA", "/root/autodl-tmp/eedi-data"))
HF_CACHE = os.environ.get("HF_HOME", "/root/autodl-tmp/hf_cache")
DATA_DIR.mkdir(parents=True, exist_ok=True)

IR_REPO = "cdtmc/eedi-ir"


def parse_query_text(text: str) -> dict:
    """
    Parse IR query text into structured fields.
    Format:
      "### Question: {Subject} | {Construct} | {QuestionText}
       ### Correct: {CorrectAnswer}
       ### Incorrect: {WrongAnswer}"
    """
    subject, construct, question, correct, wrong = "", "", "", "", ""

    # Split with regex
    q_match = re.search(r"### Question:\s*(.*?)(?=\n### Correct:|\Z)", text, re.DOTALL)
    c_match = re.search(r"### Correct:\s*(.*?)(?=\n### Incorrect:|\Z)", text, re.DOTALL)
    i_match = re.search(r"### Incorrect:\s*(.*?)\Z", text, re.DOTALL)

    if q_match:
        q_block = q_match.group(1).strip()
        # First two " | " separate Subject | Construct | QuestionText
        parts = q_block.split(" | ", 2)
        if len(parts) == 3:
            subject, construct, question = parts[0].strip(), parts[1].strip(), parts[2].strip()
        elif len(parts) == 2:
            subject, question = parts[0].strip(), parts[1].strip()
        else:
            question = q_block
    if c_match:
        correct = c_match.group(1).strip()
    if i_match:
        wrong = i_match.group(1).strip()

    return {
        "SubjectName": subject,
        "ConstructName": construct,
        "QuestionText": question,
        "CorrectAnswerText": correct,
        "WrongAnswerText": wrong,
    }


def main():
    console.rule("[bold blue]阶段0：数据准备（HF 镜像源 cdtmc/eedi-ir）")

    from huggingface_hub import hf_hub_download

    # ── Download three parquet shards ──
    console.print("[cyan]从 hf-mirror 下载 cdtmc/eedi-ir ...")
    corpus_p = hf_hub_download(
        IR_REPO, "corpus/train-00000-of-00001.parquet", repo_type="dataset", cache_dir=HF_CACHE
    )
    queries_p = hf_hub_download(
        IR_REPO, "queries/train-00000-of-00001.parquet", repo_type="dataset", cache_dir=HF_CACHE
    )
    qrels_p = hf_hub_download(
        IR_REPO, "qrels/train-00000-of-00001.parquet", repo_type="dataset", cache_dir=HF_CACHE
    )

    corpus = pl.read_parquet(corpus_p)
    queries = pl.read_parquet(queries_p)
    qrels = pl.read_parquet(qrels_p)
    console.print(f"  corpus={corpus.shape}  queries={queries.shape}  qrels={qrels.shape}")

    # ── 1. misconception_mapping.csv ──────────────
    misc_df = corpus.rename({"id_": "MisconceptionId", "text": "MisconceptionName"})
    misc_df = misc_df.select(["MisconceptionId", "MisconceptionName"]).sort("MisconceptionId")
    misc_df.write_csv(str(DATA_DIR / "misconception_mapping.csv"))
    console.print(f"[green]✓ misconception_mapping.csv: {len(misc_df)} 条")

    # ── 2. Merge queries + qrels → long_table ──
    qrels_map = {r["qid"]: r["mid"] for r in qrels.iter_rows(named=True)}

    rows = []
    for r in queries.iter_rows(named=True):
        qid_ans = r["id_"]  # e.g. "0_D"
        parts = qid_ans.rsplit("_", 1)
        question_id = parts[0]
        answer = parts[1] if len(parts) > 1 else "?"
        parsed = parse_query_text(r["text"])
        misc_id = qrels_map.get(qid_ans, -1)
        rows.append(
            {
                "QuestionId": question_id,
                "QuestionId_Answer": qid_ans,
                "Answer": answer,
                **parsed,
                "MisconceptionId": int(misc_id),
                "orig_fold": r["fold"],
            }
        )

    long_df = pl.DataFrame(rows)

    # Join MisconceptionName
    misc_name_map = {
        r["MisconceptionId"]: r["MisconceptionName"] for r in misc_df.iter_rows(named=True)
    }
    long_df = long_df.with_columns(
        pl.col("MisconceptionId")
        .map_elements(lambda x: misc_name_map.get(x, ""), return_dtype=pl.String)
        .alias("MisconceptionName")
    )

    # Build AllText (unified retrieval input)
    long_df = long_df.with_columns(
        (
            "Subject: "
            + pl.col("SubjectName")
            + "\n"
            + "Topic: "
            + pl.col("ConstructName")
            + "\n"
            + "Question: "
            + pl.col("QuestionText")
            + "\n"
            + "Correct Answer: "
            + pl.col("CorrectAnswerText")
            + "\n"
            + "Incorrect Answer: "
            + pl.col("WrongAnswerText")
        ).alias("AllText")
    )

    # ── 3. Rebuild 5-fold GroupKFold by QuestionId ──
    question_ids = long_df["QuestionId"].to_numpy()
    gkf = GroupKFold(n_splits=5)
    fold_col = np.full(len(long_df), -1, dtype=np.int32)
    dummy = np.zeros(len(long_df))
    for fold_idx, (_, val_idx) in enumerate(gkf.split(dummy, dummy, groups=question_ids)):
        fold_col[val_idx] = fold_idx
    long_df = long_df.with_columns(pl.Series("fold", fold_col))

    long_df.write_parquet(str(DATA_DIR / "folds.parquet"))
    long_df.write_parquet(str(DATA_DIR / "long_table.parquet"))
    console.print(f"[green]✓ long_table.parquet / folds.parquet: {len(long_df)} 行")

    # ── 4. EDA stats ──
    n_total = len(misc_df)
    n_seen = long_df.filter(pl.col("MisconceptionId") >= 0)["MisconceptionId"].n_unique()
    n_unseen = n_total - n_seen
    console.print("\n[bold]错因统计：")
    console.print(f"  错因总数: {n_total}")
    console.print(f"  训练中出现: {n_seen}")
    console.print(f"  未见错因:  {n_unseen} ({n_unseen / n_total * 100:.1f}%)")

    subj_dist = (
        long_df.group_by("SubjectName").agg(pl.len().alias("count")).sort("count", descending=True)
    )
    console.print("\n[bold]学科分布（Top10）：")
    console.print(subj_dist.head(10))

    # Unseen misconceptions per fold
    console.print("\n[bold]5 折统计：")
    for fold in range(5):
        tr = long_df.filter(pl.col("fold") != fold)
        va = long_df.filter(pl.col("fold") == fold)
        tr_misc = set(tr["MisconceptionId"].to_list())
        va_misc = set(va["MisconceptionId"].to_list())
        unseen_in_val = len(va_misc - tr_misc)
        console.print(
            f"  Fold {fold}: train={len(tr)}, val={len(va)}, val中未见错因={unseen_in_val}"
        )

    # ── 5. seen_misc_ids.json (score scaling) ──
    seen_ids = long_df.filter(pl.col("MisconceptionId") >= 0)["MisconceptionId"].unique().to_list()
    with open(DATA_DIR / "seen_misc_ids.json", "w") as f:
        json.dump(seen_ids, f)
    console.print(f"\n[green]✓ seen_misc_ids.json: {len(seen_ids)} 条")

    # Sample rows
    console.print("\n[bold]样例（解析后）：")
    sample = long_df.head(2).to_dicts()
    for s in sample:
        console.print(
            f"  QID_Ans={s['QuestionId_Answer']}  Subject={s['SubjectName']!r}  Construct={s['ConstructName'][:40]!r}"
        )
        console.print(f"    Q={s['QuestionText'][:60]!r}")
        console.print(
            f"    Correct={s['CorrectAnswerText'][:30]!r}  Wrong={s['WrongAnswerText'][:30]!r}"
        )
        console.print(
            f"    → Misconception[{s['MisconceptionId']}]={s['MisconceptionName'][:50]!r}"
        )

    console.rule("[bold green]数据准备完成")


if __name__ == "__main__":
    main()
