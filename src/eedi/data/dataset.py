"""
Eedi data loading, wide-to-long conversion, and 5-fold CV.

Wide table: one row per question (QuestionId, ConstructName, SubjectName,
QuestionText, CorrectAnswer, AnswerA-D, MisconceptionA-D).
Long table: one row per distractor; primary key QuestionId_Answer.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

if TYPE_CHECKING:
    from datasets import Dataset

# ─────────────────────────────────────────────
# 1. Raw data loading
# ─────────────────────────────────────────────


def load_raw_data(
    data_dir: str | Path,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return (train_df, misconception_df, test_df) as polars DataFrames."""
    data_dir = Path(data_dir)
    train_df = pl.read_csv(data_dir / "train.csv")
    misconception_df = pl.read_csv(data_dir / "misconception_mapping.csv")
    test_df = pl.read_csv(data_dir / "test.csv")
    return train_df, misconception_df, test_df


# ─────────────────────────────────────────────
# 2. Wide → long table
# ─────────────────────────────────────────────


def build_long_table(
    df: pl.DataFrame,
    misconception_df: pl.DataFrame,
    include_correct: bool = False,
) -> pl.DataFrame:
    """
    Expand wide table (one row per question) to long (one row per distractor).

    Output columns:
        QuestionId, QuestionId_Answer, SubjectName, ConstructName,
        QuestionText, CorrectAnswerText, WrongAnswerText,
        MisconceptionId, MisconceptionName, AllText
    """
    answer_cols = ["A", "B", "C", "D"]
    rows: list[dict] = []

    for row in df.iter_rows(named=True):
        correct_ans = row["CorrectAnswer"]
        for ans in answer_cols:
            if not include_correct and ans == correct_ans:
                continue
            misc_id = row.get(f"Misconception{ans}Id")
            if misc_id is None:
                continue
            answer_text = row.get(f"Answer{ans}Text") or row.get(f"Answer{ans}")
            if answer_text is None:
                continue
            rows.append(
                {
                    "QuestionId": row["QuestionId"],
                    "QuestionId_Answer": f"{row['QuestionId']}_{ans}",
                    "Answer": ans,
                    "SubjectName": row.get("SubjectName", ""),
                    "ConstructName": row.get("ConstructName", ""),
                    "QuestionText": row.get("QuestionText", ""),
                    "CorrectAnswerText": row.get(f"Answer{correct_ans}Text")
                    or row.get(f"Answer{correct_ans}", ""),
                    "WrongAnswerText": answer_text,
                    "MisconceptionId": int(misc_id) if misc_id == misc_id else -1,
                }
            )

    long_df = pl.DataFrame(rows)

    # Join MisconceptionName
    misc_map = {
        int(r["MisconceptionId"]): r["MisconceptionName"]
        for r in misconception_df.iter_rows(named=True)
    }
    long_df = long_df.with_columns(
        pl.col("MisconceptionId")
        .map_elements(lambda x: misc_map.get(x, ""), return_dtype=pl.String)
        .alias("MisconceptionName")
    )

    # Build AllText (unified model input)
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

    return long_df


# ─────────────────────────────────────────────
# 3. 5-fold CV
# ─────────────────────────────────────────────


def build_cv_folds(
    long_df: pl.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
    save_path: str | Path | None = None,
) -> pl.DataFrame:
    """
    GroupKFold by QuestionId so all distractors for a question share a fold.
    Returns long_df with a fold column.
    """
    question_ids = long_df["QuestionId"].to_numpy()
    dummy_X = np.zeros(len(long_df))
    dummy_y = np.zeros(len(long_df))

    gkf = GroupKFold(n_splits=n_folds)
    fold_col = np.full(len(long_df), -1, dtype=np.int32)
    for fold_idx, (_, val_idx) in enumerate(gkf.split(dummy_X, dummy_y, groups=question_ids)):
        fold_col[val_idx] = fold_idx

    long_df = long_df.with_columns(pl.Series("fold", fold_col))

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        long_df.write_parquet(str(save_path))

    return long_df


def load_folds(folds_path: str | Path) -> pl.DataFrame:
    return pl.read_parquet(str(folds_path))


# ─────────────────────────────────────────────
# 4. PyTorch Dataset wrapper
# ─────────────────────────────────────────────


class EediDataset:
    """
    Generic dataset for retriever / reranker modes.

    mode:
        'retriever'  → (query_text, pos_text, neg_texts)
        'pointwise'  → (query_text, candidate_text, label)
        'listwise'   → (query_text, [candidate_texts], correct_rank)
    """

    def __init__(
        self,
        long_df: pl.DataFrame,
        misconception_df: pl.DataFrame,
        mode: str = "retriever",
        neg_per_pos: int = 8,
        hard_neg_ids: dict[str, list[int]] | None = None,
        fold: int | None = None,
        split: str = "train",
    ) -> None:
        self.mode = mode
        self.neg_per_pos = neg_per_pos
        self.hard_neg_ids = hard_neg_ids or {}

        # Fold filter
        if fold is not None:
            if split == "train":
                df = long_df.filter(pl.col("fold") != fold)
            else:
                df = long_df.filter(pl.col("fold") == fold)
        else:
            df = long_df

        # Rows with gold labels only
        self.df = df.filter(pl.col("MisconceptionId") >= 0)

        # Misconception lookup
        self.misc_texts: dict[int, str] = {
            int(r["MisconceptionId"]): r["MisconceptionName"]
            for r in misconception_df.iter_rows(named=True)
        }
        self.misc_ids = list(self.misc_texts.keys())
        self.rng = np.random.default_rng(42)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.row(idx, named=True)
        query = row["AllText"]
        pos_id = row["MisconceptionId"]
        pos_text = self.misc_texts[pos_id]

        if self.mode == "retriever":
            # Hard negatives first, then random
            qa_key = row["QuestionId_Answer"]
            if qa_key in self.hard_neg_ids:
                neg_pool = [i for i in self.hard_neg_ids[qa_key] if i != pos_id]
            else:
                neg_pool = [i for i in self.misc_ids if i != pos_id]
            neg_ids = self.rng.choice(
                neg_pool, size=min(self.neg_per_pos, len(neg_pool)), replace=False
            )
            neg_texts = [self.misc_texts[i] for i in neg_ids]
            return {"query": query, "pos": pos_text, "negs": neg_texts, "pos_id": pos_id}

        elif self.mode == "pointwise":
            return {"query": query, "candidate": pos_text, "label": 1.0, "pos_id": pos_id}

        elif self.mode == "listwise":
            return {"query": query, "pos_text": pos_text, "pos_id": pos_id}

        raise ValueError(f"Unknown mode: {self.mode}")

    def to_hf_dataset(self) -> Dataset:
        """Convert to HuggingFace Dataset for SFTTrainer/GRPOTrainer."""
        from datasets import Dataset

        records = [self[i] for i in range(len(self))]
        return Dataset.from_list(records)
