"""
MAP@K / Recall@K / nDCG@K 评测器。
支持离线评测（给定 predictions + ground_truth）和在线实时指标。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────
# 核心指标函数
# ─────────────────────────────────────────────

def average_precision_at_k(predicted_ids: list[int], true_id: int, k: int = 25) -> float:
    """MAP@K 中单条样本的 AP 值。只有一个正例。"""
    for rank, pid in enumerate(predicted_ids[:k], start=1):
        if pid == true_id:
            return 1.0 / rank
    return 0.0


def recall_at_k(predicted_ids: list[int], true_id: int, k: int = 25) -> float:
    return 1.0 if true_id in predicted_ids[:k] else 0.0


def ndcg_at_k(predicted_ids: list[int], true_id: int, k: int = 25) -> float:
    """nDCG@K（二元相关度，ideal DCG = 1.0）。"""
    for rank, pid in enumerate(predicted_ids[:k], start=1):
        if pid == true_id:
            return 1.0 / math.log2(rank + 1)
    return 0.0


# ─────────────────────────────────────────────
# 批量评测
# ─────────────────────────────────────────────

class EediEvaluator:
    """
    用法：
        evaluator = EediEvaluator(k=25)
        evaluator.update(predicted_ids=[1, 42, 7, ...], true_id=42)
        metrics = evaluator.compute()
    """

    def __init__(self, k: int = 25) -> None:
        self.k = k
        self._aps: list[float] = []
        self._recalls: list[float] = []
        self._ndcgs: list[float] = []
        self._n: int = 0

    def update(self, predicted_ids: list[int], true_id: int) -> dict[str, float]:
        ap = average_precision_at_k(predicted_ids, true_id, self.k)
        rec = recall_at_k(predicted_ids, true_id, self.k)
        nd = ndcg_at_k(predicted_ids, true_id, self.k)
        self._aps.append(ap)
        self._recalls.append(rec)
        self._ndcgs.append(nd)
        self._n += 1
        return {"ap": ap, "recall": rec, "ndcg": nd}

    def compute(self) -> dict[str, float]:
        if self._n == 0:
            return {}
        return {
            f"MAP@{self.k}": sum(self._aps) / self._n,
            f"Recall@{self.k}": sum(self._recalls) / self._n,
            f"nDCG@{self.k}": sum(self._ndcgs) / self._n,
            "n_samples": self._n,
        }

    def reset(self) -> None:
        self._aps.clear()
        self._recalls.clear()
        self._ndcgs.clear()
        self._n = 0

    def __repr__(self) -> str:
        m = self.compute()
        return (
            f"EediEvaluator(n={m.get('n_samples', 0)}, "
            f"MAP@{self.k}={m.get(f'MAP@{self.k}', 0):.4f}, "
            f"Recall@{self.k}={m.get(f'Recall@{self.k}', 0):.4f})"
        )


def evaluate_pipeline(
    predictions: list[dict],
    k: int = 25,
    report_path: Optional[str | Path] = None,
    per_subject: bool = True,
) -> dict[str, float | dict]:
    """
    predictions: list of {
        "QuestionId_Answer": str,
        "predicted_ids": [int, ...],   # 降序排列的 MisconceptionId
        "true_id": int,
        "SubjectName": str  (可选)
    }
    """
    overall = EediEvaluator(k=k)
    subject_evals: dict[str, EediEvaluator] = {}

    for pred in predictions:
        true_id = pred["true_id"]
        predicted_ids = pred["predicted_ids"]
        overall.update(predicted_ids, true_id)

        if per_subject and "SubjectName" in pred:
            subj = pred["SubjectName"]
            if subj not in subject_evals:
                subject_evals[subj] = EediEvaluator(k=k)
            subject_evals[subj].update(predicted_ids, true_id)

    result: dict = {"overall": overall.compute()}
    if per_subject:
        result["by_subject"] = {s: e.compute() for s, e in subject_evals.items()}

    if report_path is not None:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def main() -> None:
    """CLI entry point: python -m eval.evaluator --predictions path/to/preds.json"""
    import argparse

    parser = argparse.ArgumentParser(description="Eedi Evaluator")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSON")
    parser.add_argument("--k", type=int, default=25)
    parser.add_argument("--report", default=None, help="Path to save report JSON")
    args = parser.parse_args()

    with open(args.predictions) as f:
        predictions = json.load(f)

    result = evaluate_pipeline(predictions, k=args.k, report_path=args.report)
    print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__":
    main()
