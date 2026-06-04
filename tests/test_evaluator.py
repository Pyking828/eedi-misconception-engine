"""Unit tests: evaluator"""

import pytest
from eval.evaluator import (
    EediEvaluator,
    average_precision_at_k,
    evaluate_pipeline,
    ndcg_at_k,
    recall_at_k,
)


def test_ap_perfect():
    assert average_precision_at_k([1, 2, 3], true_id=1, k=25) == 1.0


def test_ap_second():
    assert average_precision_at_k([2, 1, 3], true_id=1, k=25) == pytest.approx(0.5)


def test_ap_not_found():
    assert average_precision_at_k([2, 3, 4], true_id=1, k=25) == 0.0


def test_ap_beyond_k():
    # id=1 at rank 3 (1-based), within k=25 → AP=1/3
    assert average_precision_at_k([2, 3] + [1] * 30, true_id=1, k=25) == pytest.approx(1 / 3)
    # id=1 not in top-k
    assert average_precision_at_k([2, 3, 4, 5], true_id=1, k=3) == 0.0


def test_recall_found():
    assert recall_at_k([1, 2, 3], true_id=1, k=25) == 1.0


def test_recall_not_found():
    assert recall_at_k([2, 3, 4], true_id=1, k=25) == 0.0


def test_ndcg_rank1():
    import math

    assert ndcg_at_k([1, 2, 3], true_id=1, k=25) == pytest.approx(1.0 / math.log2(2))


def test_ndcg_not_found():
    assert ndcg_at_k([2, 3, 4], true_id=1, k=25) == 0.0


def test_evaluator_update_and_compute():
    ev = EediEvaluator(k=25)
    ev.update([1, 2, 3], true_id=1)
    ev.update([2, 1, 3], true_id=1)
    metrics = ev.compute()
    assert metrics["n_samples"] == 2
    assert 0 < metrics["MAP@25"] <= 1
    assert 0 < metrics["Recall@25"] <= 1


def test_evaluate_pipeline():
    predictions = [
        {
            "predicted_ids": [1, 2, 3],
            "true_id": 1,
            "QuestionId_Answer": "q1_A",
            "SubjectName": "Number",
        },
        {
            "predicted_ids": [2, 3, 4],
            "true_id": 1,
            "QuestionId_Answer": "q2_B",
            "SubjectName": "Algebra",
        },
    ]
    result = evaluate_pipeline(predictions, k=25)
    assert "overall" in result
    assert result["overall"]["MAP@25"] == pytest.approx(0.5)


def test_evaluator_reset():
    ev = EediEvaluator(k=5)
    ev.update([1], 1)
    ev.reset()
    assert ev._n == 0
    assert ev.compute() == {}
