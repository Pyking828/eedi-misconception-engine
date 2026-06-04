"""Unit tests: query router"""

import pytest
from src.eedi.router.router import PipelineMode, QueryRouter, SubjectCategory


@pytest.mark.asyncio
async def test_high_confidence_retrieve_only():
    router = QueryRouter(cost_threshold=0.7, high_confidence_threshold=0.90)
    decision = await router.route("Solve x + 1 = 2", "q1", retriever_scores=[0.99, 0.55])
    assert decision.mode == PipelineMode.RETRIEVE_ONLY


@pytest.mark.asyncio
async def test_low_confidence_full_pipeline():
    router = QueryRouter(cost_threshold=0.7, high_confidence_threshold=0.90)
    decision = await router.route("Solve x + 1 = 2", "q2", retriever_scores=[0.60, 0.58])
    assert decision.mode == PipelineMode.FULL_PIPELINE


@pytest.mark.asyncio
async def test_medium_confidence_retrieve_rerank():
    # confidence = min(1.0, top1 * (1 + gap)) = min(1.0, 0.82 * (1 + 0.32)) = min(1.0, 1.082) = 1.0 → RETRIEVE_ONLY
    # gap small enough for confidence in [0.7, 0.9)
    router = QueryRouter(cost_threshold=0.7, high_confidence_threshold=0.90)
    # top1=0.75, gap=0.05 → confidence = 0.75 * 1.05 = 0.7875 → RETRIEVE_RERANK
    decision = await router.route("Solve x + 1 = 2", "q3", retriever_scores=[0.75, 0.70])
    assert decision.mode == PipelineMode.RETRIEVE_RERANK


def test_subject_classification_algebra():
    router = QueryRouter()
    cat = router.classify_subject("Solve the quadratic equation x^2 + 2x + 1 = 0")
    assert cat == SubjectCategory.ALGEBRA


def test_subject_classification_number():
    router = QueryRouter()
    cat = router.classify_subject("Calculate the fraction 3/4 + 1/2")
    assert cat == SubjectCategory.NUMBER


def test_subject_classification_geometry():
    router = QueryRouter()
    cat = router.classify_subject("Find the area of a triangle with base 5 and height 3")
    assert cat == SubjectCategory.GEOMETRY
