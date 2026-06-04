"""
Query router (subject classification, cost-aware pipeline, memory cache).

1. Subject via keywords (future: embedding classifier)
2. Cost-aware: rerank/reason only when retrieval confidence is low
   - confidence uses top-1 score and gap vs top-2
3. Memory: return cached result on hit
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SubjectCategory(str, Enum):
    NUMBER = "Number"
    ALGEBRA = "Algebra"
    GEOMETRY = "Geometry and Measure"
    DATA = "Data and Statistics"
    UNKNOWN = "Unknown"


SUBJECT_KEYWORDS: dict[SubjectCategory, list[str]] = {
    SubjectCategory.NUMBER: [
        "fraction",
        "decimal",
        "integer",
        "arithmetic",
        "ratio",
        "percentage",
        "prime",
        "factor",
        "multiple",
        "square root",
        "cube",
        "power",
    ],
    SubjectCategory.ALGEBRA: [
        "equation",
        "inequality",
        "expression",
        "variable",
        "polynomial",
        "quadratic",
        "linear",
        "algebra",
        "formula",
        "expand",
        "factorise",
        "simplify",
        "substitut",
    ],
    SubjectCategory.GEOMETRY: [
        "angle",
        "triangle",
        "circle",
        "polygon",
        "area",
        "perimeter",
        "volume",
        "surface",
        "coordinate",
        "vector",
        "transformation",
        "rotation",
        "reflection",
        "translation",
        "geometry",
        "congruent",
    ],
    SubjectCategory.DATA: [
        "mean",
        "median",
        "mode",
        "range",
        "probability",
        "statistics",
        "graph",
        "histogram",
        "frequency",
        "cumulative",
        "quartile",
        "data",
    ],
}


class PipelineMode(str, Enum):
    """Pipeline mode (lowest to highest cost)."""

    RETRIEVE_ONLY = "retrieve_only"  # retrieve only, very high confidence
    RETRIEVE_RERANK = "retrieve_rerank"  # retrieve + pointwise (default)
    FULL_PIPELINE = "full_pipeline"  # retrieve + pointwise + listwise + CoT


@dataclass
class RoutingDecision:
    subject: SubjectCategory
    mode: PipelineMode
    confidence: float  # retrieval confidence (0-1)
    from_cache: bool = False
    cached_result: list[int] | None = None
    metadata: dict = field(default_factory=dict)


class QueryRouter:
    """Query router.

    Example:
        router = QueryRouter(cost_threshold=0.7, memory=memory_module)
        decision = await router.route(query, retriever_scores=[0.95, 0.60])
    """

    def __init__(
        self,
        cost_threshold: float = 0.7,
        high_confidence_threshold: float = 0.90,
        memory=None,  # MemoryModule (optional)
    ) -> None:
        self.cost_threshold = cost_threshold
        self.high_conf_threshold = high_confidence_threshold
        self.memory = memory

    def classify_subject(self, query: str) -> SubjectCategory:
        """Lightweight keyword subject classification (O(1))."""
        q_lower = query.lower()
        scores: dict[SubjectCategory, int] = {}
        for cat, keywords in SUBJECT_KEYWORDS.items():
            scores[cat] = sum(1 for kw in keywords if kw in q_lower)
        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        if scores[best] == 0:
            return SubjectCategory.UNKNOWN
        return best

    def compute_confidence(
        self,
        retriever_scores: list[float],
        top_k: int = 2,
    ) -> float:
        """Confidence from top-1 score and gap vs top-2."""
        if not retriever_scores:
            return 0.0
        sorted_scores = sorted(retriever_scores, reverse=True)
        top1 = sorted_scores[0]
        if len(sorted_scores) >= 2:
            gap = top1 - sorted_scores[1]
            confidence = min(1.0, top1 * (1 + gap))
        else:
            confidence = top1
        return float(confidence)

    async def route(
        self,
        query: str,
        qa_key: str,
        retriever_scores: list[float] | None = None,
    ) -> RoutingDecision:
        """Async route: cache first, then confidence-based mode."""
        subject = self.classify_subject(query)

        # 1. Memory cache
        if self.memory is not None:
            cached = await self.memory.get_result(qa_key)
            if cached is not None:
                return RoutingDecision(
                    subject=subject,
                    mode=PipelineMode.RETRIEVE_ONLY,
                    confidence=1.0,
                    from_cache=True,
                    cached_result=cached,
                )

        # 2. Cost-aware mode
        if retriever_scores:
            confidence = self.compute_confidence(retriever_scores)
        else:
            confidence = 0.5  # unknown → conservative full pipeline

        if confidence >= self.high_conf_threshold:
            mode = PipelineMode.RETRIEVE_ONLY
        elif confidence >= self.cost_threshold:
            mode = PipelineMode.RETRIEVE_RERANK
        else:
            mode = PipelineMode.FULL_PIPELINE

        return RoutingDecision(
            subject=subject,
            mode=mode,
            confidence=confidence,
            metadata={"subject_str": subject.value},
        )
