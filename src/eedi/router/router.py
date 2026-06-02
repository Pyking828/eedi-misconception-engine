"""
智能路由模块（复刻 vivo 蓝心小v 中控 3.0 "智能路由" JD 要点）。

路由决策：
1. 学科分类（keyword / 未来可升级为 embedding 分类器）
2. 成本感知升级：召回置信度低于阈值才触发重排/推理
   - 置信度 = top-1 召回分数 vs top-2 分数之差（gap），gap 大 → 高置信，可跳过重排
3. 感知记忆：查询 SQLite，命中缓存则直接返回

这与 JD "对 query 做精准化调度、提升小v使用体验" 完全对应。
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SubjectCategory(str, Enum):
    NUMBER = "Number"
    ALGEBRA = "Algebra"
    GEOMETRY = "Geometry and Measure"
    DATA = "Data and Statistics"
    UNKNOWN = "Unknown"


SUBJECT_KEYWORDS: dict[SubjectCategory, list[str]] = {
    SubjectCategory.NUMBER: [
        "fraction", "decimal", "integer", "arithmetic", "ratio", "percentage",
        "prime", "factor", "multiple", "square root", "cube", "power",
    ],
    SubjectCategory.ALGEBRA: [
        "equation", "inequality", "expression", "variable", "polynomial",
        "quadratic", "linear", "algebra", "formula", "expand", "factorise",
        "simplify", "substitut",
    ],
    SubjectCategory.GEOMETRY: [
        "angle", "triangle", "circle", "polygon", "area", "perimeter",
        "volume", "surface", "coordinate", "vector", "transformation",
        "rotation", "reflection", "translation", "geometry", "congruent",
    ],
    SubjectCategory.DATA: [
        "mean", "median", "mode", "range", "probability", "statistics",
        "graph", "histogram", "frequency", "cumulative", "quartile", "data",
    ],
}


class PipelineMode(str, Enum):
    """路由决策的执行模式（成本从低到高）。"""
    RETRIEVE_ONLY = "retrieve_only"          # 仅召回，极高置信
    RETRIEVE_RERANK = "retrieve_rerank"      # 召回 + 粗排（默认）
    FULL_PIPELINE = "full_pipeline"          # 召回 + 粗排 + 精排 + CoT 推理（最强）


@dataclass
class RoutingDecision:
    subject: SubjectCategory
    mode: PipelineMode
    confidence: float          # 召回置信度（0-1）
    from_cache: bool = False
    cached_result: Optional[list[int]] = None
    metadata: dict = field(default_factory=dict)


class QueryRouter:
    """
    中控路由器。

    示例：
        router = QueryRouter(cost_threshold=0.7, memory=memory_module)
        decision = await router.route(query, retriever_scores=[0.95, 0.60, ...])
    """

    def __init__(
        self,
        cost_threshold: float = 0.7,
        high_confidence_threshold: float = 0.90,
        memory=None,  # MemoryModule（可 None）
    ) -> None:
        self.cost_threshold = cost_threshold
        self.high_conf_threshold = high_confidence_threshold
        self.memory = memory

    def classify_subject(self, query: str) -> SubjectCategory:
        """基于关键词的轻量学科分类（O(1)，延迟极低）。"""
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
        """
        置信度 = top-1 分数，辅以 top-1 vs top-2 的 gap。
        gap 越大，越确信，赋予更高置信度。
        """
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
        retriever_scores: Optional[list[float]] = None,
    ) -> RoutingDecision:
        """异步路由决策（先查缓存，再按置信度决定 mode）。"""
        subject = self.classify_subject(query)

        # 1. 查记忆缓存
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

        # 2. 成本感知决策
        if retriever_scores:
            confidence = self.compute_confidence(retriever_scores)
        else:
            confidence = 0.5  # 未知，保守走全流程

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
