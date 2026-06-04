"""
Orchestrator — async end-to-end pipeline.

Flow:
  query → Router → Retriever → Pointwise rerank → Listwise rerank → CoT reasoner → response

Design:
  - Fully async (asyncio)
  - Memory-first (cache hit skips GPU)
  - Cost-aware routing (high confidence skips rerank/reasoning)
  - Versioned prompts (Jinja under prompts/)
  - SSE streaming from the reasoner subagent
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path

from src.eedi.router.router import PipelineMode, QueryRouter, RoutingDecision

# ─────────────────────────────────────────────
# Request / response types
# ─────────────────────────────────────────────


@dataclass
class DiagnoseRequest:
    """Orchestrator input (FastAPI /diagnose)."""

    question_text: str
    correct_answer: str
    wrong_answer: str
    subject_name: str = ""
    construct_name: str = ""
    session_id: str = "default"
    top_k: int = 25
    include_rationale: bool = True
    stream: bool = False

    @property
    def query(self) -> str:
        """Build unified AllText-style query."""
        return (
            f"Subject: {self.subject_name}\n"
            f"Topic: {self.construct_name}\n"
            f"Question: {self.question_text}\n"
            f"Correct Answer: {self.correct_answer}\n"
            f"Incorrect Answer: {self.wrong_answer}"
        )

    @property
    def qa_key(self) -> str:
        import hashlib

        return hashlib.md5(self.query.encode()).hexdigest()[:12]


@dataclass
class MisconceptionCandidate:
    misconception_id: int
    misconception_name: str
    score: float
    rank: int


@dataclass
class DiagnoseResponse:
    request_id: str
    candidates: list[MisconceptionCandidate]
    rationale: str
    subject: str
    pipeline_mode: str
    from_cache: bool
    latency_ms: float
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────
# Prompt registry (versioned)
# ─────────────────────────────────────────────


class PromptRegistry:
    """Manage Jinja2 templates under prompts/ with version switching and A/B."""

    def __init__(self, prompts_dir: str | Path = "prompts") -> None:
        from jinja2 import Environment, FileSystemLoader

        self.env = Environment(
            loader=FileSystemLoader(str(prompts_dir)),
            autoescape=False,
        )
        self._ab_versions: dict[str, str] = {}  # prompt_name → version

    def render(self, name: str, version: str | None = None, **kwargs) -> str:
        v = version or self._ab_versions.get(name, "v1")
        template = self.env.get_template(f"{name}/{v}.jinja")
        return template.render(**kwargs)

    def set_ab_version(self, name: str, version: str) -> None:
        self._ab_versions[name] = version

    def list_versions(self, name: str) -> list[str]:
        import glob

        pattern = str(Path("prompts") / name / "*.jinja")
        return [Path(p).stem for p in glob.glob(pattern)]


# ─────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────


class Orchestrator:
    """Async orchestrator.

    Example:
        orch = Orchestrator.from_config("configs/base.yaml")
        await orch.init()
        response = await orch.diagnose(request)
        async for chunk in orch.diagnose_stream(request):
            print(chunk)
    """

    def __init__(
        self,
        retriever,  # EediRetriever
        pointwise_reranker=None,  # PointwiseReranker (optional)
        listwise_reranker=None,  # ListwiseReranker (optional)
        reasoner=None,  # CoTReasoner (optional)
        router: QueryRouter | None = None,
        memory=None,  # MemoryModule (optional)
        misc_texts: dict[int, str] | None = None,
        prompt_registry: PromptRegistry | None = None,
        seen_misc_ids: set[int] | None = None,
        unseen_score_scale: float = 0.4,  # 3rd-place unseen-misc trick
        force_full: bool = False,  # Demo: always full pipeline (rerank+CoT), no cost routing / cache short-circuit
    ) -> None:
        self.retriever = retriever
        self.pointwise = pointwise_reranker
        self.listwise = listwise_reranker
        self.reasoner = reasoner
        self.force_full = force_full
        self.router = router or QueryRouter()
        self.memory = memory
        self.misc_texts = misc_texts or {}
        self.prompts = prompt_registry
        self.seen_misc_ids = seen_misc_ids or set()
        self.unseen_scale = unseen_score_scale

    @classmethod
    def from_config(cls, config_path: str | Path) -> Orchestrator:
        """Lazy-load all components from YAML (production)."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(str(config_path))
        # Actual component loading happens in init()
        instance = cls.__new__(cls)
        instance._cfg = cfg
        instance._initialized = False
        return instance

    async def init(self) -> None:
        """Lazy-load GPU components when created via from_config."""
        if getattr(self, "_initialized", True):
            return
        # Extension point for model loading in production
        self._initialized = True

    async def diagnose(self, req: DiagnoseRequest) -> DiagnoseResponse:
        """Full diagnosis (non-streaming)."""
        t0 = time.time()
        query = req.query
        qa_key = req.qa_key

        # ① Route (cache first). force_full skips cache short-circuit for full demo + CoT.
        decision = await self.router.route(query, qa_key, retriever_scores=None)
        if not self.force_full and decision.from_cache and decision.cached_result:
            return self._build_response(
                req, decision.cached_result, "", decision, t0, from_cache=True
            )

        # ② Retrieve
        retrieved_ids, retriever_scores = self.retriever.retrieve(query, top_k=50, with_scores=True)

        # Re-route with retriever scores
        decision = await self.router.route(query, qa_key, retriever_scores)
        # force_full: full pipeline for demo
        if self.force_full:
            decision.mode = PipelineMode.FULL_PIPELINE

        # ③ Pointwise rerank
        if decision.mode in (PipelineMode.RETRIEVE_RERANK, PipelineMode.FULL_PIPELINE):
            if self.pointwise is not None:
                candidate_ids = self.pointwise.rerank(
                    query, retrieved_ids[:50], self.misc_texts, top_k=10
                )
                # Unseen-misc score adjustment (3rd place)
                if self.seen_misc_ids:
                    candidate_ids = [
                        cid for cid in candidate_ids
                    ]  # scores handled inside pointwise
            else:
                candidate_ids = retrieved_ids[:10]
        else:
            candidate_ids = retrieved_ids[: req.top_k]

        # ④ CoT reasoner subagent
        rationale = ""
        if (
            decision.mode == PipelineMode.FULL_PIPELINE
            and self.reasoner is not None
            and req.include_rationale
        ):
            rationale = await self.reasoner.async_generate(
                req.question_text, req.correct_answer, req.wrong_answer
            )

        # ⑤ Listwise rerank
        if decision.mode == PipelineMode.FULL_PIPELINE and self.listwise is not None:
            final_ids = self.listwise.rerank(
                query, candidate_ids, self.misc_texts, top_k=req.top_k, cot_rationale=rationale
            )
        else:
            final_ids = candidate_ids[: req.top_k]

        # ⑥ Cache result & hard negatives
        if self.memory is not None:
            await self.memory.set_result(qa_key, final_ids)
            await self.memory.log_session(req.session_id, qa_key, query, final_ids)

        return self._build_response(req, final_ids, rationale, decision, t0)

    async def diagnose_stream(self, req: DiagnoseRequest) -> AsyncGenerator[str, None]:
        """SSE diagnosis: retrieval/rerank first, then streamed CoT."""

        t0 = time.time()
        query = req.query
        qa_key = req.qa_key

        # ① Route + retrieve (sync, fast)
        decision = await self.router.route(query, qa_key)
        if decision.from_cache and decision.cached_result:
            yield json.dumps(
                {"event": "result", "data": decision.cached_result, "from_cache": True}
            )
            return

        retrieved_ids, _ = self.retriever.retrieve(query, top_k=50, with_scores=True)

        if self.pointwise is not None:
            candidate_ids = self.pointwise.rerank(query, retrieved_ids, self.misc_texts, top_k=10)
        else:
            candidate_ids = retrieved_ids[:10]

        # ② Stream intermediate candidates
        yield json.dumps(
            {
                "event": "candidates",
                "data": candidate_ids,
                "latency_ms": (time.time() - t0) * 1000,
            }
        )

        # ③ Stream CoT (word-chunk simulation)
        if self.reasoner is not None and req.include_rationale:
            rationale = await self.reasoner.async_generate(
                req.question_text, req.correct_answer, req.wrong_answer
            )
            for word in rationale.split():
                yield json.dumps({"event": "rationale_token", "data": word + " "})
                await asyncio.sleep(0.02)

        # ④ Listwise + final result
        if self.listwise is not None:
            final_ids = self.listwise.rerank(query, candidate_ids, self.misc_texts, top_k=req.top_k)
        else:
            final_ids = candidate_ids[: req.top_k]

        yield json.dumps(
            {
                "event": "final",
                "data": final_ids,
                "latency_ms": (time.time() - t0) * 1000,
            }
        )

    def _build_response(
        self,
        req: DiagnoseRequest,
        final_ids: list[int],
        rationale: str,
        decision: RoutingDecision,
        t0: float,
        from_cache: bool = False,
    ) -> DiagnoseResponse:
        import uuid

        candidates = [
            MisconceptionCandidate(
                misconception_id=mid,
                misconception_name=self.misc_texts.get(mid, ""),
                score=1.0 / (rank + 1),  # simplified score
                rank=rank + 1,
            )
            for rank, mid in enumerate(final_ids)
        ]
        return DiagnoseResponse(
            request_id=str(uuid.uuid4())[:8],
            candidates=candidates,
            rationale=rationale,
            subject=decision.subject.value,
            pipeline_mode=decision.mode.value,
            from_cache=from_cache,
            latency_ms=(time.time() - t0) * 1000,
        )
