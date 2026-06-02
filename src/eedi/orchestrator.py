"""
中控主流程（Orchestrator）— 异步串联全链路。

对应 JD：
  "负责蓝心小v中控3.0主流程架构设计研发及维护，prompt调优及管理，
   上游感知记忆数据及下游mcp、subagent的快速接入"

流程：
  query → 路由(Router) → 召回(Retriever) → 粗排(Pointwise) → 精排(Listwise) → 推理SubAgent → 返回

设计原则：
  - 全异步（asyncio）
  - 感知记忆前置（命中缓存直接返回，跳过 GPU 推理）
  - 成本感知路由（置信度高则跳过重排/推理）
  - Prompt 版本化管理（从 prompts/ 加载 Jinja 模板）
  - 流式输出（SSE）：推理 subagent 边生成边推流
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, Optional

from src.eedi.router.router import PipelineMode, QueryRouter, RoutingDecision


# ─────────────────────────────────────────────
# 请求 / 响应结构
# ─────────────────────────────────────────────

@dataclass
class DiagnoseRequest:
    """中控入口请求（对应 FastAPI /diagnose 端点）。"""
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
        """组装 AllText 格式的统一 query。"""
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
# Prompt 注册表（版本化）
# ─────────────────────────────────────────────

class PromptRegistry:
    """
    管理 prompts/ 目录下的 Jinja2 模板，支持版本切换与 A/B 对比。
    """

    def __init__(self, prompts_dir: str | Path = "prompts") -> None:
        from jinja2 import Environment, FileSystemLoader

        self.env = Environment(
            loader=FileSystemLoader(str(prompts_dir)),
            autoescape=False,
        )
        self._ab_versions: dict[str, str] = {}  # prompt_name → version

    def render(self, name: str, version: Optional[str] = None, **kwargs) -> str:
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
# Orchestrator 主类
# ─────────────────────────────────────────────

class Orchestrator:
    """
    异步中控主流程。

    示例：
        orch = Orchestrator.from_config("configs/base.yaml")
        await orch.init()
        response = await orch.diagnose(request)
        # 或流式：
        async for chunk in orch.diagnose_stream(request):
            print(chunk)
    """

    def __init__(
        self,
        retriever,               # EediRetriever
        pointwise_reranker=None, # PointwiseReranker（可 None）
        listwise_reranker=None,  # ListwiseReranker（可 None）
        reasoner=None,           # CoTReasoner（可 None）
        router: Optional[QueryRouter] = None,
        memory=None,             # MemoryModule（可 None）
        misc_texts: Optional[dict[int, str]] = None,
        prompt_registry: Optional[PromptRegistry] = None,
        seen_misc_ids: Optional[set[int]] = None,
        unseen_score_scale: float = 0.4,  # 复刻 3rd place 技巧
    ) -> None:
        self.retriever = retriever
        self.pointwise = pointwise_reranker
        self.listwise = listwise_reranker
        self.reasoner = reasoner
        self.router = router or QueryRouter()
        self.memory = memory
        self.misc_texts = misc_texts or {}
        self.prompts = prompt_registry
        self.seen_misc_ids = seen_misc_ids or set()
        self.unseen_scale = unseen_score_scale

    @classmethod
    def from_config(cls, config_path: str | Path) -> "Orchestrator":
        """从 YAML 配置文件懒加载所有组件（适合生产部署）。"""
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(str(config_path))
        # 组件的实际加载在 init() 中完成
        instance = cls.__new__(cls)
        instance._cfg = cfg
        instance._initialized = False
        return instance

    async def init(self) -> None:
        """（如果从 from_config 创建）懒加载所有 GPU 组件。"""
        if getattr(self, "_initialized", True):
            return
        # 实际项目中在此加载模型，这里留作扩展点
        self._initialized = True

    async def diagnose(self, req: DiagnoseRequest) -> DiagnoseResponse:
        """完整诊断流程（非流式）。"""
        t0 = time.time()
        query = req.query
        qa_key = req.qa_key

        # ① 路由决策（先查缓存）
        decision = await self.router.route(
            query, qa_key, retriever_scores=None
        )
        if decision.from_cache and decision.cached_result:
            return self._build_response(
                req, decision.cached_result, "", decision, t0, from_cache=True
            )

        # ② 召回
        retrieved_ids, retriever_scores = self.retriever.retrieve(
            query, top_k=50, with_scores=True
        )

        # 更新路由置信度（有了分数后重决策）
        decision = await self.router.route(query, qa_key, retriever_scores)

        # ③ 粗排（Pointwise）
        if decision.mode in (PipelineMode.RETRIEVE_RERANK, PipelineMode.FULL_PIPELINE):
            if self.pointwise is not None:
                candidate_ids = self.pointwise.rerank(
                    query, retrieved_ids[:50], self.misc_texts, top_k=10
                )
                # 未见错因分数调整（复刻 3rd place）
                if self.seen_misc_ids:
                    candidate_ids = [
                        cid for cid in candidate_ids
                    ]  # 分数已在 pointwise 内部处理
            else:
                candidate_ids = retrieved_ids[:10]
        else:
            candidate_ids = retrieved_ids[:req.top_k]

        # ④ CoT 推理 Subagent
        rationale = ""
        if decision.mode == PipelineMode.FULL_PIPELINE and self.reasoner is not None and req.include_rationale:
            rationale = await self.reasoner.async_generate(
                req.question_text, req.correct_answer, req.wrong_answer
            )

        # ⑤ 精排（Listwise）
        if decision.mode == PipelineMode.FULL_PIPELINE and self.listwise is not None:
            final_ids = self.listwise.rerank(
                query, candidate_ids, self.misc_texts,
                top_k=req.top_k, cot_rationale=rationale
            )
        else:
            final_ids = candidate_ids[:req.top_k]

        # ⑥ 写入缓存 & 难负例
        if self.memory is not None:
            await self.memory.set_result(qa_key, final_ids)
            await self.memory.log_session(req.session_id, qa_key, query, final_ids)

        return self._build_response(req, final_ids, rationale, decision, t0)

    async def diagnose_stream(
        self, req: DiagnoseRequest
    ) -> AsyncGenerator[str, None]:
        """
        流式诊断（SSE）：
        先快速返回召回/粗排结果，再流式输出 CoT 推理。
        """
        t0 = time.time()
        query = req.query
        qa_key = req.qa_key

        # ① 路由 + 召回（同步，快）
        decision = await self.router.route(query, qa_key)
        if decision.from_cache and decision.cached_result:
            yield json.dumps(
                {"event": "result", "data": decision.cached_result, "from_cache": True}
            )
            return

        retrieved_ids, _ = self.retriever.retrieve(query, top_k=50, with_scores=True)

        if self.pointwise is not None:
            candidate_ids = self.pointwise.rerank(
                query, retrieved_ids, self.misc_texts, top_k=10
            )
        else:
            candidate_ids = retrieved_ids[:10]

        # ② 先推流中间结果
        yield json.dumps(
            {
                "event": "candidates",
                "data": candidate_ids,
                "latency_ms": (time.time() - t0) * 1000,
            }
        )

        # ③ CoT 推理流式（模拟 token-by-token）
        if self.reasoner is not None and req.include_rationale:
            rationale = await self.reasoner.async_generate(
                req.question_text, req.correct_answer, req.wrong_answer
            )
            for word in rationale.split():
                yield json.dumps({"event": "rationale_token", "data": word + " "})
                await asyncio.sleep(0.02)

        # ④ 精排 + 最终结果
        if self.listwise is not None:
            final_ids = self.listwise.rerank(
                query, candidate_ids, self.misc_texts, top_k=req.top_k
            )
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
                score=1.0 / (rank + 1),  # 简化打分
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
