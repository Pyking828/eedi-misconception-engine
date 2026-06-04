"""
Async FastAPI service (asyncio / FastAPI / type annotations).

Endpoints:
  POST /diagnose           → full diagnosis (non-streaming)
  POST /diagnose/stream    → SSE streaming diagnosis
  POST /search             → vector search only (lightweight)
  POST /feedback           → submit user feedback
  GET  /health             → health check
  GET  /metrics            → simple metrics
  GET  /docs               → Swagger UI

Design:
  - Pydantic v2 request/response models with full type hints
  - Lazy-load GPU components at startup (avoid OOM on import)
  - lifespan context manager for resources (SQLite, etc.)
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

# ─────────────────────────────────────────────
# Global component container (lazy-loaded)
# ─────────────────────────────────────────────


class AppState:
    orchestrator = None
    memory = None
    misc_texts: dict[int, str] = {}
    seen_misc_ids: set[int] = set()
    ready: bool = False


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: load components on startup, release on shutdown."""
    print("[Service] 初始化中...")
    await _load_components()
    state.ready = True
    print("[Service] 就绪 ✓")
    yield
    if state.memory:
        await state.memory.close()
    print("[Service] 已关闭")


async def _load_components() -> None:
    """Lazy-load GPU components when config and model artifacts exist."""
    from omegaconf import OmegaConf

    cfg_path = Path(__file__).parent.parent / "configs" / "base.yaml"
    if not cfg_path.exists():
        print("[Service] 警告：configs/base.yaml 不存在，以 mock 模式运行")
        return

    cfg = OmegaConf.load(str(cfg_path))

    # Load misconception texts (lightweight, required)
    misc_csv = Path(cfg.data.misconception_csv)
    if misc_csv.exists():
        import polars as pl

        misc_df = pl.read_csv(str(misc_csv))
        state.misc_texts = {
            r["MisconceptionId"]: r["MisconceptionName"] for r in misc_df.iter_rows(named=True)
        }

    # Load memory module
    from src.eedi.memory.memory import MemoryModule

    state.memory = MemoryModule(cfg.memory.db_path)
    await state.memory.init()

    # Heavy GPU models (8B retriever + 8B reranker). Set EEDI_LIGHT=1 to skip for CPU/light deploy.
    light = os.environ.get("EEDI_LIGHT", "0") == "1"

    # Load retriever (STRetriever + multistage adapter, aligned with FAISS index)
    index_path = Path(cfg.retriever.faiss_index_path)
    misc_ids_path = Path(cfg.retriever.get("misc_ids_path", ""))
    retriever = None
    if index_path.exists() and not light:
        import json as _json

        from src.eedi.retriever.st_engine import STRetriever

        misc_ids = (
            _json.loads(misc_ids_path.read_text())
            if misc_ids_path.exists()
            else list(state.misc_texts.keys())
        )
        retriever = STRetriever.from_pretrained(
            model_name=cfg.retriever.model_name,
            index_path=str(index_path),
            misc_ids=[int(x) for x in misc_ids],
            misc_texts=state.misc_texts,
            adapter_path=cfg.retriever.get("adapter_path"),
            cache_dir=cfg.paths.hf_cache,
        )
        print(
            f"[Service] STRetriever 就绪（{cfg.retriever.model_name} + {Path(cfg.retriever.get('adapter_path', '')).name}）"
        )
    elif light:
        print("[Service] EEDI_LIGHT=1，跳过重型召回器（仅 API 骨架可用）")
    else:
        print("[Service] FAISS 索引未找到，向量检索不可用")

    # Load pointwise reranker (yes/no logit scoring, matches training)
    pointwise = None
    rr_adapter = cfg.reranker.pointwise.get("adapter_path")
    if retriever is not None and rr_adapter and Path(rr_adapter).exists():
        from src.eedi.reranker.pointwise import LogitReranker

        pointwise = LogitReranker.from_pretrained(
            model_name=cfg.reranker.pointwise.model_name,
            adapter_path=rr_adapter,
            max_length=cfg.reranker.pointwise.get("max_seq_len", 768),
            cache_dir=cfg.paths.hf_cache,
        )
        print("[Service] LogitReranker 就绪（Qwen3-Reranker-8B + 最优 adapter）")

    # Load CoT reasoner subagent (default Qwen2.5-3B-Instruct; override with EEDI_REASONER)
    reasoner = None
    reasoner_model = os.environ.get("EEDI_REASONER", "Qwen/Qwen2.5-3B-Instruct")
    if retriever is not None and reasoner_model.lower() != "none":
        try:
            from src.eedi.reasoner.reasoner import CoTReasoner

            reasoner = CoTReasoner.from_pretrained(
                model_name=reasoner_model,
                max_new_tokens=200,
                temperature=0.3,
                cache_dir=cfg.paths.hf_cache,
                cache_db=str(Path(cfg.paths.outputs) / "cot_cache.db"),
            )
            print(f"[Service] CoTReasoner 就绪（{reasoner_model}）")
        except Exception as e:  # Reasoner is optional; failure does not block main path
            print(f"[Service] CoTReasoner 跳过: {e}")

    # Assemble orchestrator
    if retriever is not None:
        from src.eedi.orchestrator import Orchestrator, PromptRegistry
        from src.eedi.router.router import QueryRouter

        prompts_dir = Path(__file__).parent.parent / "prompts"
        prompt_reg = PromptRegistry(prompts_dir) if prompts_dir.exists() else None
        router = QueryRouter(memory=state.memory)
        state.orchestrator = Orchestrator(
            retriever=retriever,
            pointwise_reranker=pointwise,
            reasoner=reasoner,
            force_full=os.environ.get("EEDI_FORCE_FULL", "0") == "1",
            router=router,
            memory=state.memory,
            misc_texts=state.misc_texts,
            prompt_registry=prompt_reg,
        )


# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────

app = FastAPI(
    title="Eedi Misconception Engine API",
    description=(
        "数学错因诊断中控系统 — 召回 → 粗排 → 精排 → CoT 推理 级联服务。\n\n"
        "工程能力：工具检索召回/排序 + 智能路由 + Prompt 管理 + "
        "感知记忆 + MCP / SubAgent 接入 + SSE 流式。"
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Pydantic models (v2 + full type hints)
# ─────────────────────────────────────────────


class DiagnoseInput(BaseModel):
    question_text: str = Field(..., description="Math question text")
    correct_answer: str = Field(..., description="Correct answer text")
    wrong_answer: str = Field(..., description="Student's incorrect answer text")
    subject_name: str = Field(default="", description="Subject (Number/Algebra/...)")
    construct_name: str = Field(default="", description="Topic / construct name")
    session_id: str = Field(default="default", description="Session id (for memory)")
    top_k: int = Field(default=25, ge=1, le=50, description="Number of candidates to return")
    include_rationale: bool = Field(default=True, description="Whether to generate CoT rationale")


class MisconceptionResult(BaseModel):
    misconception_id: int
    misconception_name: str
    score: float
    rank: int


class DiagnoseOutput(BaseModel):
    request_id: str
    candidates: list[MisconceptionResult]
    rationale: str
    subject: str
    pipeline_mode: str
    from_cache: bool
    latency_ms: float


class SearchInput(BaseModel):
    query: str = Field(..., description="Free-text search query")
    top_k: int = Field(default=10, ge=1, le=50)


class SearchResult(BaseModel):
    misconception_id: int
    misconception_name: str
    score: float


class FeedbackInput(BaseModel):
    qa_key: str
    user_id: str = "anonymous"
    rating: int = Field(ge=1, le=5)
    comment: str = ""


class HealthOutput(BaseModel):
    status: str
    ready: bool
    n_misconceptions: int
    timestamp: float


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────


@app.get("/health", response_model=HealthOutput, summary="健康检查")
async def health() -> HealthOutput:
    return HealthOutput(
        status="ok" if state.ready else "initializing",
        ready=state.ready,
        n_misconceptions=len(state.misc_texts),
        timestamp=time.time(),
    )


@app.post("/diagnose", response_model=DiagnoseOutput, summary="错因诊断（完整流程）")
async def diagnose(body: DiagnoseInput) -> DiagnoseOutput:
    if not state.ready:
        raise HTTPException(status_code=503, detail="Service initializing")
    if state.orchestrator is None:
        raise HTTPException(status_code=503, detail="Models not loaded (index not found)")

    from src.eedi.orchestrator import DiagnoseRequest

    req = DiagnoseRequest(
        question_text=body.question_text,
        correct_answer=body.correct_answer,
        wrong_answer=body.wrong_answer,
        subject_name=body.subject_name,
        construct_name=body.construct_name,
        session_id=body.session_id,
        top_k=body.top_k,
        include_rationale=body.include_rationale,
        stream=False,
    )
    response = await state.orchestrator.diagnose(req)
    return DiagnoseOutput(
        request_id=response.request_id,
        candidates=[
            MisconceptionResult(
                misconception_id=c.misconception_id,
                misconception_name=c.misconception_name,
                score=c.score,
                rank=c.rank,
            )
            for c in response.candidates
        ],
        rationale=response.rationale,
        subject=response.subject,
        pipeline_mode=response.pipeline_mode,
        from_cache=response.from_cache,
        latency_ms=response.latency_ms,
    )


@app.post("/diagnose/stream", summary="错因诊断（SSE 流式）")
async def diagnose_stream(body: DiagnoseInput) -> EventSourceResponse:
    """SSE stream: return retrieval results first, then stream CoT rationale."""
    if state.orchestrator is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    from src.eedi.orchestrator import DiagnoseRequest

    req = DiagnoseRequest(
        question_text=body.question_text,
        correct_answer=body.correct_answer,
        wrong_answer=body.wrong_answer,
        subject_name=body.subject_name,
        construct_name=body.construct_name,
        session_id=body.session_id,
        top_k=body.top_k,
        include_rationale=body.include_rationale,
        stream=True,
    )

    async def event_generator() -> AsyncGenerator[dict, None]:
        async for chunk in state.orchestrator.diagnose_stream(req):
            data = json.loads(chunk)
            yield {"event": data.get("event", "message"), "data": json.dumps(data)}

    return EventSourceResponse(event_generator())


@app.post("/search", response_model=list[SearchResult], summary="纯向量检索（轻量）")
async def search(body: SearchInput) -> list[SearchResult]:
    """Skip reranking; return vector retrieval only (HF Spaces CPU demo)."""
    if state.orchestrator is None or state.orchestrator.retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not loaded")

    ids, scores = state.orchestrator.retriever.retrieve(
        body.query, top_k=body.top_k, with_scores=True
    )
    return [
        SearchResult(
            misconception_id=mid,
            misconception_name=state.misc_texts.get(mid, ""),
            score=float(s),
        )
        for mid, s in zip(ids, scores)
    ]


@app.post("/feedback", summary="提交用户反馈")
async def submit_feedback(body: FeedbackInput) -> dict:
    if state.memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    await state.memory.add_feedback(
        qa_key=body.qa_key,
        user_id=body.user_id,
        rating=body.rating,
        comment=body.comment,
    )
    return {"status": "ok"}


@app.get("/metrics", summary="服务指标")
async def metrics() -> dict:
    result: dict = {
        "n_misconceptions": len(state.misc_texts),
        "ready": state.ready,
    }
    if state.memory:
        stats = await state.memory.get_feedback_stats()
        result["feedback"] = stats
    return result


# ─────────────────────────────────────────────
# Gradio UI (mounted at /ui)
# ─────────────────────────────────────────────


def create_gradio_app():
    import gradio as gr

    async def diagnose_fn(question, correct, wrong, subject, top_k):
        if state.orchestrator is None:
            return "⚠️ Models not loaded yet.", "", ""
        from src.eedi.orchestrator import DiagnoseRequest

        req = DiagnoseRequest(
            question_text=question,
            correct_answer=correct,
            wrong_answer=wrong,
            subject_name=subject,
            top_k=int(top_k),
            include_rationale=True,
        )
        resp = await state.orchestrator.diagnose(req)

        cand_md = "\n".join(
            f"{c.rank}. **[{c.misconception_id}]** {c.misconception_name} (score={c.score:.3f})"
            for c in resp.candidates[:10]
        )
        meta = (
            f"📡 mode: `{resp.pipeline_mode}` | "
            f"subject: `{resp.subject}` | "
            f"cached: `{resp.from_cache}` | "
            f"latency: `{resp.latency_ms:.0f}ms`"
        )
        return cand_md, resp.rationale, meta

    with gr.Blocks(title="Eedi Misconception Engine", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🎓 Eedi Math Misconception Diagnosis Engine\n"
            "> Retrieve → Rerank → Listwise → CoT pipeline · "
            "Qwen3-Embedding + LoRA + GRPO"
        )

        with gr.Row():
            with gr.Column(scale=2):
                question = gr.Textbox(
                    label="Question",
                    placeholder="e.g. Simplify: 5 × 4 + 6 ÷ 2",
                    lines=3,
                )
                with gr.Row():
                    correct = gr.Textbox(label="Correct Answer", placeholder="e.g. 23")
                    wrong = gr.Textbox(label="Student's Wrong Answer", placeholder="e.g. 13")
                with gr.Row():
                    subject = gr.Dropdown(
                        choices=[
                            "",
                            "Number",
                            "Algebra",
                            "Geometry and Measure",
                            "Data and Statistics",
                        ],
                        label="Subject",
                        value="",
                    )
                    top_k = gr.Slider(5, 25, value=10, step=1, label="Top-K")
                btn = gr.Button("🔍 Diagnose", variant="primary")

            with gr.Column(scale=3):
                candidates_out = gr.Markdown(label="Candidate Misconceptions (Top-K)")
                rationale_out = gr.Textbox(label="Chain-of-Thought Explanation", lines=6)
                meta_out = gr.Markdown()

        btn.click(
            fn=diagnose_fn,
            inputs=[question, correct, wrong, subject, top_k],
            outputs=[candidates_out, rationale_out, meta_out],
        )

        gr.Examples(
            examples=[
                [
                    "Simplify: 5 × 4 + 6 ÷ 2",
                    "23",
                    "13",
                    "Number",
                    10,
                ],
                [
                    "Solve for x: 2x + 3 = 7",
                    "x = 2",
                    "x = 5",
                    "Algebra",
                    10,
                ],
            ],
            inputs=[question, correct, wrong, subject, top_k],
            label="Examples (click to load)",
        )

    return demo


# Mount Gradio at /ui
try:
    import gradio as gr

    gradio_app = create_gradio_app()
    app = gr.mount_gradio_app(app, gradio_app, path="/ui")
except Exception as e:
    print(f"[Service] Gradio 挂载跳过: {e}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────


def main() -> None:
    uvicorn.run(
        "service.app:app",
        host="0.0.0.0",
        port=6006,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
