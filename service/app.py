"""
FastAPI 异步服务（对应 JD：asyncio/fastapi/函数注解）。

端点：
  POST /diagnose           → 完整诊断（非流式）
  POST /diagnose/stream    → SSE 流式诊断
  POST /search             → 纯向量检索（轻量）
  POST /feedback           → 用户反馈提交
  GET  /health             → 健康检查
  GET  /metrics            → 简单指标统计
  GET  /docs               → Swagger 自动文档

设计：
  - Pydantic v2 请求/响应模型，完整类型注解
  - 启动时懒加载 GPU 组件（避免 import 时 OOM）
  - lifespan context manager 管理资源（SQLite 连接等）
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse


# ─────────────────────────────────────────────
# 全局组件容器（懒加载）
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
    """FastAPI lifespan：启动时加载组件，关闭时释放资源。"""
    print("[Service] 初始化中...")
    await _load_components()
    state.ready = True
    print("[Service] 就绪 ✓")
    yield
    if state.memory:
        await state.memory.close()
    print("[Service] 已关闭")


async def _load_components() -> None:
    """懒加载 GPU 组件（仅在配置文件和模型存在时加载）。"""
    from omegaconf import OmegaConf

    cfg_path = Path(__file__).parent.parent / "configs" / "base.yaml"
    if not cfg_path.exists():
        print("[Service] 警告：configs/base.yaml 不存在，以 mock 模式运行")
        return

    cfg = OmegaConf.load(str(cfg_path))

    # 加载 misconception 文本（轻量，必须）
    misc_csv = Path(cfg.data.misconception_csv)
    if misc_csv.exists():
        import polars as pl
        misc_df = pl.read_csv(str(misc_csv))
        state.misc_texts = {
            r["MisconceptionId"]: r["MisconceptionName"]
            for r in misc_df.iter_rows(named=True)
        }

    # 加载记忆模块
    from src.eedi.memory.memory import MemoryModule
    state.memory = MemoryModule(cfg.memory.db_path)
    await state.memory.init()

    # 加载 Retriever（索引存在时）
    index_path = Path(cfg.retriever.faiss_index_path)
    if index_path.exists():
        from src.eedi.retriever.retriever import EediRetriever
        retriever = EediRetriever.from_pretrained(
            model_name=cfg.retriever.model_name,
            index_path=str(index_path),
            misc_ids=list(state.misc_texts.keys()),
            misc_texts=state.misc_texts,
            cache_dir=cfg.paths.hf_cache,
        )
    else:
        retriever = None
        print("[Service] FAISS 索引未找到，向量检索不可用")

    # 组装 Orchestrator
    if retriever is not None:
        from src.eedi.orchestrator import Orchestrator, PromptRegistry
        from src.eedi.router.router import QueryRouter

        prompts_dir = Path(__file__).parent.parent / "prompts"
        prompt_reg = PromptRegistry(prompts_dir) if prompts_dir.exists() else None
        router = QueryRouter(memory=state.memory)
        state.orchestrator = Orchestrator(
            retriever=retriever,
            router=router,
            memory=state.memory,
            misc_texts=state.misc_texts,
            prompt_registry=prompt_reg,
        )


# ─────────────────────────────────────────────
# FastAPI 应用
# ─────────────────────────────────────────────

app = FastAPI(
    title="Eedi Misconception Engine API",
    description=(
        "数学错因诊断中控系统 — 企业级召回→粗排→精排→CoT推理 级联服务。\n\n"
        "对应 vivo 蓝心小v 中控 3.0 架构：工具检索召回/排序 + 智能路由 + "
        "prompt管理 + 感知记忆 + MCP/SubAgent接入。"
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
# Pydantic 模型（pydantic v2 + 完整类型注解）
# ─────────────────────────────────────────────

class DiagnoseInput(BaseModel):
    question_text: str = Field(..., description="数学题目文本")
    correct_answer: str = Field(..., description="正确答案文本")
    wrong_answer: str = Field(..., description="学生的错误答案文本")
    subject_name: str = Field(default="", description="学科（Number/Algebra/...）")
    construct_name: str = Field(default="", description="知识点")
    session_id: str = Field(default="default", description="会话 ID（用于记忆）")
    top_k: int = Field(default=25, ge=1, le=50, description="返回候选数")
    include_rationale: bool = Field(default=True, description="是否生成 CoT 推理解释")


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
    query: str = Field(..., description="自由文本检索 query")
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
# 路由
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
    """Server-Sent Events 流式接口：先快速返回召回结果，再推流 CoT 推理。"""
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
    """跳过重排，直接返回向量召回结果（适合 HF Spaces CPU demo）。"""
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
# Gradio UI（挂载到 /ui）
# ─────────────────────────────────────────────

def create_gradio_app():
    import gradio as gr

    async def diagnose_fn(question, correct, wrong, subject, top_k):
        if state.orchestrator is None:
            return "⚠️ 模型尚未加载，请先运行向量索引构建脚本。", ""
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
            f"📡 模式: `{resp.pipeline_mode}` | "
            f"学科: `{resp.subject}` | "
            f"缓存: `{resp.from_cache}` | "
            f"延迟: `{resp.latency_ms:.0f}ms`"
        )
        return cand_md, resp.rationale, meta

    with gr.Blocks(title="Eedi 错因诊断中控", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🎓 Eedi 数学错因诊断中控系统\n"
            "> 召回 → 粗排 → 精排 → CoT推理 全链路 | "
            "Qwen3-Embedding + LoRA + GRPO | vivo蓝心中控3.0风格"
        )

        with gr.Row():
            with gr.Column(scale=2):
                question = gr.Textbox(
                    label="题目 (QuestionText)",
                    placeholder="e.g. Simplify: 5 × 4 + 6 ÷ 2",
                    lines=3,
                )
                with gr.Row():
                    correct = gr.Textbox(label="正确答案", placeholder="e.g. 23")
                    wrong = gr.Textbox(label="学生错误答案", placeholder="e.g. 13")
                with gr.Row():
                    subject = gr.Dropdown(
                        choices=["", "Number", "Algebra", "Geometry and Measure", "Data and Statistics"],
                        label="学科", value=""
                    )
                    top_k = gr.Slider(5, 25, value=10, step=1, label="Top-K")
                btn = gr.Button("🔍 诊断错因", variant="primary")

            with gr.Column(scale=3):
                candidates_out = gr.Markdown(label="候选错因（Top-K）")
                rationale_out = gr.Textbox(label="CoT 推理解释", lines=6)
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
        )

    return demo


# 挂载 Gradio 到 /ui
try:
    import gradio as gr
    gradio_app = create_gradio_app()
    app = gr.mount_gradio_app(app, gradio_app, path="/ui")
except Exception as e:
    print(f"[Service] Gradio 挂载跳过: {e}")


# ─────────────────────────────────────────────
# 启动入口
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
