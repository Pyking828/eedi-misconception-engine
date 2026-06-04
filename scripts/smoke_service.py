"""End-to-end smoke test of the production pipeline (retriever -> pointwise rerank).

Loads the real multistage 8B retriever + 8B yes/no reranker + orchestrator and runs a
couple of sample diagnoses, printing ranked misconceptions + latency. Validates the P5
wiring before launching the FastAPI/Gradio service.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import polars as pl
from omegaconf import OmegaConf
from rich.console import Console

console = Console()
ROOT = Path("/root/autodl-tmp/eedi-misconception-engine")


async def main() -> None:
    cfg = OmegaConf.load(str(ROOT / "configs/base.yaml"))
    misc_df = pl.read_csv(cfg.data.misconception_csv)
    misc_texts = {
        int(r["MisconceptionId"]): r["MisconceptionName"] for r in misc_df.iter_rows(named=True)
    }

    import json

    from src.eedi.orchestrator import DiagnoseRequest, Orchestrator
    from src.eedi.reranker.pointwise import LogitReranker
    from src.eedi.retriever.st_engine import STRetriever
    from src.eedi.router.router import QueryRouter

    misc_ids = [int(x) for x in json.loads(Path(cfg.retriever.misc_ids_path).read_text())]

    console.rule("[bold blue]Loading retriever (8B + multistage adapter)")
    t = time.time()
    retriever = STRetriever.from_pretrained(
        model_name=cfg.retriever.model_name,
        index_path=cfg.retriever.faiss_index_path,
        misc_ids=misc_ids,
        misc_texts=misc_texts,
        adapter_path=cfg.retriever.adapter_path,
        cache_dir=cfg.paths.hf_cache,
    )
    console.print(f"[green]retriever loaded in {time.time() - t:.0f}s")

    console.rule("[bold blue]Loading pointwise reranker (8B + best adapter)")
    t = time.time()
    pointwise = LogitReranker.from_pretrained(
        model_name=cfg.reranker.pointwise.model_name,
        adapter_path=cfg.reranker.pointwise.adapter_path,
        max_length=cfg.reranker.pointwise.max_seq_len,
        cache_dir=cfg.paths.hf_cache,
    )
    console.print(f"[green]reranker loaded in {time.time() - t:.0f}s")

    orch = Orchestrator(
        retriever=retriever,
        pointwise_reranker=pointwise,
        router=QueryRouter(),
        misc_texts=misc_texts,
    )

    samples = [
        dict(
            question_text="Simplify fully: 6/8",
            correct_answer="3/4",
            wrong_answer="6/8",
            subject_name="Number",
            construct_name="Simplify fractions",
        ),
        dict(
            question_text="Work out 3.2 x 10",
            correct_answer="32",
            wrong_answer="3.20",
            subject_name="Number",
            construct_name="Multiply decimals by powers of 10",
        ),
    ]
    for s in samples:
        req = DiagnoseRequest(**s, top_k=5, include_rationale=False)
        t = time.time()
        resp = await orch.diagnose(req)
        console.rule(
            f"[bold]Q: {s['question_text']}  (correct={s['correct_answer']}, wrong={s['wrong_answer']})"
        )
        console.print(
            f"[cyan]mode={resp.pipeline_mode} subject={resp.subject} latency={resp.latency_ms:.0f}ms"
        )
        for c in resp.candidates[:5]:
            console.print(f"  {c.rank}. [{c.misconception_id}] {c.misconception_name}")

    console.print("\n[bold green]✓ Production pipeline smoke OK")


if __name__ == "__main__":
    asyncio.run(main())
