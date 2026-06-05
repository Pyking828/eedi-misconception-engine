"""
HuggingFace Spaces free CPU demo.
Runs lightweight retrieval only (bge-m3 + FAISS CPU); no GPU required.
Users can search the misconception bank by keyword or run zero-shot diagnosis from a math question.

Cold-start: if precomputed embeddings (misc_embs_bge_m3.npy) exist in the repo, load them and
encode only the user query at runtime (~0.5s on CPU); otherwise encode all 2587 items on first request (~1–2 min).
"""

import json
import os
from pathlib import Path

import faiss
import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).parent
DATA_FILE = HERE / "misconceptions.txt"
IDS_FILE = HERE / "misconception_ids.json"
EMB_FILE = HERE / "misc_embs_bge_m3.npy"
GITHUB_URL = os.environ.get(
    "EEDI_GITHUB_URL", "https://github.com/Pyking828/eedi-misconception-engine"
)


def load_misconceptions() -> tuple[list[str], list[int]]:
    if DATA_FILE.exists():
        names = DATA_FILE.read_text().strip().split("\n")
        ids = json.loads(IDS_FILE.read_text()) if IDS_FILE.exists() else list(range(len(names)))
        return names, [int(x) for x in ids]
    demo = [
        "Carries out operations from left to right regardless of priority order",
        "Thinks that fractions with different denominators cannot be added",
        "Believes that multiplying by a fraction always makes a number smaller",
        "When adding fractions, adds the numerators and denominators",
    ]
    return demo, list(range(len(demo)))


MISCONCEPTIONS, MISC_IDS = load_misconceptions()

_model = None
_index = None


def get_model_and_index():
    global _model, _index
    if _model is None:
        print("Loading bge-m3 (first time)...")
        _model = SentenceTransformer("BAAI/bge-m3", device="cpu")
        if EMB_FILE.exists():
            print("Loading precomputed embeddings (fast cold start)...")
            embs = np.load(EMB_FILE).astype(np.float32)
        else:
            print(f"Encoding {len(MISCONCEPTIONS)} misconceptions (~1-2 min)...")
            embs = _model.encode(
                MISCONCEPTIONS, normalize_embeddings=True, show_progress_bar=False
            ).astype(np.float32)
        _index = faiss.IndexFlatIP(embs.shape[1])
        _index.add(embs)
    return _model, _index


def search_fn(query: str, top_k: int = 10) -> str:
    if not query.strip():
        return "请输入检索词"
    model, index = get_model_and_index()
    q_emb = model.encode([query], normalize_embeddings=True)
    D, I = index.search(q_emb.astype(np.float32), top_k)
    results = []
    for rank, (idx, score) in enumerate(zip(I[0], D[0]), 1):
        results.append(f"{rank}. **[{MISC_IDS[idx]}]** {MISCONCEPTIONS[idx]} (score={score:.3f})")
    return "\n\n".join(results)


def diagnose_fn(question: str, correct: str, wrong: str, top_k: int = 5) -> str:
    query = (
        f"Question: {question}\n"
        f"Correct Answer: {correct}\n"
        f"Incorrect Answer: {wrong}\n"
        f"Identify the mathematical misconception:"
    )
    return search_fn(query, top_k)


# ─── UI ──────────────────────────────────────────────────────────────
with gr.Blocks(title="Eedi Misconception Retrieval Demo") as demo:
    gr.Markdown(
        "# 🎓 Eedi Math Misconception Retrieval Demo\n"
        "> Lightweight CPU demo (bge-m3 + FAISS over 2587 real misconceptions). "
        "The full GPU pipeline (Qwen3-Embedding-8B LoRA retriever + Qwen3-Reranker-8B + ensemble, "
        "CV MAP@25=0.597) runs locally.\n"
        f"> [Full project on GitHub]({GITHUB_URL})"
    )

    with gr.Tab("Diagnose"):
        with gr.Row():
            with gr.Column():
                q = gr.Textbox(
                    label="Question", placeholder="e.g. Simplify: 5 × 4 + 6 ÷ 2", lines=2
                )
                correct = gr.Textbox(label="Correct Answer", placeholder="e.g. 23")
                wrong = gr.Textbox(label="Student's Wrong Answer", placeholder="e.g. 13")
                top_k_slider = gr.Slider(3, 15, value=5, step=1, label="Top-K")
                btn = gr.Button("Diagnose", variant="primary")
            with gr.Column():
                out = gr.Markdown(label="Candidate Misconceptions")

        btn.click(diagnose_fn, inputs=[q, correct, wrong, top_k_slider], outputs=out)
        gr.Examples(
            [
                ["Simplify: 5 × 4 + 6 ÷ 2", "23", "13", 5],
                ["Solve: 2x + 3 = 7", "x = 2", "x = 5", 5],
            ],
            inputs=[q, correct, wrong, top_k_slider],
        )

    with gr.Tab("Search misconception bank"):
        with gr.Row():
            search_q = gr.Textbox(label="Query", placeholder="e.g. order of operations fraction")
            search_k = gr.Slider(3, 15, value=8, step=1, label="Top-K")
        search_btn = gr.Button("Search")
        search_out = gr.Markdown()
        search_btn.click(search_fn, inputs=[search_q, search_k], outputs=search_out)

    gr.Markdown(
        "---\n"
        "⚡ **Full pipeline (local GPU):** Qwen3-Embedding-8B LoRA + Qwen3-Reranker-8B + "
        "DeepSeek-R1-Distill-Qwen-14B listwise/GRPO + CoT reasoning + FastAPI SSE + MCP Server."
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
