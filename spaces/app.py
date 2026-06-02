"""
HuggingFace Spaces 免费 CPU demo。
只跑轻量召回（bge-m3 + FAISS CPU），不需要 GPU。
允许用户输入 misconception 关键词检索错因库，或输入数学问题做零样本诊断。
"""
import os
from pathlib import Path

import gradio as gr
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ─── 数据加载（从本 repo 内的预存文件中加载，避免 Kaggle 依赖）──────
DATA_FILE = Path(__file__).parent / "misconceptions.txt"

def load_misconceptions() -> list[str]:
    if DATA_FILE.exists():
        return DATA_FILE.read_text().strip().split("\n")
    # 示例数据（正式部署时替换为完整 2587 条）
    return [
        "Carries out operations from left to right regardless of priority order",
        "Confuses the order of operations when brackets are involved",
        "Does not know that the gradient of a horizontal line is 0",
        "Thinks that fractions with different denominators cannot be added",
        "Believes that multiplying by a fraction always makes a number smaller",
        "Confuses the meaning of the equals sign with assignment",
        "Does not understand the concept of a variable",
        "Thinks that a negative times a negative is negative",
    ]

MISCONCEPTIONS = load_misconceptions()

# 懒加载模型（冷启动时下载）
_model = None
_index = None
_embs = None


def get_model_and_index():
    global _model, _index, _embs
    if _model is None:
        print("Loading model (first time, may take ~1 min)...")
        _model = SentenceTransformer("BAAI/bge-m3", device="cpu")
        _embs = _model.encode(MISCONCEPTIONS, normalize_embeddings=True, show_progress_bar=False)
        _index = faiss.IndexFlatIP(_embs.shape[1])
        _index.add(_embs.astype(np.float32))
    return _model, _index


def search_fn(query: str, top_k: int = 10) -> str:
    if not query.strip():
        return "请输入检索词"
    model, index = get_model_and_index()
    q_emb = model.encode([query], normalize_embeddings=True)
    D, I = index.search(q_emb.astype(np.float32), top_k)
    results = []
    for rank, (idx, score) in enumerate(zip(I[0], D[0]), 1):
        results.append(f"{rank}. **{MISCONCEPTIONS[idx]}** (score={score:.3f})")
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
with gr.Blocks(title="Eedi 错因检索 Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🎓 Eedi 数学错因检索 Demo\n"
        "> CPU 版轻量 Demo（bge-m3 + FAISS）· 完整 GPU 版含 Qwen3-Embedding LoRA 微调 + GRPO 精排\n"
        "> [GitHub 完整项目](https://github.com/YOUR_GITHUB/eedi-misconception-engine)"
    )

    with gr.Tab("错因诊断"):
        with gr.Row():
            with gr.Column():
                q = gr.Textbox(label="题目", placeholder="e.g. Simplify: 5 × 4 + 6 ÷ 2", lines=2)
                correct = gr.Textbox(label="正确答案", placeholder="e.g. 23")
                wrong = gr.Textbox(label="学生错误答案", placeholder="e.g. 13")
                top_k_slider = gr.Slider(3, 15, value=5, step=1, label="Top-K")
                btn = gr.Button("诊断错因", variant="primary")
            with gr.Column():
                out = gr.Markdown(label="候选错因")

        btn.click(diagnose_fn, inputs=[q, correct, wrong, top_k_slider], outputs=out)
        gr.Examples(
            [
                ["Simplify: 5 × 4 + 6 ÷ 2", "23", "13", 5],
                ["Solve: 2x + 3 = 7", "x = 2", "x = 5", 5],
            ],
            inputs=[q, correct, wrong, top_k_slider],
        )

    with gr.Tab("错因库检索"):
        with gr.Row():
            search_q = gr.Textbox(label="检索词", placeholder="e.g. order of operations fraction")
            search_k = gr.Slider(3, 15, value=8, step=1, label="Top-K")
        search_btn = gr.Button("检索")
        search_out = gr.Markdown()
        search_btn.click(search_fn, inputs=[search_q, search_k], outputs=search_out)

    gr.Markdown(
        "---\n"
        "⚡ **完整版功能**（本地 GPU）：Qwen3-Embedding-8B LoRA + Qwen3-Reranker-8B + "
        "DeepSeek-R1-Distill-Qwen-14B GRPO精排/CoT推理 + 32B离线teacher/judge + FastAPI SSE + MCP"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
