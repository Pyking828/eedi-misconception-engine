"""Launch the full-pipeline Gradio demo locally (8B retrieve + rerank).

Loads STRetriever 8B + LogitReranker 8B on your own GPU, then serves the Gradio UI.
Set EEDI_SHARE=1 to expose a temporary public https://xxxx.gradio.live tunnel (~72h)
for recording a demo; otherwise it serves on the local port (EEDI_DEMO_PORT, default 6006).

This is a reproduction entry point — it runs entirely on the machine you launch it on,
so the URL lives only as long as your own process/instance does.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from service.app import _load_components, create_gradio_app, state


async def _setup() -> None:
    await _load_components()
    state.ready = True


def main() -> None:
    print("[demo] 加载真实模型（8B 召回 + 8B 粗排，约 15-20s）...")
    asyncio.run(_setup())
    if state.orchestrator is None:
        print("[demo] 组件未就绪，退出")
        return
    print("[demo] 组件就绪，启动 Gradio...")
    port = int(os.environ.get("EEDI_DEMO_PORT", "6006"))
    share = (
        os.environ.get("EEDI_SHARE", "0") == "1"
    )  # set EEDI_SHARE=1 for a temporary public tunnel
    demo = create_gradio_app()
    demo.queue()
    print(f"[demo] launching on http://localhost:{port}/ (share={share})")
    demo.launch(server_name="0.0.0.0", server_port=port, share=share, show_error=True)


if __name__ == "__main__":
    main()
