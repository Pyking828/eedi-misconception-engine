"""
MCP Server（Model Context Protocol）— 对应 JD "下游mcp接入"。

暴露工具：
  - diagnose_misconception：给定数学题+错误选项，返回 top-K 错因候选
  - search_misconceptions：纯文本检索错因库
  - get_misconception_detail：获取指定 misconception 的详细信息

协议：stdio（JSON-RPC 2.0 over stdin/stdout）
使用方：可在 Cursor / Claude Desktop 的 MCP 配置中直接接入。

Cursor 接入配置（写入 .cursor/mcp.json）：
{
  "mcpServers": {
    "eedi-misconception-engine": {
      "command": "python",
      "args": ["/root/autodl-tmp/eedi-misconception-engine/mcp_server/server.py"],
      "env": {
        "HF_HOME": "/root/autodl-tmp/hf_cache",
        "EEDI_CONFIG": "/root/autodl-tmp/eedi-misconception-engine/configs/base.yaml"
      }
    }
  }
}
"""
from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import httpx


# 对接本地 FastAPI 服务（MCP → HTTP → FastAPI）
FASTAPI_BASE = "http://localhost:6006"


# ─────────────────────────────────────────────
# MCP 协议工具函数
# ─────────────────────────────────────────────

def mcp_response(request_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def mcp_error(request_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


# ─────────────────────────────────────────────
# 工具定义（MCP tools/list 返回值）
# ─────────────────────────────────────────────

TOOLS = [
    {
        "name": "diagnose_misconception",
        "description": (
            "给定一道数学题及学生的错误答案，从 2587 条数学错因库中检索最可能的错因，"
            "返回 top-K 候选（含 ID、名称、置信度）及 CoT 推理解释。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question_text": {"type": "string", "description": "数学题目文本"},
                "correct_answer": {"type": "string", "description": "正确答案"},
                "wrong_answer": {"type": "string", "description": "学生的错误答案"},
                "subject_name": {"type": "string", "description": "学科（可选）"},
                "top_k": {"type": "integer", "default": 5, "description": "返回候选数"},
                "include_rationale": {"type": "boolean", "default": True},
            },
            "required": ["question_text", "correct_answer", "wrong_answer"],
        },
    },
    {
        "name": "search_misconceptions",
        "description": "用自由文本检索数学错因库，返回语义最相关的错因列表。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索文本"},
                "top_k": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_misconception_detail",
        "description": "获取指定 MisconceptionId 的详细描述。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "misconception_id": {"type": "integer"},
            },
            "required": ["misconception_id"],
        },
    },
]


# ─────────────────────────────────────────────
# 工具执行
# ─────────────────────────────────────────────

async def call_tool(name: str, args: dict) -> Any:
    async with httpx.AsyncClient(base_url=FASTAPI_BASE, timeout=60.0) as client:
        if name == "diagnose_misconception":
            resp = await client.post("/diagnose", json=args)
            resp.raise_for_status()
            data = resp.json()
            # 格式化为 MCP 文本内容
            lines = [f"## 错因诊断结果（{data['pipeline_mode']} | {data['latency_ms']:.0f}ms）\n"]
            for c in data["candidates"]:
                lines.append(f"{c['rank']}. [{c['misconception_id']}] {c['misconception_name']} (score={c['score']:.3f})")
            if data.get("rationale"):
                lines.append(f"\n**推理解释：** {data['rationale']}")
            return "\n".join(lines)

        elif name == "search_misconceptions":
            resp = await client.post("/search", json=args)
            resp.raise_for_status()
            results = resp.json()
            lines = [f"## 检索结果（query: {args['query'][:50]}...）\n"]
            for r in results:
                lines.append(f"- [{r['misconception_id']}] {r['misconception_name']} ({r['score']:.3f})")
            return "\n".join(lines)

        elif name == "get_misconception_detail":
            mid = args["misconception_id"]
            resp = await client.get(f"/metrics")
            # 简化：从 /health 获取全量 misc_texts 不现实，这里从本地 CSV 读
            return f"MisconceptionId={mid}（需本地 CSV 查询，或通过 /search 检索）"

        else:
            raise ValueError(f"Unknown tool: {name}")


# ─────────────────────────────────────────────
# MCP stdio 主循环
# ─────────────────────────────────────────────

async def main() -> None:
    """读取 stdin JSON-RPC，处理后写 stdout。"""
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

    writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
        asyncio.BaseProtocol, sys.stdout
    )

    def write(obj: dict) -> None:
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        sys.stdout.write(line)
        sys.stdout.flush()

    while True:
        try:
            line = await reader.readline()
            if not line:
                break
            request = json.loads(line.decode().strip())
        except (EOFError, json.JSONDecodeError):
            break

        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params", {})

        if method == "initialize":
            write(
                mcp_response(
                    req_id,
                    {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "eedi-misconception-engine", "version": "0.1.0"},
                    },
                )
            )

        elif method == "tools/list":
            write(mcp_response(req_id, {"tools": TOOLS}))

        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})
            try:
                result = await call_tool(tool_name, tool_args)
                write(
                    mcp_response(
                        req_id,
                        {"content": [{"type": "text", "text": result}]},
                    )
                )
            except Exception as e:
                write(mcp_error(req_id, -32000, str(e)))

        elif method == "notifications/initialized":
            pass  # no-op

        else:
            write(mcp_error(req_id, -32601, f"Method not found: {method}"))


if __name__ == "__main__":
    asyncio.run(main())
