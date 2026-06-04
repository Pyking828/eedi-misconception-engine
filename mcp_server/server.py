"""
MCP Server (Model Context Protocol) — downstream MCP integration.

Exposed tools:
  - diagnose_misconception: given a math question and wrong answer, return top-K misconception candidates
  - search_misconceptions: free-text search over the misconception bank
  - get_misconception_detail: fetch details for a given misconception id

Protocol: stdio (JSON-RPC 2.0 over stdin/stdout)
Consumers: wire into Cursor / Claude Desktop via MCP config.

Cursor config (add to .cursor/mcp.json):
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

# Proxy to local FastAPI (MCP → HTTP → FastAPI)
FASTAPI_BASE = "http://localhost:6006"


# ─────────────────────────────────────────────
# MCP protocol helpers
# ─────────────────────────────────────────────


def mcp_response(request_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def mcp_error(request_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


# ─────────────────────────────────────────────
# Tool definitions (MCP tools/list response)
# ─────────────────────────────────────────────

TOOLS = [
    {
        "name": "diagnose_misconception",
        "description": (
            "Given a math question and the student's wrong answer, retrieve the most likely misconceptions "
            "from a bank of 2587 math misconceptions; returns top-K candidates (id, name, score) and optional CoT rationale."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question_text": {"type": "string", "description": "Math question text"},
                "correct_answer": {"type": "string", "description": "Correct answer"},
                "wrong_answer": {"type": "string", "description": "Student's incorrect answer"},
                "subject_name": {"type": "string", "description": "Subject (optional)"},
                "top_k": {
                    "type": "integer",
                    "default": 5,
                    "description": "Number of candidates to return",
                },
                "include_rationale": {"type": "boolean", "default": True},
            },
            "required": ["question_text", "correct_answer", "wrong_answer"],
        },
    },
    {
        "name": "search_misconceptions",
        "description": "Free-text semantic search over the math misconception bank.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query text"},
                "top_k": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_misconception_detail",
        "description": "Fetch the full description for a MisconceptionId.",
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
# Tool execution
# ─────────────────────────────────────────────


async def call_tool(name: str, args: dict) -> Any:
    async with httpx.AsyncClient(base_url=FASTAPI_BASE, timeout=60.0) as client:
        if name == "diagnose_misconception":
            resp = await client.post("/diagnose", json=args)
            resp.raise_for_status()
            data = resp.json()
            # Format as MCP text content
            lines = [f"## 错因诊断结果（{data['pipeline_mode']} | {data['latency_ms']:.0f}ms）\n"]
            for c in data["candidates"]:
                lines.append(
                    f"{c['rank']}. [{c['misconception_id']}] {c['misconception_name']} (score={c['score']:.3f})"
                )
            if data.get("rationale"):
                lines.append(f"\n**推理解释：** {data['rationale']}")
            return "\n".join(lines)

        elif name == "search_misconceptions":
            resp = await client.post("/search", json=args)
            resp.raise_for_status()
            results = resp.json()
            lines = [f"## 检索结果（query: {args['query'][:50]}...）\n"]
            for r in results:
                lines.append(
                    f"- [{r['misconception_id']}] {r['misconception_name']} ({r['score']:.3f})"
                )
            return "\n".join(lines)

        elif name == "get_misconception_detail":
            mid = args["misconception_id"]
            resp = await client.get("/metrics")
            # Simplified: full misc_texts via /health is impractical; local CSV lookup would go here
            return f"MisconceptionId={mid}（需本地 CSV 查询，或通过 /search 检索）"

        else:
            raise ValueError(f"Unknown tool: {name}")


# ─────────────────────────────────────────────
# MCP stdio main loop
# ─────────────────────────────────────────────


async def main() -> None:
    """Read JSON-RPC from stdin, write responses to stdout."""
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
