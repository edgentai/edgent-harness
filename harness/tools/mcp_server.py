"""Stdio JSON-RPC 2.0 Model Context Protocol (MCP) server.

Exposes any ``ToolRegistry`` plus guided workflow prompts over stdin/stdout so
external hosts (Antigravity ``agy``, Claude Code, Codex, Cursor) can drive
Edgent's tools and workspace conventions directly in Host Mode.
"""
from __future__ import annotations

import json
import sys
from typing import Any, Callable, IO, Optional

from .registry import ToolContext, ToolRegistry

PROTOCOL_VERSION = "2024-11-05"

DEFAULT_PROMPTS: dict[str, dict[str, Any]] = {
    "plan_task": {
        "name": "plan_task",
        "description": "Inspect the workspace, identify target files and risks, propose a step-by-step plan, and pause for approval.",
        "arguments": [{"name": "goal", "description": "Task goal", "required": True}],
        "template": (
            "Orient using list_files, glob_files, and read_file on relevant workspace files and AGENTS.md. "
            "Draft a concrete step-by-step plan with verification criteria, then call request_review before editing. Goal: {goal}"
        ),
    },
    "review_work": {
        "name": "review_work",
        "description": "Independently audit a modified file or deliverable against requirements and executable checks.",
        "arguments": [{"name": "target", "description": "Workspace path or deliverable to review", "required": True}],
        "template": (
            "Inspect {target} using read_file and verify any code or behavior with bash / run_python / run_tests. "
            "Return a concise, concrete list of blocking defects and suggested fixes."
        ),
    },
    "investigate": {
        "name": "investigate",
        "description": "Search workspace files and artifacts to answer a technical question with exact file paths and quotes.",
        "arguments": [{"name": "question", "description": "Question to investigate", "required": True}],
        "template": (
            "Use glob_files, search_files, and read_file to gather exact evidence for the question below. "
            "Cite file paths and line numbers for every claim, and state clearly what could not be verified. Question: {question}"
        ),
    },
}


class MCPServer:
    """Line-delimited JSON-RPC 2.0 MCP server over a ToolRegistry."""

    def __init__(
        self,
        registry: ToolRegistry,
        ctx_factory: Callable[[], ToolContext],
        *,
        server_name: str = "edgent-harness",
        server_version: str = "0.2.0",
        compact_schemas: bool = False,
        allowlist: Optional[set[str]] = None,
        prompts: Optional[dict[str, dict[str, Any]]] = None,
    ) -> None:
        self.registry = registry
        self.ctx_factory = ctx_factory
        self.server_name = server_name
        self.server_version = server_version
        self.compact_schemas = compact_schemas
        self.allowlist = allowlist
        self.prompts = dict(DEFAULT_PROMPTS if prompts is None else prompts)

    def handle_message(self, msg: dict[str, Any]) -> Optional[dict[str, Any]]:
        req_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}

        # Notifications have no id and expect no reply.
        if req_id is None and method and method.startswith("notifications/"):
            return None

        def ok(result: Any) -> dict[str, Any]:
            return {"jsonrpc": "2.0", "id": req_id, "result": result}

        def err(code: int, message: str, data: Any = None) -> dict[str, Any]:
            e: dict[str, Any] = {"code": code, "message": message}
            if data is not None:
                e["data"] = data
            return {"jsonrpc": "2.0", "id": req_id, "error": e}

        if method == "initialize":
            return ok({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}, "prompts": {}},
                "serverInfo": {"name": self.server_name, "version": self.server_version},
            })
        if method == "ping":
            return ok({})
        if method == "tools/list":
            return ok(self.registry.mcp_manifest(compact=self.compact_schemas, allowlist=self.allowlist))
        if method == "tools/call":
            name = str(params.get("name") or "")
            args = params.get("arguments") or {}
            ctx = self.ctx_factory()
            res = self.registry.call(name, args, ctx, allowlist=self.allowlist)
            payload = res.output if res.ok else {"error": res.error.to_dict() if res.error else "failed"}
            return ok({
                "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, default=str)}],
                "isError": not res.ok,
            })
        if method == "prompts/list":
            return ok({
                "prompts": [
                    {"name": p["name"], "description": p["description"], "arguments": p.get("arguments", [])}
                    for p in self.prompts.values()
                ]
            })
        if method == "prompts/get":
            pname = str(params.get("name") or "")
            p = self.prompts.get(pname)
            if p is None:
                return err(-32602, f"unknown prompt '{pname}'")
            pargs = {a["name"]: str((params.get("arguments") or {}).get(a["name"], "")) for a in p.get("arguments", [])}
            text = p["template"].format(**pargs)
            return ok({
                "description": p["description"],
                "messages": [{"role": "user", "content": {"type": "text", "text": text}}],
            })
        return err(-32601, f"method not found: {method}")

    def serve_stdio(self, stdin: Optional[IO[str]] = None, stdout: Optional[IO[str]] = None) -> None:
        inp = stdin or sys.stdin
        out = stdout or sys.stdout
        for line in inp:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as e:
                resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"parse error: {e}"}}
                out.write(json.dumps(resp) + "\n")
                out.flush()
                continue
            resp = self.handle_message(msg)
            if resp is not None:
                out.write(json.dumps(resp, ensure_ascii=False, default=str) + "\n")
                out.flush()
