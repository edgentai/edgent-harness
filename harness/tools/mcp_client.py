"""Stdio JSON-RPC 2.0 MCP Client for connecting external MCP servers.

Spawns any stdio MCP server process, performs the ``initialize`` handshake,
discovers tools via ``tools/list``, invokes them via ``tools/call``, and
registers them dynamically onto a ``ToolRegistry`` as first-class ``ToolSpec``
entries so hooks, permissions, timeouts and tracing apply uniformly.
"""
from __future__ import annotations

import json
import subprocess
import threading
from typing import Any, Optional

from .registry import ToolContext, ToolFailure, ToolRegistry, ToolSpec

PROTOCOL_VERSION = "2024-11-05"


class MCPClient:
    """Line-delimited JSON-RPC 2.0 stdio MCP client."""

    def __init__(
        self,
        command: list[str],
        *,
        server_name: str = "external",
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        timeout_s: float = 60.0,
    ) -> None:
        self.command = list(command)
        self.server_name = server_name
        self.cwd = cwd
        self.env = env
        self.timeout_s = timeout_s
        self._proc: Optional[subprocess.Popen[str]] = None
        self._lock = threading.Lock()
        self._next_id = 0
        self.server_info: dict[str, Any] = {}

    def connect(self) -> dict[str, Any]:
        if self._proc is not None and self._proc.poll() is None:
            return self.server_info
        self._proc = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            env=self.env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        res = self._request("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": {"name": "edgent-harness", "version": "0.2.0"}})
        self.server_info = res
        self._notify("notifications/initialized", {})
        return res

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        assert self._proc and self._proc.stdin
        payload = json.dumps({"jsonrpc": "2.0", "method": method, "params": params}, ensure_ascii=False)
        with self._lock:
            self._proc.stdin.write(payload + "\n")
            self._proc.stdin.flush()

    def _request(self, method: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        if self._proc is None or self._proc.poll() is not None:
            raise ToolFailure("unavailable", f"MCP server '{self.server_name}' is not running")
        assert self._proc.stdin and self._proc.stdout
        with self._lock:
            self._next_id += 1
            req_id = self._next_id
            payload = json.dumps({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}}, ensure_ascii=False)
            self._proc.stdin.write(payload + "\n")
            self._proc.stdin.flush()

            holder: dict[str, Any] = {}

            def reader() -> None:
                while True:
                    line = self._proc.stdout.readline()  # type: ignore[union-attr]
                    if not line:
                        holder["eof"] = True
                        return
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if msg.get("id") == req_id:
                        holder["msg"] = msg
                        return

            t = threading.Thread(target=reader, daemon=True)
            t.start()
            t.join(self.timeout_s)
            if t.is_alive():
                raise ToolFailure("timeout", f"MCP server '{self.server_name}' timed out on {method}", retryable=True)
            if holder.get("eof") or "msg" not in holder:
                raise ToolFailure("unavailable", f"MCP server '{self.server_name}' closed stdout unexpectedly")
            msg = holder["msg"]
            if "error" in msg and msg["error"]:
                err = msg["error"]
                raise ToolFailure("execution_error", f"MCP error {err.get('code')}: {err.get('message')}")
            return msg.get("result") or {}

    def list_tools(self) -> list[dict[str, Any]]:
        res = self._request("tools/list", {})
        return list(res.get("tools") or [])

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        res = self._request("tools/call", {"name": name, "arguments": arguments})
        blocks = res.get("content") or []
        texts = [str(b.get("text") or "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
        joined = "\n".join(texts)
        if res.get("isError"):
            raise ToolFailure("execution_error", joined or f"MCP tool '{name}' returned error")
        try:
            return json.loads(joined)
        except Exception:
            return {"text": joined, "untrusted": True}

    def register_into(self, registry: ToolRegistry, *, prefix: Optional[str] = None) -> list[str]:
        """Discover all remote tools and register them onto ``registry``."""
        self.connect()
        registered: list[str] = []
        pfx = f"mcp__{self.server_name}__" if prefix is None else prefix
        for tool in self.list_tools():
            remote_name = str(tool.get("name") or "")
            if not remote_name:
                continue
            local_name = f"{pfx}{remote_name}"
            ann = tool.get("annotations") or {}
            perm = str(ann.get("x-permission") or ("read" if ann.get("readOnlyHint") else ("delete" if ann.get("destructiveHint") else ("network" if ann.get("openWorldHint") else "compute"))))
            if perm not in ("read", "compute", "write", "network", "spawn", "interactive", "delete"):
                perm = "compute"
            schema = tool.get("inputSchema") or {"type": "object", "properties": {}}

            def _make_fn(rname: str):
                def _fn(ctx: ToolContext, args: dict[str, Any]) -> Any:
                    return self.call_tool(rname, args)
                return _fn

            registry.register(ToolSpec(
                name=local_name,
                description=f"[MCP:{self.server_name}] {tool.get('description') or remote_name}",
                parameters=schema,
                fn=_make_fn(remote_name),
                permission=perm,
                timeout_s=float(ann.get("x-timeout_s") or self.timeout_s),
                side_effects=not bool(ann.get("readOnlyHint", False)),
            ))
            registered.append(local_name)
        return registered

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.poll() is None:
                self._proc.terminate()
                self._proc.wait(timeout=2.0)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        for stream in (self._proc.stdin, self._proc.stdout, self._proc.stderr):
            try:
                if stream:
                    stream.close()
            except Exception:
                pass
        self._proc = None
