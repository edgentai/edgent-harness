"""Tool registry: one definition per tool, MCP-compatible export, enforcement in code.

Every call goes through ``ToolRegistry.call`` which validates arguments
against the JSON schema, checks the caller's allowlist and the permission
policy, applies the timeout, consults the idempotency store for
side-effecting tools, compacts oversized outputs and returns a typed
``ToolResult`` (never a stack trace).
"""
from __future__ import annotations

import concurrent.futures
import json
import sqlite3
from contextlib import closing
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..core.types import ToolError, ToolResult
from ..core.utils import IdGenerator, json_dumps, truncate

PERMISSION_CLASSES = ("read", "compute", "write", "network", "spawn", "interactive", "delete")
DANGEROUS = {"write", "network", "delete"}
MAX_INLINE_OUTPUT_CHARS = 6000


@dataclass
class ToolContext:
    """Everything a tool may touch. Built per sub-agent, never shared across runs."""

    run_id: str
    node_id: str
    store: Any  # ArtifactStore
    workspace: Any  # Workspace
    sidecars: Any  # Sidecars
    config: Any  # HarnessConfig
    memory: Any = None  # MemorySystem
    tracer: Any = None
    ask_user: Optional[Callable[[str], str]] = None
    spawn: Optional[Callable[[str, str, dict[str, Any]], dict[str, Any]]] = None
    approve: Optional[Callable[[str, dict[str, Any]], bool]] = None  # policy approval for dangerous tools
    role: str = "executor"
    extra: dict[str, Any] = field(default_factory=dict)


ToolFn = Callable[[ToolContext, dict[str, Any]], Any]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: ToolFn
    returns: dict[str, Any] = field(default_factory=lambda: {"type": "object"})
    permission: str = "read"
    timeout_s: float = 60.0
    idempotent: bool = True
    side_effects: bool = False
    cost_class: str = "cheap"  # cheap | moderate | expensive
    when_not_to_use: str = ""
    examples: list[dict[str, Any]] = field(default_factory=list)
    version: int = 1

    def openai_schema(self) -> dict[str, Any]:
        desc = self.description
        if self.when_not_to_use:
            desc += f" Do not use when: {self.when_not_to_use}"
        if self.examples:
            desc += " Example: " + json.dumps(self.examples[0])
        return {"type": "function", "function": {"name": self.name, "description": desc, "parameters": self.parameters}}

    def mcp_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.parameters,
            "outputSchema": self.returns,
            "annotations": {
                "readOnlyHint": not self.side_effects,
                "idempotentHint": self.idempotent,
                "destructiveHint": self.permission == "delete",
                "openWorldHint": self.permission == "network",
                "x-permission": self.permission,
                "x-timeout_s": self.timeout_s,
                "x-cost_class": self.cost_class,
            },
        }


def validate_args(schema: dict[str, Any], args: dict[str, Any]) -> list[str]:
    """Small JSON-schema validator: types, required, enum, bounds, nested objects/arrays."""
    errors: list[str] = []

    def check(sch: dict[str, Any], val: Any, path: str) -> None:
        t = sch.get("type")
        if "enum" in sch and val not in sch["enum"]:
            errors.append(f"{path}: must be one of {sch['enum']}")
            return
        if t is None:
            return
        types = t if isinstance(t, list) else [t]
        ok = False
        for tt in types:
            if tt == "string" and isinstance(val, str):
                ok = True
            elif tt == "integer" and isinstance(val, int) and not isinstance(val, bool):
                ok = True
            elif tt == "number" and isinstance(val, (int, float)) and not isinstance(val, bool):
                ok = True
            elif tt == "boolean" and isinstance(val, bool):
                ok = True
            elif tt == "array" and isinstance(val, list):
                ok = True
            elif tt == "object" and isinstance(val, dict):
                ok = True
            elif tt == "null" and val is None:
                ok = True
        if not ok:
            errors.append(f"{path}: expected {t}, got {type(val).__name__}")
            return
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            if "minimum" in sch and val < sch["minimum"]:
                errors.append(f"{path}: below minimum {sch['minimum']}")
            if "maximum" in sch and val > sch["maximum"]:
                errors.append(f"{path}: above maximum {sch['maximum']}")
        if isinstance(val, str) and "maxLength" in sch and len(val) > sch["maxLength"]:
            errors.append(f"{path}: longer than {sch['maxLength']}")
        if isinstance(val, dict):
            for req in sch.get("required", []):
                if req not in val:
                    errors.append(f"{path}.{req}: required")
            props = sch.get("properties", {})
            for k, v in val.items():
                if k in props:
                    check(props[k], v, f"{path}.{k}")
                elif sch.get("additionalProperties") is False:
                    errors.append(f"{path}.{k}: unexpected property")
        if isinstance(val, list) and "items" in sch:
            for i, item in enumerate(val):
                check(sch["items"], item, f"{path}[{i}]")

    check(schema, args, "$")
    return errors


class IdempotencyStore:
    """Persisted results of side-effecting tool calls, keyed by (run, key)."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        with closing(self._conn()) as c:
            c.execute("CREATE TABLE IF NOT EXISTS idempotency(run_id TEXT, key TEXT, result TEXT, PRIMARY KEY(run_id, key))")

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def get(self, run_id: str, key: str) -> Optional[dict[str, Any]]:
        with self._lock, closing(self._conn()) as c:
            row = c.execute("SELECT result FROM idempotency WHERE run_id=? AND key=?", (run_id, key)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, run_id: str, key: str, result: dict[str, Any]) -> None:
        with self._lock, closing(self._conn()) as c:
            c.execute("INSERT OR REPLACE INTO idempotency VALUES (?,?,?)", (run_id, key, json_dumps(result)))


class ToolRegistry:
    def __init__(self, idempotency: Optional[IdempotencyStore] = None, ids: Optional[IdGenerator] = None, metrics: Any = None) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self.idempotency = idempotency
        self.ids = ids or IdGenerator()
        self.metrics = metrics
        self.failures: dict[str, list[dict[str, Any]]] = {}  # tool misuse log for description self-improvement (§9.3)
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=16, thread_name_prefix="tool")

    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.permission not in PERMISSION_CLASSES:
            raise ValueError(f"unknown permission class {spec.permission}")
        self._tools[spec.name] = spec
        return spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def openai_schemas(self, allowlist: Optional[set[str]] = None) -> list[dict[str, Any]]:
        return [t.openai_schema() for n, t in sorted(self._tools.items()) if allowlist is None or n in allowlist]

    def mcp_manifest(self) -> dict[str, Any]:
        """``tools/list`` response shape of the Model Context Protocol."""
        return {"tools": [t.mcp_schema() for _, t in sorted(self._tools.items())]}

    def improve_description(self, name: str, new_description: str, when_not_to_use: str = "") -> None:
        spec = self._tools[name]
        spec.description = new_description
        if when_not_to_use:
            spec.when_not_to_use = when_not_to_use
        spec.version += 1

    def _record_failure(self, name: str, args: dict[str, Any], error: ToolError) -> None:
        self.failures.setdefault(name, []).append({"arguments": args, "error": error.to_dict()})

    def call(self, name: str, arguments: dict[str, Any], ctx: ToolContext, allowlist: Optional[set[str]] = None,
             idempotency_key: Optional[str] = None, call_id: Optional[str] = None) -> ToolResult:
        call_id = call_id or self.ids.new("call")
        t0 = time.monotonic()
        spec = self._tools.get(name)

        def fail(code: str, msg: str, retryable: bool = False, **details: Any) -> ToolResult:
            err = ToolError(code, msg, retryable, details)
            if spec is not None:
                self._record_failure(name, arguments, err)
            res = ToolResult(call_id, name, False, None, err, duration_s=time.monotonic() - t0)
            self._trace(ctx, res, arguments)
            return res

        if spec is None:
            return fail("not_found", f"unknown tool '{name}'. Available: {', '.join(self.names())}")
        if allowlist is not None and name not in allowlist:
            return fail("permission_denied", f"tool '{name}' is not in the allowlist for role '{ctx.role}'")
        if not isinstance(arguments, dict):
            return fail("invalid_arguments", "arguments must be a JSON object")
        errs = validate_args(spec.parameters, arguments)
        if errs:
            return fail("invalid_arguments", "; ".join(errs), schema=spec.parameters)
        if spec.permission in DANGEROUS:
            if ctx.approve is None or not ctx.approve(name, arguments):
                return fail("permission_denied", f"tool '{name}' ({spec.permission}) requires policy approval; none was granted")
        if spec.side_effects and self.idempotency is not None:
            key = idempotency_key or f"{name}:{json_dumps(arguments)}"
            cached = self.idempotency.get(ctx.run_id, key)
            if cached is not None:
                res = ToolResult.from_dict(cached)
                res.call_id = call_id
                self._trace(ctx, res, arguments, replayed=True)
                return res
        fut = self._pool.submit(spec.fn, ctx, arguments)
        try:
            output = fut.result(timeout=spec.timeout_s)
        except concurrent.futures.TimeoutError:
            return fail("timeout", f"tool '{name}' exceeded {spec.timeout_s}s", retryable=True)
        except ToolFailure as e:
            return fail(e.code, str(e), e.retryable, **e.details)
        except FileNotFoundError as e:
            return fail("not_found", str(e))
        except PermissionError as e:
            return fail("permission_denied", str(e))
        except Exception as e:  # noqa: BLE001
            return fail("execution_error", f"{type(e).__name__}: {e}"[:500])
        artifact_ids: list[str] = []
        if isinstance(output, dict):
            artifact_ids = [a for a in output.get("artifact_ids", []) if isinstance(a, str)]
            if output.get("artifact_id"):
                artifact_ids.append(output["artifact_id"])
        output, compacted = self._compact(output, ctx, name)
        res = ToolResult(call_id, name, True, output, None, artifact_ids, time.monotonic() - t0, compacted)
        if spec.side_effects and self.idempotency is not None:
            self.idempotency.put(ctx.run_id, idempotency_key or f"{name}:{json_dumps(arguments)}", res.to_dict())
        self._trace(ctx, res, arguments)
        return res

    def _compact(self, output: Any, ctx: ToolContext, name: str) -> tuple[Any, bool]:
        """Result compaction (§9.4): big outputs become an artifact plus a truncated preview."""
        text = json_dumps(output) if not isinstance(output, str) else output
        if len(text) <= MAX_INLINE_OUTPUT_CHARS:
            return output, False
        from ..core.types import Modality

        art = ctx.store.put_bytes(text.encode("utf-8"), "application/json" if not isinstance(output, str) else "text/plain", Modality.TEXT,
                                  source=f"tool:{name}", ext=".txt")
        ctx.store.put_text_view(art.id, "text", text)
        ctx.store.put_json_view(art.id, "summary", {"modality": "text", "tool_output": name, "chars": len(text)})
        preview = truncate(text, MAX_INLINE_OUTPUT_CHARS)
        return {"compacted": True, "preview": preview, "full_output_artifact_id": art.id, "chars": len(text),
                "hint": "use fetch_artifact_view with offset/length to page through the full output"}, True

    def _trace(self, ctx: ToolContext, res: ToolResult, arguments: dict[str, Any], replayed: bool = False) -> None:
        if self.metrics is not None:
            self.metrics.record_tool_call(res.name, res.ok, res.duration_s)
        if ctx.tracer is not None:
            ctx.tracer.record("tool_call", {"name": res.name, "arguments": arguments, "ok": res.ok, "error": res.error.to_dict() if res.error else None,
                                            "duration_s": round(res.duration_s, 3), "artifact_ids": res.artifact_ids, "replayed": replayed,
                                            "output_preview": truncate(json_dumps(res.output), 500)}, node_id=ctx.node_id)

    def call_parallel(self, calls: list[tuple[str, dict[str, Any], Optional[str]]], ctx: ToolContext, allowlist: Optional[set[str]] = None) -> list[ToolResult]:
        """Independent calls dispatched together (§8.5)."""
        futs = [self._pool.submit(self.call, n, a, ctx, allowlist, None, cid) for n, a, cid in calls]
        return [f.result() for f in futs]


class ToolFailure(Exception):
    """Raise inside a tool to return a typed error."""

    def __init__(self, code: str, message: str, retryable: bool = False, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.details = details
