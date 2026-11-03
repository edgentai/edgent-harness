"""Tool registry: one definition per tool, enforcement in code, MCP export.

Every call goes through ``ToolRegistry.call``, which validates arguments
against the JSON schema, checks the caller's allowlist and the permission
policy, applies the timeout, consults the idempotency store for
side-effecting tools, compacts oversized outputs into persisted files, and
returns a typed ``ToolResult`` whose error follows the canonical contract.
"""
from __future__ import annotations

import concurrent.futures
import json
import sqlite3
import threading
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from ..domain.errors import ErrorCategory, HarnessError, ToolFailure
from ..domain.types import ContentPart
from ..domain.utils import IdGenerator, ensure_dir, json_dumps, truncate

PERMISSION_CLASSES = ("read", "compute", "write", "network", "spawn", "interactive", "delete")
DANGEROUS = {"network", "delete"}
MAX_INLINE_OUTPUT_CHARS = 6000
COMPACT_SCHEMA_MAX_DEPTH = 10

Decision = Literal["allow", "deny", "ask"]


class PermissionPolicy(Protocol):
    def decide(self, spec: ToolSpec, args: dict[str, Any]) -> Decision: ...


class AllowAll:
    def decide(self, spec: ToolSpec, args: dict[str, Any]) -> Decision:
        return "allow"


class ReadOnly:
    def decide(self, spec: ToolSpec, args: dict[str, Any]) -> Decision:
        return "allow" if spec.permission == "read" else "deny"


class AskDangerous:
    """Allow everything except network and delete, which ask."""

    def decide(self, spec: ToolSpec, args: dict[str, Any]) -> Decision:
        return "ask" if spec.permission in DANGEROUS else "allow"


@dataclass
class ToolContext:
    """Everything a tool may touch. Built per session or sub-agent; never shared across workspaces."""

    workspace: Any  # domain.workspace.Workspace
    session_id: str = "session"
    run_id: str = ""
    node_id: str = ""  # unit id or sub-agent id, for traces
    role: str = "main"
    provider: Any = None  # a Provider, for tools that need the model (judge, rewrite)
    evidence: Any = None  # domain.evidence.EvidenceStore
    settings: dict[str, Any] = field(default_factory=dict)
    toolchain: Any = None  # adapters.toolchain.Toolchain
    permissions: Any = None  # PermissionPolicy
    ask_user: Callable[[str], str] | None = None
    ask_permission: Callable[[ToolSpec, dict[str, Any]], bool] | None = None
    spawn: Callable[..., dict[str, Any]] | None = None  # spawn(agent, task, **opts) -> result dict
    tracer: Any = None
    metrics: Any = None
    session: Any = None  # application.sessions.Session, for todo and recall tools
    skills: Any = None  # application.skills.SkillIndex
    extra: dict[str, Any] = field(default_factory=dict)

    def tool_results_dir(self) -> Path | None:
        if self.workspace is None:
            return None
        try:
            return Path(self.workspace.path("tool_results"))
        except Exception:
            return None


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
    group: str = "core"  # core | sources | book | media | assess | learner | checks | session | custom
    attaches_media: bool = False
    max_inline_chars: int | None = None  # per-tool inline output ceiling (chars); None means MAX_INLINE_OUTPUT_CHARS

    def full_description(self) -> str:
        desc = self.description.strip()
        if self.when_not_to_use:
            desc += f" Do not use when: {self.when_not_to_use.strip()}"
        if self.examples:
            desc += " Example arguments: " + json.dumps(self.examples[0], ensure_ascii=False)
        return desc

    def openai_schema(self, compact: bool = False) -> dict[str, Any]:
        params = compact_schema(self.parameters) if compact else self.parameters
        desc = self.description.strip() if compact else self.full_description()
        return {"type": "function", "function": {"name": self.name, "description": desc, "parameters": params}}

    def mcp_schema(self, compact: bool = False) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description.strip() if compact else self.full_description(),
            "inputSchema": compact_schema(self.parameters) if compact else self.parameters,
            "outputSchema": self.returns,
            "annotations": {
                "readOnlyHint": not self.side_effects,
                "idempotentHint": self.idempotent,
                "destructiveHint": self.permission == "delete",
                "openWorldHint": self.permission == "network",
                "x-permission": self.permission,
                "x-timeout_s": self.timeout_s,
                "x-cost_class": self.cost_class,
                "x-group": self.group,
            },
        }


_ANNOTATIONS = {"description", "examples", "default", "title", "when_not_to_use"}
_SCHEMA_MAPS = {"properties", "patternProperties", "$defs", "definitions"}
_SCHEMA_CHILDREN = {"items", "additionalProperties", "not", "if", "then", "else", "contains", "propertyNames"}
_SCHEMA_ARRAYS = {"anyOf", "oneOf", "allOf", "prefixItems"}


def compact_schema(schema: dict[str, Any], keep_root_description: bool = True) -> dict[str, Any]:
    """Strip annotation prose from subschemas while keeping every validation keyword."""

    def rebuild(node: Any, depth: int, root: bool) -> Any:
        if isinstance(node, list):
            return [rebuild(x, depth + 1, False) for x in node]
        if not isinstance(node, dict):
            return node
        if depth > COMPACT_SCHEMA_MAX_DEPTH:
            return {"type": node.get("type", "object")}
        out: dict[str, Any] = {}
        for k, v in node.items():
            if k in _ANNOTATIONS and not (root and k == "description" and keep_root_description):
                continue
            if k in _SCHEMA_MAPS and isinstance(v, dict):
                out[k] = {name: rebuild(child, depth + 1, False) for name, child in v.items()}
            elif k in _SCHEMA_ARRAYS and isinstance(v, list):
                out[k] = [rebuild(child, depth + 1, False) for child in v]
            elif k in _SCHEMA_CHILDREN:
                out[k] = rebuild(v, depth + 1, False) if isinstance(v, (dict, list)) else v
            else:
                out[k] = v
        return out

    return rebuild(schema, 0, True)


def validate_args(schema: dict[str, Any], args: Any) -> list[str]:
    """Small JSON-schema validator: types, required, enum, bounds, nested objects and arrays."""
    errors: list[str] = []

    def check(sch: dict[str, Any], val: Any, path: str) -> None:
        if "enum" in sch and val not in sch["enum"]:
            errors.append(f"{path}: must be one of {sch['enum']}")
            return
        if "anyOf" in sch:
            sub_errors = []
            for alt in sch["anyOf"]:
                before = len(errors)
                check(alt, val, path)
                if len(errors) == before:
                    return
                sub_errors.extend(errors[before:])
                del errors[before:]
            errors.append(f"{path}: no alternative matched ({'; '.join(sub_errors)[:200]})")
            return
        t = sch.get("type")
        if t is None:
            return
        types = t if isinstance(t, list) else [t]
        ok = False
        for tt in types:
            if tt == "string" and isinstance(val, str):
                ok = True
            elif tt == "integer" and isinstance(val, int) and not isinstance(val, bool):
                ok = True
            elif tt == "integer" and isinstance(val, float) and val.is_integer():
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
        if isinstance(val, str):
            if "maxLength" in sch and len(val) > sch["maxLength"]:
                errors.append(f"{path}: longer than {sch['maxLength']}")
            if "minLength" in sch and len(val) < sch["minLength"]:
                errors.append(f"{path}: shorter than {sch['minLength']}")
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
        if isinstance(val, list):
            if "items" in sch:
                for i, item in enumerate(val):
                    check(sch["items"], item, f"{path}[{i}]")
            if "maxItems" in sch and len(val) > sch["maxItems"]:
                errors.append(f"{path}: more than {sch['maxItems']} items")
            if "minItems" in sch and len(val) < sch["minItems"]:
                errors.append(f"{path}: fewer than {sch['minItems']} items")

    check(schema, args, "$")
    return errors


@dataclass
class ToolResult:
    call_id: str
    name: str
    ok: bool
    output: Any = None
    error: HarnessError | None = None
    artifact_paths: list[str] = field(default_factory=list)
    attachments: list[ContentPart] = field(default_factory=list)  # media to show the model (frames, screenshots)
    duration_s: float = 0.0
    compacted: bool = False
    replayed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "name": self.name,
            "ok": self.ok,
            "output": self.output,
            "error": self.error.to_dict() if self.error else None,
            "artifact_paths": list(self.artifact_paths),
            "attachments": [a.to_dict() for a in self.attachments],
            "duration_s": round(self.duration_s, 3),
            "compacted": self.compacted,
            "replayed": self.replayed,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolResult:
        return cls(
            call_id=d["call_id"],
            name=d["name"],
            ok=bool(d["ok"]),
            output=d.get("output"),
            error=HarnessError.from_dict(d["error"]) if d.get("error") else None,
            artifact_paths=list(d.get("artifact_paths") or []),
            attachments=[ContentPart.from_dict(a) for a in d.get("attachments") or []],
            duration_s=float(d.get("duration_s", 0.0)),
            compacted=bool(d.get("compacted", False)),
            replayed=bool(d.get("replayed", False)),
        )

    def as_text(self) -> str:
        """What goes into the tool message the model reads."""
        if self.ok:
            return self.output if isinstance(self.output, str) else json_dumps(self.output)
        assert self.error is not None
        return json_dumps({"error": self.error.to_dict()})


class IdempotencyStore:
    """Persisted results of side-effecting tool calls, keyed by (session, key). A result is replayed only when a caller
    repeats the same explicit key (a resumed turn re-sending the model's call id); the arguments alone never form a key."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        ensure_dir(Path(self.db_path).parent)
        self._lock = threading.Lock()
        with closing(self._conn()) as c:
            c.execute("CREATE TABLE IF NOT EXISTS idempotency(session_id TEXT, key TEXT, result TEXT, PRIMARY KEY(session_id, key))")

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def get(self, session_id: str, key: str) -> dict[str, Any] | None:
        with self._lock, closing(self._conn()) as c:
            row = c.execute("SELECT result FROM idempotency WHERE session_id=? AND key=?", (session_id, key)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, session_id: str, key: str, result: dict[str, Any]) -> None:
        with self._lock, closing(self._conn()) as c:
            c.execute("INSERT OR REPLACE INTO idempotency VALUES (?,?,?)", (session_id, key, json_dumps(result)))


class ToolRegistry:
    def __init__(self, idempotency: IdempotencyStore | None = None, ids: IdGenerator | None = None, metrics: Any = None, max_workers: int = 16) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self.idempotency = idempotency
        self.ids = ids or IdGenerator()
        self.metrics = metrics
        self.failures: dict[str, list[dict[str, Any]]] = {}
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tool")
        # Dispatch of parallel batches gets its own pool: a tool call occupies one worker while it runs and, when a
        # subagent calls tools from inside it, its batch must not wait on the same workers (a nested tree would deadlock).
        self._dispatch = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tool-dispatch")

    # ------------------------------------------------------------------ definition
    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.permission not in PERMISSION_CLASSES:
            raise ValueError(f"unknown permission class {spec.permission!r} for tool {spec.name}")
        if spec.name in self._tools:
            raise ValueError(f"tool {spec.name!r} is already registered")
        self._tools[spec.name] = spec
        return spec

    def tool(self, name: str, description: str, parameters: dict[str, Any], **kw: Any) -> Callable[[ToolFn], ToolFn]:
        """Decorator form: ``@registry.tool("read_file", "...", schema, permission="read")``."""

        def deco(fn: ToolFn) -> ToolFn:
            self.register(ToolSpec(name=name, description=description, parameters=parameters, fn=fn, **kw))
            return fn

        return deco

    def include(self, other: ToolRegistry) -> None:
        for spec in other.specs():
            if spec.name not in self._tools:
                self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self, group: str | None = None) -> list[ToolSpec]:
        return [t for _, t in sorted(self._tools.items()) if group is None or t.group == group]

    def openai_schemas(self, allowlist: set[str] | None = None, compact: bool = False) -> list[dict[str, Any]]:
        return [t.openai_schema(compact) for n, t in sorted(self._tools.items()) if allowlist is None or n in allowlist]

    def mcp_manifest(self, compact: bool = False, allowlist: set[str] | None = None) -> dict[str, Any]:
        return {"tools": [t.mcp_schema(compact) for n, t in sorted(self._tools.items()) if allowlist is None or n in allowlist]}

    def schema_bytes(self, compact: bool = False) -> int:
        return len(json_dumps(self.openai_schemas(compact=compact)).encode("utf-8"))

    def improve_description(self, name: str, new_description: str, when_not_to_use: str = "") -> None:
        spec = self._tools[name]
        spec.description = new_description
        if when_not_to_use:
            spec.when_not_to_use = when_not_to_use
        spec.version += 1

    # ------------------------------------------------------------------ execution
    def _record_failure(self, name: str, args: dict[str, Any], error: HarnessError) -> None:
        self.failures.setdefault(name, []).append({"arguments": args, "error": error.to_dict()})

    def call(
        self,
        name: str,
        arguments: Any,
        ctx: ToolContext,
        allowlist: set[str] | None = None,
        idempotency_key: str | None = None,
        call_id: str | None = None,
    ) -> ToolResult:
        call_id = call_id or self.ids.new("call")
        t0 = time.monotonic()
        spec = self._tools.get(name)

        def fail(error: HarnessError) -> ToolResult:
            if spec is not None:
                self._record_failure(name, arguments if isinstance(arguments, dict) else {}, error)
            res = ToolResult(call_id, name, False, None, error, duration_s=time.monotonic() - t0)
            if error.partial is not None:
                res.output = error.partial
            self._trace(ctx, res, arguments)
            return res

        if spec is None:
            return fail(HarnessError("not_found", f"unknown tool {name!r}. Available: {', '.join(self.names())}", ErrorCategory.not_found,
                                     remediation="Choose a tool from the list; do not repeat the unchanged call."))
        if allowlist is not None and name not in allowlist:
            return fail(HarnessError("permission_denied", f"tool {name!r} is not available to role {ctx.role!r}", ErrorCategory.permission_denied,
                                     remediation="Use a tool in your allowlist, or return what you have and let the parent decide."))
        if not isinstance(arguments, dict):
            return fail(HarnessError("invalid_input", "arguments must be a JSON object", ErrorCategory.invalid_input,
                                     remediation="Send arguments as a JSON object matching the tool schema.",
                                     recovery={"kind": "inspect_input_contract", "operation": name}))
        errs = validate_args(spec.parameters, arguments)
        if errs:
            return fail(HarnessError("invalid_input", "; ".join(errs), ErrorCategory.invalid_input,
                                     remediation="Correct the listed fields from the tool schema; do not repeat the unchanged call.",
                                     recovery={"kind": "inspect_input_contract", "operation": name}, details={"schema": compact_schema(spec.parameters)}))
        decision: Decision = ctx.permissions.decide(spec, arguments) if ctx.permissions is not None else ("ask" if spec.permission in DANGEROUS else "allow")
        if decision == "deny":
            return fail(HarnessError("permission_denied", f"tool {name!r} ({spec.permission}) is denied by the permission policy", ErrorCategory.permission_denied,
                                     remediation="Choose an approach that does not need this permission, or ask the user to allow it in settings."))
        if decision == "ask":
            granted = False
            if ctx.ask_permission is not None:
                try:
                    granted = bool(ctx.ask_permission(spec, arguments))
                except Exception as e:  # noqa: BLE001
                    granted = False
                    ctx.extra["last_permission_error"] = str(e)
            if not granted:
                how = "the user declined" if ctx.ask_permission is not None else "there is no way to ask in this session"
                return fail(HarnessError("permission_denied", f"tool {name!r} ({spec.permission}) needs approval and {how}", ErrorCategory.permission_denied,
                                         remediation="Do not retry. Explain what you wanted to do and continue with what is allowed."))
        # Replay happens only for an explicit key (the loop passes the model's call id): keying on the arguments
        # replayed a repeated write_file with identical content after the file had changed under it.
        key = idempotency_key
        if key is not None and spec.side_effects and self.idempotency is not None:
            cached = self.idempotency.get(ctx.session_id, key)
            if cached is not None:
                res = ToolResult.from_dict(cached)
                res.call_id = call_id
                res.replayed = True
                self._trace(ctx, res, arguments)
                return res
        fut = self._pool.submit(spec.fn, ctx, arguments)
        try:
            output = fut.result(timeout=spec.timeout_s)
        except concurrent.futures.TimeoutError:
            return fail(HarnessError("timeout", f"tool {name!r} exceeded {spec.timeout_s:g} s", ErrorCategory.timeout, retryable=True,
                                     remediation="Narrow the request (smaller range, fewer items) before retrying once.",
                                     recovery={"kind": "inspect_timeout_state", "operation": name, "timeout_s": spec.timeout_s}))
        except ToolFailure as e:
            return fail(e.error)
        except FileNotFoundError as e:
            return fail(HarnessError("not_found", str(e), ErrorCategory.not_found, remediation="Check the path with list_files or glob first."))
        except PermissionError as e:
            return fail(HarnessError("permission_denied", str(e), ErrorCategory.permission_denied))
        except Exception as e:  # noqa: BLE001
            return fail(HarnessError("execution_error", f"{type(e).__name__}: {e}"[:600], ErrorCategory.execution_failure,
                                     remediation="Read the message, change the input or approach, and do not repeat the unchanged call."))
        artifact_paths: list[str] = []
        attachments: list[ContentPart] = []
        if isinstance(output, dict):
            artifact_paths = [a for a in output.get("artifact_paths", []) if isinstance(a, str)]
            if isinstance(output.get("artifact_path"), str):
                artifact_paths.append(output["artifact_path"])
            for att in output.pop("_attach", []) or []:
                if isinstance(att, dict) and att.get("path"):
                    attachments.append(ContentPart.of_media(att.get("type", "image"), att["path"], att.get("mime"), att.get("text", "")))
        output, compacted = self._compact(output, ctx, name, call_id, spec.max_inline_chars)
        res = ToolResult(call_id, name, True, output, None, artifact_paths, attachments, time.monotonic() - t0, compacted)
        if key is not None and spec.side_effects and self.idempotency is not None:
            self.idempotency.put(ctx.session_id, key, res.to_dict())
        self._trace(ctx, res, arguments)
        return res

    def call_parallel(self, calls: list[tuple[str, dict[str, Any], str | None]], ctx: ToolContext, allowlist: set[str] | None = None) -> list[ToolResult]:
        """Independent calls dispatched together; order of results matches order of calls."""
        # The replay key is the model's call id together with the call itself: a resumed turn that re-sends the
        # same call replays, while a server that reuses ids across turns (call_0, call_1, ...) never gets a stale result.
        keyed = [(n, a, cid, f"{cid}:{n}:{json_dumps(a)}" if cid else None) for n, a, cid in calls]
        if len(keyed) == 1:  # the common case runs on the caller's thread and costs no dispatch worker
            n, a, cid, key = keyed[0]
            return [self.call(n, a, ctx, allowlist, key, cid)]
        futs = [self._dispatch.submit(self.call, n, a, ctx, allowlist, key, cid) for n, a, cid, key in keyed]
        return [f.result() for f in futs]

    def _compact(self, output: Any, ctx: ToolContext, name: str, call_id: str, limit: int | None = None) -> tuple[Any, bool]:
        limit = limit or MAX_INLINE_OUTPUT_CHARS
        text = output if isinstance(output, str) else json_dumps(output)
        if len(text) <= limit:
            return output, False
        d = ctx.tool_results_dir()
        pointer: str | None = None
        if d is not None:
            ensure_dir(d)
            p = d / f"{call_id}.{'txt' if isinstance(output, str) else 'json'}"
            p.write_text(text, encoding="utf-8")
            try:
                pointer = ctx.workspace.relpath(p)
            except Exception:
                pointer = str(p)
        preview = truncate(text, limit)
        return {
            "compacted": True,
            "preview": preview,
            "chars": len(text),
            "full_output_path": pointer,
            "hint": "the full output was persisted; use read_file with offset/length to page through it" if pointer else "output truncated; narrow the request",
        }, True

    def _trace(self, ctx: ToolContext, res: ToolResult, arguments: Any) -> None:
        if self.metrics is not None:
            try:
                self.metrics.record_tool_call(res.name, res.ok, res.duration_s)
            except Exception:
                pass
        if ctx.tracer is not None:
            try:
                ctx.tracer.record(
                    "tool_call",
                    {
                        "name": res.name,
                        "call_id": res.call_id,
                        "arguments": arguments if isinstance(arguments, dict) else {"_raw": str(arguments)[:200]},
                        "ok": res.ok,
                        "error": res.error.to_dict() if res.error else None,
                        "duration_s": round(res.duration_s, 3),
                        "artifact_paths": res.artifact_paths,
                        "replayed": res.replayed,
                        "output_preview": truncate(res.as_text(), 500),
                    },
                    node_id=ctx.node_id or None,
                )
            except Exception:
                pass


# Convenience schema helpers used by every tool module.
S: dict[str, Any] = {"type": "string"}
I: dict[str, Any] = {"type": "integer"}  # noqa: E741
N: dict[str, Any] = {"type": "number"}
B: dict[str, Any] = {"type": "boolean"}


def obj(props: dict[str, Any], required: list[str] | None = None, **kw: Any) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "object", "properties": props, "required": list(required or []), "additionalProperties": False}
    schema.update(kw)
    return schema


def arr(items: dict[str, Any], **kw: Any) -> dict[str, Any]:
    return {"type": "array", "items": items, **kw}


def opt(schema: dict[str, Any], description: str, **kw: Any) -> dict[str, Any]:
    return {**schema, "description": description, **kw}
