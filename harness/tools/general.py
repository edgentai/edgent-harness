"""General-purpose coding, file-editing, shell, web, skill & HITL tools.

Provides a Claude-Code / Antigravity-class toolset on top of the starter
registry: line-range & multimodal `read_file`, surgical `edit_file`,
`delete_file`, `glob_files`, `read_numbered`, sandboxed `bash` with persistent
cwd & background tasks (`task_status`, `task_output`, `task_stop`), `web_fetch`,
`web_search`, progressive-disclosure `load_skill`, `todo_manage`,
`session_recall`, `request_review`, and isolated `spawn_subagent`.
"""
from __future__ import annotations

import difflib
import json
import mimetypes
import os
import re
from typing import Any, Optional

from ..modality import image as image_mod
from ..sandbox.runner import run_subprocess
from .registry import ToolContext, ToolFailure, ToolRegistry, ToolSpec
from .shell import bash_tool, task_output_tool, task_status_tool, task_stop_tool
from .web import web_fetch, web_search

S = {"type": "string"}
I = {"type": "integer"}
N = {"type": "number"}
B = {"type": "boolean"}

_IMAGE_EXTS = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}
MAX_IMAGE_INLINE_BYTES = 5 * 1024 * 1024


def _obj(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


# ---------------------------------------------------------------- paragraphs & anchors
def paragraph_spans(body: str) -> list[tuple[int, int]]:
    """Blank-line separated paragraphs, keeping fenced code and $$ math blocks intact."""
    spans: list[tuple[int, int]] = []
    start: Optional[int] = None
    in_fence = in_math = False
    pos = 0
    for line in body.split("\n"):
        end = pos + len(line)
        stripped = line.strip()
        if start is None and stripped:
            start = pos
        if stripped.startswith("```"):
            in_fence = not in_fence
        elif not in_fence:
            if stripped == "$$":
                in_math = not in_math
            elif stripped.startswith("$$") and not (stripped.endswith("$$") and len(stripped) > 4):
                in_math = True
            elif in_math and stripped.endswith("$$"):
                in_math = False
        if not stripped and start is not None and not in_fence and not in_math:
            spans.append((start, pos - 1 if pos > start else pos))
            start = None
        pos = end + 1
    if start is not None:
        trimmed = body.rstrip("\n")
        spans.append((start, len(trimmed) if trimmed else len(body)))
    return [(s, e) for s, e in spans if body[s:e].strip()]


def _locate_anchor(text: str, anchor: str, *, replace_all: bool = False) -> tuple[list[tuple[int, int]], str]:
    if not anchor or not anchor.strip():
        return [], "empty anchor"
    hits: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = text.find(anchor, start)
        if idx == -1:
            break
        hits.append((idx, idx + len(anchor)))
        start = idx + len(anchor)
    if hits:
        if len(hits) > 1 and not replace_all:
            return [], f"anchor matched {len(hits)} times; supply a more specific quote or set replace_all=true"
        return hits, ""
    pattern = re.compile(r"\s+".join(re.escape(p) for p in anchor.split()))
    matches = [m.span() for m in pattern.finditer(text)]
    if len(matches) == 1 or (len(matches) > 1 and replace_all):
        return matches, ""
    if len(matches) > 1:
        return [], f"anchor matched {len(matches)} times (whitespace-tolerant); provide more context"
    return [], "anchor not found in file"


def _nearest_paragraph(body: str, quote: str) -> str:
    best, score = "", 0.0
    q = quote[:400].lower()
    for s, e in paragraph_spans(body):
        para = body[s:e]
        r = difflib.SequenceMatcher(None, q, para[:600].lower()).ratio()
        if r > score:
            best, score = para, r
    return best[:300] if score > 0.30 else ""


def _resolve_para_span(body: str, para_ref: str) -> Optional[tuple[int, int]]:
    m = re.fullmatch(r"¶?\s*(\d+)", str(para_ref or "").strip())
    if not m:
        return None
    spans = paragraph_spans(body)
    idx = int(m.group(1))
    if not 1 <= idx <= len(spans):
        return None
    return spans[idx - 1]


# ---------------------------------------------------------------- file tools
def read_file_smart(ctx: ToolContext, a: dict[str, Any]) -> Any:
    """Read a workspace file: supports line ranges, char paging, and multimodal images."""
    rel = a["path"]
    if not ctx.workspace.exists(rel):
        raise ToolFailure("not_found", f"file not found: {rel}", remediation="Use list_files or glob_files to check the path.")
    full = ctx.workspace.resolve(rel)
    if os.path.isdir(full):
        raise ToolFailure("invalid_arguments", f"'{rel}' is a directory; use list_files instead")

    ext = os.path.splitext(full)[1].lower()
    if ext in _IMAGE_EXTS:
        mime = _IMAGE_EXTS[ext] or mimetypes.guess_type(full)[0] or "image/png"
        size = os.path.getsize(full)
        with open(full, "rb") as f:
            raw = f.read(MAX_IMAGE_INLINE_BYTES + 1)
        info = image_mod.describe(raw, mime)
        out: dict[str, Any] = {
            "path": rel,
            "modality": "image",
            "bytes": size,
            **info,
            "untrusted": True,
        }
        if size <= MAX_IMAGE_INLINE_BYTES:
            out["attach_image_path"] = rel
        else:
            out["note"] = f"image is {size:,} bytes (over {MAX_IMAGE_INLINE_BYTES:,}-byte inline cap); use crop_zoom or downscale first"
        return out

    text = ctx.workspace.read_text(rel)
    lines = text.splitlines()
    if a.get("start_line") is not None or a.get("end_line") is not None:
        s_line = max(1, int(a.get("start_line") or 1))
        e_line = min(len(lines), int(a.get("end_line") or (s_line + 199)))
        if e_line < s_line:
            e_line = s_line
        chunk = lines[s_line - 1 : e_line]
        numbered = "\n".join(f"{i}: {ln}" for i, ln in enumerate(chunk, start=s_line))
        return {
            "path": rel,
            "start_line": s_line,
            "end_line": e_line,
            "total_lines": len(lines),
            "total_chars": len(text),
            "content": numbered[: int(a.get("length", 20000))],
            "untrusted": True,
        }

    off, ln = int(a.get("offset", 0)), int(a.get("length", 20000))
    return {
        "path": rel,
        "offset": off,
        "length": min(ln, max(0, len(text) - off)),
        "total_lines": len(lines),
        "total_chars": len(text),
        "content": text[off : off + ln],
        "untrusted": True,
    }


def edit_file(ctx: ToolContext, a: dict[str, Any]) -> Any:
    """Surgical edit on a workspace file: replace, insert, or delete by quote or ¶n."""
    path = a["path"]
    if not ctx.workspace.exists(path):
        raise ToolFailure("not_found", f"file not found: {path}", remediation="Use write_file to create a new file or list_files to check the path.")
    body = ctx.workspace.read_text(path)
    op = str(a.get("operation") or "replace").strip().lower()
    replace_all = bool(a.get("replace_all", False))

    if op == "replace":
        if "new" not in a:
            raise ToolFailure("invalid_arguments", "replace requires 'new'")
        new_text = str(a["new"])
        if a.get("para") is not None:
            span = _resolve_para_span(body, str(a["para"]))
            if span is None:
                raise ToolFailure("not_found", f"paragraph {a['para']} out of range (file has {len(paragraph_spans(body))} paragraphs)")
            updated = body[: span[0]] + new_text + body[span[1] :]
            replaced_chars = span[1] - span[0]
            count = 1
        else:
            old = str(a.get("old") or "")
            spans, err = _locate_anchor(body, old, replace_all=replace_all)
            if not spans:
                hint = _nearest_paragraph(body, old)
                msg = err + (f". Closest paragraph:\n{hint}" if hint else "")
                raise ToolFailure("not_found", msg, remediation="Re-read the target file and copy an exact unique snippet.")
            updated = body
            replaced_chars = 0
            for s, e in reversed(spans):
                updated = updated[:s] + new_text + updated[e:]
                replaced_chars += e - s
            count = len(spans)
        ctx.workspace.write_text(path, updated)
        return {"path": path, "operation": "replace", "replacements": count, "replaced_chars": replaced_chars, "new_total_chars": len(updated)}

    if op == "insert":
        text = str(a.get("new") if a.get("new") is not None else a.get("text") or "")
        if not text.strip():
            raise ToolFailure("invalid_arguments", "insert requires non-empty 'new' text")
        after = str(a.get("after") or "end").strip()
        if after == "start":
            pos = 0
        elif after == "end":
            pos = len(body.rstrip("\n"))
        elif re.fullmatch(r"¶?\s*\d+", after):
            span = _resolve_para_span(body, after)
            if span is None:
                raise ToolFailure("not_found", f"paragraph {after} out of range (file has {len(paragraph_spans(body))} paragraphs)")
            pos = span[1]
        else:
            spans, err = _locate_anchor(body, after, replace_all=False)
            if not spans:
                hint = _nearest_paragraph(body, after)
                raise ToolFailure("not_found", err + (f". Closest paragraph:\n{hint}" if hint else ""))
            pos = spans[0][1]
        left, right = body[:pos].rstrip("\n"), body[pos:].lstrip("\n")
        chunks = [c for c in (left, text.strip("\n"), right) if c.strip()]
        updated = "\n\n".join(chunks) + ("\n" if body.endswith("\n") else "")
        ctx.workspace.write_text(path, updated)
        return {"path": path, "operation": "insert", "after": after, "inserted_chars": len(text), "new_total_chars": len(updated)}

    if op == "delete":
        if a.get("para") is not None:
            span = _resolve_para_span(body, str(a["para"]))
            if span is None:
                raise ToolFailure("not_found", f"paragraph {a['para']} out of range")
            spans = [span]
        else:
            old = str(a.get("old") or "")
            spans, err = _locate_anchor(body, old, replace_all=replace_all)
            if not spans:
                hint = _nearest_paragraph(body, old)
                raise ToolFailure("not_found", err + (f". Closest paragraph:\n{hint}" if hint else ""))
        updated = body
        removed = 0
        for s, e in reversed(spans):
            left, right = updated[:s].rstrip(" \t"), updated[e:]
            updated = left.rstrip("\n") + ("\n\n" if left.strip() and right.strip() else "") + right.lstrip("\n")
            removed += e - s
        ctx.workspace.write_text(path, updated)
        return {"path": path, "operation": "delete", "deleted_chars": removed, "new_total_chars": len(updated)}

    raise ToolFailure("invalid_arguments", f"unknown operation '{op}'; expected replace, insert, or delete")


def delete_file(ctx: ToolContext, a: dict[str, Any]) -> Any:
    ok = ctx.workspace.delete(a["path"])
    if not ok:
        raise ToolFailure("not_found", f"path does not exist: {a['path']}")
    return {"path": a["path"], "deleted": True}


def glob_files(ctx: ToolContext, a: dict[str, Any]) -> Any:
    hits = ctx.workspace.glob(a["pattern"], a.get("path", "."), int(a.get("limit", 200)))
    return {"pattern": a["pattern"], "files": hits}


def read_numbered(ctx: ToolContext, a: dict[str, Any]) -> Any:
    """Read a workspace text file with ¶n paragraph markers and word count."""
    text = ctx.workspace.read_text(a["path"])
    spans = paragraph_spans(text)
    numbered = "\n\n".join(f"¶{i} {text[s:e]}" for i, (s, e) in enumerate(spans, 1)) if spans else "(empty)"
    off, ln = int(a.get("offset", 0)), int(a.get("length", 12000))
    return {
        "path": a["path"],
        "paragraphs": len(spans),
        "words": len(text.split()),
        "total_chars": len(text),
        "content": numbered[off : off + ln],
        "untrusted": True,
    }


# ---------------------------------------------------------------- argv command
def run_command(ctx: ToolContext, a: dict[str, Any]) -> Any:
    """Run an argv command inside the sandboxed workspace (no shell interpolation)."""
    argv = [str(x) for x in a["argv"]]
    if not argv:
        raise ToolFailure("invalid_arguments", "argv must be a non-empty list")
    sb = ctx.config.sandbox
    timeout = min(float(a.get("timeout_s", sb.timeout_s)), max(sb.timeout_s, 300.0))
    allowed = None if ctx.extra.get("allow_any_executable") else sb.allowed_executables
    res = run_subprocess(
        argv,
        ctx.workspace.path,
        timeout_s=timeout,
        cpu_seconds=sb.cpu_seconds,
        memory_bytes=sb.memory_bytes,
        allow_network=sb.allow_network,
        allowed_executables=allowed,
    )
    out = res.to_dict()
    out["ok"] = res.ok
    out["argv"] = argv
    if res.returncode == 126:
        raise ToolFailure(
            "permission_denied",
            res.stderr,
            remediation=f"Allowed executables: {list(sb.allowed_executables)}. Use `bash` or add to sandbox.allowed_executables.",
        )
    return out


# ---------------------------------------------------------------- skills, todo, recall, HITL review
def load_skill(ctx: ToolContext, a: dict[str, Any]) -> Any:
    reg = ctx.extra.get("skills")
    if reg is None:
        raise ToolFailure("unavailable", "no skill registry attached to this session")
    skill = reg.get(a["name"])
    if skill is None:
        raise ToolFailure("not_found", f"unknown skill '{a['name']}'. Available: {', '.join(reg.names())}")
    loaded: list[str] = ctx.extra.setdefault("loaded_skills", [])
    if skill.name not in loaded:
        loaded.append(skill.name)
    return {"name": skill.name, "description": skill.description, "body": skill.body, "references": skill.references}


def todo_manage(ctx: ToolContext, a: dict[str, Any]) -> Any:
    """Read or update the persistent task list; runs any attached unit_done gate on 'done'."""
    rel = a.get("path") or "todo.json"
    action = str(a["action"]).strip().lower()
    data: dict[str, Any] = {"items": []}
    if ctx.workspace.exists(rel):
        try:
            data = json.loads(ctx.workspace.read_text(rel))
        except Exception:
            data = {"items": []}
    items: list[dict[str, Any]] = list(data.get("items") or [])

    if action == "list":
        return {"path": rel, "items": items}
    if action == "set":
        new_items = a.get("items")
        if not isinstance(new_items, list):
            raise ToolFailure("invalid_arguments", "action='set' requires 'items' array")
        existing_ids = {str(it.get("id")) for it in items if it.get("id")}
        incoming_ids = {str(it.get("id")) for it in new_items if isinstance(it, dict) and it.get("id")}
        dropped = existing_ids - incoming_ids
        if dropped and not a.get("allow_remove"):
            raise ToolFailure("permission_denied", f"cannot drop tracked items {sorted(dropped)}; mark them skipped instead")
        data["items"] = new_items
        ctx.workspace.write_text(rel, json.dumps(data, indent=2, ensure_ascii=False))
        return {"path": rel, "count": len(new_items), "items": new_items}
    if action == "update":
        tid = str(a.get("id") or "")
        target = next((it for it in items if str(it.get("id")) == tid), None)
        if target is None:
            raise ToolFailure("not_found", f"todo item '{tid}' not found")
        new_status = str(a.get("status") or target.get("status") or "pending")
        if new_status == "done":
            gate = ctx.extra.get("unit_done_gate")
            if callable(gate):
                ok, findings = gate(target, ctx)
                if not ok:
                    target["status"] = "partial"
                    target["open_findings"] = findings
                    data["items"] = items
                    ctx.workspace.write_text(rel, json.dumps(data, indent=2, ensure_ascii=False))
                    return {"ok": False, "path": rel, "id": tid, "status": "partial", "blocked_by": findings}
        target["status"] = new_status
        if a.get("note"):
            target["note"] = str(a["note"])
        data["items"] = items
        ctx.workspace.write_text(rel, json.dumps(data, indent=2, ensure_ascii=False))
        return {"ok": True, "path": rel, "id": tid, "status": new_status}
    raise ToolFailure("invalid_arguments", f"unknown todo action '{action}'; expected list, set, or update")


def request_review(ctx: ToolContext, a: dict[str, Any]) -> Any:
    """Pause the session cleanly to request human review of an artifact or plan."""
    payload = {
        "kind": str(a.get("kind") or "artifact"),
        "title": str(a["title"]),
        "summary": str(a["summary"]),
        "artifact_paths": list(a.get("artifact_paths") or []),
        "questions": list(a.get("questions") or []),
    }
    ctx.extra["pending_review"] = payload
    return {"status": "review_requested", **payload}


def session_recall(ctx: ToolContext, a: dict[str, Any]) -> Any:
    """Grep-based verbatim recall over earlier turns in the session transcript."""
    store = ctx.extra.get("session_store")
    session_id = ctx.extra.get("session_id")
    if store is None or not session_id:
        raise ToolFailure("unavailable", "no persistent session store attached")
    hits = store.recall(session_id, str(a["query"]), limit=int(a.get("limit", 8)))
    return {"query": a["query"], "hits": hits, "untrusted": True}


def register_general_tools(registry: ToolRegistry) -> ToolRegistry:
    """Register general-purpose coding, editing, shell, web, skill & HITL tools."""
    specs = [
        ToolSpec(
            "read_file",
            "Read a workspace file: supports line ranges (start_line/end_line), char paging (offset/length), and multimodal images (PNG/JPEG/GIF/WebP/BMP).",
            _obj({
                "path": S,
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
                "offset": {"type": "integer", "minimum": 0},
                "length": {"type": "integer", "minimum": 1, "maximum": 40000},
            }, ["path"]),
            read_file_smart,
            permission="read",
        ),
        ToolSpec(
            "edit_file",
            "Surgically edit a text file in the workspace: operation='replace' (by exact/unique old quote or para='¶n'), 'insert' (after='start'|'end'|'¶n'|quote), or 'delete'. Untouched text is preserved verbatim.",
            _obj({"path": S, "operation": {"type": "string", "enum": ["replace", "insert", "delete"]}, "old": S, "new": S, "para": S, "after": S, "replace_all": B}, ["path"]),
            edit_file,
            permission="write",
            side_effects=True,
            idempotent=False,
        ),
        ToolSpec(
            "delete_file",
            "Delete a file or directory inside the workspace.",
            _obj({"path": S}, ["path"]),
            delete_file,
            permission="delete",
            side_effects=True,
            idempotent=False,
        ),
        ToolSpec(
            "glob_files",
            "Find files matching a glob pattern (e.g. '*.py', 'docs/*.md') inside the workspace.",
            _obj({"pattern": S, "path": S, "limit": I}, ["pattern"]),
            glob_files,
            permission="read",
        ),
        ToolSpec(
            "read_numbered",
            "Read a workspace text file with ¶1, ¶2… paragraph numbers and word count.",
            _obj({"path": S, "offset": {"type": "integer", "minimum": 0}, "length": {"type": "integer", "minimum": 1, "maximum": 40000}}, ["path"]),
            read_numbered,
            permission="read",
        ),
        ToolSpec(
            "bash",
            "Run a shell command in the sandboxed workspace (supports pipes, redirection, persistent cwd, output offloading to .edgent/tool_outputs/, and background=true).",
            _obj({
                "command": S,
                "cwd": S,
                "timeout_s": {"type": "number", "minimum": 1, "maximum": 600},
                "background": B,
                "allow_network": B,
            }, ["command"]),
            bash_tool,
            permission="write",
            side_effects=True,
            idempotent=False,
            timeout_s=620,
            cost_class="moderate",
        ),
        ToolSpec(
            "task_status",
            "Check the status of one or all background shell tasks started with bash(background=true).",
            _obj({"task_id": S}, []),
            task_status_tool,
            permission="read",
        ),
        ToolSpec(
            "task_output",
            "Read the tail output of a background shell task.",
            _obj({"task_id": S, "tail_chars": {"type": "integer", "minimum": 100, "maximum": 40000}}, ["task_id"]),
            task_output_tool,
            permission="read",
        ),
        ToolSpec(
            "task_stop",
            "Terminate a running background shell task.",
            _obj({"task_id": S}, ["task_id"]),
            task_stop_tool,
            permission="write",
            side_effects=True,
            idempotent=False,
        ),
        ToolSpec(
            "run_command",
            "Run an allow-listed executable inside the sandboxed workspace with an argv array (no shell interpolation).",
            _obj({"argv": {"type": "array", "items": S}, "timeout_s": {"type": "number", "minimum": 1, "maximum": 300}}, ["argv"]),
            run_command,
            permission="compute",
            timeout_s=320,
            cost_class="moderate",
        ),
        ToolSpec(
            "web_fetch",
            "Fetch an http/https URL (SSRF-checked), extract readable text, save full text to .edgent/web_cache/, and return a bounded untrusted view.",
            _obj({"url": S, "max_chars": {"type": "integer", "minimum": 200, "maximum": 40000}, "timeout_s": {"type": "number", "minimum": 1, "maximum": 60}}, ["url"]),
            web_fetch,
            permission="network",
            timeout_s=70,
            cost_class="moderate",
        ),
        ToolSpec(
            "web_search",
            "Search the web and return titles, URLs, and snippets (untrusted data).",
            _obj({"query": S, "limit": {"type": "integer", "minimum": 1, "maximum": 20}}, ["query"]),
            web_search,
            permission="network",
            timeout_s=30,
            cost_class="moderate",
        ),
        ToolSpec(
            "load_skill",
            "Load the full instructions of a named skill from the skill registry.",
            _obj({"name": S}, ["name"]),
            load_skill,
            permission="read",
        ),
        ToolSpec(
            "todo_manage",
            "Read or update the persistent task list (list, set, update). Marking an item 'done' runs any attached obligation gate.",
            _obj({"action": {"type": "string", "enum": ["list", "set", "update"]}, "path": S, "id": S, "status": S, "note": S, "items": {"type": "array", "items": {"type": "object"}}, "allow_remove": B}, ["action"]),
            todo_manage,
            permission="write",
            side_effects=True,
            idempotent=False,
        ),
        ToolSpec(
            "request_review",
            "Pause the run cleanly and hand an artifact or plan to the human user for approval or feedback.",
            _obj({"kind": S, "title": S, "summary": S, "artifact_paths": {"type": "array", "items": S}, "questions": {"type": "array", "items": S}}, ["title", "summary"]),
            request_review,
            permission="interactive",
            side_effects=False,
        ),
        ToolSpec(
            "session_recall",
            "Search earlier turns of the current session transcript verbatim by keyword/regex (no embeddings).",
            _obj({"query": S, "limit": {"type": "integer", "minimum": 1, "maximum": 30}}, ["query"]),
            session_recall,
            permission="read",
        ),
        ToolSpec(
            "spawn_subagent",
            "Delegate a focused sub-task to an isolated fresh-context subagent (e.g. explore, coder, reviewer, planner, researcher) and receive only its final summary.",
            _obj({"role": S, "task": S, "max_turns": {"type": "integer", "minimum": 1, "maximum": 30}, "max_tool_calls": {"type": "integer", "minimum": 1, "maximum": 60}}, ["role", "task"]),
            lambda ctx, a: _spawn_general(ctx, a),
            permission="spawn",
            timeout_s=900,
            cost_class="expensive",
        ),
    ]
    for s in specs:
        registry.register(s)
    return registry


def _spawn_general(ctx: ToolContext, a: dict[str, Any]) -> Any:
    if ctx.spawn is None:
        raise ToolFailure("permission_denied", "subagents are not enabled in this session")
    return ctx.spawn(str(a["role"]), str(a["task"]), {"max_turns": int(a.get("max_turns", 12)), "max_tool_calls": int(a.get("max_tool_calls", 20))})
