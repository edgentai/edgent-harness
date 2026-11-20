"""LaTeX- and HTML-safe text tool-call protocol for non-native-tool bridges.

CLI bridges such as `agy --print` or `claude --print` do not expose native tool
calling, and JSON tool encodings corrupt LaTeX backslashes (`\\frac`, `\\text`)
while bare `<text>`/`<code>` XML tags collide with HTML/SVG teaching content.

This module uses namespaced `<edgent_call>` / `<edgent_arg>` envelopes with
optional CDATA (`<![CDATA[...]]>`) or custom sentinel boundaries so arbitrary
LaTeX, Markdown, HTML, SVG, JSX and code survive untouched.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from .types import ToolCall
from .utils import IdGenerator

_CALL_RE = re.compile(r"<edgent_call\b([^>]*?)(/>|>(.*?)</edgent_call>)", re.DOTALL)
_ATTR_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')
_ARG_OPEN_RE = re.compile(r"<edgent_arg\b([^>]*)>", re.DOTALL)
_CDATA_RE = re.compile(r"^\s*<!\[CDATA\[(.*?)\]\]>\s*$", re.DOTALL)


def compact_param_summary(schema: dict[str, Any]) -> str:
    """One-line human/model-readable summary of a JSON-schema object."""
    props = schema.get("properties") or {}
    req = set(schema.get("required") or [])
    parts = []
    for name, ps in props.items():
        t = ps.get("type", "any")
        if isinstance(t, list):
            t = "|".join(t)
        if "enum" in ps:
            t = "|".join(str(e) for e in ps["enum"])
        star = "" if name in req else "?"
        parts.append(f"{name}{star}: {t}")
    return ", ".join(parts)


def render_tools_prompt(openai_schemas: list[dict[str, Any]], *, compact: bool = True) -> str:
    """Render tool declarations + sentinel-safe invocation instructions."""
    lines = [
        "## Tools (text protocol)",
        "To call a tool, emit one or more `<edgent_call>` blocks. Wrap multi-line, LaTeX, HTML or code arguments in `<![CDATA[...]]>` (or pass `sentinel=\"END_TAG\"` on `<edgent_arg>` if content contains `]]>`):",
        "",
        '<edgent_call tool="edit_file">',
        '<edgent_arg name="path">book/ch1.md</edgent_arg>',
        '<edgent_arg name="old"><![CDATA[exact text]]></edgent_arg>',
        '<edgent_arg name="new"><![CDATA[replacement with $\\frac{a}{b}$ and <code>x</code>]]></edgent_arg>',
        "</edgent_call>",
        "",
        "Available tools:",
    ]
    for item in openai_schemas:
        fn = item.get("function") or item
        name = fn.get("name", "")
        desc = (fn.get("description") or "").strip()
        params = fn.get("parameters") or {}
        if compact:
            short_desc = desc.split("\n", 1)[0][:220]
            lines.append(f"- `{name}`({compact_param_summary(params)}): {short_desc}")
        else:
            lines.append(f"- `{name}`: {desc}\n  parameters: {json.dumps(params, ensure_ascii=False)}")
    lines.append("\nIf no tool is needed, reply normally with no `<edgent_call>` blocks.")
    return "\n".join(lines)


def _coerce_value(raw: str, param_schema: Optional[dict[str, Any]]) -> Any:
    if not param_schema:
        return raw
    t = param_schema.get("type")
    types = t if isinstance(t, list) else ([t] if t else [])
    if "string" in types and len(types) == 1:
        return raw
    stripped = raw.strip()
    if "boolean" in types:
        if stripped.lower() in ("true", "1", "yes"):
            return True
        if stripped.lower() in ("false", "0", "no"):
            return False
    if "integer" in types:
        try:
            return int(stripped)
        except ValueError:
            pass
    if "number" in types:
        try:
            return float(stripped)
        except ValueError:
            pass
    if "array" in types or "object" in types:
        try:
            parsed = json.loads(stripped)
            if ("array" in types and isinstance(parsed, list)) or ("object" in types and isinstance(parsed, dict)):
                return parsed
        except Exception:
            pass
    return raw


def _parse_args(body: str, param_schemas: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Parse `<edgent_arg>` blocks supporting CDATA and custom sentinels."""
    args: dict[str, Any] = {}
    pos = 0
    n = len(body)
    while pos < n:
        m = _ARG_OPEN_RE.search(body, pos)
        if not m:
            break
        attrs = dict(_ATTR_RE.findall(m.group(1)))
        name = (attrs.get("name") or "").strip()
        sentinel = attrs.get("sentinel")
        start = m.end()
        if sentinel:
            close_marker = f"{sentinel}</edgent_arg>"
            end = body.find(close_marker, start)
            if end == -1:
                # Tolerate optional newline before sentinel
                end = body.find("</edgent_arg>", start)
                if end == -1:
                    break
                raw = body[start:end]
                pos = end + len("</edgent_arg>")
            else:
                raw = body[start:end]
                if raw.startswith("\n"):
                    raw = raw[1:]
                if raw.endswith("\n"):
                    raw = raw[:-1]
                pos = end + len(close_marker)
        else:
            # Check if CDATA starts immediately; if so jump past ]]> before finding </edgent_arg>
            tail = body[start:]
            cdata_open = re.match(r"\s*<!\[CDATA\[", tail)
            if cdata_open:
                c_start = start + cdata_open.end()
                c_end = body.find("]]>", c_start)
                if c_end != -1:
                    raw = body[c_start:c_end]
                    close_end = body.find("</edgent_arg>", c_end + 3)
                    pos = (close_end + len("</edgent_arg>")) if close_end != -1 else (c_end + 3)
                else:
                    end = body.find("</edgent_arg>", start)
                    if end == -1:
                        break
                    raw = body[start:end]
                    pos = end + len("</edgent_arg>")
            else:
                end = body.find("</edgent_arg>", start)
                if end == -1:
                    break
                raw = body[start:end]
                m_cd = _CDATA_RE.match(raw)
                if m_cd:
                    raw = m_cd.group(1)
                elif raw.startswith("\n") and raw.endswith("\n") and len(raw) >= 2:
                    raw = raw[1:-1]
                pos = end + len("</edgent_arg>")
        if name:
            ps = (param_schemas or {}).get(name)
            args[name] = _coerce_value(raw, ps)
    return args


def parse_tool_calls(
    text: str,
    ids: Optional[IdGenerator] = None,
    schemas_by_name: Optional[dict[str, dict[str, Any]]] = None,
) -> list[ToolCall]:
    """Extract `<edgent_call>` blocks from model text without mangling LaTeX/HTML."""
    if not text or "<edgent_call" not in text:
        return []
    gen = ids or IdGenerator()
    calls: list[ToolCall] = []
    for m in _CALL_RE.finditer(text):
        attrs = dict(_ATTR_RE.findall(m.group(1)))
        tool = (attrs.get("tool") or attrs.get("name") or "").strip()
        if not tool:
            continue
        call_id = (attrs.get("id") or "").strip() or gen.new("call")
        body = m.group(3) or ""
        param_schemas = None
        if schemas_by_name and tool in schemas_by_name:
            param_schemas = (schemas_by_name[tool].get("properties") or {})
        args = _parse_args(body, param_schemas)
        # Allow inline attribute shortcuts on self-closing calls: <edgent_call tool="load_skill" name="explaining"/>
        skip_attrs = {"tool", "id"} if "tool" in attrs else {"name", "id"}
        for k, v in attrs.items():
            if k not in skip_attrs and k not in args:
                ps = (param_schemas or {}).get(k)
                args[k] = _coerce_value(v, ps)
        calls.append(ToolCall(id=call_id, name=tool, arguments=args))
    return calls


def strip_tool_calls(text: str) -> str:
    """Remove `<edgent_call>` blocks to leave only human-facing assistant text."""
    if not text:
        return ""
    return _CALL_RE.sub("", text).strip()
