"""The text tool protocol for providers without native tool calling.

The model is asked to emit one fenced block with the language tag ``tool``
containing ``{"name": ..., "arguments": {...}}`` (or a JSON list of such
objects for parallel calls). The harness parses the blocks, validates the
arguments against the schema exactly as native calls are, and returns the
results as text. Malformed blocks become typed errors the model can fix.
"""
from __future__ import annotations

import re
from typing import Any

from ..domain.types import ToolCall
from ..domain.utils import IdGenerator, extract_json, json_dumps

_ids = IdGenerator()
_OPEN_RE = re.compile(r"^[ \t]*```tool[ \t]*(.*)$")  # the tag line; JSON may start on the same line
_CLOSE_RE = re.compile(r"^[ \t]*```[ \t]*$")  # a closing fence is a line of its own: fences inside JSON strings never are

PROTOCOL_INSTRUCTIONS = """## Tool calls (text protocol)

This model is driven without native tool calling. To call a tool, write a fenced block with the
language tag `tool` containing a JSON object `{"name": "<tool>", "arguments": {...}}`. To call
several independent tools at once, put a JSON list of such objects in one block. Put nothing else
in the block. After the block, stop and wait for the results, which arrive as the next message.
When you have finished and need no more tools, reply in plain prose with no `tool` block.

Example:

```tool
{"name": "read_file", "arguments": {"path": "sources/INDEX.md"}}
```
"""


def render_tool_catalog(schemas: list[dict[str, Any]]) -> str:
    """A compact, human-readable catalogue of the tools for the system prompt."""
    lines = ["## Available tools", ""]
    for s in schemas:
        fn = s.get("function", s)
        name = fn.get("name", "?")
        desc = (fn.get("description") or "").strip()
        params = fn.get("parameters") or {}
        props = params.get("properties") or {}
        required = set(params.get("required") or [])
        sig = ", ".join(f"{k}{'' if k in required else '?'}: {_type_of(v)}" for k, v in props.items())
        lines.append(f"- `{name}({sig})`: {desc}")
    return "\n".join(lines) + "\n"


def _type_of(schema: dict[str, Any]) -> str:
    t = schema.get("type")
    if isinstance(t, list):
        return "|".join(str(x) for x in t)
    if "enum" in schema:
        return "|".join(str(x) for x in schema["enum"])
    if t == "array":
        return f"{_type_of(schema.get('items') or {})}[]"
    if t == "object" and isinstance(schema.get("properties"), dict):
        required = set(schema.get("required") or [])
        inner = ", ".join(f"{k}{'' if k in required else '?'}: {_type_of(v)}" for k, v in schema["properties"].items() if isinstance(v, dict))
        return "{" + inner + "}"
    return str(t or "any")


def tool_instructions(schemas: list[dict[str, Any]]) -> str:
    return PROTOCOL_INSTRUCTIONS + "\n" + render_tool_catalog(schemas)


def parse_tool_calls(text: str) -> tuple[str, list[ToolCall], list[str]]:
    """Return (text without tool blocks, parsed tool calls, parse errors)."""
    calls: list[ToolCall] = []
    errors: list[str] = []

    def one(obj: Any) -> None:
        if not isinstance(obj, dict) or "name" not in obj:
            errors.append(f"tool block must be an object with a 'name' field; got {json_dumps(obj)[:120]}")
            return
        args = obj.get("arguments", obj.get("args", {}))
        if args is None:
            args = {}
        if not isinstance(args, dict):
            errors.append(f"'arguments' for {obj['name']} must be a JSON object")
            return
        calls.append(ToolCall(id=str(obj.get("id") or _ids.new("call")), name=str(obj["name"]), arguments=args))

    def handle(block: str) -> None:
        data = extract_json(block)
        if data is None:
            errors.append(f"could not parse tool block as JSON: {block.strip()[:120]}")
            return
        if isinstance(data, dict) and isinstance(data.get("tool_calls"), list):
            data = data["tool_calls"]
        if isinstance(data, list):
            for item in data:
                one(item)
        else:
            one(data)

    kept: list[str] = []
    block: list[str] | None = None
    for line in (text or "").splitlines():
        if block is None:
            m = _OPEN_RE.match(line)
            if m:
                block = [m.group(1)] if m.group(1).strip() else []
            else:
                kept.append(line)
        elif _CLOSE_RE.match(line):
            handle("\n".join(block))
            block = None
        else:
            block.append(line)
    if block is not None:  # the model stopped before closing the fence: parse what is there rather than lose the call
        handle("\n".join(block))
    clean = "\n".join(kept).strip()
    return clean, calls, errors


def render_tool_result(name: str, call_id: str, payload: Any) -> str:
    body = payload if isinstance(payload, str) else json_dumps(payload)
    return f"TOOL RESULT {name} ({call_id}):\n{body}"


def render_results_message(results: list[tuple[str, str, Any]]) -> str:
    return "\n\n".join(render_tool_result(n, c, p) for n, c, p in results)
