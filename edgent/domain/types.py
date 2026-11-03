"""Messages, content parts, tool calls, usage, replies and completion requests.

These are the only shapes that cross the boundary between the loop, the
providers and the tools. Everything serialises to plain JSON so transcripts
on disk are readable and resumable.
"""
from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass, field
from typing import Any, Literal

from .utils import IdGenerator, estimate_tokens

Role = Literal["system", "user", "assistant", "tool"]
PartType = Literal["text", "image", "audio", "video", "file"]
Effort = Literal["low", "medium", "high"]

_ids = IdGenerator()


@dataclass
class ContentPart:
    """One piece of a message: text, or a media reference.

    Media carries a local ``path`` (and optionally inline base64 ``data_b64``) plus ``text``,
    a textual stand-in used when the provider cannot accept the medium natively.
    """

    type: PartType = "text"
    text: str = ""
    path: str | None = None
    mime: str | None = None
    data_b64: str | None = None

    @classmethod
    def of_text(cls, text: str) -> ContentPart:
        return cls(type="text", text=text)

    @classmethod
    def of_media(cls, type_: PartType, path: str, mime: str | None = None, text: str = "") -> ContentPart:
        return cls(type=type_, path=path, mime=mime or mimetypes.guess_type(path)[0], text=text or f"[{type_}: {path}]")

    def load_b64(self) -> str:
        if self.data_b64:
            return self.data_b64
        if not self.path:
            raise ValueError("media part has neither data nor path")
        with open(self.path, "rb") as f:
            self.data_b64 = base64.b64encode(f.read()).decode("ascii")
        return self.data_b64

    def data_url(self) -> str:
        return f"data:{self.mime or 'application/octet-stream'};base64,{self.load_b64()}"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type}
        if self.text:
            d["text"] = self.text
        if self.path:
            d["path"] = self.path
        if self.mime:
            d["mime"] = self.mime
        return d  # inline data is never written to transcripts

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ContentPart:
        return cls(type=d.get("type", "text"), text=d.get("text", ""), path=d.get("path"), mime=d.get("mime"))


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolCall:
        return cls(id=d["id"], name=d["name"], arguments=dict(d.get("arguments") or {}))

    @classmethod
    def new(cls, name: str, arguments: dict[str, Any] | None = None) -> ToolCall:
        return cls(id=_ids.new("call"), name=name, arguments=dict(arguments or {}))


@dataclass
class Message:
    role: Role
    content: list[ContentPart] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # for role == "tool"
    name: str | None = None  # tool name for role == "tool"
    reasoning: str | None = None  # kept in traces, never re-sent as user input
    meta: dict[str, Any] = field(default_factory=dict)  # persisted_path, cleared, ts, ...

    # -- constructors ---------------------------------------------------
    @classmethod
    def system(cls, text: str) -> Message:
        return cls(role="system", content=[ContentPart.of_text(text)])

    @classmethod
    def user(cls, text: str, parts: list[ContentPart] | None = None) -> Message:
        content = [ContentPart.of_text(text)] if text else []
        return cls(role="user", content=content + list(parts or []))

    @classmethod
    def assistant(cls, text: str = "", tool_calls: list[ToolCall] | None = None, reasoning: str | None = None) -> Message:
        return cls(role="assistant", content=[ContentPart.of_text(text)] if text else [], tool_calls=list(tool_calls or []), reasoning=reasoning)

    @classmethod
    def tool(cls, call_id: str, name: str, text: str, parts: list[ContentPart] | None = None, **meta: Any) -> Message:
        return cls(role="tool", content=[ContentPart.of_text(text)] + list(parts or []), tool_call_id=call_id, name=name, meta=dict(meta))

    # -- helpers --------------------------------------------------------
    def text(self) -> str:
        return "\n".join(p.text for p in self.content if p.type == "text" and p.text)

    def plain_text(self) -> str:
        """All text including textual stand-ins for media (what a text-only provider sees)."""
        return "\n".join(p.text for p in self.content if p.text)

    def media(self) -> list[ContentPart]:
        return [p for p in self.content if p.type != "text"]

    def tokens(self) -> int:
        return estimate_tokens(self.plain_text()) + sum(estimate_tokens(str(tc.arguments)) + 10 for tc in self.tool_calls)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": [p.to_dict() for p in self.content]}
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        if self.reasoning:
            d["reasoning"] = self.reasoning
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Message:
        content = d.get("content", [])
        if isinstance(content, str):
            parts = [ContentPart.of_text(content)]
        else:
            parts = [ContentPart.from_dict(p) for p in content]
        return cls(
            role=d["role"],
            content=parts,
            tool_calls=[ToolCall.from_dict(tc) for tc in d.get("tool_calls", [])],
            tool_call_id=d.get("tool_call_id"),
            name=d.get("name"),
            reasoning=d.get("reasoning"),
            meta=dict(d.get("meta") or {}),
        )


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.reasoning_tokens + other.reasoning_tokens,
            self.cached_tokens + other.cached_tokens,
            self.cost_usd + other.cost_usd,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_tokens": self.cached_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Usage:
        return cls(
            int(d.get("prompt_tokens", 0) or 0),
            int(d.get("completion_tokens", 0) or 0),
            int(d.get("reasoning_tokens", 0) or 0),
            int(d.get("cached_tokens", 0) or 0),
            float(d.get("cost_usd", 0.0) or 0.0),
        )


@dataclass
class Reply:
    """What a provider returns for one completion."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    reasoning: str | None = None
    finish_reason: str = "stop"  # stop | tool_calls | length | error
    model: str = ""
    latency_s: float = 0.0
    request_id: str = ""
    raw: dict[str, Any] | None = None

    def as_message(self) -> Message:
        return Message.assistant(self.text, self.tool_calls, self.reasoning)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "usage": self.usage.to_dict(),
            "reasoning": self.reasoning,
            "finish_reason": self.finish_reason,
            "model": self.model,
            "latency_s": round(self.latency_s, 3),
            "request_id": self.request_id,
        }


@dataclass
class CompletionRequest:
    """One call to a provider. ``tools`` are OpenAI-style function schemas (see tools.registry)."""

    messages: list[Message]
    tools: list[dict[str, Any]] = field(default_factory=list)
    effort: Effort = "medium"
    max_tokens: int = 4096
    temperature: float = 0.2
    tool_choice: Any | None = None  # "auto" | "none" | "required" | {"name": ...}
    json_schema: dict[str, Any] | None = None  # ask for structured output
    stop: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)  # role, session_id, unit_id (for traces and mocks)

    def system_text(self) -> str:
        return "\n\n".join(m.plain_text() for m in self.messages if m.role == "system")

    def non_system(self) -> list[Message]:
        return [m for m in self.messages if m.role != "system"]

    def last_user_text(self) -> str:
        for m in reversed(self.messages):
            if m.role == "user":
                return m.plain_text()
        return ""

    def prompt_tokens_estimate(self) -> int:
        return sum(m.tokens() for m in self.messages) + 60 * len(self.tools)

    def prompt_bytes_estimate(self) -> int:
        return sum(len(m.plain_text().encode("utf-8")) for m in self.messages) + sum(len(str(t)) for t in self.tools)
