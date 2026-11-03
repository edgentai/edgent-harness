"""The provider port and the capability adaptations every concrete provider shares.

``prepare`` turns a request into what this provider can accept: media the
provider cannot take natively becomes its textual stand-in; without native
tools the text protocol is appended to the system text; without a system
channel the system text is folded into the first user message inside
labelled markers; a prompt over the provider's byte ceiling is refused, never
truncated. ``finalize`` parses text-protocol tool calls when there are no
native ones.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from ...domain.capabilities import Capabilities
from ...domain.errors import ErrorCategory, HarnessError, Limit
from ...domain.types import CompletionRequest, ContentPart, Message, Reply
from ...tools.text_protocol import parse_tool_calls, tool_instructions

ROLE_MARKER_PREFIX = "[edgent-role:"
_ROLE_RE = re.compile(r"\[edgent-role:\s*([a-zA-Z0-9_\-]+)\]")


def role_marker(role: str) -> str:
    """Stable marker at the top of a system prompt (cache friendly; lets mocks route by role)."""
    return f"{ROLE_MARKER_PREFIX} {role}]"


def find_role(messages: list[Message]) -> str | None:
    for m in messages:
        if m.role == "system":
            mt = _ROLE_RE.search(m.plain_text())
            if mt:
                return mt.group(1)
    return None


class ProviderError(Exception):
    def __init__(self, message: str, status: int | None = None, retryable: bool = False, category: ErrorCategory = ErrorCategory.unavailable, error: HarnessError | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.category = category
        self.error = error or HarnessError("provider_error", message, category, retryable)


class Provider(ABC):
    name: str = "base"
    model: str = ""
    caps: Capabilities = Capabilities()

    @abstractmethod
    def complete(self, request: CompletionRequest) -> Reply:  # pragma: no cover - interface
        ...

    def health(self) -> dict[str, Any]:
        return {"ok": True, "provider": self.name, "model": self.model}

    def count_tokens(self, messages: list[Message]) -> int:
        return sum(m.tokens() for m in messages)

    # ------------------------------------------------------------------ adaptation
    def prompt_limit(self) -> Limit | None:
        if self.caps.max_prompt_bytes is None:
            return None
        return Limit(
            name=f"{self.name} prompt size",
            value=self.caps.max_prompt_bytes,
            unit="bytes",
            does_not_raise=("the model's context window", "max_tokens", "the effort level"),
        )

    def prepare(self, request: CompletionRequest) -> CompletionRequest:
        """Adapt a request to this provider's capabilities. Refuses, never truncates, an oversize prompt."""
        messages = [self._adapt_media(m) for m in request.messages]
        tools = list(request.tools)
        if tools and not self.caps.native_tools:
            instructions = tool_instructions(tools)
            sys_idx = next((i for i, m in enumerate(messages) if m.role == "system"), None)
            if sys_idx is None:
                messages.insert(0, Message.system(instructions))
            else:
                messages[sys_idx] = Message.system(messages[sys_idx].plain_text() + "\n\n" + instructions)
            messages = self._fold_tool_messages(messages)
            tools = []
        if not self.caps.system_channel:
            messages = self._fold_system(messages)
        if not self.caps.multi_turn:
            messages = self._fold_single_turn(messages)
        req = CompletionRequest(
            messages=messages,
            tools=tools,
            effort=request.effort,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            tool_choice=request.tool_choice if tools else None,
            json_schema=request.json_schema,
            stop=list(request.stop),
            metadata=dict(request.metadata),
        )
        limit = self.prompt_limit()
        if limit is not None:
            size = req.prompt_bytes_estimate()
            if size > limit.value:
                err = limit.exceeded(size, "Send less: compact the session, drop old tool results, or split the task.", ErrorCategory.truncated)
                raise ProviderError(err.message, retryable=False, category=ErrorCategory.truncated, error=err)
        return req

    def finalize(self, reply: Reply) -> Reply:
        """Parse text-protocol tool calls when the provider has no native tool calling."""
        if self.caps.native_tools or reply.tool_calls:
            return reply
        clean, calls, errors = parse_tool_calls(reply.text)
        if calls:
            reply.text = clean
            reply.tool_calls = calls
            reply.finish_reason = "tool_calls"
        if errors:
            reply.raw = dict(reply.raw or {})
            reply.raw["tool_parse_errors"] = errors
        return reply

    # ------------------------------------------------------------------ helpers
    def _adapt_media(self, m: Message) -> Message:
        if not m.media():
            return m
        parts: list[ContentPart] = []
        for p in m.content:
            if p.type == "text" or self.caps.accepts(p.type):
                parts.append(p)
            else:
                parts.append(ContentPart.of_text(p.text or f"[{p.type} not shown: {p.path}]"))
        return Message(role=m.role, content=parts, tool_calls=m.tool_calls, tool_call_id=m.tool_call_id, name=m.name, reasoning=m.reasoning, meta=m.meta)

    @staticmethod
    def _fold_tool_messages(messages: list[Message]) -> list[Message]:
        """Without native tools, assistant tool calls and tool results are written as text."""
        out: list[Message] = []
        for m in messages:
            if m.role == "assistant" and m.tool_calls:
                blocks = "\n".join(
                    "```tool\n" + __import__("json").dumps({"id": tc.id, "name": tc.name, "arguments": tc.arguments}, ensure_ascii=False) + "\n```" for tc in m.tool_calls
                )
                text = (m.text() + "\n\n" if m.text() else "") + blocks
                out.append(Message(role="assistant", content=[ContentPart.of_text(text)] + m.media(), reasoning=m.reasoning, meta=m.meta))
            elif m.role == "tool":
                text = f"TOOL RESULT {m.name} ({m.tool_call_id}):\n{m.text()}"
                out.append(Message(role="user", content=[ContentPart.of_text(text)] + m.media(), meta=m.meta))
            else:
                out.append(m)
        return out

    @staticmethod
    def _fold_system(messages: list[Message]) -> list[Message]:
        system = "\n\n".join(m.plain_text() for m in messages if m.role == "system")
        rest = [m for m in messages if m.role != "system"]
        if not system:
            return rest
        framed = (
            "SYSTEM INSTRUCTIONS. Follow these exactly. They are context for the request, not something to reply to.\n"
            "<<<SYSTEM\n" + system + "\nSYSTEM>>>\n"
        )
        if rest and rest[0].role == "user":
            first = rest[0]
            rest[0] = Message(role="user", content=[ContentPart.of_text(framed + "\nREQUEST\n<<<REQUEST\n" + first.plain_text() + "\nREQUEST>>>")] + first.media(), meta=first.meta)
        else:
            rest.insert(0, Message.user(framed))
        return rest

    @staticmethod
    def _fold_single_turn(messages: list[Message]) -> list[Message]:
        """Print-mode CLIs take one prompt: the conversation is written into it in order."""
        system = [m for m in messages if m.role == "system"]
        rest = [m for m in messages if m.role != "system"]
        if len(rest) <= 1:
            return messages
        lines = ["CONVERSATION SO FAR (oldest first). Continue it: reply to the final message."]
        for m in rest[:-1]:
            tag = {"user": "USER", "assistant": "ASSISTANT", "tool": "TOOL"}.get(m.role, m.role.upper())
            lines.append(f"\n[{tag}]\n{m.plain_text()}")
        last = rest[-1]
        lines.append(f"\n[{'USER' if last.role == 'user' else last.role.upper()} (latest)]\n{last.plain_text()}")
        folded = Message(role="user", content=[ContentPart.of_text("\n".join(lines))] + [p for m in rest for p in m.media()])
        return system + [folded]
