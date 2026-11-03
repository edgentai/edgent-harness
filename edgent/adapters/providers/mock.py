"""Offline provider for tests, demos and CI.

Resolution order per request: a scripted reply pushed with ``push``; a
handler registered for the request's role (``metadata['role']`` or the role
marker in the system prompt); the default handler; the built-in policy, which
echoes the last user message. Scripted items may be ``Reply`` objects,
strings, or callables ``(request) -> Reply``.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from ...domain.capabilities import Capabilities
from ...domain.types import CompletionRequest, Reply, ToolCall, Usage
from ...domain.utils import IdGenerator, estimate_tokens
from .base import Provider, find_role

Handler = Callable[[CompletionRequest], Reply]


class MockProvider(Provider):
    name = "mock"
    model = "mock"

    def __init__(
        self,
        caps: Capabilities | None = None,
        handlers: dict[str, Handler] | None = None,
        default: Handler | None = None,
        ids: IdGenerator | None = None,
        fail_on_roles: set[str] | None = None,
    ) -> None:
        self.caps = caps or Capabilities(native_tools=True, system_channel=True, images=True, audio=False, video=False, json_schema=True, context_tokens=200_000, prompt_caching=True, concurrency=8)
        self.handlers: dict[str, Handler] = dict(handlers or {})
        self.default = default
        self.ids = ids or IdGenerator()
        self.script: list[Any] = []
        self.calls: list[CompletionRequest] = []
        self.fail_on_roles = set(fail_on_roles or ())
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ scripting
    def push(self, *items: Any) -> MockProvider:
        for it in items:
            self.script.append(Reply(text=it) if isinstance(it, str) else it)
        return self

    def on_role(self, role: str, handler: Handler) -> MockProvider:
        self.handlers[role] = handler
        return self

    @staticmethod
    def tool_reply(name: str, arguments: dict[str, Any] | None = None, text: str = "", call_id: str | None = None) -> Reply:
        tc = ToolCall(id=call_id or ToolCall.new(name).id, name=name, arguments=dict(arguments or {}))
        return Reply(text=text, tool_calls=[tc], finish_reason="tool_calls")

    @staticmethod
    def tools_reply(calls: list[tuple[str, dict[str, Any]]], text: str = "") -> Reply:
        return Reply(text=text, tool_calls=[ToolCall.new(n, a) for n, a in calls], finish_reason="tool_calls")

    @staticmethod
    def text_reply(text: str) -> Reply:
        return Reply(text=text)

    # ------------------------------------------------------------------ completion
    def complete(self, request: CompletionRequest) -> Reply:
        req = self.prepare(request)
        with self._lock:
            self.calls.append(req)
            role = req.metadata.get("role") or find_role(req.messages) or "main"
            if role in self.fail_on_roles:
                from .base import ProviderError

                raise ProviderError(f"mock failure for role {role}", retryable=False)
            if self.script:
                item = self.script.pop(0)
                reply = item(req) if callable(item) else item
            elif role in self.handlers:
                reply = self.handlers[role](req)
            elif self.default is not None:
                reply = self.default(req)
            else:
                reply = self._policy(req)
        if not reply.request_id:
            reply.request_id = self.ids.new("req")
        if reply.usage.total_tokens == 0:
            pt = req.prompt_tokens_estimate()
            reply.usage = Usage(prompt_tokens=pt, completion_tokens=estimate_tokens(reply.text) + 8 * len(reply.tool_calls), cached_tokens=pt // 2)
        reply.model = self.model
        return self.finalize(reply)

    def _policy(self, req: CompletionRequest) -> Reply:
        last = req.last_user_text()
        return Reply(text=f"[mock] {last[:200]}" if last else "[mock] ok")

    def health(self) -> dict[str, Any]:
        return {"ok": True, "provider": self.name, "model": self.model, "calls": len(self.calls)}
