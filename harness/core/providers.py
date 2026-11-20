"""Additional model providers & per-stage routing.

- ``AnthropicMessagesClient``: talks to Anthropic's ``/v1/messages`` API as well
  as local Anthropic-shaped subscription bridges (``claude_bridge.py`` and
  ``gemini_bridge.py`` / ``agy``). Enforces a hard ``max_prompt_bytes`` ceiling
  (refusing rather than letting a bridge silently truncate) and supports both
  native tool calling and the LaTeX/HTML-safe text protocol in ``tool_protocol``.
- ``ModelRouter``: routes each ``ModelRequest`` by ``metadata["stage"]`` or
  ``metadata["role"]`` to a dedicated client + optional parameter overrides.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .client import ModelClient, ModelError, ModelRequest, ModelResponse, StreamCallback, StreamEvent, split_thinking
from .tool_protocol import parse_tool_calls, render_tools_prompt, strip_tool_calls
from .types import ContentPart, Message, ToolCall, Usage
from .utils import IdGenerator, extract_json


class PromptTooLargeError(ModelError):
    """Raised before sending when a serialized request exceeds the provider's byte cap."""

    def __init__(self, actual_bytes: int, limit_bytes: int, provider: str) -> None:
        self.actual_bytes = actual_bytes
        self.limit_bytes = limit_bytes
        self.provider = provider
        super().__init__(
            f"prompt is {actual_bytes:,} bytes, exceeding {provider} limit of {limit_bytes:,} bytes "
            f"(compact context or trim tool results; raising max_tokens does not raise this limit)",
            status=413,
            retryable=False,
        )


class AnthropicMessagesClient(ModelClient):
    """Anthropic Messages API client (also compatible with claude_bridge & gemini_bridge)."""

    name = "anthropic_messages"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        *,
        effort: Optional[str] = None,
        timeout_s: float = 300.0,
        max_retries: int = 3,
        max_prompt_bytes: Optional[int] = None,
        native_tool_calls: bool = True,
        native_image: bool = True,
        single_turn_flatten: bool = False,
        prompt_caching: bool = True,
        anthropic_version: str = "2023-06-01",
        price_per_million_prompt: float = 0.0,
        price_per_million_completion: float = 0.0,
        ids: Optional[IdGenerator] = None,
        opener: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.effort = effort
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.max_prompt_bytes = max_prompt_bytes
        self.native_tool_calls = native_tool_calls
        self.native_image = native_image
        self.single_turn_flatten = single_turn_flatten
        self.prompt_caching = prompt_caching
        self.anthropic_version = anthropic_version
        self.price_prompt = price_per_million_prompt
        self.price_completion = price_per_million_completion
        self.ids = ids or IdGenerator()
        self._open = opener or urllib.request.urlopen

    def _endpoint(self) -> str:
        if self.base_url.endswith("/v1/messages"):
            return self.base_url
        if self.base_url.endswith("/v1"):
            return self.base_url + "/messages"
        return self.base_url + "/v1/messages"

    def _to_anthropic_tools(self, openai_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for item in openai_tools:
            fn = item.get("function") or item
            out.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            })
        return out

    def _convert_content(self, parts: list[ContentPart]) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for p in parts:
            if p.type == "text" and p.text:
                blocks.append({"type": "text", "text": p.text})
            elif p.type == "image" and p.data_url and self.native_image:
                if p.data_url.startswith("data:") and "," in p.data_url:
                    header, b64 = p.data_url.split(",", 1)
                    media_type = header[5:].split(";", 1)[0] or (p.mime or "image/png")
                    blocks.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}})
                elif p.text:
                    blocks.append({"type": "text", "text": p.text})
            elif p.text:
                blocks.append({"type": "text", "text": p.text})
        return blocks

    def build_payload(self, req: ModelRequest) -> dict[str, Any]:
        system_chunks: list[str] = []
        messages: list[dict[str, Any]] = []
        use_native_tools = bool(req.tools and self.native_tool_calls)

        for m in req.messages:
            if m.role == "system":
                txt = m.plain_text()
                if txt:
                    system_chunks.append(txt)
                continue
            if m.role == "tool":
                if use_native_tools:
                    block = {"type": "tool_result", "tool_use_id": m.tool_call_id or "call", "content": m.plain_text()}
                else:
                    block = {"type": "text", "text": f"[tool_result name={m.name or 'tool'} id={m.tool_call_id or ''}]\n{m.plain_text()}"}
                if messages and messages[-1]["role"] == "user":
                    messages[-1]["content"].append(block)
                else:
                    messages.append({"role": "user", "content": [block]})
                continue
            blocks = self._convert_content(m.content)
            if m.role == "assistant" and m.tool_calls:
                if use_native_tools:
                    for tc in m.tool_calls:
                        blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
                else:
                    rendered = []
                    for tc in m.tool_calls:
                        args_xml = "".join(f'<edgent_arg name="{k}"><![CDATA[{v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}]]></edgent_arg>' for k, v in tc.arguments.items())
                        rendered.append(f'<edgent_call tool="{tc.name}" id="{tc.id}">{args_xml}</edgent_call>')
                    blocks.append({"type": "text", "text": "\n".join(rendered)})
            if not blocks:
                blocks = [{"type": "text", "text": ""}]
            # Merge adjacent same-role turns (Anthropic requires alternating user/assistant)
            if messages and messages[-1]["role"] == m.role:
                messages[-1]["content"].extend(blocks)
            else:
                messages.append({"role": m.role, "content": blocks})

        if req.tools and not use_native_tools:
            system_chunks.append(render_tools_prompt(req.tools, compact=True))

        if not messages:
            messages = [{"role": "user", "content": [{"type": "text", "text": "Continue."}]}]

        if self.single_turn_flatten and messages:
            # Single-turn CLI bridges (e.g. gemini_bridge.py, claude_bridge.py) ignore
            # role=="assistant" messages. Always apply the same [USER]/[ASSISTANT] framing
            # from turn 1 onward so the user-message prefix stays cache-stable across turns.
            flat_blocks: list[dict[str, Any]] = []
            for msg in messages:
                tag = "[ASSISTANT]" if msg["role"] == "assistant" else "[USER]"
                for blk in msg["content"]:
                    if blk.get("type") == "text":
                        txt = blk.get("text") or ""
                        if txt.strip():
                            flat_blocks.append({"type": "text", "text": f"{tag}\n{txt}"})
                    else:
                        flat_blocks.append(blk)
            messages = [{"role": "user", "content": flat_blocks or [{"type": "text", "text": "Continue."}]}]

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "messages": messages,
        }
        use_cache = self.prompt_caching and not self.single_turn_flatten
        if system_chunks:
            sys_text = "\n\n".join(system_chunks)
            if use_cache:
                payload["system"] = [{"type": "text", "text": sys_text, "cache_control": {"type": "ephemeral"}}]
            else:
                payload["system"] = sys_text
        if use_native_tools:
            atools = self._to_anthropic_tools(req.tools)
            if use_cache and atools:
                atools[-1]["cache_control"] = {"type": "ephemeral"}
            payload["tools"] = atools
        if use_cache and messages:
            # Mark the tail of the latest user turn as a conversation prefix cache breakpoint
            # (total <= 3 breakpoints: system, last tool, latest user turn).
            for msg in reversed(messages):
                if msg.get("role") == "user" and isinstance(msg.get("content"), list) and msg["content"]:
                    last_blk = msg["content"][-1]
                    if isinstance(last_blk, dict) and last_blk.get("type") in ("text", "tool_result"):
                        last_blk["cache_control"] = {"type": "ephemeral"}
                    break
        effort = req.metadata.get("effort") or self.effort
        if effort:
            payload["output_config"] = {"effort": effort}
        if req.stop:
            payload["stop_sequences"] = list(req.stop)
        return payload

    def _post(self, payload: dict[str, Any], request_id: str) -> dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if self.max_prompt_bytes is not None and len(data) > self.max_prompt_bytes:
            raise PromptTooLargeError(len(data), self.max_prompt_bytes, self.name)
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": self.anthropic_version,
            "X-Request-Id": request_id,
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
            headers["Authorization"] = f"Bearer {self.api_key}"
        http_req = urllib.request.Request(self._endpoint(), data=data, headers=headers, method="POST")
        delay = 1.0
        last: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                with self._open(http_req, timeout=self.timeout_s) as resp:
                    body = resp.read().decode("utf-8")
                return json.loads(body)
            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8", "replace")[:500]
                except Exception:
                    pass
                retryable = e.code in (408, 409, 425, 429, 500, 502, 503, 504, 529)
                last = ModelError(f"HTTP {e.code}: {body}", status=e.code, retryable=retryable)
                if not retryable:
                    raise last
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last = ModelError(f"transport error: {e}", retryable=True)
            if attempt < self.max_retries:
                time.sleep(delay)
                delay = min(delay * 2, 16.0)
        assert last is not None
        raise last

    def _usage(self, raw: dict[str, Any]) -> Usage:
        u = raw.get("usage") or {}
        bridge = raw.get("_bridge") or {}
        pt = int(u.get("input_tokens", 0) or 0)
        ct = int(u.get("output_tokens", 0) or 0)
        cached = int(u.get("cache_read_input_tokens", 0) or 0)
        rt = int(u.get("reasoning_tokens") or bridge.get("thinking_tokens") or 0)
        cost = (pt * self.price_prompt + ct * self.price_completion) / 1e6
        if cost == 0.0 and bridge.get("cost_usd"):
            try:
                cost = float(bridge["cost_usd"])
            except (TypeError, ValueError):
                cost = 0.0
        return Usage(
            prompt_tokens=pt,
            completion_tokens=ct,
            reasoning_tokens=rt,
            cached_tokens=cached,
            cost_usd=cost,
        )

    def _parse_response(self, raw: dict[str, Any], req: ModelRequest, request_id: str, latency: float) -> ModelResponse:
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for blk in raw.get("content") or []:
            if not isinstance(blk, dict):
                continue
            btype = blk.get("type")
            if btype == "text":
                text_parts.append(str(blk.get("text") or ""))
            elif btype == "thinking":
                reasoning_parts.append(str(blk.get("thinking") or blk.get("text") or ""))
            elif btype == "tool_use":
                inp = blk.get("input")
                tool_calls.append(ToolCall(
                    id=str(blk.get("id") or self.ids.new("call")),
                    name=str(blk.get("name") or ""),
                    arguments=inp if isinstance(inp, dict) else {"value": inp},
                ))
        text = "\n".join(text_parts).strip()
        reasoning = "\n".join(reasoning_parts).strip() or None
        if not reasoning:
            r2, text = split_thinking(text)
            reasoning = r2 or reasoning

        if not tool_calls and req.tools and not self.native_tool_calls:
            schemas = {(t.get("function") or t).get("name", ""): (t.get("function") or t).get("parameters") or {} for t in req.tools}
            tool_calls = parse_tool_calls(text, self.ids, schemas)
            if tool_calls:
                text = strip_tool_calls(text)

        return ModelResponse(
            text=text,
            reasoning=reasoning,
            tool_calls=tool_calls,
            usage=self._usage(raw),
            request_id=request_id,
            finish_reason=str(raw.get("stop_reason") or "stop"),
            latency_s=latency,
            model=str(raw.get("model") or self.model),
            raw=raw,
        )

    def complete(self, req: ModelRequest) -> ModelResponse:
        request_id = self.ids.new("req")
        payload = self.build_payload(req)
        t0 = time.monotonic()
        raw = self._post(payload, request_id)
        return self._parse_response(raw, req, request_id, time.monotonic() - t0)

    def complete_stream(self, req: ModelRequest, on_event: Optional[StreamCallback] = None) -> ModelResponse:
        """Stream Anthropic SSE events (falls back to non-streaming for single-turn or text-tool bridges)."""
        if self.single_turn_flatten or (req.tools and not self.native_tool_calls):
            return super().complete_stream(req, on_event=on_event)

        request_id = self.ids.new("req")
        payload = self.build_payload(req)
        payload["stream"] = True
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if self.max_prompt_bytes is not None and len(data) > self.max_prompt_bytes:
            raise PromptTooLargeError(len(data), self.max_prompt_bytes, self.name)
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "anthropic-version": self.anthropic_version,
            "X-Request-Id": request_id,
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
            headers["Authorization"] = f"Bearer {self.api_key}"
        http_req = urllib.request.Request(self._endpoint(), data=data, headers=headers, method="POST")

        t0 = time.monotonic()
        delay = 1.0
        last: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                with self._open(http_req, timeout=self.timeout_s) as resp:
                    text_parts: list[str] = []
                    reasoning_parts: list[str] = []
                    blocks: dict[int, dict[str, Any]] = {}
                    usage_acc: dict[str, Any] = {}
                    finish_reason = "stop"
                    model_name = self.model
                    buffered_json: list[str] = []
                    saw_sse = False

                    for raw_line in resp:
                        line = raw_line.decode("utf-8", "replace") if isinstance(raw_line, (bytes, bytearray)) else str(raw_line)
                        stripped = line.strip()
                        if not stripped or stripped.startswith(":"):
                            continue
                        if not saw_sse and stripped.startswith("{"):
                            buffered_json.append(stripped)
                            continue
                        if buffered_json:
                            buffered_json.append(stripped)
                            continue
                        if stripped.startswith("event:"):
                            saw_sse = True
                            continue
                        if not stripped.startswith("data:"):
                            continue
                        saw_sse = True
                        body = stripped[5:].strip()
                        if not body or body == "[DONE]":
                            continue
                        try:
                            ev = json.loads(body)
                        except json.JSONDecodeError:
                            continue
                        etype = ev.get("type")
                        if etype == "message_start":
                            msg = ev.get("message") or {}
                            if msg.get("model"):
                                model_name = str(msg["model"])
                            if isinstance(msg.get("usage"), dict):
                                usage_acc.update(msg["usage"])
                        elif etype == "content_block_start":
                            idx = int(ev.get("index", 0))
                            cb = ev.get("content_block") or {}
                            blocks[idx] = {"type": cb.get("type", ""), "id": cb.get("id", ""), "name": cb.get("name", ""), "input": cb.get("input"), "json_parts": []}
                            if cb.get("type") == "tool_use" and on_event:
                                on_event(StreamEvent("tool_call_start", data={"name": cb.get("name", ""), "id": cb.get("id", "")}))
                        elif etype == "content_block_delta":
                            idx = int(ev.get("index", 0))
                            delta = ev.get("delta") or {}
                            dtype = delta.get("type")
                            if dtype == "text_delta":
                                chunk = str(delta.get("text") or "")
                                text_parts.append(chunk)
                                if on_event and chunk:
                                    on_event(StreamEvent("text_delta", text=chunk))
                            elif dtype == "thinking_delta":
                                chunk = str(delta.get("thinking") or "")
                                reasoning_parts.append(chunk)
                                if on_event and chunk:
                                    on_event(StreamEvent("thinking_delta", text=chunk))
                            elif dtype == "input_json_delta":
                                chunk = str(delta.get("partial_json") or "")
                                blk = blocks.setdefault(idx, {"type": "tool_use", "id": "", "name": "", "input": None, "json_parts": []})
                                blk["json_parts"].append(chunk)
                                if on_event and chunk:
                                    on_event(StreamEvent("tool_call_delta", text=chunk, data={"index": idx}))
                        elif etype == "message_delta":
                            delta = ev.get("delta") or {}
                            if delta.get("stop_reason"):
                                finish_reason = str(delta["stop_reason"])
                            if isinstance(ev.get("usage"), dict):
                                usage_acc.update(ev["usage"])

                    latency = time.monotonic() - t0
                    if buffered_json and not saw_sse:
                        raw = json.loads("\n".join(buffered_json))
                        parsed_resp = self._parse_response(raw, req, request_id, latency)
                        if on_event:
                            if parsed_resp.reasoning:
                                on_event(StreamEvent("thinking_delta", text=parsed_resp.reasoning))
                            if parsed_resp.text:
                                on_event(StreamEvent("text_delta", text=parsed_resp.text))
                            on_event(StreamEvent("done", data={"finish_reason": parsed_resp.finish_reason, "usage": parsed_resp.usage.to_dict()}))
                        return parsed_resp

                    tool_calls: list[ToolCall] = []
                    for idx in sorted(blocks):
                        blk = blocks[idx]
                        if blk.get("type") != "tool_use":
                            continue
                        if blk["json_parts"]:
                            raw_json = "".join(blk["json_parts"])
                            parsed = extract_json(raw_json)
                            args = parsed if isinstance(parsed, dict) else {"_raw": raw_json}
                        elif isinstance(blk.get("input"), dict):
                            args = blk["input"]
                        else:
                            args = {}
                        tool_calls.append(ToolCall(id=blk.get("id") or self.ids.new("call"), name=blk.get("name") or "", arguments=args))

                    text = "".join(text_parts).strip()
                    reasoning = "".join(reasoning_parts).strip() or None
                    if not reasoning:
                        r2, text = split_thinking(text)
                        reasoning = r2 or reasoning
                    usage = self._usage({"usage": usage_acc})
                    if on_event:
                        on_event(StreamEvent("done", data={"finish_reason": finish_reason, "usage": usage.to_dict()}))
                    return ModelResponse(
                        text=text,
                        reasoning=reasoning,
                        tool_calls=tool_calls,
                        usage=usage,
                        request_id=request_id,
                        finish_reason=finish_reason,
                        latency_s=latency,
                        model=model_name,
                    )
            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8", "replace")[:500]
                except Exception:
                    pass
                retryable = e.code in (408, 409, 425, 429, 500, 502, 503, 504, 529)
                last = ModelError(f"HTTP {e.code}: {body}", status=e.code, retryable=retryable)
                if not retryable:
                    raise last
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last = ModelError(f"transport error: {e}", retryable=True)
            if attempt < self.max_retries:
                time.sleep(delay)
                delay = min(delay * 2, 16.0)
        assert last is not None
        raise last

    def health(self) -> dict[str, Any]:
        for path in ("/healthz", "/v1/models"):
            try:
                url = self.base_url.rstrip("/") + path
                http_req = urllib.request.Request(url, headers={"x-api-key": self.api_key} if self.api_key else {})
                with self._open(http_req, timeout=min(self.timeout_s, 10.0)) as resp:
                    body = resp.read().decode("utf-8", "replace")
                return {"ok": True, "backend": self.name, "base_url": self.base_url, "detail": body[:300]}
            except Exception:
                continue
        return {"ok": False, "backend": self.name, "base_url": self.base_url, "error": "unreachable"}


@dataclass
class StageRoute:
    client: ModelClient
    effort: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    thinking_budget: Optional[int] = None


class ModelRouter(ModelClient):
    """Per-stage / per-role model router implementing ``ModelClient``.

    Lets one harness send planning to one model, implementation to a second,
    independent review to a third, and fast exploration to a fourth.
    """

    name = "router"

    def __init__(self, default: ModelClient, routes: Optional[dict[str, StageRoute]] = None) -> None:
        self.default = default
        self.routes: dict[str, StageRoute] = dict(routes or {})

    def set_route(
        self,
        stage: str,
        client: ModelClient,
        *,
        effort: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        thinking_budget: Optional[int] = None,
    ) -> None:
        self.routes[stage] = StageRoute(client, effort=effort, temperature=temperature, max_tokens=max_tokens, thinking_budget=thinking_budget)

    def resolve(self, req: ModelRequest) -> tuple[ModelClient, ModelRequest]:
        stage = req.metadata.get("stage") or req.metadata.get("role") or ""
        route = self.routes.get(str(stage))
        if route is None:
            return self.default, req
        meta = dict(req.metadata)
        if route.effort and "effort" not in meta:
            meta["effort"] = route.effort
        patched = ModelRequest(
            messages=req.messages,
            tools=req.tools,
            tool_choice=req.tool_choice,
            temperature=req.temperature if route.temperature is None else route.temperature,
            max_tokens=req.max_tokens if route.max_tokens is None else route.max_tokens,
            thinking_budget=req.thinking_budget if route.thinking_budget is None else route.thinking_budget,
            response_format=req.response_format,
            seed=req.seed,
            stop=req.stop,
            metadata=meta,
        )
        return route.client, patched

    def complete(self, req: ModelRequest) -> ModelResponse:
        client, patched = self.resolve(req)
        return client.complete(patched)

    def complete_stream(self, req: ModelRequest, on_event: Optional[StreamCallback] = None) -> ModelResponse:
        client, patched = self.resolve(req)
        return client.complete_stream(patched, on_event=on_event)

    def health(self) -> dict[str, Any]:
        out = {"default": self.default.health(), "routes": {k: r.client.health() for k, r in self.routes.items()}}
        out["ok"] = bool(out["default"].get("ok"))
        return out
