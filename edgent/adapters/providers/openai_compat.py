"""OpenAI-compatible chat completions provider, stdlib only.

One adapter for local servers (vLLM, SGLang, Ollama, LM Studio, llama.cpp) and
any hosted API with the OpenAI shape. ``base_url`` is the API root, usually
ending in ``/v1``; ``/chat/completions`` and ``/models`` are appended. The
server decides what the model can do, so ``caps`` is configurable; the default
assumes native tools, a system channel, images and JSON-schema output.

Effort reaches the model in whichever way the server understands:
``thinking_switch=True`` sends ``chat_template_kwargs.enable_thinking`` (vLLM and
SGLang serving Qwen-style models; ``low`` switches thinking off), and
``reasoning_effort_param=True`` sends ``reasoning_effort`` (OpenAI o-series and
servers that copy it) and omits ``temperature``, which those models refuse; a
server that takes both can get it back through ``extra_body``. Left unset, it
switches itself on for the o-series and gpt-5 families (``is_reasoning_model``),
and ``api.openai.com`` always gets ``max_completion_tokens``, the only spelling
every current model there accepts. Reasoning comes back from ``reasoning_content`` or
``reasoning`` fields, or from a ``<think>...</think>`` block in the content,
and is split out of the visible text either way.
"""
from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from ...domain.capabilities import Capabilities
from ...domain.errors import ConfigurationError, ErrorCategory, HarnessError
from ...domain.types import CompletionRequest, ContentPart, Message, Reply, ToolCall, Usage
from ...domain.utils import IdGenerator
from .base import Provider, ProviderError
from .http import (
    DEFAULT_BACKOFF_S,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT_S,
    Opener,
    Sleep,
    get_json,
    post_json,
    priced,
    unreadable_response,
    validate_base_url,
)

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_CAPS = Capabilities(
    native_tools=True,
    system_channel=True,
    images=True,
    json_schema=True,
    context_tokens=128_000,
    effort_control="switch",
)
FINISH_MAP = {"stop": "stop", "length": "length", "tool_calls": "tool_calls", "function_call": "tool_calls", "content_filter": "error"}
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
AUDIO_FORMATS = {"audio/wav": "wav", "audio/x-wav": "wav", "audio/wave": "wav", "audio/mpeg": "mp3", "audio/mp3": "mp3"}
HEALTH_MODELS_SHOWN = 20
OPENAI_HOSTS = frozenset({"api.openai.com"})  # hosts whose every current model wants max_completion_tokens
REASONING_MODEL_RE = re.compile(r"^(o\d|gpt-5)", re.IGNORECASE)  # families that refuse max_tokens and temperature
UNSUPPORTED_PARAMETER_RE = re.compile(r"unsupported parameter|max_tokens|temperature|reasoning_effort", re.IGNORECASE)
MAX_COMPLETION_TOKENS_PARAM = "max_completion_tokens"
MAX_TOKENS_PARAM = "max_tokens"


def is_reasoning_model(model: str) -> bool:
    """OpenAI's o-series and gpt-5 family: they take ``reasoning_effort`` and refuse ``max_tokens`` and ``temperature``."""
    return bool(REASONING_MODEL_RE.match((model or "").strip()))


def hosted_by_openai(base_url: str) -> bool:
    try:
        host = (urlsplit(base_url or "").hostname or "").lower()
    except ValueError:
        return False
    return host in OPENAI_HOSTS


def reasoning_defaults(base_url: str | None, model: str | None) -> dict[str, Any]:
    """Constructor options the factory and the provider both derive from the base URL and the model (explicit settings win)."""
    out: dict[str, Any] = {}
    if is_reasoning_model(model or ""):
        out["reasoning_effort_param"] = True
    if out.get("reasoning_effort_param") or hosted_by_openai(base_url or ""):
        out["max_tokens_param"] = MAX_COMPLETION_TOKENS_PARAM
    return out


def split_think(text: str) -> tuple[str, str | None]:
    """Separate a ``<think>...</think>`` block from the visible text.

    A closing tag without an opening one (chat templates often pre-fill ``<think>``) treats everything
    before it as reasoning; an unclosed opening tag at the start (cut off while thinking) means there is no
    visible text yet.
    """
    if not text:
        return text, None
    close = text.find(THINK_CLOSE)
    if close != -1:
        head = text[:close]
        open_ = head.find(THINK_OPEN)
        before = head[:open_] if open_ > 0 else ""
        thought = head[open_ + len(THINK_OPEN):] if open_ != -1 else head
        visible = (before + text[close + len(THINK_CLOSE):]).strip()
        return visible, thought.strip() or None
    if text.lstrip().startswith(THINK_OPEN):
        return "", text.lstrip()[len(THINK_OPEN):].strip() or None
    return text, None


def openai_tool_choice(tool_choice: Any) -> Any:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            return tool_choice
        name = tool_choice.get("name")
        if name:
            return {"type": "function", "function": {"name": name}}
    raise ProviderError(f"unsupported tool_choice shape: {tool_choice!r}", retryable=False, category=ErrorCategory.invalid_input)


class OpenAICompatProvider(Provider):
    name = "openai"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_RETRIES,
        opener: Opener | None = None,
        caps: Capabilities | None = None,
        thinking_switch: bool = False,
        reasoning_effort_param: bool | None = None,
        prices: Mapping[str, float] | None = None,
        max_tokens_param: str | None = None,
        extra_body: Mapping[str, Any] | None = None,
        backoff_s: float = DEFAULT_BACKOFF_S,
        ids: IdGenerator | None = None,
        sleep: Sleep | None = None,
    ) -> None:
        if not base_url:
            raise ConfigurationError("the OpenAI-compatible provider needs a base URL (for example http://localhost:8000/v1).")
        if not model:
            raise ConfigurationError("the OpenAI-compatible provider needs a model id; GET {base_url}/models lists what the server serves.")
        self.base_url = validate_base_url(base_url, "http://localhost:8000/v1")
        self.model = model
        self.api_key = api_key or ""
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.opener = opener
        self.thinking_switch = thinking_switch
        defaults = reasoning_defaults(self.base_url, model)
        if reasoning_effort_param is None:  # unset: the model family decides
            reasoning_effort_param = bool(defaults.get("reasoning_effort_param", False))
        self.reasoning_effort_param = reasoning_effort_param
        if caps is None:
            control = "switch" if (thinking_switch or reasoning_effort_param) else "none"
            caps = Capabilities(
                native_tools=DEFAULT_CAPS.native_tools,
                system_channel=DEFAULT_CAPS.system_channel,
                images=DEFAULT_CAPS.images,
                json_schema=DEFAULT_CAPS.json_schema,
                context_tokens=DEFAULT_CAPS.context_tokens,
                effort_control=control,
            )
        self.caps = caps
        self.prices = dict(prices) if prices else None
        # OpenAI's reasoning models reject max_tokens, and api.openai.com takes max_completion_tokens for every model;
        # other servers still speak max_tokens.
        self.max_tokens_param = max_tokens_param or (
            MAX_COMPLETION_TOKENS_PARAM if (reasoning_effort_param or hosted_by_openai(self.base_url)) else MAX_TOKENS_PARAM
        )
        self.extra_body = dict(extra_body or {})
        self.backoff_s = backoff_s
        self.ids = ids or IdGenerator()
        self.sleep = sleep

    def __repr__(self) -> str:
        return f"OpenAICompatProvider(model={self.model!r}, base_url={self.base_url!r})"

    # ------------------------------------------------------------------ completion
    def complete(self, request: CompletionRequest) -> Reply:
        req = self.prepare(request)
        payload = self.build_payload(req)
        request_id = self.ids.new("req")
        t0 = time.monotonic()
        try:
            data = post_json(
                f"{self.base_url}/chat/completions",
                payload,
                headers=self._headers(),
                timeout_s=self.timeout_s,
                opener=self.opener,
                retries=self.max_retries,
                backoff_s=self.backoff_s,
                request_id=request_id,
                sleep=self.sleep,
            )
        except ProviderError as e:
            raise self._with_parameter_hint(e) from e
        try:
            reply = self.parse_response(data)
        except (KeyError, TypeError, AttributeError, IndexError, ValueError) as e:
            raise unreadable_response("the server", data, e) from e
        reply.model = str(data.get("model") or self.model)
        reply.latency_s = time.monotonic() - t0
        reply.request_id = request_id
        return self.finalize(reply)

    def build_payload(self, req: CompletionRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "messages": self.convert_messages(req.messages), self.max_tokens_param: req.max_tokens}
        if not self.reasoning_effort_param:  # OpenAI's reasoning models refuse any temperature but their default
            payload["temperature"] = req.temperature
        if req.stop:
            payload["stop"] = list(req.stop)
        if req.tools:
            payload["tools"] = list(req.tools)
            choice = openai_tool_choice(req.tool_choice)
            if choice is not None:
                payload["tool_choice"] = choice
        if req.json_schema and self.caps.json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": str(req.json_schema.get("title") or "response"), "schema": req.json_schema},
            }
        if self.thinking_switch:
            payload["chat_template_kwargs"] = {"enable_thinking": req.effort != "low"}
        if self.reasoning_effort_param:
            payload["reasoning_effort"] = req.effort
        payload.update(self.extra_body)
        return payload

    def _with_parameter_hint(self, e: ProviderError) -> ProviderError:
        """A 400 that names max_tokens, temperature or reasoning_effort gets the options that switch them."""
        if e.status not in (400, 422) or not UNSUPPORTED_PARAMETER_RE.search(str(e)):
            return e
        other = MAX_TOKENS_PARAM if self.max_tokens_param == MAX_COMPLETION_TOKENS_PARAM else MAX_COMPLETION_TOKENS_PARAM
        sent = self.max_tokens_param + ("" if self.reasoning_effort_param else " and temperature")
        hint = (
            f" This request sent {sent}; set settings.provider.options.reasoning_effort_param to "
            f"{'false' if self.reasoning_effort_param else 'true (o-series and gpt-5 models)'} or max_tokens_param to {other!r}."
        )
        e.error.remediation = (e.error.remediation or "").rstrip() + hint
        return e

    # ------------------------------------------------------------------ request shape
    def convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        pending_media: list[dict[str, Any]] = []  # attachments of the tool results answering one assistant turn

        def flush_media() -> None:
            # tool messages carry text only in this API, and every tool result must directly follow the assistant's
            # tool_calls message, so the attachments of the whole run ride on ONE user turn after the last result
            if pending_media:
                out.append({"role": "user", "content": list(pending_media)})
                pending_media.clear()

        for m in messages:
            if m.role != "tool":
                flush_media()
            if m.role == "system":
                out.append({"role": "system", "content": m.plain_text()})
            elif m.role == "user":
                out.append({"role": "user", "content": self._content(m.content)})
            elif m.role == "assistant":
                out.append(self._assistant(m))
            elif m.role == "tool":
                out.append({"role": "tool", "tool_call_id": m.tool_call_id or "", "content": m.text()})
                media = [p for p in (self._media_part(p) for p in m.media()) if p is not None]
                if media:
                    pending_media.append({"type": "text", "text": f"[attachment from tool {m.name or '?'}]"})
                    pending_media.extend(media)
        flush_media()
        return out

    def _assistant(self, m: Message) -> dict[str, Any]:
        d: dict[str, Any] = {"role": "assistant", "content": m.text() or None}
        if m.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)}}
                for tc in m.tool_calls
            ]
        if d["content"] is None and not m.tool_calls:
            d["content"] = ""
        return d

    def _content(self, parts: list[ContentPart]) -> str | list[dict[str, Any]]:
        if all(p.type == "text" for p in parts):
            return "\n".join(p.text for p in parts if p.text)
        out: list[dict[str, Any]] = []
        for p in parts:
            if p.type == "text":
                if p.text:
                    out.append({"type": "text", "text": p.text})
                continue
            media = self._media_part(p)
            out.append(media if media is not None else {"type": "text", "text": p.text or f"[{p.type}: {p.path}]"})
        return out

    def _media_part(self, p: ContentPart) -> dict[str, Any] | None:
        try:
            if p.type == "image" and self.caps.images:
                return {"type": "image_url", "image_url": {"url": p.data_url()}}
            if p.type == "audio" and self.caps.audio and (p.mime or "") in AUDIO_FORMATS:
                return {"type": "input_audio", "input_audio": {"data": p.load_b64(), "format": AUDIO_FORMATS[p.mime or ""]}}
        except (OSError, ValueError) as e:
            raise ProviderError(f"cannot read media {p.path!r}: {e}", retryable=False, category=ErrorCategory.not_found) from e
        return None

    # ------------------------------------------------------------------ response shape
    def parse_response(self, data: dict[str, Any]) -> Reply:
        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise ProviderError(
                "the server returned no choices",
                retryable=True,
                category=ErrorCategory.unavailable,
                error=HarnessError("provider_empty", "the server returned no choices", ErrorCategory.unavailable, True,
                                   "Retry once; if it persists, check the server logs and the model id."),
            )
        choice = choices[0]
        msg = choice.get("message") or {}
        text = _text_of(msg.get("content"))
        text, think = split_think(text)
        reasoning_field = msg.get("reasoning_content") or msg.get("reasoning")
        reasoning = "\n".join(x for x in (reasoning_field if isinstance(reasoning_field, str) else None, think) if x) or None
        calls, parse_errors = self._tool_calls(msg)
        finish_raw = str(choice.get("finish_reason") or ("tool_calls" if calls else "stop"))
        usage_data = data.get("usage") or {}
        usage = priced(
            Usage(
                prompt_tokens=int(usage_data.get("prompt_tokens") or 0),
                completion_tokens=int(usage_data.get("completion_tokens") or 0),
                reasoning_tokens=int((usage_data.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0),
                cached_tokens=int((usage_data.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
            ),
            self.prices,
            reasoning_in_completion=True,
        )
        raw: dict[str, Any] = {"id": data.get("id"), "finish_reason": finish_raw, "model": data.get("model")}
        if parse_errors:
            raw["tool_parse_errors"] = parse_errors
        if finish_raw not in FINISH_MAP:
            raw["note"] = f"unexpected finish_reason {finish_raw}"
        return Reply(
            text=text,
            tool_calls=calls,
            usage=usage,
            reasoning=reasoning,
            finish_reason="tool_calls" if calls else FINISH_MAP.get(finish_raw, "stop"),
            raw=raw,
        )

    def _tool_calls(self, msg: dict[str, Any]) -> tuple[list[ToolCall], list[str]]:
        calls: list[ToolCall] = []
        errors: list[str] = []
        raw_calls = list(msg.get("tool_calls") or [])
        legacy = msg.get("function_call")
        if isinstance(legacy, dict) and legacy.get("name"):
            raw_calls.append({"function": legacy})
        for tc in raw_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            name = str(fn.get("name") or "")
            raw_args = fn.get("arguments")
            args: dict[str, Any]
            if isinstance(raw_args, dict):
                args = raw_args
            elif raw_args in (None, ""):
                args = {}
            else:
                try:
                    parsed = json.loads(raw_args)
                except (TypeError, ValueError) as e:
                    errors.append(f"{name}: arguments are not valid JSON ({e})")
                    parsed = None
                if isinstance(parsed, dict):
                    args = parsed
                else:
                    if parsed is not None:
                        errors.append(f"{name}: arguments must be a JSON object")
                    args = {"_raw": raw_args}
            calls.append(ToolCall(id=str(tc.get("id") or self.ids.new("call")), name=name, arguments=args))
        return calls, errors

    # ------------------------------------------------------------------ discovery
    def health(self) -> dict[str, Any]:
        try:
            data = get_json(
                f"{self.base_url}/models",
                headers=self._headers(),
                timeout_s=self.timeout_s,
                opener=self.opener,
                retries=self.max_retries,
                backoff_s=self.backoff_s,
                request_id=self.ids.new("req"),
                sleep=self.sleep,
            )
        except ProviderError as e:
            return {"ok": False, "provider": self.name, "model": self.model, "status": e.status, "error": str(e)}
        rows = data.get("data") if isinstance(data.get("data"), list) else data.get("_value") or data.get("models") or []
        ids = [str(r.get("id") or r.get("name") or "") for r in rows if isinstance(r, dict)]
        return {
            "ok": True,
            "provider": self.name,
            "model": self.model,
            "model_listed": (self.model in ids) if ids else None,
            "models_seen": ids[:HEALTH_MODELS_SHOWN],
        }

    # ------------------------------------------------------------------ plumbing
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type", "text") == "text")
    return ""
