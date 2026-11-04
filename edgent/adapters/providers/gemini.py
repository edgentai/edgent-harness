"""Gemini API provider: ``generateContent`` over REST (v1beta), stdlib only.

Model ids change; set ``EDGENT_GEMINI_MODEL`` (or ``model=``) to the one you want
and use ``list_models()`` to see which ids your key can call. The default,
``gemini-2.5-pro``, is a reasonable choice at the time of writing.

Effort maps to a thinking budget (``thinking_budgets``: tokens per effort level;
``-1`` means dynamic, ``0`` turns thinking off where the model allows it, ``None``
sends no thinking config). Media the model accepts natively (images, audio, video)
goes inline as base64 under a named 20 MB limit; larger inputs need the Files API,
which this release does not use. Tool schemas are reduced to the OpenAPI subset
Gemini accepts; a free-form object parameter (no ``properties``) or a typeless one
(any JSON value) is advertised as a JSON string and decoded again when the model
calls the tool. Media attached to a tool result travels on a user turn of its own
right after the function-response turn, which may hold nothing but function
responses. Gemini has no tool
call ids, so this provider mints ``call_...`` ids and remembers name and thought
signature per id so later turns can be rebuilt faithfully.
"""
from __future__ import annotations

import json
import mimetypes
import os
import threading
import time
from collections.abc import Mapping
from typing import Any

from ...domain.capabilities import Capabilities
from ...domain.errors import ConfigurationError, ErrorCategory, HarnessError, Limit
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
    with_query,
)

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-2.5-pro"
MODEL_ENV = "EDGENT_GEMINI_MODEL"
API_KEY_HEADER = "x-goog-api-key"

DEFAULT_THINKING_BUDGETS: dict[str, int | None] = {"low": 0, "medium": 2048, "high": 8192}
PRO_MIN_THINKING_BUDGET_TOKENS = 128  # the pro models cannot switch thinking off; 128 is their floor
DYNAMIC_THINKING_BUDGET = -1
GEMINI_MAX_OUTPUT_TOKENS = 65_536  # the output ceiling of the thinking models; a larger maxOutputTokens is refused
_NO_THINKING_MODEL_PREFIXES = ("gemini-1.", "gemini-2.0", "gemma", "embedding", "text-embedding")

INLINE_MEDIA_LIMIT = Limit(
    name="Gemini inline media size",
    value=20 * 1024 * 1024,
    unit="bytes",
    does_not_raise=("max_tokens", "the thinking budget", "the context window"),
)
SIGNATURE_CACHE_ENTRIES = 512  # tool calls remembered for name and thought-signature lookup
MODELS_PAGE_SIZE = 100
MODELS_PAGES_MAX = 5

FINISH_MAP = {"STOP": "stop", "MAX_TOKENS": "length", "MALFORMED_FUNCTION_CALL": "error", "UNEXPECTED_TOOL_CALL": "error"}
BLOCK_FINISH = frozenset({"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY"})

# The OpenAPI subset Gemini accepts in function declarations and response schemas.
_SCHEMA_KEEP = frozenset(
    {
        "type", "format", "description", "nullable", "enum", "maxItems", "minItems", "properties", "required",
        "minProperties", "maxProperties", "minLength", "maxLength", "pattern", "items", "minimum", "maximum",
        "anyOf", "propertyOrdering",
    }
)
JSON_STRING_NOTE = " (a JSON object, encoded as a string)"
ANY_JSON_STRING_NOTE = " (any JSON value, encoded as a string)"
TOOL_MEDIA_LABEL = "[attachment from tool {name}]"  # text part that opens the user turn carrying a tool result's media
# ``format`` is only legal for these values per type; anything else (uri, email, ...) is a 400.
_FORMATS_BY_TYPE = {"string": ("enum", "date-time"), "number": ("float", "double"), "integer": ("int32", "int64")}
# Gemini has no "any": a typeless node in a response schema becomes this union. It cannot include object or array,
# because Gemini rejects an OBJECT without ``properties`` and an ARRAY without typed ``items``; typeless *tool parameters*
# are advertised as JSON strings instead (see ``clean_schema``) so the model can still pass objects and lists.
ANY_VALUE_SCHEMA: dict[str, Any] = {"anyOf": [{"type": "string"}, {"type": "number"}, {"type": "boolean"}]}

GEMINI_CAPS = Capabilities(
    native_tools=True,
    system_channel=True,
    images=True,
    audio=True,
    video=True,
    streaming=False,
    json_schema=True,
    context_tokens=1_000_000,
    effort_control="thinking_budget",
    concurrency=4,
    prompt_caching=True,
    multi_turn=True,
)

_NO_JSON = object()
SchemaPath = tuple[str, ...]


# ----------------------------------------------------------------------------- schemas
def clean_schema(schema: Any, stringify_free_objects: bool = False, _path: SchemaPath = (), _paths: list[SchemaPath] | None = None) -> Any:
    """Reduce a JSON schema to what Gemini accepts; optionally record free-form objects and typeless nodes turned into strings."""
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for k, v in schema.items():
        if k == "properties" and isinstance(v, dict):
            out[k] = {name: clean_schema(sub, stringify_free_objects, (*_path, name), _paths) for name, sub in v.items()}
        elif k == "items":
            out[k] = clean_schema(v, stringify_free_objects, (*_path, "[]"), _paths) if isinstance(v, dict) else {"type": "string"}
        elif k in ("anyOf", "oneOf") and isinstance(v, list):
            out["anyOf"] = [clean_schema(x, stringify_free_objects, _path, _paths) for x in v]
        elif k == "const":
            out["enum"] = [v]
        elif k == "type":
            if isinstance(v, list):
                non_null = [t for t in v if t != "null"]
                out["type"] = non_null[0] if non_null else "string"
                if "null" in v:
                    out["nullable"] = True
            else:
                out["type"] = v
        elif k in _SCHEMA_KEEP:
            out[k] = v
    typeless = _is_typeless(out)
    _legalise(out)
    free_object = out.get("type") == "object" and not out.get("properties")
    if (free_object or typeless) and stringify_free_objects and _path:
        # Gemini rejects an OBJECT with no properties and has no "any" type; advertise a JSON string and decode it on
        # the way back, so the model can still hand over an object or a list.
        if _paths is not None:
            _paths.append(_path)
        note = JSON_STRING_NOTE if free_object else ANY_JSON_STRING_NOTE
        return {"type": "string", "description": ((out.get("description") or "").rstrip() + note).strip()}
    return out


def _is_typeless(out: dict[str, Any]) -> bool:
    """A node that says nothing about its type: any JSON value is meant."""
    return not any(k in out for k in ("type", "anyOf", "enum", "properties", "items"))


def _legalise(out: dict[str, Any]) -> None:
    """Gemini's ``Schema.type`` is required, ``enum`` is for strings only and ``format`` is per type; fix the node in place."""
    if "type" not in out and "anyOf" not in out:
        if "enum" in out:
            out["type"] = "string"
        elif "properties" in out:
            out["type"] = "object"
        elif "items" in out:
            out["type"] = "array"
        else:
            out["anyOf"] = [dict(x) for x in ANY_VALUE_SCHEMA["anyOf"]]
    if "enum" in out:
        values = out["enum"] if isinstance(out["enum"], list) else [out["enum"]]
        if None in values:
            out["nullable"] = True
            values = [x for x in values if x is not None]
        if out.get("type") == "string":
            out["enum"] = [str(x) for x in values]
        else:  # keep the type the tool validates against; the allowed values travel in the description
            out.pop("enum")
            note = "one of: " + ", ".join(json.dumps(x) for x in values)
            desc = (out.get("description") or "").rstrip()
            out["description"] = f"{desc} ({note})" if desc else note
    if "format" in out and out["format"] not in _FORMATS_BY_TYPE.get(str(out.get("type")), ()):
        out.pop("format")


def decode_json_strings(args: Any, paths: list[SchemaPath]) -> Any:
    """Undo ``clean_schema``'s string encoding of free-form objects and typeless values in the arguments the model sent.

    A string that parses as JSON is replaced by the parsed value; prose that is not JSON stays a string.
    """

    def walk(node: Any, path: SchemaPath) -> Any:
        if not path:
            if isinstance(node, str):
                try:
                    return json.loads(node)
                except ValueError:
                    return node
            return node
        seg, rest = path[0], path[1:]
        if seg == "[]":
            return [walk(x, rest) for x in node] if isinstance(node, list) else node
        if isinstance(node, dict) and seg in node:
            node[seg] = walk(node[seg], rest)
        return node

    for p in paths:
        args = walk(args, p)
    return args


def function_declaration(tool: dict[str, Any]) -> tuple[dict[str, Any], list[SchemaPath]]:
    """An OpenAI-style function schema -> a Gemini function declaration plus the JSON-string parameter paths."""
    fn = tool.get("function", tool)
    paths: list[SchemaPath] = []
    params = clean_schema(fn.get("parameters") or {}, stringify_free_objects=True, _paths=paths)
    decl: dict[str, Any] = {"name": fn["name"], "description": fn.get("description", "")}
    if isinstance(params, dict) and params.get("properties"):
        decl["parameters"] = params  # an OBJECT with no properties is rejected, so it is omitted
    return decl, paths


def tool_config(tool_choice: Any) -> dict[str, Any] | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        mode = {"auto": "AUTO", "none": "NONE", "required": "ANY", "any": "ANY"}.get(tool_choice.lower())
        if mode is None:
            raise ProviderError(f"unknown tool_choice {tool_choice!r}", retryable=False, category=ErrorCategory.invalid_input)
        return {"functionCallingConfig": {"mode": mode}}
    if isinstance(tool_choice, dict):
        name = tool_choice.get("name") or (tool_choice.get("function") or {}).get("name")
        if name:
            return {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [name]}}
    raise ProviderError(f"unsupported tool_choice shape: {tool_choice!r}", retryable=False, category=ErrorCategory.invalid_input)


def default_thinking_budgets(model: str) -> dict[str, int | None]:
    budgets = dict(DEFAULT_THINKING_BUDGETS)
    if "-pro" in model:
        budgets["low"] = PRO_MIN_THINKING_BUDGET_TOKENS
    return budgets


def supports_thinking(model: str) -> bool:
    return not model.startswith(_NO_THINKING_MODEL_PREFIXES)


class _InlineBudget:
    def __init__(self, limit: Limit) -> None:
        self.limit = limit
        self.total = 0

    def add(self, b64_len: int) -> None:
        self.total += b64_len * 3 // 4
        if self.total > self.limit.value:
            err = self.limit.exceeded(
                self.total,
                "Send less media: downscale images, trim clips, or attach fewer files in one turn.",
                ErrorCategory.truncated,
                compactable=False,  # one turn's media, not the history: compaction cannot help, so the loop must not force it
            )
            raise ProviderError(err.message, retryable=False, category=ErrorCategory.truncated, error=err)


# ----------------------------------------------------------------------------- provider
class GeminiProvider(Provider):
    name = "gemini"
    caps = GEMINI_CAPS

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_RETRIES,
        opener: Opener | None = None,
        thinking_budgets: Mapping[str, int | None] | None = None,
        prices: Mapping[str, float] | None = None,
        include_thoughts: bool = True,
        backoff_s: float = DEFAULT_BACKOFF_S,
        ids: IdGenerator | None = None,
        sleep: Sleep | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        if not api_key:
            raise ConfigurationError("Gemini needs an API key: set EDGENT_GEMINI_API_KEY (or GEMINI_API_KEY) in the environment that starts Edgent.")
        env = os.environ if environ is None else environ
        self.api_key = api_key
        self.model = (model or env.get(MODEL_ENV) or DEFAULT_MODEL).removeprefix("models/")
        self.base_url = validate_base_url(base_url, DEFAULT_BASE_URL)
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.opener = opener
        self._thinking_budgets: dict[str, int | None] | None = dict(thinking_budgets) if thinking_budgets is not None else None
        self.prices = dict(prices) if prices else None
        self.include_thoughts = include_thoughts
        self.backoff_s = backoff_s
        self.ids = ids or IdGenerator()
        self.sleep = sleep
        self._calls: dict[str, tuple[str, str | None]] = {}  # call id -> (tool name, thought signature)
        self._json_string_params: dict[str, list[SchemaPath]] = {}  # tool name -> free-form object parameter paths
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"GeminiProvider(model={self.model!r}, base_url={self.base_url!r})"

    @property
    def thinking_budgets(self) -> dict[str, int | None]:
        """Configured budgets, else the defaults for the *current* model (``model`` may be reassigned after construction)."""
        return dict(self._thinking_budgets) if self._thinking_budgets is not None else default_thinking_budgets(self.model)

    @thinking_budgets.setter
    def thinking_budgets(self, value: Mapping[str, int | None] | None) -> None:
        self._thinking_budgets = dict(value) if value is not None else None

    # ------------------------------------------------------------------ completion
    def complete(self, request: CompletionRequest) -> Reply:
        req = self.prepare(request)
        payload = self.build_payload(req)
        request_id = self.ids.new("req")
        t0 = time.monotonic()
        data = post_json(
            self._url(f"models/{self.model}:generateContent"),
            payload,
            headers=self._headers(),
            timeout_s=self.timeout_s,
            opener=self.opener,
            retries=self.max_retries,
            backoff_s=self.backoff_s,
            request_id=request_id,
            sleep=self.sleep,
        )
        try:
            reply = self.parse_response(data)
        except (KeyError, TypeError, AttributeError, IndexError, ValueError) as e:
            raise unreadable_response("Gemini", data, e) from e
        reply.model = str(data.get("modelVersion") or self.model)
        reply.latency_s = time.monotonic() - t0
        reply.request_id = request_id
        return self.finalize(reply)

    def build_payload(self, req: CompletionRequest) -> dict[str, Any]:
        system = req.system_text()
        contents = self._contents(req.messages)
        if not contents:
            raise ProviderError("the request has no user, assistant or tool messages to send", retryable=False, category=ErrorCategory.invalid_input)
        payload: dict[str, Any] = {"contents": contents}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        generation: dict[str, Any] = {"temperature": req.temperature, "maxOutputTokens": req.max_tokens}
        if req.stop:
            generation["stopSequences"] = list(req.stop)
        if req.json_schema:
            # Gemini refuses JSON mode together with function declarations; structured output is the explicit ask.
            generation["responseMimeType"] = "application/json"
            generation["responseSchema"] = clean_schema(req.json_schema)
        elif req.tools:
            declarations = []
            for tool in req.tools:
                decl, paths = function_declaration(tool)
                declarations.append(decl)
                with self._lock:
                    self._json_string_params[decl["name"]] = paths
            payload["tools"] = [{"functionDeclarations": declarations}]
            cfg = tool_config(req.tool_choice)
            if cfg:
                payload["toolConfig"] = cfg
        thinking = self.thinking_config(req.effort)
        if thinking:
            generation["thinkingConfig"] = thinking
            # Thought tokens count against maxOutputTokens on the thinking models: without headroom a high effort
            # can spend the whole allowance thinking and return an empty "length" reply.
            generation["maxOutputTokens"] = min(req.max_tokens + self._thinking_headroom(thinking["thinkingBudget"]), GEMINI_MAX_OUTPUT_TOKENS)
        payload["generationConfig"] = generation
        return payload

    def _thinking_headroom(self, budget: int) -> int:
        if budget >= 0:
            return budget  # 0 is thinking off: nothing to reserve
        positive = [b for b in self.thinking_budgets.values() if isinstance(b, int) and b > 0]
        return max(positive, default=0)  # dynamic (-1): reserve the largest configured budget

    def thinking_config(self, effort: str) -> dict[str, Any] | None:
        if not supports_thinking(self.model):
            return None
        budget = self.thinking_budgets.get(effort, self.thinking_budgets.get("medium"))
        if budget is None:
            return None
        cfg: dict[str, Any] = {"thinkingBudget": int(budget)}
        if self.include_thoughts:
            cfg["includeThoughts"] = True
        return cfg

    # ------------------------------------------------------------------ request shape
    def _contents(self, messages: list[Message]) -> list[dict[str, Any]]:
        budget = _InlineBudget(INLINE_MEDIA_LIMIT)
        contents: list[dict[str, Any]] = []
        pending_tool_parts: list[dict[str, Any]] = []
        pending_media_parts: list[dict[str, Any]] = []

        def flush_tools() -> None:
            # the turn after a function call may hold function responses only, one per call; the media those results
            # attached follows on a user turn of its own
            if pending_tool_parts:
                contents.append({"role": "user", "parts": list(pending_tool_parts)})
                pending_tool_parts.clear()
            if pending_media_parts:
                contents.append({"role": "user", "parts": list(pending_media_parts)})
                pending_media_parts.clear()

        for m in messages:
            if m.role == "system":
                continue
            if m.role == "tool":
                pending_tool_parts.append(self._function_response_part(m))
                if m.media():
                    pending_media_parts.append({"text": TOOL_MEDIA_LABEL.format(name=m.name or "?")})
                    pending_media_parts.extend(self._inline_part(p, budget) for p in m.media())
                continue
            flush_tools()
            parts = self._user_or_model_parts(m, budget)
            if parts:
                contents.append({"role": "model" if m.role == "assistant" else "user", "parts": parts})
        flush_tools()
        return contents

    def _user_or_model_parts(self, m: Message, budget: _InlineBudget) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        for p in m.content:
            if p.type == "text":
                if p.text:
                    parts.append({"text": p.text})
            else:
                parts.append(self._inline_part(p, budget))
        signatures = m.meta.get("thought_signatures") if isinstance(m.meta.get("thought_signatures"), dict) else {}
        for tc in m.tool_calls:
            part: dict[str, Any] = {"functionCall": {"name": tc.name, "args": tc.arguments}}
            sig = signatures.get(tc.id) or self._remembered(tc.id)[1]
            if sig:
                part["thoughtSignature"] = sig
            parts.append(part)
        return parts

    def _function_response_part(self, m: Message) -> dict[str, Any]:
        name = m.name or self._remembered(m.tool_call_id or "")[0]
        if not name:
            raise ProviderError(
                f"tool message {m.tool_call_id!r} has no tool name and the call id is unknown to this provider",
                retryable=False,
                category=ErrorCategory.invalid_input,
            )
        text = m.text()
        parsed = _parse_json(text)
        result = parsed if parsed is not _NO_JSON else {"text": text}
        return {"functionResponse": {"name": name, "response": {"result": result}}}

    def _inline_part(self, p: ContentPart, budget: _InlineBudget) -> dict[str, Any]:
        mime = p.mime or (mimetypes.guess_type(p.path)[0] if p.path else None)
        if not mime:
            raise ProviderError(
                f"cannot determine the media type of {p.path!r}; set ContentPart.mime",
                retryable=False,
                category=ErrorCategory.invalid_input,
            )
        try:
            b64 = p.load_b64()
        except (OSError, ValueError) as e:
            raise ProviderError(f"cannot read media {p.path!r}: {e}", retryable=False, category=ErrorCategory.not_found) from e
        budget.add(len(b64))
        return {"inlineData": {"mimeType": mime, "data": b64}}

    # ------------------------------------------------------------------ response shape
    def parse_response(self, data: dict[str, Any]) -> Reply:
        feedback = data.get("promptFeedback") or {}
        if feedback.get("blockReason"):
            raise self._block_error(str(feedback["blockReason"]), feedback.get("safetyRatings"), partial=None, where="prompt")
        candidates = data.get("candidates") or []
        if not candidates:
            raise ProviderError(
                "Gemini returned no candidates",
                retryable=True,
                category=ErrorCategory.unavailable,
                error=HarnessError("provider_empty", "Gemini returned no candidates", ErrorCategory.unavailable, True,
                                   "Retry once; if it persists, change the prompt or the model."),
            )
        cand = candidates[0]
        finish_raw = str(cand.get("finishReason") or "STOP")
        parts = (cand.get("content") or {}).get("parts") or []
        text_chunks: list[str] = []
        thought_chunks: list[str] = []
        calls: list[ToolCall] = []
        signatures: dict[str, str] = {}
        for part in parts:
            if "functionCall" in part:
                fc = part["functionCall"] or {}
                name = str(fc.get("name") or "")
                args = fc.get("args") or {}
                with self._lock:
                    paths = list(self._json_string_params.get(name, []))
                args = decode_json_strings(args if isinstance(args, dict) else {"_raw": args}, paths)
                call = ToolCall(id=self.ids.new("call"), name=name, arguments=args)
                sig = part.get("thoughtSignature")
                self._remember(call.id, name, sig)
                if sig:
                    signatures[call.id] = sig
                calls.append(call)
            elif "text" in part:
                (thought_chunks if part.get("thought") else text_chunks).append(str(part["text"]))
        text = "".join(text_chunks)
        if finish_raw in BLOCK_FINISH:
            raise self._block_error(finish_raw, cand.get("safetyRatings"), partial=text or None, where="response")
        usage = priced(
            Usage(
                prompt_tokens=int((data.get("usageMetadata") or {}).get("promptTokenCount") or 0),
                completion_tokens=int((data.get("usageMetadata") or {}).get("candidatesTokenCount") or 0),
                reasoning_tokens=int((data.get("usageMetadata") or {}).get("thoughtsTokenCount") or 0),
                cached_tokens=int((data.get("usageMetadata") or {}).get("cachedContentTokenCount") or 0),
            ),
            self.prices,
            reasoning_in_completion=False,
        )
        raw: dict[str, Any] = {"finish_reason": finish_raw, "response_id": data.get("responseId"), "model_version": data.get("modelVersion")}
        if signatures:
            raw["thought_signatures"] = signatures
        if finish_raw not in FINISH_MAP:
            raw["note"] = f"unexpected finishReason {finish_raw}"
        return Reply(
            text=text,
            tool_calls=calls,
            usage=usage,
            reasoning="".join(thought_chunks) or None,
            finish_reason="tool_calls" if calls else FINISH_MAP.get(finish_raw, "stop"),
            raw=raw,
        )

    def _block_error(self, reason: str, ratings: Any, partial: str | None, where: str) -> ProviderError:
        flagged = [r.get("category") for r in ratings or [] if isinstance(r, dict) and r.get("blocked")]
        msg = f"Gemini blocked the {where}: {reason}" + (f" (categories: {', '.join(map(str, flagged))})" if flagged else "")
        err = HarnessError(
            "provider_blocked",
            msg,
            ErrorCategory.invalid_input,
            False,
            "Rephrase the content or the request; do not repeat the unchanged call.",
            details={"block_reason": reason, "where": where},
            partial=partial,
        )
        return ProviderError(msg, retryable=False, category=ErrorCategory.invalid_input, error=err)

    # ------------------------------------------------------------------ call bookkeeping
    def _remember(self, call_id: str, name: str, signature: str | None) -> None:
        with self._lock:
            self._calls[call_id] = (name, signature)
            while len(self._calls) > SIGNATURE_CACHE_ENTRIES:
                self._calls.pop(next(iter(self._calls)))

    def _remembered(self, call_id: str) -> tuple[str, str | None]:
        with self._lock:
            return self._calls.get(call_id, ("", None))

    # ------------------------------------------------------------------ discovery
    def health(self) -> dict[str, Any]:
        try:
            data = self._get("models", {"pageSize": 1})
        except ProviderError as e:
            return {"ok": False, "provider": self.name, "model": self.model, "status": e.status, "error": str(e)}
        names = [str(m.get("name", "")).removeprefix("models/") for m in data.get("models") or []]
        listed: bool | None = True if self.model in names else (False if not data.get("nextPageToken") else None)
        return {"ok": True, "provider": self.name, "model": self.model, "model_listed": listed, "models_seen": names}

    def list_models(self) -> list[dict[str, Any]]:
        """Models this key can call with generateContent (a few pages at most)."""
        out: list[dict[str, Any]] = []
        token: str | None = None
        for _ in range(MODELS_PAGES_MAX):
            data = self._get("models", {"pageSize": MODELS_PAGE_SIZE, "pageToken": token})
            for m in data.get("models") or []:
                if "generateContent" not in (m.get("supportedGenerationMethods") or []):
                    continue
                out.append(
                    {
                        "id": str(m.get("name", "")).removeprefix("models/"),
                        "display_name": m.get("displayName"),
                        "input_token_limit": m.get("inputTokenLimit"),
                        "output_token_limit": m.get("outputTokenLimit"),
                    }
                )
            token = data.get("nextPageToken")
            if not token:
                break
        return out

    # ------------------------------------------------------------------ plumbing
    def _url(self, path: str, params: Mapping[str, Any] | None = None) -> str:
        return with_query(f"{self.base_url}/{path}", params)

    def _headers(self) -> dict[str, str]:
        return {API_KEY_HEADER: self.api_key}

    def _get(self, path: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return get_json(
            self._url(path, params),
            headers=self._headers(),
            timeout_s=self.timeout_s,
            opener=self.opener,
            retries=self.max_retries,
            backoff_s=self.backoff_s,
            request_id=self.ids.new("req"),
            sleep=self.sleep,
        )


def _parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return _NO_JSON
