"""GeminiProvider: request shapes, response parsing, errors and retries, all through a fake opener."""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest

from edgent.adapters.providers.base import ProviderError
from edgent.adapters.providers.gemini import (
    ANY_JSON_STRING_NOTE,
    ANY_VALUE_SCHEMA,
    INLINE_MEDIA_LIMIT,
    TOOL_MEDIA_LABEL,
    GeminiProvider,
    clean_schema,
    decode_json_strings,
    function_declaration,
)
from edgent.adapters.providers.http import HttpRequest, HttpResponse
from edgent.domain.errors import ConfigurationError
from edgent.domain.types import CompletionRequest, ContentPart, Message, ToolCall
from edgent.domain.utils import SequentialIdGenerator

KEY = "AIzaSy-test-key-0123456789"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "title": "ReadFileArgs",
                "properties": {
                    "path": {"type": "string", "description": "Relative path.", "examples": ["a.md"], "default": ""},
                    "offset": {"type": ["integer", "null"], "minimum": 0},
                    "tags": {"type": "array", "items": {"type": "string", "title": "Tag"}, "additionalProperties": False},
                    "mode": {"const": "text"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {"type": "function", "function": {"name": "ping", "description": "No arguments.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
]


class FakeOpener:
    """Queues responses (dict -> 200 JSON, HttpResponse, or an exception to raise) and records requests."""

    def __init__(self, *responses: Any) -> None:
        self.queue = list(responses)
        self.requests: list[HttpRequest] = []

    def __call__(self, req: HttpRequest) -> HttpResponse:
        self.requests.append(req)
        item = self.queue.pop(0) if self.queue else {"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}]}
        if isinstance(item, Exception):
            raise item
        if isinstance(item, dict):
            return HttpResponse(200, json.dumps(item).encode("utf-8"))
        return item

    @property
    def last(self) -> HttpRequest:
        return self.requests[-1]


def text_response(text: str, finish: str = "STOP", thoughts: str | None = None, usage: dict[str, int] | None = None) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    if thoughts:
        parts.append({"text": thoughts, "thought": True})
    parts.append({"text": text})
    out: dict[str, Any] = {"candidates": [{"content": {"parts": parts, "role": "model"}, "finishReason": finish}], "modelVersion": "gemini-2.5-flash"}
    if usage:
        out["usageMetadata"] = usage
    return out


def calls_response(*calls: tuple[str, dict[str, Any], str | None]) -> dict[str, Any]:
    parts = []
    for name, args, sig in calls:
        part: dict[str, Any] = {"functionCall": {"name": name, "args": args}}
        if sig:
            part["thoughtSignature"] = sig
        parts.append(part)
    return {"candidates": [{"content": {"parts": parts, "role": "model"}, "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 6}}


def make(opener: FakeOpener, model: str = "gemini-2.5-flash", **kw: Any) -> tuple[GeminiProvider, list[float]]:
    sleeps: list[float] = []
    kw.setdefault("ids", SequentialIdGenerator())
    p = GeminiProvider(KEY, model=model, opener=opener, backoff_s=0.0, sleep=sleeps.append, environ={}, **kw)
    return p, sleeps


# ----------------------------------------------------------------------------- request shape
def test_payload_text_system_and_generation_config():
    opener = FakeOpener(text_response("hi"))
    p, _ = make(opener)
    req = CompletionRequest([Message.system("Be brief."), Message.system("Cite sources."), Message.user("hello")], max_tokens=512, temperature=0.3, stop=["END"])
    reply = p.complete(req)
    sent = opener.last
    assert sent.method == "POST" and sent.url == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    assert sent.headers["x-goog-api-key"] == KEY and sent.headers["X-Request-Id"] == reply.request_id
    body = sent.json()
    assert body["systemInstruction"] == {"parts": [{"text": "Be brief.\n\nCite sources."}]}
    assert body["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
    gen = body["generationConfig"]
    assert gen["temperature"] == 0.3 and gen["stopSequences"] == ["END"]
    assert gen["maxOutputTokens"] == 512 + 2048  # thoughts count against maxOutputTokens: the medium budget is reserved on top
    assert gen["thinkingConfig"] == {"thinkingBudget": 2048, "includeThoughts": True}
    assert "tools" not in body and "toolConfig" not in body
    assert reply.text == "hi" and reply.finish_reason == "stop" and reply.model == "gemini-2.5-flash"


@pytest.mark.parametrize(
    ("model", "effort", "budget"),
    [("gemini-2.5-flash", "low", 0), ("gemini-2.5-flash", "high", 8192), ("gemini-2.5-pro", "low", 128), ("gemini-2.5-pro", "medium", 2048)],
)
def test_thinking_budget_by_effort_and_model(model: str, effort: str, budget: int):
    opener = FakeOpener(text_response("x"))
    p, _ = make(opener, model=model)
    p.complete(CompletionRequest([Message.user("q")], effort=effort))  # type: ignore[arg-type]
    assert opener.last.json()["generationConfig"]["thinkingConfig"]["thinkingBudget"] == budget


def test_thinking_config_is_configurable_and_absent_for_models_without_thinking():
    opener = FakeOpener(text_response("x"), text_response("x"), text_response("x"))
    p, _ = make(opener, thinking_budgets={"low": -1, "medium": None, "high": 1024}, include_thoughts=False)
    p.complete(CompletionRequest([Message.user("q")], effort="low"))
    assert opener.last.json()["generationConfig"]["thinkingConfig"] == {"thinkingBudget": -1}
    p.complete(CompletionRequest([Message.user("q")], effort="medium"))
    assert "thinkingConfig" not in opener.last.json()["generationConfig"]
    old, _ = make(FakeOpener(text_response("x")), model="gemini-2.0-flash")
    old.complete(CompletionRequest([Message.user("q")], effort="high"))
    assert "thinkingConfig" not in old.opener.last.json()["generationConfig"]  # type: ignore[attr-defined]


def test_model_default_from_env_and_models_prefix(monkeypatch: pytest.MonkeyPatch):
    assert GeminiProvider(KEY, environ={}).model == "gemini-2.5-pro"
    assert GeminiProvider(KEY, environ={"EDGENT_GEMINI_MODEL": "models/gemini-9-ultra"}).model == "gemini-9-ultra"
    assert GeminiProvider(KEY, model="gemini-2.5-flash", environ={"EDGENT_GEMINI_MODEL": "other"}).model == "gemini-2.5-flash"
    with pytest.raises(ConfigurationError):
        GeminiProvider("")
    assert KEY not in repr(GeminiProvider(KEY, environ={}))


def test_function_declarations_are_cleaned_and_tool_choice_mapped():
    opener = FakeOpener(text_response("x"))
    p, _ = make(opener)
    p.complete(CompletionRequest([Message.user("q")], tools=TOOLS, tool_choice="auto"))
    body = opener.last.json()
    decls = body["tools"][0]["functionDeclarations"]
    assert [d["name"] for d in decls] == ["read_file", "ping"]
    params = decls[0]["parameters"]
    assert set(params) == {"type", "properties", "required"}
    assert params["properties"]["path"] == {"type": "string", "description": "Relative path."}
    assert params["properties"]["offset"] == {"type": "integer", "nullable": True, "minimum": 0}
    assert params["properties"]["tags"] == {"type": "array", "items": {"type": "string"}}
    assert params["properties"]["mode"] == {"type": "string", "enum": ["text"]}  # Schema.type is required
    assert "parameters" not in decls[1]  # an OBJECT without properties is rejected by Gemini
    assert body["toolConfig"] == {"functionCallingConfig": {"mode": "AUTO"}}
    for choice, cfg in (
        ("none", {"mode": "NONE"}),
        ("required", {"mode": "ANY"}),
        ({"name": "ping"}, {"mode": "ANY", "allowedFunctionNames": ["ping"]}),
        ({"type": "function", "function": {"name": "read_file"}}, {"mode": "ANY", "allowedFunctionNames": ["read_file"]}),
    ):
        opener.queue.append(text_response("x"))
        p.complete(CompletionRequest([Message.user("q")], tools=TOOLS, tool_choice=choice))
        assert opener.last.json()["toolConfig"] == {"functionCallingConfig": cfg}
    opener.queue.append(text_response("x"))
    p.complete(CompletionRequest([Message.user("q")], tools=TOOLS))
    assert "toolConfig" not in opener.last.json()


def test_free_form_object_parameter_round_trips_as_json_string():
    tool = {"type": "function", "function": {"name": "set_meta", "description": "d", "parameters": {
        "type": "object",
        "properties": {"data": {"type": "object", "description": "Any JSON."}, "rows": {"type": "array", "items": {"type": "object"}}},
        "required": ["data"],
    }}}
    decl, paths = function_declaration(tool)
    assert decl["parameters"]["properties"]["data"] == {"type": "string", "description": "Any JSON. (a JSON object, encoded as a string)"}
    assert decl["parameters"]["properties"]["rows"]["items"]["type"] == "string"
    assert paths == [("data",), ("rows", "[]")]
    args = decode_json_strings({"data": '{"a": 1}', "rows": ['{"b": 2}', "not json"]}, paths)
    assert args == {"data": {"a": 1}, "rows": [{"b": 2}, "not json"]}
    opener = FakeOpener(text_response("x"), calls_response(("set_meta", {"data": '{"k": [1, 2]}'}, None)))
    p, _ = make(opener)
    p.complete(CompletionRequest([Message.user("q")], tools=[tool]))
    reply = p.complete(CompletionRequest([Message.user("q")], tools=[tool]))
    assert reply.tool_calls[0].arguments == {"data": {"k": [1, 2]}}


def test_clean_schema_keeps_validation_keywords_and_maps_oneof():
    cleaned = clean_schema({"oneOf": [{"type": "string", "maxLength": 3}, {"type": "number", "maximum": 2, "exclusiveMaximum": 1}], "title": "T"})
    assert cleaned == {"anyOf": [{"type": "string", "maxLength": 3}, {"type": "number", "maximum": 2}]}
    assert clean_schema({"type": "object", "properties": {}, "additionalProperties": False}) == {"type": "object", "properties": {}}


def test_tool_calls_round_trip_function_response_grouping_and_signatures():
    opener = FakeOpener(calls_response(("read_file", {"path": "a.md"}, "sig-A"), ("read_file", {"path": "b.md"}, None)), text_response("done"))
    p, _ = make(opener)
    first = p.complete(CompletionRequest([Message.user("read both")], tools=TOOLS))
    assert first.finish_reason == "tool_calls" and [tc.name for tc in first.tool_calls] == ["read_file", "read_file"]
    assert all(tc.id.startswith("call_") for tc in first.tool_calls) and first.tool_calls[0].id != first.tool_calls[1].id
    assert first.raw["thought_signatures"] == {first.tool_calls[0].id: "sig-A"}
    assert first.usage.prompt_tokens == 12 and first.usage.completion_tokens == 6
    history = [
        Message.user("read both"),
        first.as_message(),
        Message.tool(first.tool_calls[0].id, "read_file", '{"content": "A", "lines": 1}'),
        Message.tool(first.tool_calls[1].id, None, "plain text, not json"),  # type: ignore[arg-type]
        Message.user("thanks"),
    ]
    p.complete(CompletionRequest(history, tools=TOOLS))
    contents = opener.last.json()["contents"]
    assert [c["role"] for c in contents] == ["user", "model", "user", "user"]
    model_parts = contents[1]["parts"]
    assert model_parts[0] == {"functionCall": {"name": "read_file", "args": {"path": "a.md"}}, "thoughtSignature": "sig-A"}
    assert model_parts[1] == {"functionCall": {"name": "read_file", "args": {"path": "b.md"}}}
    assert contents[2]["parts"] == [
        {"functionResponse": {"name": "read_file", "response": {"result": {"content": "A", "lines": 1}}}},
        {"functionResponse": {"name": "read_file", "response": {"result": {"text": "plain text, not json"}}}},
    ]
    assert contents[3] == {"role": "user", "parts": [{"text": "thanks"}]}


def test_signature_from_message_meta_and_unknown_tool_name_is_refused():
    opener = FakeOpener(text_response("x"))
    p, _ = make(opener)
    assistant = Message.assistant("", [ToolCall("call_zz", "ping", {})])
    assistant.meta["thought_signatures"] = {"call_zz": "sig-meta"}
    p.complete(CompletionRequest([Message.user("q"), assistant, Message.tool("call_zz", "ping", "{}")], tools=TOOLS))
    parts = opener.last.json()["contents"][1]["parts"]
    assert parts == [{"functionCall": {"name": "ping", "args": {}}, "thoughtSignature": "sig-meta"}]
    with pytest.raises(ProviderError) as ei:
        p.complete(CompletionRequest([Message.user("q"), Message.tool("call_unknown", None, "{}")], tools=TOOLS))  # type: ignore[arg-type]
    assert ei.value.category.value == "invalid_input" and "call_unknown" in str(ei.value)


def test_inline_image_data_and_media_limit(tmp_path: Path):
    png = tmp_path / "pic.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    opener = FakeOpener(text_response("a cat"))
    p, _ = make(opener)
    msg = Message.user("what is this?", [ContentPart.of_media("image", str(png))])
    p.complete(CompletionRequest([msg]))
    parts = opener.last.json()["contents"][0]["parts"]
    assert parts[0] == {"text": "what is this?"}
    assert parts[1] == {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(png.read_bytes()).decode("ascii")}}
    big = ContentPart(type="image", path="/nonexistent/big.png", mime="image/png", data_b64="A" * (int(INLINE_MEDIA_LIMIT.value) * 4 // 3 + 8))
    with pytest.raises(ProviderError) as ei:
        p.complete(CompletionRequest([Message.user("x", [big])]))
    err = ei.value.error
    assert err.category.value == "truncated" and err.details["limit"] == INLINE_MEDIA_LIMIT.name and "Send less media" in err.remediation
    with pytest.raises(ProviderError) as ei2:
        p.complete(CompletionRequest([Message.user("x", [ContentPart.of_media("image", str(tmp_path / "missing.png"))])]))
    assert ei2.value.category.value == "not_found"


def test_tool_message_media_follows_the_function_response_turn(tmp_path: Path):
    """The turn after a function call holds function responses only; attached media rides on the next user turn."""
    shot = tmp_path / "shot.jpg"
    shot.write_bytes(b"\xff\xd8\xff" + b"\x00" * 8)
    opener = FakeOpener(text_response("looks fine"))
    p, _ = make(opener)
    history = [
        Message.user("check the page"),
        Message.assistant("", [ToolCall("call_1", "screenshot", {}), ToolCall("call_2", "grep", {"pattern": "x"})]),
        Message.tool("call_1", "screenshot", '{"ok": true}', [ContentPart.of_media("image", str(shot))]),
        Message.tool("call_2", "grep", '{"hits": []}'),
    ]
    p.complete(CompletionRequest(history, tools=TOOLS))
    contents = opener.last.json()["contents"]
    assert [c["role"] for c in contents] == ["user", "model", "user", "user"]
    responses = contents[2]["parts"]
    assert len(responses) == 2 and all(set(part) == {"functionResponse"} for part in responses)
    assert [part["functionResponse"]["name"] for part in responses] == ["screenshot", "grep"]
    media = contents[3]["parts"]
    assert media[0] == {"text": TOOL_MEDIA_LABEL.format(name="screenshot")} and media[1]["inlineData"]["mimeType"] == "image/jpeg"
    # a following user message still comes after the media turn, and a result without media adds no media turn
    opener.queue.append(text_response("ok"))
    p.complete(CompletionRequest([*history, Message.user("thanks")], tools=TOOLS))
    assert [c["role"] for c in opener.last.json()["contents"]] == ["user", "model", "user", "user", "user"]
    opener.queue.append(text_response("ok"))
    p.complete(CompletionRequest(history[:2] + [Message.tool("call_1", "screenshot", "{}"), history[3]], tools=TOOLS))
    assert [c["role"] for c in opener.last.json()["contents"]] == ["user", "model", "user"]


def test_media_limit_error_tells_the_loop_not_to_compact():
    big = ContentPart(type="image", path="/nonexistent/big.png", mime="image/png", data_b64="A" * (int(INLINE_MEDIA_LIMIT.value) * 4 // 3 + 8))
    p, _ = make(FakeOpener())
    with pytest.raises(ProviderError) as ei:
        p.complete(CompletionRequest([Message.user("x", [big])]))
    assert ei.value.category.value == "truncated" and ei.value.error.details["compactable"] is False


def test_typeless_tool_parameter_round_trips_as_a_json_string():
    """A parameter with no type means any JSON value; a string|number|boolean union could never carry an object or a list."""
    tool = {"type": "function", "function": {"name": "set_value", "description": "d", "parameters": {
        "type": "object",
        "properties": {"value": {"description": "Any JSON."}, "name": {"type": "string"}, "items": {"type": "array", "items": {}}},
        "required": ["value"],
    }}}
    decl, paths = function_declaration(tool)
    props = decl["parameters"]["properties"]
    assert props["value"] == {"type": "string", "description": "Any JSON." + ANY_JSON_STRING_NOTE}
    assert props["items"]["items"] == {"type": "string", "description": ANY_JSON_STRING_NOTE.strip()}
    assert props["name"] == {"type": "string"} and paths == [("value",), ("items", "[]")]
    decoded = decode_json_strings({"value": '{"a": [1, 2]}', "name": "42", "items": ["[1]", "3.5", "true", "plain prose"]}, paths)
    assert decoded == {"value": {"a": [1, 2]}, "name": "42", "items": [[1], 3.5, True, "plain prose"]}
    opener = FakeOpener(calls_response(("set_value", {"value": "[1, {\"b\": 2}]", "name": "n"}, None)))
    p, _ = make(opener)
    reply = p.complete(CompletionRequest([Message.user("q")], tools=[tool]))
    assert reply.tool_calls[0].arguments == {"value": [1, {"b": 2}], "name": "n"}
    # response schemas have no decode step, so a typeless node there keeps the scalar union Gemini can type
    assert clean_schema({"properties": {"v": {}}})["properties"]["v"] == ANY_VALUE_SCHEMA


def test_json_schema_mode_wins_over_tools():
    opener = FakeOpener(text_response('{"a": 1}'))
    p, _ = make(opener)
    schema = {"type": "object", "properties": {"a": {"type": "integer", "default": 0}}, "additionalProperties": False, "$schema": "x"}
    p.complete(CompletionRequest([Message.user("q")], tools=TOOLS, json_schema=schema))
    body = opener.last.json()
    gen = body["generationConfig"]
    assert gen["responseMimeType"] == "application/json"
    assert gen["responseSchema"] == {"type": "object", "properties": {"a": {"type": "integer"}}}
    assert "tools" not in body


def test_empty_request_is_refused_before_the_network():
    opener = FakeOpener()
    p, _ = make(opener)
    with pytest.raises(ProviderError) as ei:
        p.complete(CompletionRequest([Message.system("only system")]))
    assert ei.value.category.value == "invalid_input" and not opener.requests


# ----------------------------------------------------------------------------- response parsing
def test_parse_text_thoughts_usage_and_cost():
    usage = {"promptTokenCount": 1000, "candidatesTokenCount": 200, "thoughtsTokenCount": 300, "cachedContentTokenCount": 400, "totalTokenCount": 1500}
    opener = FakeOpener(text_response("Answer.", thoughts="Let me think.", usage=usage))
    p, _ = make(opener, prices={"input": 1.0, "output": 10.0, "cached_input": 0.1})
    r = p.complete(CompletionRequest([Message.user("q")]))
    assert r.text == "Answer." and r.reasoning == "Let me think."
    assert (r.usage.prompt_tokens, r.usage.completion_tokens, r.usage.reasoning_tokens, r.usage.cached_tokens) == (1000, 200, 300, 400)
    # 600 uncached * 1.0 + 400 cached * 0.1 + (200 + 300) output * 10.0, per million tokens
    assert r.usage.cost_usd == pytest.approx((600 * 1.0 + 400 * 0.1 + 500 * 10.0) / 1_000_000)
    assert r.raw["finish_reason"] == "STOP" and r.request_id.startswith("req_") and r.latency_s >= 0


def test_text_parts_are_concatenated_and_finish_reasons_mapped():
    multi = {"candidates": [{"content": {"parts": [{"text": "Hello, "}, {"text": "world"}]}, "finishReason": "MAX_TOKENS"}]}
    opener = FakeOpener(multi, {"candidates": [{"content": {"parts": []}, "finishReason": "MALFORMED_FUNCTION_CALL"}]})
    p, _ = make(opener)
    r = p.complete(CompletionRequest([Message.user("q")]))
    assert r.text == "Hello, world" and r.finish_reason == "length"
    r2 = p.complete(CompletionRequest([Message.user("q")]))
    assert r2.finish_reason == "error" and r2.raw["finish_reason"] == "MALFORMED_FUNCTION_CALL"


def test_safety_blocks_and_empty_candidates():
    opener = FakeOpener(
        {"promptFeedback": {"blockReason": "SAFETY", "safetyRatings": [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "blocked": True}]}},
        {"candidates": [{"content": {"parts": [{"text": "partial"}]}, "finishReason": "SAFETY"}]},
        {"candidates": []},
    )
    p, _ = make(opener)
    with pytest.raises(ProviderError) as ei:
        p.complete(CompletionRequest([Message.user("q")]))
    assert not ei.value.retryable and ei.value.category.value == "invalid_input"
    assert "SAFETY" in str(ei.value) and "HARM_CATEGORY_DANGEROUS_CONTENT" in str(ei.value)
    with pytest.raises(ProviderError) as ei2:
        p.complete(CompletionRequest([Message.user("q")]))
    assert ei2.value.error.partial == "partial" and ei2.value.error.details["where"] == "response"
    with pytest.raises(ProviderError) as ei3:
        p.complete(CompletionRequest([Message.user("q")]))
    assert ei3.value.category.value == "unavailable" and ei3.value.retryable


# ----------------------------------------------------------------------------- transport
def test_retry_on_503_then_success_and_retry_after():
    opener = FakeOpener(
        HttpResponse(503, b'{"error": {"message": "overloaded", "status": "UNAVAILABLE"}}'),
        HttpResponse(429, b"slow down", {"Retry-After": "2"}),
        ConnectionResetError("reset by peer"),
        text_response("finally"),
    )
    p, sleeps = make(opener, max_retries=3)
    r = p.complete(CompletionRequest([Message.user("q")]))
    assert r.text == "finally" and len(opener.requests) == 4
    assert sleeps == [0.0, 2.0, 0.0]
    assert len({req.headers["X-Request-Id"] for req in opener.requests}) == 1  # one id across retries


def test_retries_exhausted_is_retryable_error():
    opener = FakeOpener(HttpResponse(503, b"down"), HttpResponse(502, b"bad gateway"))
    p, _ = make(opener, max_retries=1)
    with pytest.raises(ProviderError) as ei:
        p.complete(CompletionRequest([Message.user("q")]))
    e = ei.value
    assert e.retryable and e.status == 502 and e.category.value == "unavailable" and "2 attempt(s)" in str(e)
    assert e.error.details["attempts"] == 2 and e.error.code == "provider_http_502"
    opener2 = FakeOpener(TimeoutError("timed out"), TimeoutError("timed out"))
    p2, _ = make(opener2, max_retries=1)
    with pytest.raises(ProviderError) as ei2:
        p2.complete(CompletionRequest([Message.user("q")]))
    assert ei2.value.retryable and ei2.value.category.value == "timeout" and ei2.value.status is None


def test_non_retryable_400_with_redacted_secret():
    body = json.dumps({"error": {"code": 400, "message": f"Bad key {KEY} in header x-goog-api-key: {KEY}; Bearer {KEY}", "status": "INVALID_ARGUMENT"}})
    opener = FakeOpener(HttpResponse(400, body.encode()))
    p, sleeps = make(opener, max_retries=3)
    with pytest.raises(ProviderError) as ei:
        p.complete(CompletionRequest([Message.user("q")]))
    e = ei.value
    assert e.status == 400 and not e.retryable and e.category.value == "invalid_input"
    assert len(opener.requests) == 1 and sleeps == []
    everything = str(e) + json.dumps(e.error.to_dict())
    assert KEY not in everything and "[REDACTED]" in everything and "Bad key" in str(e)
    assert "do not repeat" in e.error.remediation.lower()
    for status, category in ((401, "permission_denied"), (404, "not_found"), (413, "truncated")):
        opener.queue.append(HttpResponse(status, b"nope"))
        with pytest.raises(ProviderError) as ei2:
            p.complete(CompletionRequest([Message.user("q")]))
        assert ei2.value.status == status and ei2.value.category.value == category and not ei2.value.retryable


def test_non_json_success_body_and_oversize_response():
    opener = FakeOpener(HttpResponse(200, b"<html>proxy login</html>"))
    p, _ = make(opener)
    with pytest.raises(ProviderError) as ei:
        p.complete(CompletionRequest([Message.user("q")]))
    assert "non-JSON" in str(ei.value) and not ei.value.retryable
    from edgent.adapters.providers.http import RESPONSE_SIZE_LIMIT

    opener.queue.append(HttpResponse(200, b"x" * (int(RESPONSE_SIZE_LIMIT.value) + 1)))
    with pytest.raises(ProviderError) as ei2:
        p.complete(CompletionRequest([Message.user("q")]))
    assert ei2.value.category.value == "resource_constraint" and RESPONSE_SIZE_LIMIT.name in str(ei2.value)


# ----------------------------------------------------------------------------- discovery
def test_health_and_list_models():
    opener = FakeOpener(
        {"models": [{"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]}], "nextPageToken": "t2"},
        {"models": [{"name": "models/gemini-2.5-flash", "displayName": "Flash", "inputTokenLimit": 1_000_000, "outputTokenLimit": 65536, "supportedGenerationMethods": ["generateContent"]},
                    {"name": "models/embedding-001", "supportedGenerationMethods": ["embedContent"]}], "nextPageToken": "p2"},
        {"models": [{"name": "models/gemini-2.5-pro", "supportedGenerationMethods": ["generateContent", "countTokens"]}]},
        HttpResponse(403, b'{"error": {"message": "API key not valid"}}'),
    )
    p, _ = make(opener)
    h = p.health()
    assert h == {"ok": True, "provider": "gemini", "model": "gemini-2.5-flash", "model_listed": True, "models_seen": ["gemini-2.5-flash"]}
    assert opener.last.method == "GET" and opener.last.url.endswith("/models?pageSize=1") and opener.last.headers["x-goog-api-key"] == KEY
    models = p.list_models()
    assert [m["id"] for m in models] == ["gemini-2.5-flash", "gemini-2.5-pro"] and models[0]["input_token_limit"] == 1_000_000
    assert "pageToken=p2" in opener.last.url
    bad = p.health()
    assert bad["ok"] is False and bad["status"] == 403 and "API key not valid" in bad["error"] and KEY not in bad["error"]


def test_health_reports_unknown_listing_when_paginated():
    opener = FakeOpener({"models": [{"name": "models/other"}], "nextPageToken": "more"})
    p, _ = make(opener)
    assert p.health()["model_listed"] is None


# ----------------------------------------------------------------------------- review additions
def _every_node_is_typed(node: Any) -> bool:
    if not isinstance(node, dict):
        return True
    if "type" not in node and "anyOf" not in node:
        return False
    children = list((node.get("properties") or {}).values()) + [node.get("items")] + list(node.get("anyOf") or [])
    return all(_every_node_is_typed(c) for c in children if c is not None)


def test_response_schema_nodes_get_the_types_gemini_requires():
    from edgent.tools.edu.assess import ITEM_SET_SCHEMA  # the real examiner schema: typeless enum and "any" nodes

    cleaned = clean_schema(ITEM_SET_SCHEMA)
    assert _every_node_is_typed(cleaned)
    item = cleaned["properties"]["items"]["items"]["properties"]
    assert item["kind"]["type"] == "string" and item["kind"]["enum"] == list(ITEM_SET_SCHEMA["properties"]["items"]["items"]["properties"]["kind"]["enum"])
    assert item["answer"]["anyOf"] == [{"type": "string"}, {"type": "number"}, {"type": "boolean"}] and item["answer"]["description"]
    assert cleaned["properties"]["level"] == {"type": "string", "nullable": True}
    # enum on a non-string keeps the type the registry validates against; the values travel in the description
    assert clean_schema({"type": "integer", "enum": [1, 2], "description": "Level."}) == {"type": "integer", "description": "Level. (one of: 1, 2)"}
    assert clean_schema({"enum": ["a", None]}) == {"type": "string", "enum": ["a"], "nullable": True}
    # format survives only where Gemini allows it
    assert clean_schema({"type": "string", "format": "uri"}) == {"type": "string"}
    assert clean_schema({"type": "string", "format": "date-time"}) == {"type": "string", "format": "date-time"}
    assert clean_schema({"type": "integer", "format": "int64"}) == {"type": "integer", "format": "int64"}
    assert clean_schema({"type": "number", "format": "int64"}) == {"type": "number"}
    opener = FakeOpener(text_response("{}"))
    p, _ = make(opener)
    p.complete(CompletionRequest([Message.user("q")], json_schema=ITEM_SET_SCHEMA))
    assert _every_node_is_typed(opener.last.json()["generationConfig"]["responseSchema"])


def test_thinking_floor_follows_a_model_reassigned_after_construction():
    opener = FakeOpener(text_response("x"), text_response("x"))
    p, _ = make(opener, model="gemini-2.5-flash")
    p.model = "gemini-2.5-pro"  # App sets provider.model from --model after make_provider() built it
    p.complete(CompletionRequest([Message.user("q")], effort="low"))
    assert opener.last.url.endswith("/models/gemini-2.5-pro:generateContent")
    assert opener.last.json()["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 128  # pro cannot switch thinking off
    explicit, _ = make(opener, model="gemini-2.5-flash", thinking_budgets={"low": 0, "medium": 0, "high": 0})
    explicit.model = "gemini-2.5-pro"
    explicit.complete(CompletionRequest([Message.user("q")], effort="low"))
    assert opener.last.json()["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 0  # explicit budgets are the user's call
    explicit.thinking_budgets = {"low": 7}
    assert explicit.thinking_budgets == {"low": 7}


def test_max_output_tokens_reserves_the_thinking_budget():
    opener = FakeOpener(text_response("x"), text_response("x"), text_response("x"), text_response("x"))
    p, _ = make(opener)
    p.complete(CompletionRequest([Message.user("q")], effort="high", max_tokens=1000))
    assert opener.last.json()["generationConfig"]["maxOutputTokens"] == 1000 + 8192
    p.complete(CompletionRequest([Message.user("q")], effort="low", max_tokens=1000))
    assert opener.last.json()["generationConfig"]["maxOutputTokens"] == 1000  # budget 0: nothing to reserve
    p.complete(CompletionRequest([Message.user("q")], effort="high", max_tokens=60_000))
    assert opener.last.json()["generationConfig"]["maxOutputTokens"] == 65_536  # the model's ceiling is never exceeded
    dyn, _ = make(opener, thinking_budgets={"low": -1, "medium": 512, "high": 4096})
    dyn.complete(CompletionRequest([Message.user("q")], effort="low", max_tokens=100))
    assert opener.last.json()["generationConfig"]["maxOutputTokens"] == 100 + 4096  # dynamic: the largest configured budget
    old, _ = make(FakeOpener(text_response("x")), model="gemini-2.0-flash")
    old.complete(CompletionRequest([Message.user("q")], effort="high", max_tokens=300))
    assert old.opener.last.json()["generationConfig"]["maxOutputTokens"] == 300  # type: ignore[attr-defined]


def test_error_object_in_a_2xx_body_is_a_typed_redacted_error():
    body = {"error": {"code": 400, "message": f"API key not valid: {KEY}", "status": "INVALID_ARGUMENT"}}
    opener = FakeOpener(body, body)
    p, sleeps = make(opener)
    with pytest.raises(ProviderError) as ei:
        p.complete(CompletionRequest([Message.user("q")]))
    e = ei.value
    assert e.status == 400 and not e.retryable and e.category.value == "invalid_input" and len(opener.requests) == 1 and sleeps == []
    everything = str(e) + json.dumps(e.error.to_dict())
    assert "API key not valid" in everything and KEY not in everything and "[REDACTED]" in everything
    h = p.health()
    assert h["ok"] is False and h["status"] == 400 and KEY not in json.dumps(h)


def test_unreadable_response_shape_is_a_typed_error():
    opener = FakeOpener({"candidates": [5]}, {"candidates": {"content": {}}}, {"candidates": [{"content": {"parts": [{"functionCall": "x"}]}}]})
    p, _ = make(opener)
    for _ in range(3):
        with pytest.raises(ProviderError) as ei:
            p.complete(CompletionRequest([Message.user("q")]))
        assert not ei.value.retryable and ei.value.category.value == "execution_failure" and ei.value.error.code == "provider_bad_response"
        assert "candidates" in str(ei.value) and "base URL" in ei.value.error.remediation


def test_base_url_needs_a_scheme_and_the_message_hides_query_secrets():
    with pytest.raises(ConfigurationError, match="http"):
        GeminiProvider(KEY, base_url="generativelanguage.googleapis.com/v1beta", environ={})
    with pytest.raises(ConfigurationError) as ei:
        GeminiProvider(KEY, base_url="proxy.local/v1beta?key=abc-secret-123456", environ={})
    assert "abc-secret-123456" not in str(ei.value) and "[REDACTED]" in str(ei.value)
    assert GeminiProvider(KEY, base_url="https://proxy.local/gemini/", environ={}).base_url == "https://proxy.local/gemini"


def test_value_error_from_the_opener_is_a_configuration_error_not_a_retry():
    opener = FakeOpener(ValueError("unknown url type: 'localhost'"))
    p, sleeps = make(opener, max_retries=3)
    with pytest.raises(ProviderError) as ei:
        p.complete(CompletionRequest([Message.user("q")]))
    e = ei.value
    assert not e.retryable and e.category.value == "configuration_invalid" and e.status is None
    assert len(opener.requests) == 1 and sleeps == [] and "http://" in e.error.remediation
