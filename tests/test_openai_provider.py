"""OpenAICompatProvider: message conversion both ways, reasoning splitting, errors and discovery."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from edgent.adapters.providers.base import ProviderError
from edgent.adapters.providers.http import HttpRequest, HttpResponse
from edgent.adapters.providers.openai_compat import (
    DEFAULT_BASE_URL,
    OpenAICompatProvider,
    hosted_by_openai,
    is_reasoning_model,
    reasoning_defaults,
    split_think,
)
from edgent.domain.capabilities import Capabilities
from edgent.domain.errors import ConfigurationError
from edgent.domain.types import CompletionRequest, ContentPart, Message, ToolCall
from edgent.domain.utils import SequentialIdGenerator

KEY = "sk-local-secret-0123456789"
BASE = "http://localhost:8000/v1"
TOOLS = [{"type": "function", "function": {"name": "grep", "description": "Search.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"], "additionalProperties": False}}}]


class FakeOpener:
    def __init__(self, *responses: Any) -> None:
        self.queue = list(responses)
        self.requests: list[HttpRequest] = []

    def __call__(self, req: HttpRequest) -> HttpResponse:
        self.requests.append(req)
        item = self.queue.pop(0) if self.queue else chat("ok")
        if isinstance(item, Exception):
            raise item
        if isinstance(item, dict):
            return HttpResponse(200, json.dumps(item).encode("utf-8"))
        return item

    @property
    def last(self) -> HttpRequest:
        return self.requests[-1]


def chat(content: Any, finish: str = "stop", tool_calls: list[dict[str, Any]] | None = None, usage: dict[str, Any] | None = None, **message_extra: Any) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content, **message_extra}
    if tool_calls:
        message["tool_calls"] = tool_calls
    out: dict[str, Any] = {"id": "chatcmpl-1", "model": "served-model", "choices": [{"index": 0, "message": message, "finish_reason": finish}]}
    if usage:
        out["usage"] = usage
    return out


def make(opener: FakeOpener, **kw: Any) -> tuple[OpenAICompatProvider, list[float]]:
    sleeps: list[float] = []
    kw.setdefault("api_key", KEY)
    kw.setdefault("ids", SequentialIdGenerator())
    p = OpenAICompatProvider(BASE, "qwen3-32b", opener=opener, backoff_s=0.0, sleep=sleeps.append, **kw)
    return p, sleeps


# ----------------------------------------------------------------------------- request shape
def test_payload_messages_tools_and_auth(tmp_path: Path):
    png = tmp_path / "a.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 4)
    opener = FakeOpener(chat("fine"))
    p, _ = make(opener)
    history = [
        Message.system("You are terse."),
        Message.user("look", [ContentPart.of_media("image", str(png))]),
        Message.assistant("I will search.", [ToolCall("call_7", "grep", {"pattern": "x y"})]),
        Message.tool("call_7", "grep", "a.md:1: x y"),
        Message.user("ok"),
    ]
    reply = p.complete(CompletionRequest(history, tools=TOOLS, tool_choice="auto", max_tokens=300, temperature=0.1, stop=["\n\n"]))
    sent = opener.last
    assert sent.url == f"{BASE}/chat/completions" and sent.headers["Authorization"] == f"Bearer {KEY}"
    assert sent.headers["X-Request-Id"] == reply.request_id and sent.headers["Content-Type"].startswith("application/json")
    body = sent.json()
    assert body["model"] == "qwen3-32b" and body["max_tokens"] == 300 and body["temperature"] == 0.1 and body["stop"] == ["\n\n"]
    assert body["tools"] == TOOLS and body["tool_choice"] == "auto"
    assert "chat_template_kwargs" not in body and "reasoning_effort" not in body and "response_format" not in body
    msgs = body["messages"]
    assert msgs[0] == {"role": "system", "content": "You are terse."}
    assert msgs[1]["role"] == "user" and msgs[1]["content"][0] == {"type": "text", "text": "look"}
    assert msgs[1]["content"][1]["type"] == "image_url" and msgs[1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert msgs[2] == {"role": "assistant", "content": "I will search.", "tool_calls": [{"id": "call_7", "type": "function", "function": {"name": "grep", "arguments": '{"pattern": "x y"}'}}]}
    assert msgs[3] == {"role": "tool", "tool_call_id": "call_7", "content": "a.md:1: x y"}
    assert msgs[4] == {"role": "user", "content": "ok"}
    assert reply.text == "fine" and reply.model == "served-model"


def test_tool_choice_shapes_and_no_auth_header_without_key():
    opener = FakeOpener(chat("x"), chat("x"), chat("x"))
    p, _ = make(opener, api_key="")
    p.complete(CompletionRequest([Message.user("q")], tools=TOOLS, tool_choice={"name": "grep"}))
    assert opener.last.json()["tool_choice"] == {"type": "function", "function": {"name": "grep"}}
    assert "Authorization" not in opener.last.headers
    p.complete(CompletionRequest([Message.user("q")], tools=TOOLS, tool_choice="required"))
    assert opener.last.json()["tool_choice"] == "required"
    p.complete(CompletionRequest([Message.user("q")], tool_choice="required"))
    assert "tool_choice" not in opener.last.json() and "tools" not in opener.last.json()


def test_effort_switches_and_max_tokens_param():
    opener = FakeOpener(chat("x"), chat("x"), chat("x"))
    p, _ = make(opener, thinking_switch=True)
    assert p.caps.effort_control == "switch"
    p.complete(CompletionRequest([Message.user("q")], effort="low"))
    assert opener.last.json()["chat_template_kwargs"] == {"enable_thinking": False}
    p.complete(CompletionRequest([Message.user("q")], effort="high"))
    assert opener.last.json()["chat_template_kwargs"] == {"enable_thinking": True}
    o, _ = make(opener, reasoning_effort_param=True, extra_body={"top_k": 20})
    o.complete(CompletionRequest([Message.user("q")], effort="high", max_tokens=99))
    body = opener.last.json()
    assert body["reasoning_effort"] == "high" and body["max_completion_tokens"] == 99 and "max_tokens" not in body and body["top_k"] == 20
    plain, _ = make(FakeOpener(), api_key="")
    assert plain.caps.effort_control == "none" and plain.max_tokens_param == "max_tokens"


def test_response_format_json_schema_and_caps_adaptation():
    opener = FakeOpener(chat('{"a": 1}'))
    p, _ = make(opener)
    schema = {"type": "object", "title": "Answer", "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": False}
    p.complete(CompletionRequest([Message.user("q")], json_schema=schema))
    assert opener.last.json()["response_format"] == {"type": "json_schema", "json_schema": {"name": "Answer", "schema": schema}}
    caps = Capabilities(native_tools=False, system_channel=False, images=False, json_schema=False)
    opener2 = FakeOpener(chat('I will.\n```tool\n{"name": "grep", "arguments": {"pattern": "z"}}\n```'))
    text_only, _ = make(opener2, caps=caps)
    r = text_only.complete(CompletionRequest([Message.system("S"), Message.user("look", [ContentPart.of_media("image", "/x.png", text="[image x.png]")])], tools=TOOLS, json_schema=schema))
    body = opener2.last.json()
    assert "tools" not in body and "response_format" not in body and body["messages"][0]["role"] == "user"
    assert "<<<SYSTEM" in body["messages"][0]["content"] and "[image x.png]" in body["messages"][0]["content"]
    assert r.tool_calls and r.tool_calls[0].arguments == {"pattern": "z"} and r.finish_reason == "tool_calls" and r.text == "I will."


def test_tool_message_media_becomes_a_user_turn(tmp_path: Path):
    shot = tmp_path / "s.png"
    shot.write_bytes(b"\x89PNG\r\n\x1a\n")
    opener = FakeOpener(chat("x"))
    p, _ = make(opener)
    history = [Message.user("q"), Message.assistant("", [ToolCall("c1", "screenshot", {})]), Message.tool("c1", "screenshot", "{}", [ContentPart.of_media("image", str(shot))])]
    p.complete(CompletionRequest(history, tools=TOOLS))
    msgs = opener.last.json()["messages"]
    assert msgs[1] == {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "screenshot", "arguments": "{}"}}]}
    assert msgs[2] == {"role": "tool", "tool_call_id": "c1", "content": "{}"}
    assert msgs[3]["role"] == "user" and msgs[3]["content"][0]["text"] == "[attachment from tool screenshot]" and msgs[3]["content"][1]["type"] == "image_url"


def test_constructor_validation():
    with pytest.raises(ConfigurationError):
        OpenAICompatProvider("", "m")
    with pytest.raises(ConfigurationError):
        OpenAICompatProvider(BASE, "")
    assert KEY not in repr(OpenAICompatProvider(BASE + "/", "m", api_key=KEY)) and OpenAICompatProvider(BASE + "/", "m").base_url == BASE


# ----------------------------------------------------------------------------- response parsing
@pytest.mark.parametrize(
    ("text", "visible", "reasoning"),
    [
        ("<think>\nplan\n</think>\n\nAnswer", "Answer", "plan"),
        ("plan only</think>Answer", "Answer", "plan only"),
        ("<think>cut off while thinking", "", "cut off while thinking"),
        ("no tags here", "no tags here", None),
        ("", "", None),
        ("<think></think>x", "x", None),
    ],
)
def test_split_think(text: str, visible: str, reasoning: str | None):
    assert split_think(text) == (visible, reasoning)


def test_parse_reasoning_fields_parts_content_usage_and_cost():
    usage = {"prompt_tokens": 1000, "completion_tokens": 500, "prompt_tokens_details": {"cached_tokens": 250}, "completion_tokens_details": {"reasoning_tokens": 100}}
    opener = FakeOpener(
        chat("<think>a</think>Final", reasoning_content="b", usage=usage),
        chat([{"type": "text", "text": "part one "}, {"type": "text", "text": "part two"}], reasoning="field only"),
    )
    p, _ = make(opener, prices={"input": 2.0, "output": 8.0})
    r = p.complete(CompletionRequest([Message.user("q")]))
    assert r.text == "Final" and r.reasoning == "b\na"
    assert (r.usage.prompt_tokens, r.usage.completion_tokens, r.usage.cached_tokens, r.usage.reasoning_tokens) == (1000, 500, 250, 100)
    assert r.usage.cost_usd == pytest.approx((750 * 2.0 + 250 * 2.0 + 500 * 8.0) / 1_000_000)  # cached priced as input when no cached price
    r2 = p.complete(CompletionRequest([Message.user("q")]))
    assert r2.text == "part one part two" and r2.reasoning == "field only" and r2.raw["id"] == "chatcmpl-1"


def test_parse_tool_calls_with_bad_arguments_and_finish_reasons():
    calls = [
        {"id": "call_a", "type": "function", "function": {"name": "grep", "arguments": '{"pattern": "x"}'}},
        {"type": "function", "function": {"name": "grep", "arguments": "{not json"}},
        {"id": "call_c", "type": "function", "function": {"name": "grep", "arguments": {"pattern": "dict"}}},
        {"id": "call_d", "type": "function", "function": {"name": "grep", "arguments": "[1, 2]"}},
    ]
    opener = FakeOpener(chat(None, finish="tool_calls", tool_calls=calls), chat("cut", finish="length"), chat("", finish="content_filter"), chat("legacy", function_call={"name": "grep", "arguments": "{}"}))
    p, _ = make(opener)
    r = p.complete(CompletionRequest([Message.user("q")], tools=TOOLS))
    assert r.finish_reason == "tool_calls" and r.text == ""
    assert [tc.id for tc in r.tool_calls][0] == "call_a" and r.tool_calls[1].id.startswith("call_")
    assert r.tool_calls[0].arguments == {"pattern": "x"} and r.tool_calls[2].arguments == {"pattern": "dict"}
    assert r.tool_calls[1].arguments == {"_raw": "{not json"} and r.tool_calls[3].arguments == {"_raw": "[1, 2]"}
    assert len(r.raw["tool_parse_errors"]) == 2 and "not valid JSON" in r.raw["tool_parse_errors"][0]
    assert p.complete(CompletionRequest([Message.user("q")])).finish_reason == "length"
    assert p.complete(CompletionRequest([Message.user("q")])).finish_reason == "error"
    legacy = p.complete(CompletionRequest([Message.user("q")], tools=TOOLS))
    assert legacy.finish_reason == "tool_calls" and legacy.tool_calls[0].name == "grep" and legacy.text == "legacy"


def test_empty_choices_is_retryable_unavailable():
    opener = FakeOpener({"choices": []})
    p, _ = make(opener)
    with pytest.raises(ProviderError) as ei:
        p.complete(CompletionRequest([Message.user("q")]))
    assert ei.value.retryable and ei.value.category.value == "unavailable"


# ----------------------------------------------------------------------------- transport
def test_retry_then_success_and_exhaustion():
    opener = FakeOpener(ConnectionRefusedError("refused"), HttpResponse(500, b"boom"), chat("there"))
    p, sleeps = make(opener, max_retries=2)
    assert p.complete(CompletionRequest([Message.user("q")])).text == "there" and len(opener.requests) == 3 and sleeps == [0.0, 0.0]
    opener2 = FakeOpener(ConnectionRefusedError("refused"), ConnectionRefusedError("refused"))
    p2, _ = make(opener2, max_retries=1)
    with pytest.raises(ProviderError) as ei:
        p2.complete(CompletionRequest([Message.user("q")]))
    assert ei.value.retryable and ei.value.category.value == "unavailable" and "refused" in str(ei.value)


def test_401_is_not_retried_and_bearer_is_redacted():
    body = json.dumps({"error": {"message": f"invalid Authorization: Bearer {KEY}", "type": "invalid_request_error"}}).encode()
    opener = FakeOpener(HttpResponse(401, body))
    p, sleeps = make(opener)
    with pytest.raises(ProviderError) as ei:
        p.complete(CompletionRequest([Message.user("q")]))
    e = ei.value
    assert e.status == 401 and not e.retryable and e.category.value == "permission_denied" and len(opener.requests) == 1 and sleeps == []
    everything = str(e) + json.dumps(e.error.to_dict())
    assert KEY not in everything and "Bearer [REDACTED]" in everything and "credentials" in e.error.remediation


def test_404_hints_at_the_v1_suffix():
    opener = FakeOpener(HttpResponse(404, b"not found"))
    p, _ = make(opener)
    with pytest.raises(ProviderError) as ei:
        p.complete(CompletionRequest([Message.user("q")]))
    assert "/v1" in ei.value.error.remediation and ei.value.category.value == "not_found"


# ----------------------------------------------------------------------------- discovery
def test_health_lists_models():
    opener = FakeOpener({"object": "list", "data": [{"id": "qwen3-32b"}, {"id": "other"}]}, HttpResponse(503, b"starting"), HttpResponse(200, b"[]"))
    p, _ = make(opener, max_retries=0)
    assert p.health() == {"ok": True, "provider": "openai", "model": "qwen3-32b", "model_listed": True, "models_seen": ["qwen3-32b", "other"]}
    assert opener.last.method == "GET" and opener.last.url == f"{BASE}/models" and opener.last.headers["Authorization"] == f"Bearer {KEY}"
    down = p.health()
    assert down["ok"] is False and down["status"] == 503
    assert p.health()["model_listed"] is None  # a bare list body is tolerated; nothing to compare against


# ----------------------------------------------------------------------------- review additions
def test_reasoning_models_get_no_temperature():
    opener = FakeOpener(chat("x"), chat("x"))
    o, _ = make(opener, reasoning_effort_param=True)
    o.complete(CompletionRequest([Message.user("q")], temperature=0.2))
    body = opener.last.json()
    assert "temperature" not in body and body["reasoning_effort"] == "medium"  # o-series/gpt-5 refuse any but the default
    plain, _ = make(opener)
    plain.complete(CompletionRequest([Message.user("q")], temperature=0.2))
    assert opener.last.json()["temperature"] == 0.2


def test_error_object_in_a_2xx_body_and_unreadable_shape():
    opener = FakeOpener({"error": {"message": f"no such model; key {KEY}", "code": "404", "type": "invalid_request_error"}}, {"choices": [{"message": "text"}]})
    p, sleeps = make(opener)
    with pytest.raises(ProviderError) as ei:
        p.complete(CompletionRequest([Message.user("q")]))
    e = ei.value
    assert e.status == 404 and not e.retryable and e.category.value == "not_found" and len(opener.requests) == 1 and sleeps == []
    assert "no such model" in str(e) and KEY not in str(e) + json.dumps(e.error.to_dict())
    with pytest.raises(ProviderError) as ei2:
        p.complete(CompletionRequest([Message.user("q")]))
    assert ei2.value.error.code == "provider_bad_response" and not ei2.value.retryable and "choices" in str(ei2.value)
    assert p.health()["ok"] is True  # the fake's default answer is a chat body: a bare dict without "data" still counts as reachable


def test_base_url_needs_a_scheme():
    with pytest.raises(ConfigurationError, match="http://"):
        OpenAICompatProvider("localhost:8000/v1", "m")
    assert OpenAICompatProvider("HTTP://localhost:8000/v1/", "m").base_url == "HTTP://localhost:8000/v1"


def test_parallel_tool_results_stay_adjacent_and_share_one_attachment_turn(tmp_path: Path):
    """Every tool message must directly follow the assistant's tool_calls message; media waits until the run ends."""
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.write_bytes(b"\x89PNG\r\n\x1a\nA")
    b.write_bytes(b"\x89PNG\r\n\x1a\nB")
    opener = FakeOpener(chat("x"), chat("x"))
    p, _ = make(opener)
    history = [
        Message.user("q"),
        Message.assistant("", [ToolCall("c1", "screenshot", {}), ToolCall("c2", "grep", {"pattern": "z"}), ToolCall("c3", "render", {})]),
        Message.tool("c1", "screenshot", "{}", [ContentPart.of_media("image", str(a))]),
        Message.tool("c2", "grep", '{"hits": 0}'),
        Message.tool("c3", "render", "{}", [ContentPart.of_media("image", str(b))]),
        Message.user("and now?"),
    ]
    p.complete(CompletionRequest(history, tools=TOOLS))
    msgs = opener.last.json()["messages"]
    assert [(m["role"], m.get("tool_call_id")) for m in msgs] == [("user", None), ("assistant", None), ("tool", "c1"), ("tool", "c2"), ("tool", "c3"), ("user", None), ("user", None)]
    attachments = msgs[5]["content"]
    assert [c["type"] for c in attachments] == ["text", "image_url", "text", "image_url"]
    assert attachments[0]["text"] == "[attachment from tool screenshot]" and attachments[2]["text"] == "[attachment from tool render]"
    assert msgs[6] == {"role": "user", "content": "and now?"}
    # media on the last message of the history is still flushed
    p.complete(CompletionRequest(history[:-1], tools=TOOLS))
    assert [m["role"] for m in opener.last.json()["messages"]][-2:] == ["tool", "user"]


def test_openai_hosted_and_reasoning_models_get_the_parameters_they_accept():
    assert is_reasoning_model("gpt-5") and is_reasoning_model("o3-mini") and is_reasoning_model("GPT-5.1-codex") and not is_reasoning_model("gpt-4o")
    assert hosted_by_openai(DEFAULT_BASE_URL) and not hosted_by_openai(BASE) and not hosted_by_openai("not a url")
    assert reasoning_defaults(BASE, "qwen3") == {}
    opener = FakeOpener(chat("x"), chat("x"), chat("x"), chat("x"))
    hosted_reasoning = OpenAICompatProvider(DEFAULT_BASE_URL, "gpt-5", api_key=KEY, opener=opener)
    hosted_reasoning.complete(CompletionRequest([Message.user("q")], max_tokens=77, temperature=0.2))
    body = opener.last.json()
    assert body["max_completion_tokens"] == 77 and "max_tokens" not in body and "temperature" not in body and body["reasoning_effort"] == "medium"
    hosted_plain = OpenAICompatProvider(DEFAULT_BASE_URL, "gpt-4o", api_key=KEY, opener=opener)
    hosted_plain.complete(CompletionRequest([Message.user("q")], max_tokens=77, temperature=0.2))
    body = opener.last.json()
    assert body["max_completion_tokens"] == 77 and body["temperature"] == 0.2 and "reasoning_effort" not in body
    local_reasoning = OpenAICompatProvider(BASE, "o4-mini", api_key=KEY, opener=opener)  # a proxy serving the o-series
    local_reasoning.complete(CompletionRequest([Message.user("q")]))
    assert "max_completion_tokens" in opener.last.json() and "temperature" not in opener.last.json()
    local_plain, _ = make(opener)
    local_plain.complete(CompletionRequest([Message.user("q")], temperature=0.2))
    assert opener.last.json()["max_tokens"] == 4096 and opener.last.json()["temperature"] == 0.2
    # an explicit setting wins over the model-based default
    forced = OpenAICompatProvider(DEFAULT_BASE_URL, "gpt-5", api_key=KEY, opener=opener, reasoning_effort_param=False, max_tokens_param="max_tokens")
    assert not forced.reasoning_effort_param and forced.max_tokens_param == "max_tokens"


def test_unsupported_parameter_400_names_the_switches():
    body = json.dumps({"error": {"message": "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.", "type": "invalid_request_error"}})
    opener = FakeOpener(HttpResponse(400, body.encode()), HttpResponse(400, json.dumps({"error": {"message": "model not found"}}).encode()))
    p, _ = make(opener)
    with pytest.raises(ProviderError) as ei:
        p.complete(CompletionRequest([Message.user("q")]))
    e = ei.value
    assert e.status == 400 and "reasoning_effort_param" in e.error.remediation and "max_completion_tokens" in e.error.remediation
    assert "max_tokens and temperature" in e.error.remediation
    with pytest.raises(ProviderError) as ei2:  # a 400 about something else keeps the plain remediation
        p.complete(CompletionRequest([Message.user("q")]))
    assert "reasoning_effort_param" not in ei2.value.error.remediation
