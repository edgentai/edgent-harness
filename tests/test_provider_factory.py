"""Provider selection from settings, environment dicts and a fake PATH lookup; no network, no real CLIs."""
from __future__ import annotations

import json
import sys
import types
from typing import Any

import pytest

from edgent.adapters.providers.factory import (
    NO_PROVIDER_WARNING,
    ProviderSpec,
    describe_options,
    make_provider,
    resolve_spec,
)
from edgent.adapters.providers.gemini import GeminiProvider
from edgent.adapters.providers.http import HttpRequest, HttpResponse
from edgent.adapters.providers.mock import MockProvider
from edgent.adapters.providers.openai_compat import OpenAICompatProvider
from edgent.domain.errors import ConfigurationError
from edgent.domain.types import CompletionRequest, Message

NOTHING = lambda name: None  # noqa: E731 - a fake PATH lookup that finds nothing


def on_path(*names: str):
    return lambda name: f"/usr/local/bin/{name}" if name in names else None


class FakeBridge:
    name = "bridge"

    def __init__(self, model: str | None = None, **kw: Any) -> None:
        self.model = model
        self.kw = kw

    def complete(self, request: CompletionRequest) -> Any:  # pragma: no cover - never called
        raise NotImplementedError


def install_fake_bridge(monkeypatch: pytest.MonkeyPatch, module: str, class_name: str) -> type:
    cls = type(class_name, (FakeBridge,), {})
    mod = types.ModuleType(module)
    setattr(mod, class_name, cls)
    monkeypatch.setitem(sys.modules, module, mod)
    return cls


# ----------------------------------------------------------------------------- selection order
def test_fallback_is_mock_with_warning():
    p = make_provider(settings={}, environ={}, which=NOTHING)
    assert isinstance(p, MockProvider)
    assert p.meta["warning"] == NO_PROVIDER_WARNING and p.meta["kind"] == "mock" and p.meta["source"] == "fallback"
    assert p.complete(CompletionRequest([Message.user("hi")])).text.startswith("[mock]")


def test_gemini_from_environment():
    env = {"GEMINI_API_KEY": "gem-key-123456", "EDGENT_GEMINI_MODEL": "gemini-2.5-flash"}
    p = make_provider(environ=env, which=on_path("agy", "claude"))
    assert isinstance(p, GeminiProvider) and p.model == "gemini-2.5-flash" and p.api_key == "gem-key-123456"
    assert p.meta == {"kind": "gemini", "source": "env:GEMINI_API_KEY", "warnings": []}
    spec = resolve_spec(environ={"EDGENT_GEMINI_API_KEY": "k-123456"}, which=NOTHING)
    assert spec.kind == "gemini" and spec.source == "env:EDGENT_GEMINI_API_KEY" and spec.model is None
    assert "k-123456" not in repr(spec) and "k-123456" not in json.dumps(spec.describe()) and spec.describe()["api_key"] == "[set]"


def test_openai_from_environment_and_defaults():
    env = {"OPENAI_BASE_URL": "http://localhost:11434/v1/", "OPENAI_MODEL": "llama3", "OPENAI_API_KEY": "sk-123456"}
    p = make_provider(environ=env, which=on_path("agy"))
    assert isinstance(p, OpenAICompatProvider) and p.base_url == "http://localhost:11434/v1" and p.model == "llama3" and p.api_key == "sk-123456"
    assert p.meta["source"] == "env:OPENAI_BASE_URL"
    hosted = resolve_spec(environ={"OPENAI_API_KEY": "sk-123456", "EDGENT_MODEL": "gpt-x"}, which=NOTHING)
    assert hosted.kind == "openai" and hosted.base_url == "https://api.openai.com/v1" and hosted.model == "gpt-x"
    with pytest.raises(ConfigurationError, match="model id"):
        make_provider(environ={"OPENAI_BASE_URL": "http://localhost:8000/v1"}, which=NOTHING)
    with pytest.raises(ConfigurationError, match="base URL"):
        make_provider(environ={}, which=NOTHING, kind="openai")


def test_gemini_beats_openai_and_clis_beat_nothing(monkeypatch: pytest.MonkeyPatch):
    env = {"GEMINI_API_KEY": "g-123456", "OPENAI_BASE_URL": "http://x/v1", "OPENAI_MODEL": "m"}
    assert resolve_spec(environ=env, which=on_path("agy", "claude")).kind == "gemini"
    assert resolve_spec(environ={"OPENAI_BASE_URL": "http://x/v1", "OPENAI_MODEL": "m"}, which=on_path("agy")).kind == "openai"
    agy_cls = install_fake_bridge(monkeypatch, "edgent.adapters.providers.agy_bridge", "AgyProvider")
    claude_cls = install_fake_bridge(monkeypatch, "edgent.adapters.providers.claude_bridge", "ClaudeCliProvider")
    p = make_provider(environ={"EDGENT_MODEL": "gemini-3-pro-high"}, which=on_path("agy", "claude"))
    assert isinstance(p, agy_cls) and p.model == "gemini-3-pro-high" and p.meta["source"] == "path:agy"
    q = make_provider(environ={}, which=on_path("claude"))
    assert isinstance(q, claude_cls) and q.model is None and q.meta["source"] == "path:claude"


def test_missing_bridge_module_is_a_configuration_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "edgent.adapters.providers.agy_bridge", None)
    with pytest.raises(ConfigurationError, match="agy_bridge"):
        make_provider(environ={}, which=on_path("agy"))


# ----------------------------------------------------------------------------- explicit choices
def test_explicit_kind_wins_over_environment():
    env = {"GEMINI_API_KEY": "g-123456"}
    assert isinstance(make_provider(environ=env, which=NOTHING, kind="mock"), MockProvider)
    assert isinstance(make_provider(environ=env, which="mock"), MockProvider)  # a string `which` is read as the kind
    assert isinstance(make_provider(settings={"provider": {"kind": "mock"}}, environ=env, which=NOTHING), MockProvider)
    assert resolve_spec(environ={**env, "EDGENT_PROVIDER": "mock"}, which=NOTHING).source == "EDGENT_PROVIDER"
    explicit = make_provider(settings={"provider": {"kind": "mock"}}, environ=env, which=NOTHING)
    assert "warning" not in explicit.meta and explicit.meta["source"] == "settings.provider.kind"
    with pytest.raises(ConfigurationError, match="unknown provider kind"):
        make_provider(environ={}, which=NOTHING, kind="bedrock")
    with pytest.raises(ConfigurationError, match="settings.provider must be an object"):
        make_provider(settings={"provider": "gemini"}, environ={}, which=NOTHING)


def test_settings_model_base_url_and_options_reach_the_constructor():
    settings = {"provider": {"kind": "openai", "model": "qwen3", "base_url": "http://srv:8000/v1", "api_key_env": "SRV_KEY", "options": {"thinking_switch": True, "timeout_s": 7}}}
    p = make_provider(settings=settings, environ={"SRV_KEY": "srv-secret-1", "OPENAI_MODEL": "ignored"}, which=NOTHING)
    assert isinstance(p, OpenAICompatProvider) and p.model == "qwen3" and p.base_url == "http://srv:8000/v1"
    assert p.api_key == "srv-secret-1" and p.thinking_switch and p.timeout_s == 7 and p.meta["warnings"] == []
    with pytest.raises(ConfigurationError, match="SRV_KEY"):
        make_provider(settings=settings, environ={}, which=NOTHING)
    bad = {"provider": {"kind": "openai", "model": "m", "base_url": "http://srv/v1", "options": {"bogus": 1}}}
    with pytest.raises(ConfigurationError, match="bogus"):
        make_provider(settings=bad, environ={}, which=NOTHING)


def test_plain_api_key_in_settings_warns_and_never_leaks():
    settings = {"provider": {"kind": "gemini", "api_key": "plain-key-123456", "model": "gemini-2.5-flash", "options": {"thinking_budgets": {"low": 0, "medium": 1, "high": 2}}}}
    p = make_provider(settings=settings, environ={}, which=NOTHING)
    assert isinstance(p, GeminiProvider) and p.api_key == "plain-key-123456" and p.thinking_budgets == {"low": 0, "medium": 1, "high": 2}
    assert len(p.meta["warnings"]) == 1 and "plain text" in p.meta["warnings"][0] and "plain-key" not in p.meta["warnings"][0]
    with pytest.raises(ConfigurationError, match="API key"):
        make_provider(environ={}, which=NOTHING, kind="gemini")


def test_opener_is_injected_into_the_built_provider():
    seen: list[HttpRequest] = []

    def opener(req: HttpRequest) -> HttpResponse:
        seen.append(req)
        return HttpResponse(200, json.dumps({"candidates": [{"content": {"parts": [{"text": "via factory"}]}, "finishReason": "STOP"}]}).encode())

    p = make_provider(environ={"GEMINI_API_KEY": "g-key-123456"}, which=NOTHING, opener=opener)
    assert p.complete(CompletionRequest([Message.user("hi")])).text == "via factory" and seen[0].headers["x-goog-api-key"] == "g-key-123456"


def test_describe_options_and_spec_defaults():
    text = describe_options()
    for kind in ("gemini", "openai", "agy", "claude", "mock"):
        assert kind in text
    assert "api_key_env" in text and "GEMINI_API_KEY" in text and "OPENAI_BASE_URL" in text
    spec = ProviderSpec(kind="mock")
    assert spec.describe()["api_key"] == "[not set]" and spec.options == {} and spec.warnings == []


# ----------------------------------------------------------------------------- review additions
def test_api_key_inside_options_gets_the_plain_text_warning_and_api_key_env_wins():
    settings = {"provider": {"kind": "openai", "model": "m", "base_url": "http://srv/v1", "options": {"api_key": "opt-secret-123456"}}}
    p = make_provider(settings=settings, environ={}, which=NOTHING)
    assert isinstance(p, OpenAICompatProvider) and p.api_key == "opt-secret-123456"
    assert len(p.meta["warnings"]) == 1 and "plain text" in p.meta["warnings"][0] and "opt-secret" not in json.dumps(p.meta)
    settings["provider"]["api_key_env"] = "SRV_KEY"
    q = make_provider(settings=settings, environ={"SRV_KEY": "env-secret-123456"}, which=NOTHING)
    assert q.api_key == "env-secret-123456" and q.meta["warnings"] == []


def test_openai_reasoning_defaults_are_decided_in_the_spec_and_overridable():
    hosted = resolve_spec(environ={"OPENAI_API_KEY": "sk-123456", "EDGENT_OPENAI_MODEL": "gpt-5"}, which=NOTHING)
    assert hosted.options == {"reasoning_effort_param": True, "max_tokens_param": "max_completion_tokens"}
    assert sorted(hosted.describe()["options"]) == ["max_tokens_param", "reasoning_effort_param"]
    p = make_provider(environ={"OPENAI_API_KEY": "sk-123456", "EDGENT_OPENAI_MODEL": "gpt-5"}, which=NOTHING)
    assert isinstance(p, OpenAICompatProvider) and p.reasoning_effort_param and p.max_tokens_param == "max_completion_tokens"
    plain_hosted = resolve_spec(environ={"OPENAI_API_KEY": "sk-123456", "OPENAI_MODEL": "gpt-4o"}, which=NOTHING)
    assert plain_hosted.options == {"max_tokens_param": "max_completion_tokens"}
    local = resolve_spec(environ={"OPENAI_BASE_URL": "http://localhost:8000/v1", "OPENAI_MODEL": "qwen3"}, which=NOTHING)
    assert local.options == {}
    settings = {"provider": {"kind": "openai", "model": "gpt-5", "options": {"reasoning_effort_param": False}}}
    forced = resolve_spec(settings=settings, environ={"OPENAI_API_KEY": "sk-123456"}, which=NOTHING)
    assert forced.options == {"reasoning_effort_param": False, "max_tokens_param": "max_completion_tokens"}
    assert not make_provider(settings=settings, environ={"OPENAI_API_KEY": "sk-123456"}, which=NOTHING).reasoning_effort_param
    assert "max_completion_tokens" in describe_options() and "gpt-5" in describe_options()
