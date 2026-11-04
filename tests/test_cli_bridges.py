"""Offline tests for the subscription CLI bridges: every CLI call goes through a fake runner.

The real ``agy`` and ``claude`` binaries are never started. ``FakeCli`` answers ``--version``, ``--help``
and ``models`` from canned text and serves scripted completions, recording every argv, stdin and env.
"""
from __future__ import annotations

import base64
import errno
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from edgent.adapters.providers import agy_bridge, claude_bridge
from edgent.adapters.providers.agy_bridge import AgyProvider, parse_models_output, split_effort_suffix
from edgent.adapters.providers.base import ProviderError
from edgent.adapters.providers.claude_bridge import ClaudeCliProvider
from edgent.adapters.providers.cli_bridge import (
    EnvelopeError,
    parse_help_flags,
    parse_json_envelope,
    redact_argv,
    subprocess_runner,
)
from edgent.domain.errors import ConfigurationError
from edgent.domain.types import CompletionRequest, ContentPart, Message, ToolCall

TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
PNG = b"\x89PNG\r\n\x1a\n" + b"edgent-test-image"


@pytest.fixture(autouse=True)
def _clean_bridge_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bridges read these at construction; a developer's shell must not leak into the assertions."""
    for name in ("EDGENT_AGY_MODEL", "EDGENT_AGY_MODELS", "EDGENT_AGY_MAX_PROMPT_BYTES", "EDGENT_CLAUDE_MODEL"):
        monkeypatch.delenv(name, raising=False)

AGY_MODELS_TABLE = """MODEL                     PROVIDER   NOTES
gemini-3.1-pro-low        google     fast
gemini-3.1-pro-medium     google
* gemini-3.1-pro-high     google     (default)
claude-sonnet-4.5         anthropic
"""

CLAUDE_HELP_MODERN = """Usage: claude [options] [command] [prompt]
Options:
  --allowedTools, --allowed-tools <tools...>  Comma or space-separated list of tool names to allow
  --effort <level>                      Effort level (low, medium, high)
  --model <model>                       Model for the current session
  --output-format <format>              "text", "json", or "stream-json"
  -p, --print                           Print response and exit
  --system-prompt <prompt>              System prompt to use for the session
  --system-prompt-file <file>           Read the system prompt from a file
  --no-session-persistence              Disable session persistence
  --disable-slash-commands              Disable all skills and commands
  --strict-mcp-config                   Only use MCP servers from --mcp-config, ignoring all other MCP configurations
  --tools <tools...>                    Specify the list of available tools. Use "" to disable all tools
  --bare                                Minimal mode: OAuth and keychain are never read
  --dangerously-skip-permissions        Bypass all permission checks
"""
# claude 2.1.x: --system-prompt-file is only listed as the "[-file]" shorthand inside the --bare description
CLAUDE_HELP_SHORTHAND = """Usage: claude [options] [command] [prompt]
Options:
  --effort <level>                      Effort level (low, medium, high)
  --model <model>                       Model for the current session
  --output-format <format>              "text", "json", or "stream-json"
  -p, --print                           Print response and exit
  --bare                                Minimal mode: skip hooks (those defined
                                        in settings and by installed plugins;
                                        features built into Claude Code are
                                        unaffected). Explicitly provide context
                                        via: --system-prompt[-file],
                                        --append-system-prompt[-file], --add-dir
                                        (CLAUDE.md dirs), --mcp-config,
                                        --settings, --agents, --plugin-dir.
  --system-prompt <prompt>              System prompt to use for the session
  --system-prompt-snapshot <on|off>     Record the system prompt once per
                                        session
  --tools <tools...>                    Specify the list of available tools. Use "" to disable all tools
"""
CLAUDE_HELP_OLD = """Usage: claude [options] [command] [prompt]
Options:
  --allowedTools <tools...>   Comma/space-separated list of tool names to allow
  --model <model>             Model for the current session
  --output-format <format>    Output format
  -p, --print                 Print response and exit
  --system-prompt <prompt>    System prompt to use for the session
"""
CLAUDE_HELP_BARE = """Usage: claude [options]
Options:
  --model <model>           Model
  -p, --print               Print response and exit
  --system-prompt <prompt>  System prompt
"""


@dataclass
class Completed:
    returncode: int = 0
    stdout: bytes | str = b""
    stderr: bytes | str = b""


def agy_envelope(response: Any, status: str = "SUCCESS", **over: Any) -> bytes:
    env: dict[str, Any] = {
        "conversation_id": "conv-42",
        "status": status,
        "response": response,
        "duration_seconds": 1.5,
        "num_turns": 1,
        "usage": {"input_tokens": 120, "output_tokens": 30, "thinking_tokens": 10, "cache_read_tokens": 40, "total_tokens": 160},
    }
    env.update(over)
    return json.dumps(env).encode()


def claude_envelope(result: Any, is_error: bool = False, **over: Any) -> bytes:
    env: dict[str, Any] = {
        "type": "result",
        "subtype": "error_during_execution" if is_error else "success",
        "is_error": is_error,
        "result": result,
        "session_id": "sess-7",
        "num_turns": 1,
        "duration_ms": 1200,
        "total_cost_usd": 0.0123,
        "usage": {"input_tokens": 100, "output_tokens": 25, "cache_read_input_tokens": 60, "cache_creation_input_tokens": 20},
    }
    env.update(over)
    return json.dumps(env).encode()


class FakeCli:
    """Routes argv to canned outputs and records every call; never touches a real CLI.

    ``completions`` items are ``Completed`` objects, exceptions to raise, or callables ``(argv, stdin, env) -> Completed``.
    """

    def __init__(
        self,
        *,
        version: str = "agy 9.9.9",
        help_text: str = "Usage: agy [options]\n  --print <prompt>\n  --model <id>\n",
        models_text: str = AGY_MODELS_TABLE,
        completions: list[Any] | None = None,
        default: Completed | None = None,
        missing: bool = False,
    ) -> None:
        self.version = version
        self.help_text = help_text
        self.models_text = models_text
        self.completions = list(completions or [])
        self.default = default or Completed(0, agy_envelope("ok"))
        self.missing = missing
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def __call__(self, argv: list[str], stdin: bytes | None, timeout_s: float, env: dict[str, str]) -> Completed:
        with self._lock:
            self.calls.append({"argv": list(argv), "stdin": stdin, "timeout_s": timeout_s, "env": dict(env)})
            if self.missing:
                raise FileNotFoundError(2, "No such file or directory", argv[0])
            sub = argv[1] if len(argv) > 1 else ""
            if sub == "--version":
                return Completed(0, self.version)
            if sub == "--help":
                return Completed(0, self.help_text)
            if sub == "models":
                return Completed(0, self.models_text)
            item = self.completions.pop(0) if self.completions else self.default
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return item(argv, stdin, env)
        return item

    def completion_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if len(c["argv"]) > 1 and c["argv"][1] in ("--print", "-p")]


def agy_prompt(call: dict[str, Any]) -> str:
    argv = call["argv"]
    return argv[argv.index("--print") + 1]


def at_refs(prompt: str) -> list[str]:
    return [ln[1:] for ln in prompt.splitlines() if ln.startswith("@")]


# =========================================================================== agy
def test_agy_argv_shape_and_markers():
    fake = FakeCli(completions=[Completed(0, agy_envelope("fine"))])
    p = AgyProvider(runner=fake, known_models=list(agy_bridge.FALLBACK_MODELS))
    req = CompletionRequest([Message.system("Be terse."), Message.user("Say hi.")], tools=TOOLS, effort="low")
    reply = p.complete(req)
    assert reply.text == "fine" and reply.model == "gemini-3.1-pro-low" and reply.finish_reason == "stop" and reply.request_id == "conv-42"
    call = fake.completion_calls()[0]
    argv = call["argv"]
    assert argv[0] == "agy" and argv[1] == "--print" and call["stdin"] is None
    assert argv[3:] == ["--model", "gemini-3.1-pro-low", "--output-format", "json", "--disable-slash-commands", "--print-timeout", "1800s"]
    assert "--effort" not in argv and "--dangerously-skip-permissions" not in argv
    prompt = argv[2]
    assert "SYSTEM INSTRUCTIONS" in prompt and "<<<SYSTEM" in prompt and "Be terse." in prompt and "SYSTEM>>>" in prompt
    assert "<<<REQUEST" in prompt and "Say hi." in prompt and "REQUEST>>>" in prompt
    assert "Tool calls (text protocol)" in prompt and "read_file" in prompt
    assert prompt.endswith("Reply with the requested output only.") and "ATTACHED IMAGES" not in prompt
    assert call["env"]["AGY_ENTRYPOINT"] == "edgent" and call["timeout_s"] == pytest.approx(1830.0)
    assert reply.raw["argv"][2].startswith("<") and reply.raw["argv"][2].endswith("bytes>")  # the prompt is elided in traces
    assert reply.raw["cli"] == "agy" and reply.raw["returncode"] == 0


def test_agy_folds_conversation_into_one_prompt():
    fake = FakeCli()
    p = AgyProvider(runner=fake, known_models=[], timeout_s=120)
    msgs = [
        Message.system("S"),
        Message.user("first"),
        Message.assistant("", [ToolCall("c1", "read_file", {"path": "a"})]),
        Message.tool("c1", "read_file", "CONTENT"),
        Message.user("now answer"),
    ]
    p.complete(CompletionRequest(msgs, tools=TOOLS))
    call = fake.completion_calls()[0]
    prompt = agy_prompt(call)
    assert "CONVERSATION SO FAR" in prompt and "TOOL RESULT read_file" in prompt and "CONTENT" in prompt and "now answer" in prompt
    assert prompt.count("Reply with the requested output only.") == 1
    assert call["argv"][-2:] == ["--print-timeout", "120s"] and call["timeout_s"] == pytest.approx(150.0)


def test_agy_ceiling_refuses_never_truncates(monkeypatch):
    monkeypatch.setenv("EDGENT_AGY_MAX_PROMPT_BYTES", "4096")
    fake = FakeCli()
    p = AgyProvider(runner=fake, known_models=[])
    assert p.caps.max_prompt_bytes == 4096 and p.prompt_limit().source == "setting"
    with pytest.raises(ProviderError) as ei:
        p.complete(CompletionRequest([Message.user("x" * 5000)]))
    err = ei.value.error
    assert err.category.value == "truncated" and not ei.value.retryable and err.code == "limit:agy prompt size"
    assert "EDGENT_AGY_MAX_PROMPT_BYTES" in err.remediation and "Send less" in err.remediation
    assert err.details["observed"] > 4096 and err.details["value"] == 4096
    assert fake.completion_calls() == []  # refused before any process started
    # the final check covers the bytes the bridge adds after prepare (the trailer, the image block)
    body = "y" * (4096 - 20)
    with pytest.raises(ProviderError) as ei2:
        p.complete(CompletionRequest([Message.user(body)]))
    assert ei2.value.error.category.value == "truncated" and fake.completion_calls() == []
    # just under the ceiling goes through
    p.complete(CompletionRequest([Message.user("z" * 3000)]))
    assert len(fake.completion_calls()) == 1


def test_agy_ceiling_setting_is_validated(monkeypatch):
    monkeypatch.setenv("EDGENT_AGY_MAX_PROMPT_BYTES", "100")
    with pytest.raises(ConfigurationError):
        AgyProvider(runner=FakeCli())
    monkeypatch.setenv("EDGENT_AGY_MAX_PROMPT_BYTES", str(2 << 20))
    with pytest.raises(ConfigurationError):
        AgyProvider(runner=FakeCli())
    monkeypatch.setenv("EDGENT_AGY_MAX_PROMPT_BYTES", "lots")
    with pytest.raises(ConfigurationError):
        AgyProvider(runner=FakeCli())
    monkeypatch.delenv("EDGENT_AGY_MAX_PROMPT_BYTES")
    p = AgyProvider(runner=FakeCli())
    assert p.caps.max_prompt_bytes == 128 * 1024 and p.prompt_limit().source == "default"
    assert p.caps.effort_control == "model_suffix" and not p.caps.native_tools and not p.caps.system_channel and p.caps.images
    assert AgyProvider.caps.images and AgyProvider.caps.max_prompt_bytes == 128 * 1024 and not AgyProvider.caps.system_channel  # the class is honest too


def test_agy_effort_resolves_to_listed_sibling():
    known = ["gemini-3.1-pro-low", "gemini-3.1-pro-high", "claude-sonnet-4.5"]
    p = AgyProvider(runner=FakeCli(), known_models=known)
    assert p.resolve_model("gemini-3.1-pro-high", "low") == "gemini-3.1-pro-low"
    assert p.resolve_model("gemini-3.1-pro-high", "medium") == "gemini-3.1-pro-high"  # no medium sibling listed: unchanged
    assert p.resolve_model("gemini-3.1-pro-high", "high") == "gemini-3.1-pro-high"
    assert p.resolve_model("claude-sonnet-4.5", "high") == "claude-sonnet-4.5"  # no suffix family: unchanged
    assert p.resolve_model("gemini-3.1-pro-low", "high", known=["gemini-3.1-pro-high"]) == "gemini-3.1-pro-high"
    assert p.resolve_model("gemini-3.1-pro-low", "") == "gemini-3.1-pro-low"
    assert split_effort_suffix("gemini-3.1-pro-medium") == ("gemini-3.1-pro", "medium")
    assert split_effort_suffix("gemini-3-flash") == ("gemini-3-flash", None)
    # an unknown effort value falls back to effort_default for the sibling lookup
    fake = FakeCli(completions=[Completed(0, agy_envelope("ok"))])
    p2 = AgyProvider("gemini-3.1-pro-high", runner=fake, known_models=known, effort_default="low")
    p2.complete(CompletionRequest([Message.user("q")], effort="extreme"))  # type: ignore[arg-type]
    assert fake.completion_calls()[0]["argv"][4] == "gemini-3.1-pro-low"


def test_agy_list_models_parses_cli_and_falls_back(monkeypatch):
    fake = FakeCli(models_text=AGY_MODELS_TABLE)
    p = AgyProvider(runner=fake)
    assert p.list_models() == ["gemini-3.1-pro-low", "gemini-3.1-pro-medium", "gemini-3.1-pro-high", "claude-sonnet-4.5"]
    assert p.models_source == "cli"
    p.list_models()
    assert len([c for c in fake.calls if c["argv"][1:] == ["models"]]) == 1  # cached
    assert parse_models_output('[{"id": "m-low", "name": "M"}, "m-high"]') == ["m-low", "m-high"]
    assert parse_models_output('{"models": [{"name": "n-medium"}]}') == ["n-medium"]
    assert parse_models_output("| Model | Notes |\n|---|---|\n| gemini-3-pro-low | x |\n") == ["gemini-3-pro-low"]
    assert parse_models_output("") == []
    broken = AgyProvider(runner=FakeCli(missing=True))
    assert broken.list_models() == list(agy_bridge.FALLBACK_MODELS) and broken.models_source == "fallback"
    empty = AgyProvider(runner=FakeCli(models_text="MODEL\n"))
    assert empty.list_models() == list(agy_bridge.FALLBACK_MODELS)
    monkeypatch.setenv("EDGENT_AGY_MODELS", "a-low, a-high,")
    from_env = AgyProvider(runner=FakeCli(missing=True))
    assert from_env.list_models() == ["a-low", "a-high"] and from_env.models_source == "environment"


def test_agy_envelope_usage_and_pricing():
    fake = FakeCli(completions=[Completed(0, agy_envelope("answer", num_turns=2))])
    p = AgyProvider(runner=fake, known_models=[], prices={"input": 2.0, "output": 12.0, "cached": 0.5})
    r = p.complete(CompletionRequest([Message.user("q")]))
    assert r.text == "answer" and r.usage.prompt_tokens == 120 and r.usage.completion_tokens == 30
    assert r.usage.reasoning_tokens == 10 and r.usage.cached_tokens == 40 and r.usage.total_tokens == 150
    assert r.usage.cost_usd == pytest.approx((80 * 2.0 + 40 * 0.5 + 30 * 12.0 + 10 * 12.0) / 1_000_000)
    assert r.raw["envelope"]["num_turns"] == 2 and r.raw["envelope"]["status"] == "SUCCESS" and r.raw["envelope"]["total_tokens"] == 160
    assert r.latency_s >= 0 and "stderr_tail" not in r.raw
    p2 = AgyProvider(runner=FakeCli(completions=[Completed(0, agy_envelope({"k": 1}))]), known_models=[])
    r2 = p2.complete(CompletionRequest([Message.user("q")]))
    assert r2.usage.cost_usd == 0.0 and json.loads(r2.text) == {"k": 1}  # no price table: a subscription costs nothing per call


def test_agy_envelope_tolerates_noise_and_rejects_garbage():
    noisy = b"Loaded cached credentials.\n" + agy_envelope("clean")
    fake = FakeCli(completions=[Completed(0, noisy), Completed(1, b"not json at all", b"boom: something broke"), Completed(0, b"", b"")])
    p = AgyProvider(runner=fake, known_models=[])
    req = CompletionRequest([Message.user("q")])
    assert p.complete(req).text == "clean"
    with pytest.raises(ProviderError) as ei:
        p.complete(req)
    assert ei.value.error.category.value == "execution_failure" and "something broke" in str(ei.value) and ei.value.error.details["returncode"] == 1
    with pytest.raises(ProviderError) as ei2:
        p.complete(req)
    assert "printed nothing" in str(ei2.value)


def test_agy_status_and_stderr_classification():
    fake = FakeCli(
        completions=[
            Completed(0, agy_envelope("", status="TIMEOUT")),
            Completed(0, agy_envelope("", status="ERROR")),
            Completed(1, b"", b"Error: you are not logged in. Run agy login first."),
            Completed(1, b"", b"Error: 429 Too Many Requests, quota exceeded"),
            Completed(2, b"", b"error: unknown option '--print-timeout'"),
            Completed(0, agy_envelope("fine", status="PARTIAL"), b"some warning"),
        ]
    )
    p = AgyProvider(runner=fake, known_models=[])
    req = CompletionRequest([Message.user("q")])
    with pytest.raises(ProviderError) as e1:
        p.complete(req)
    assert e1.value.retryable and e1.value.error.category.value == "timeout"
    with pytest.raises(ProviderError) as e2:
        p.complete(req)
    assert e2.value.error.category.value == "execution_failure" and "ERROR" in str(e2.value)
    with pytest.raises(ProviderError) as e3:
        p.complete(req)
    assert e3.value.error.category.value == "permission_denied" and "login" in e3.value.error.remediation
    with pytest.raises(ProviderError) as e4:
        p.complete(req)
    assert e4.value.retryable and e4.value.error.category.value == "resource_constraint"
    with pytest.raises(ProviderError) as e5:
        p.complete(req)
    assert e5.value.error.category.value == "configuration_invalid" and "edgent doctor" in e5.value.error.remediation
    r = p.complete(req)  # an unknown but not failing status is accepted and kept in raw
    assert r.text == "fine" and r.raw["envelope"]["status"] == "PARTIAL" and r.raw["stderr_tail"] == "some warning"


def test_agy_denial_detected_with_settings_line():
    stderr = b'warning: a tool required the "read_file" permission and was denied in headless mode\n'
    fake = FakeCli(completions=[Completed(0, agy_envelope(""), stderr)])
    p = AgyProvider(runner=fake, known_models=[])
    with pytest.raises(ProviderError) as ei:
        p.complete(CompletionRequest([Message.user("q")]))
    err = ei.value.error
    assert err.category.value == "permission_denied" and err.code == "cli_permission_denied" and not ei.value.retryable
    assert '"permissions": {"allow": ["read_file(*)"]}' in err.remediation and "~/.gemini/antigravity-cli/settings.json" in err.remediation
    assert err.details["tool"] == "read_file" and err.details["settings_line"] == '"permissions": {"allow": ["read_file(*)"]}'
    # a denial with no envelope at all is still reported as a denial, not as garbage output
    p2 = AgyProvider(runner=FakeCli(completions=[Completed(1, b"", stderr)]), known_models=[])
    with pytest.raises(ProviderError) as ei2:
        p2.complete(CompletionRequest([Message.user("q")]))
    assert ei2.value.error.category.value == "permission_denied"
    # a non-empty answer with a denial on stderr is returned with a warning when no image depended on the tool
    p3 = AgyProvider(runner=FakeCli(completions=[Completed(0, agy_envelope("partial"), stderr)]), known_models=[])
    r = p3.complete(CompletionRequest([Message.user("q")]))
    assert r.text == "partial" and any("read_file" in w for w in r.raw["warnings"])


def test_agy_images_are_staged_referenced_and_cleaned(tmp_path):
    img = tmp_path / "shot with space.png"
    img.write_bytes(PNG)
    seen: dict[str, Any] = {}

    def completion(argv: list[str], stdin: bytes | None, env: dict[str, str]) -> Completed:
        prompt = argv[argv.index("--print") + 1]
        refs = at_refs(prompt)
        seen.update(refs=refs, exists=[os.path.isfile(r) for r in refs], data=Path(refs[0]).read_bytes(), prompt=prompt)
        return Completed(0, agy_envelope("a cat"))

    p = AgyProvider(runner=FakeCli(completions=[completion]), known_models=[])
    req = CompletionRequest([Message.system("S"), Message.user("What is this?", [ContentPart.of_media("image", str(img))])])
    assert p.complete(req).text == "a cat"
    assert len(seen["refs"]) == 1 and seen["exists"] == [True] and seen["data"] == PNG
    ref = seen["refs"][0]
    assert os.path.isabs(ref) and ref.endswith(".png") and " " not in ref and "edgent-agy-" in ref
    prompt = seen["prompt"]
    assert "What is this?" in prompt and prompt.index("ATTACHED IMAGES") < prompt.index("@" + ref)
    assert prompt.rstrip().endswith("Reply with the requested output only.") and prompt.index("@" + ref) < prompt.index("Reply with the requested")
    assert not os.path.exists(os.path.dirname(ref))  # the per-call directory is removed in finally


def test_agy_inline_image_data_and_missing_file(tmp_path):
    seen: dict[str, Any] = {}

    def completion(argv: list[str], stdin: bytes | None, env: dict[str, str]) -> Completed:
        refs = at_refs(argv[2])
        seen.update(refs=refs, data=Path(refs[0]).read_bytes())
        return Completed(0, agy_envelope("ok"))

    p = AgyProvider(runner=FakeCli(completions=[completion]), known_models=[])
    inline = ContentPart(type="image", data_b64=base64.b64encode(PNG).decode(), mime="image/jpeg", text="[image]")
    p.complete(CompletionRequest([Message.user("look", [inline])]))
    assert seen["refs"][0].endswith(".jpg") and seen["data"] == PNG
    with pytest.raises(ProviderError) as ei:
        p.complete(CompletionRequest([Message.user("look", [ContentPart.of_media("image", str(tmp_path / "nope.png"))])]))
    assert ei.value.error.category.value == "not_found"
    with pytest.raises(ProviderError) as ei2:
        p.complete(CompletionRequest([Message.user("look", [ContentPart(type="image", text="[image]")])]))
    assert ei2.value.error.category.value == "invalid_input"


def test_agy_image_denial_raises_even_with_text_and_cleans_up(tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"jpegdata")
    seen: dict[str, Any] = {}

    def completion(argv: list[str], stdin: bytes | None, env: dict[str, str]) -> Completed:
        seen["dir"] = os.path.dirname(at_refs(argv[2])[0])
        return Completed(0, agy_envelope("I cannot see the image."), b'a tool required the "read_file" permission')

    p = AgyProvider(runner=FakeCli(completions=[completion]), known_models=[])
    with pytest.raises(ProviderError) as ei:
        p.complete(CompletionRequest([Message.user("look", [ContentPart.of_media("image", str(img))])]))
    assert ei.value.error.category.value == "permission_denied" and "read_file(*)" in ei.value.error.remediation
    assert not os.path.exists(seen["dir"])


def test_agy_text_protocol_tool_call_parsed():
    text = 'I will read.\n```tool\n{"name": "read_file", "arguments": {"path": "sources/INDEX.md"}}\n```'
    p = AgyProvider(runner=FakeCli(completions=[Completed(0, agy_envelope(text))]), known_models=[])
    r = p.complete(CompletionRequest([Message.system("S"), Message.user("go")], tools=TOOLS))
    assert r.finish_reason == "tool_calls" and r.text == "I will read."
    assert r.tool_calls[0].name == "read_file" and r.tool_calls[0].arguments == {"path": "sources/INDEX.md"}


def test_agy_never_passes_forbidden_flags(monkeypatch):
    fake = FakeCli()
    p = AgyProvider(runner=fake, known_models=[])
    original = p.build_argv

    def tampered(*a: Any, **kw: Any) -> tuple[list[str], bytes | None]:
        argv, stdin = original(*a, **kw)
        return [*argv, "--dangerously-skip-permissions"], stdin

    monkeypatch.setattr(p, "build_argv", tampered)
    with pytest.raises(ConfigurationError):
        p.complete(CompletionRequest([Message.user("q")]))
    assert fake.completion_calls() == []
    assert "--effort" in p.forbidden_flags


def test_agy_default_model_and_health(monkeypatch):
    monkeypatch.setenv("EDGENT_AGY_MODEL", "gemini-3-flash")
    assert AgyProvider(runner=FakeCli()).model == "gemini-3-flash"
    assert AgyProvider("explicit-id", runner=FakeCli()).model == "explicit-id"
    monkeypatch.delenv("EDGENT_AGY_MODEL")
    assert AgyProvider(runner=FakeCli()).model == agy_bridge.DEFAULT_MODEL
    fake = FakeCli(version="agy 1.4.0 (build 77)")
    p = AgyProvider(runner=fake, known_models=["x"])
    h = p.health()
    assert h["ok"] and h["version"] == "agy 1.4.0 (build 77)" and h["provider"] == "agy" and h["cli"] == "agy"
    assert h["prompt_limit_bytes"] == 128 * 1024 and h["models_source"] == "constructor" and h["concurrency"] == 3
    p.health()
    assert len([c for c in fake.calls if c["argv"][1] == "--help"]) == 1 and len([c for c in fake.calls if c["argv"][1] == "--version"]) == 1


def test_agy_denial_wins_over_failing_envelope_status():
    stderr = b'a tool required the "read_file" permission'
    fake = FakeCli(completions=[Completed(1, agy_envelope("", status="ERROR"), stderr)])
    p = AgyProvider(runner=fake, known_models=[])
    with pytest.raises(ProviderError) as ei:
        p.complete(CompletionRequest([Message.user("q")]))
    err = ei.value.error
    assert err.category.value == "permission_denied" and '"permissions": {"allow": ["read_file(*)"]}' in err.remediation
    # the same status without a denial on stderr is still the status failure
    p2 = AgyProvider(runner=FakeCli(completions=[Completed(1, agy_envelope("", status="ERROR"))]), known_models=[])
    with pytest.raises(ProviderError) as ei2:
        p2.complete(CompletionRequest([Message.user("q")]))
    assert ei2.value.error.category.value == "execution_failure"


def test_agy_print_timeout_is_whole_seconds_rounded_up():
    fake = FakeCli()
    AgyProvider(runner=fake, known_models=[], timeout_s=0.5).complete(CompletionRequest([Message.user("q")]))
    assert fake.completion_calls()[0]["argv"][-2:] == ["--print-timeout", "1s"]
    fake2 = FakeCli()
    AgyProvider(runner=fake2, known_models=[], timeout_s=90.2).complete(CompletionRequest([Message.user("q")]))
    assert fake2.completion_calls()[0]["argv"][-1] == "91s"


# =========================================================================== shared machinery
def test_image_attachment_failures_are_typed_and_cleaned(tmp_path, monkeypatch):
    made: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def recording_mkdtemp(**kw: Any) -> str:
        made.append(real_mkdtemp(**kw))
        return made[-1]

    monkeypatch.setattr(tempfile, "mkdtemp", recording_mkdtemp)
    p = AgyProvider(runner=FakeCli(), known_models=[])
    with pytest.raises(ProviderError) as ei:
        p.complete(CompletionRequest([Message.user("look", [ContentPart(type="image", data_b64="!!not base64!!", text="[image]")])]))
    assert ei.value.error.category.value == "invalid_input" and ei.value.error.code == "image_invalid"
    with pytest.raises(ProviderError) as ei2:
        p.complete(CompletionRequest([Message.user("look", [ContentPart.of_media("image", str(tmp_path))])]))  # a directory, not a file
    assert ei2.value.error.category.value == "invalid_input" and ei2.value.error.details["path"] == str(tmp_path)
    assert len(made) == 2 and not any(os.path.exists(d) for d in made)  # the per-call directory is removed on failure too


def test_nul_bytes_never_reach_argv_and_a_runner_value_error_is_typed():
    fake = FakeCli(completions=[Completed(0, agy_envelope("ok")), ValueError("embedded null byte")])
    p = AgyProvider(runner=fake, known_models=[])
    p.complete(CompletionRequest([Message.system("S\x00YS"), Message.user("a\x00b")]))
    prompt = agy_prompt(fake.completion_calls()[0])
    assert "\x00" not in prompt and "a\ufffdb" in prompt and "S\ufffdYS" in prompt
    with pytest.raises(ProviderError) as ei:
        p.complete(CompletionRequest([Message.user("q")]))
    assert ei.value.error.category.value == "invalid_input" and ei.value.error.code == "cli_argv_invalid"
    fake2 = FakeCli(help_text=CLAUDE_HELP_MODERN, completions=[Completed(0, claude_envelope("ok"))])
    ClaudeCliProvider(runner=fake2).complete(CompletionRequest([Message.system("S\x00YS"), Message.user("hi")]))
    argv = fake2.completion_calls()[0]["argv"]
    assert argv[argv.index("--system-prompt") + 1] == "S\ufffdYS"  # the system prompt travels on argv too


def test_empty_prompt_is_refused_before_any_process():
    fake = FakeCli(help_text=CLAUDE_HELP_MODERN)
    with pytest.raises(ProviderError) as ei:
        ClaudeCliProvider(runner=fake).complete(CompletionRequest([Message.system("only a system prompt")]))
    assert ei.value.error.category.value == "invalid_input" and ei.value.error.code == "empty_prompt" and fake.completion_calls() == []
    with pytest.raises(ProviderError) as ei2:
        AgyProvider(runner=FakeCli(), known_models=[]).complete(CompletionRequest([]))
    assert ei2.value.error.code == "empty_prompt"


def test_bad_cwd_is_a_configuration_error(tmp_path):
    with pytest.raises(ConfigurationError, match="cwd"):
        AgyProvider(runner=FakeCli(), cwd=tmp_path / "missing")
    assert AgyProvider(runner=FakeCli(), cwd=tmp_path).cwd == str(tmp_path)


def test_concurrency_semaphore_bounds_parallel_processes():
    active, peak = 0, 0
    lock = threading.Lock()

    def slow(argv: list[str], stdin: bytes | None, env: dict[str, str]) -> Completed:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return Completed(0, agy_envelope("ok"))

    p = AgyProvider(runner=FakeCli(completions=[slow] * 4), known_models=[], concurrency=1)
    req = CompletionRequest([Message.user("q")])
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: p.complete(req).text, range(4)))
    assert results == ["ok"] * 4 and peak == 1
    # with concurrency 2 two calls really overlap: both must be inside the runner for the barrier to open
    barrier = threading.Barrier(2)

    def paired(argv: list[str], stdin: bytes | None, env: dict[str, str]) -> Completed:
        barrier.wait(timeout=5)  # BrokenBarrierError if the semaphore let only one call in
        return Completed(0, agy_envelope("pair"))

    p2 = AgyProvider(runner=FakeCli(completions=[paired, paired]), known_models=[], concurrency=2)
    assert p2.caps.concurrency == 2
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda _: p2.complete(req).text, range(2))) == ["pair", "pair"]


def test_missing_cli_is_unavailable_with_install_hint():
    fake = FakeCli(missing=True)
    p = AgyProvider(runner=fake, known_models=[])
    with pytest.raises(ProviderError) as ei:
        p.complete(CompletionRequest([Message.user("q")]))
    err = ei.value.error
    assert err.category.value == "unavailable" and err.code == "cli_unavailable" and not ei.value.retryable
    assert "Install the Antigravity CLI" in err.remediation and "edgent doctor" in err.remediation
    h = p.health()
    assert h["ok"] is False and "not on PATH" in h["error"] and "Install" in h["remediation"]
    c = ClaudeCliProvider(runner=FakeCli(missing=True))
    with pytest.raises(ProviderError) as ei2:
        c.complete(CompletionRequest([Message.user("q")]))
    assert ei2.value.error.category.value == "unavailable" and "Install Claude Code" in ei2.value.error.remediation


def test_runner_timeout_and_argv_overflow_are_typed():
    fake = FakeCli(completions=[subprocess.TimeoutExpired(cmd=["agy"], timeout=90), OSError(errno.E2BIG, "Argument list too long"), PermissionError(13, "Permission denied")])
    p = AgyProvider(runner=fake, known_models=[], timeout_s=60)
    req = CompletionRequest([Message.user("q")])
    with pytest.raises(ProviderError) as ei:
        p.complete(req)
    assert ei.value.retryable and ei.value.error.category.value == "timeout" and "90 s" in str(ei.value) and "timeout_s" in ei.value.error.remediation
    with pytest.raises(ProviderError) as ei2:
        p.complete(req)
    assert ei2.value.error.category.value == "truncated" and "131072" in str(ei2.value)
    with pytest.raises(ProviderError) as ei3:
        p.complete(req)
    assert ei3.value.error.category.value == "unavailable"


def test_bridge_constructor_validation():
    with pytest.raises(ConfigurationError):
        AgyProvider(runner=FakeCli(), concurrency=0)
    with pytest.raises(ConfigurationError):
        AgyProvider(runner=FakeCli(), timeout_s=0)
    with pytest.raises(ConfigurationError):
        AgyProvider(runner=FakeCli(), effort_default="max")
    with pytest.raises(ConfigurationError):
        AgyProvider(runner=FakeCli(), cli="")


def test_envelope_and_argv_helpers():
    assert parse_json_envelope('{"type":"system"}\n{"type":"result","result":"x"}', ("result",))["result"] == "x"
    assert parse_json_envelope('[{"result": "a"}, {"result": "b"}]', ("result",))["result"] == "b"
    with pytest.raises(EnvelopeError):
        parse_json_envelope("", ("result",))
    with pytest.raises(EnvelopeError):
        parse_json_envelope('{"other": 1}', ("result",))
    assert redact_argv(["agy", "--print", "p" * 300, "--model", "m"]) == ["agy", "--print", "<300 bytes>", "--model", "m"]
    assert parse_help_flags("--allowedTools, --allowed-tools <tools...>\n  -p, --print") == {"--allowedTools", "--allowed-tools", "--print"}


def test_default_runner_feeds_stdin_and_kills_on_timeout(tmp_path):
    run = subprocess_runner(str(tmp_path))
    r = run([sys.executable, "-c", "import sys, os; print(sys.stdin.read().upper()); print(os.getcwd())"], b"hello", 30.0, dict(os.environ))
    assert r.returncode == 0 and r.stdout.decode().splitlines()[0] == "HELLO" and Path(r.stdout.decode().splitlines()[1]).resolve() == tmp_path.resolve()
    with pytest.raises(subprocess.TimeoutExpired):
        run([sys.executable, "-c", "import time; time.sleep(30)"], None, 0.5, dict(os.environ))


# =========================================================================== claude
def test_claude_probe_chooses_tool_flag():
    modern = ClaudeCliProvider(runner=FakeCli(version="2.1.0 (Claude Code)", help_text=CLAUDE_HELP_MODERN))
    assert modern.tool_restriction_flag() == "--tools" and modern.probe().has_flag("--effort") and modern.probe().version == "2.1.0 (Claude Code)"
    old = ClaudeCliProvider(runner=FakeCli(help_text=CLAUDE_HELP_OLD))
    assert old.tool_restriction_flag() == "--allowedTools" and not old.probe().has_flag("--effort")
    bare = ClaudeCliProvider(runner=FakeCli(help_text=CLAUDE_HELP_BARE))
    assert bare.tool_restriction_flag() is None and "warning" in bare.health() and bare.health()["tool_flag"] is None
    forced = ClaudeCliProvider(runner=FakeCli(help_text=CLAUDE_HELP_BARE), tool_restriction="--allowedTools")
    assert forced.tool_restriction_flag() == "--allowedTools"
    unrestricted = ClaudeCliProvider(runner=FakeCli(help_text=CLAUDE_HELP_MODERN), tool_restriction=None)
    assert unrestricted.tool_restriction_flag() is None
    missing = ClaudeCliProvider(runner=FakeCli(missing=True))
    assert missing.tool_restriction_flag() == "--tools"  # an unprobeable CLI gets the modern spelling
    with pytest.raises(ConfigurationError):
        ClaudeCliProvider(runner=FakeCli(), tool_restriction="tools")


def test_claude_stdin_prompt_and_system_prompt():
    fake = FakeCli(help_text=CLAUDE_HELP_MODERN, completions=[Completed(0, claude_envelope("hello"))])
    p = ClaudeCliProvider(runner=fake)
    req = CompletionRequest([Message.system("Be brief."), Message.user("Say hi.")], tools=TOOLS, effort="high")
    r = p.complete(req)
    assert r.text == "hello" and r.model == "sonnet" and r.request_id == "sess-7" and r.finish_reason == "stop"
    call = fake.completion_calls()[0]
    argv = call["argv"]
    assert argv[:6] == ["claude", "-p", "--output-format", "json", "--model", "sonnet"]
    assert argv[argv.index("--effort") + 1] == "high"
    system = argv[argv.index("--system-prompt") + 1]
    assert system.startswith("Be brief.") and "Tool calls (text protocol)" in system and "read_file" in system
    assert argv[argv.index("--tools") + 1] == "" and "--allowedTools" not in argv
    assert "--no-session-persistence" in argv and "--disable-slash-commands" in argv and "--strict-mcp-config" in argv
    assert "--dangerously-skip-permissions" not in argv and "--bare" not in argv  # --bare would skip the subscription login
    assert call["stdin"] == b"Say hi." and call["env"]["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert r.raw["argv"][argv.index("--system-prompt") + 1].endswith("bytes>")
    assert p.caps.system_channel and p.caps.effort_control == "switch" and p.caps.max_prompt_bytes is None and p.prompt_limit() is None


def test_claude_old_release_flags_and_default_system_prompt():
    fake = FakeCli(help_text=CLAUDE_HELP_OLD, completions=[Completed(0, claude_envelope("ok"))])
    p = ClaudeCliProvider(runner=fake)
    p.complete(CompletionRequest([Message.user("q")], effort="low"))
    argv = fake.completion_calls()[0]["argv"]
    assert "--effort" not in argv and argv[argv.index("--allowedTools") + 1] == "" and "--tools" not in argv
    assert "--no-session-persistence" not in argv and "--disable-slash-commands" not in argv and "--strict-mcp-config" not in argv
    assert argv[argv.index("--system-prompt") + 1] == claude_bridge.DEFAULT_SYSTEM_PROMPT  # never Claude Code's own persona


def test_claude_folds_turns_into_stdin_and_keeps_system_separate():
    fake = FakeCli(help_text=CLAUDE_HELP_MODERN, completions=[Completed(0, claude_envelope("ok"))])
    p = ClaudeCliProvider(runner=fake)
    msgs = [
        Message.system("SYS-TEXT"),
        Message.user("first"),
        Message.assistant("", [ToolCall("c1", "read_file", {"path": "a"})]),
        Message.tool("c1", "read_file", "CONTENT"),
        Message.user("now"),
    ]
    p.complete(CompletionRequest(msgs, tools=TOOLS))
    call = fake.completion_calls()[0]
    prompt = call["stdin"].decode()
    assert "CONVERSATION SO FAR" in prompt and "TOOL RESULT read_file" in prompt and "CONTENT" in prompt and prompt.rstrip().endswith("now")
    assert "SYS-TEXT" not in prompt and "<<<SYSTEM" not in prompt
    assert call["argv"][call["argv"].index("--system-prompt") + 1].startswith("SYS-TEXT")


def test_claude_is_error_envelope():
    fake = FakeCli(
        help_text=CLAUDE_HELP_MODERN,
        completions=[
            Completed(1, claude_envelope("Rate limit reached, retry later", is_error=True)),
            Completed(0, claude_envelope("Not logged in. Please run /login", is_error=True)),
            Completed(0, claude_envelope("Something exploded", is_error=True)),
        ],
    )
    p = ClaudeCliProvider(runner=fake)
    req = CompletionRequest([Message.user("q")])
    with pytest.raises(ProviderError) as e1:
        p.complete(req)
    assert e1.value.retryable and e1.value.error.category.value == "resource_constraint"
    with pytest.raises(ProviderError) as e2:
        p.complete(req)
    assert e2.value.error.category.value == "permission_denied" and "log in" in e2.value.error.remediation
    with pytest.raises(ProviderError) as e3:
        p.complete(req)
    assert e3.value.error.category.value == "execution_failure" and "Something exploded" in str(e3.value) and "is_error" in str(e3.value)


def test_claude_usage_mapping_and_reported_cost():
    fake = FakeCli(help_text=CLAUDE_HELP_MODERN, completions=[Completed(0, claude_envelope("x"))])
    r = ClaudeCliProvider(runner=fake).complete(CompletionRequest([Message.user("q")]))
    assert r.usage.prompt_tokens == 180 and r.usage.cached_tokens == 60 and r.usage.completion_tokens == 25 and r.usage.reasoning_tokens == 0
    assert r.usage.cost_usd == pytest.approx(0.0123) and r.raw["envelope"]["reported_cost_usd"] == 0.0123 and r.raw["envelope"]["num_turns"] == 1
    fake2 = FakeCli(help_text=CLAUDE_HELP_MODERN, completions=[Completed(0, claude_envelope("x"))])
    r2 = ClaudeCliProvider(runner=fake2, prices={"input": 3.0, "output": 15.0, "cached": 0.3}).complete(CompletionRequest([Message.user("q")]))
    assert r2.usage.cost_usd == pytest.approx((120 * 3.0 + 60 * 0.3 + 25 * 15.0) / 1_000_000)


def test_claude_text_protocol_tool_call_parsed():
    text = 'Reading.\n```tool\n[{"name": "read_file", "arguments": {"path": "a.md"}}, {"name": "read_file", "arguments": {"path": "b.md"}}]\n```'
    fake = FakeCli(help_text=CLAUDE_HELP_MODERN, completions=[Completed(0, claude_envelope(text))])
    r = ClaudeCliProvider(runner=fake).complete(CompletionRequest([Message.system("S"), Message.user("go")], tools=TOOLS))
    assert r.finish_reason == "tool_calls" and [c.arguments["path"] for c in r.tool_calls] == ["a.md", "b.md"] and r.text == "Reading."


def test_claude_large_system_prompt_uses_file_or_refuses():
    big = "R" * (claude_bridge.SYSTEM_PROMPT_ARGV_MAX_BYTES + 10)
    seen: dict[str, Any] = {}

    def completion(argv: list[str], stdin: bytes | None, env: dict[str, str]) -> Completed:
        path = argv[argv.index("--system-prompt-file") + 1]
        seen.update(path=path, content=Path(path).read_text(encoding="utf-8"), has_inline="--system-prompt" in argv)
        return Completed(0, claude_envelope("ok"))

    p = ClaudeCliProvider(runner=FakeCli(help_text=CLAUDE_HELP_MODERN, completions=[completion]))
    p.complete(CompletionRequest([Message.system(big), Message.user("q")]))
    assert seen["content"] == big and not seen["has_inline"] and not os.path.exists(seen["path"])
    old = ClaudeCliProvider(runner=FakeCli(help_text=CLAUDE_HELP_OLD))
    with pytest.raises(ProviderError) as ei:
        old.complete(CompletionRequest([Message.system(big), Message.user("q")]))
    assert ei.value.error.category.value == "truncated" and "system-prompt-file" in ei.value.error.remediation


def test_claude_images_as_at_references(tmp_path):
    img = tmp_path / "diagram.webp"
    img.write_bytes(b"webp-bytes")
    seen: dict[str, Any] = {}

    def completion(argv: list[str], stdin: bytes | None, env: dict[str, str]) -> Completed:
        prompt = (stdin or b"").decode()
        refs = at_refs(prompt)
        seen.update(refs=refs, exists=[os.path.isfile(r) for r in refs], prompt=prompt)
        return Completed(0, claude_envelope("a diagram"))

    p = ClaudeCliProvider(runner=FakeCli(help_text=CLAUDE_HELP_MODERN, completions=[completion]))
    r = p.complete(CompletionRequest([Message.user("describe", [ContentPart.of_media("image", str(img))])]))
    assert r.text == "a diagram" and seen["exists"] == [True] and seen["refs"][0].endswith(".webp")
    assert seen["prompt"].startswith("describe") and "ATTACHED IMAGES" in seen["prompt"] and not seen["prompt"].endswith("Reply with the requested output only.")
    assert not os.path.exists(os.path.dirname(seen["refs"][0]))


def test_claude_default_model_and_health(monkeypatch):
    monkeypatch.setenv("EDGENT_CLAUDE_MODEL", "opus")
    assert ClaudeCliProvider(runner=FakeCli()).model == "opus"
    assert ClaudeCliProvider("haiku", runner=FakeCli()).model == "haiku"
    assert ClaudeCliProvider(runner=FakeCli(), environ={}).model == "sonnet"
    p = ClaudeCliProvider(runner=FakeCli(version="2.1.317 (Claude Code)", help_text=CLAUDE_HELP_MODERN))
    h = p.health()
    assert h["ok"] and h["tool_flag"] == "--tools" and h["effort_flag"] and h["system_prompt_file"] and h["provider"] == "claude-cli"
    assert h["version"] == "2.1.317 (Claude Code)" and "warning" not in h


def test_claude_probe_expands_the_system_prompt_file_shorthand():
    """Current releases list --system-prompt-file only as ``--system-prompt[-file]`` in the --bare text."""
    assert claude_bridge.expand_flag_shorthand(CLAUDE_HELP_SHORTHAND) == {"--system-prompt", "--system-prompt-file", "--append-system-prompt", "--append-system-prompt-file"}
    assert claude_bridge.expand_flag_shorthand(CLAUDE_HELP_OLD) == frozenset()
    assert parse_help_flags(CLAUDE_HELP_SHORTHAND) >= {"--system-prompt", "--system-prompt-snapshot"} and "--system-prompt-file" not in parse_help_flags(CLAUDE_HELP_SHORTHAND)
    fake = FakeCli(version="2.1.317 (Claude Code)", help_text=CLAUDE_HELP_SHORTHAND)
    p = ClaudeCliProvider(runner=fake)
    probe = p.probe()
    assert probe.has_flag("--system-prompt-file") and probe.has_flag("--append-system-prompt-file") and probe.version == "2.1.317 (Claude Code)"
    assert [c["argv"][1] for c in fake.calls] == ["--version", "--help", "--help"]  # one extra --help read, once per process
    p.probe()
    assert len(fake.calls) == 3
    assert p.health()["system_prompt_file"] is True
    # a release that lists the flag plainly, or lacks --system-prompt altogether, is not re-read
    modern = FakeCli(help_text=CLAUDE_HELP_MODERN)
    ClaudeCliProvider(runner=modern).probe()
    assert [c["argv"][1] for c in modern.calls] == ["--version", "--help"]
    old = FakeCli(help_text=CLAUDE_HELP_OLD)
    assert not ClaudeCliProvider(runner=old).probe().has_flag("--system-prompt-file") and len(old.calls) == 3


def test_claude_large_system_prompt_uses_the_file_flag_found_by_the_shorthand():
    big = "R" * (claude_bridge.SYSTEM_PROMPT_ARGV_MAX_BYTES + 10)
    seen: dict[str, Any] = {}

    def completion(argv: list[str], stdin: bytes | None, env: dict[str, str]) -> Completed:
        path = argv[argv.index("--system-prompt-file") + 1]
        seen.update(content=Path(path).read_text(encoding="utf-8"), has_inline="--system-prompt" in argv)
        return Completed(0, claude_envelope("ok"))

    p = ClaudeCliProvider(runner=FakeCli(help_text=CLAUDE_HELP_SHORTHAND, completions=[completion]))
    assert p.complete(CompletionRequest([Message.system(big), Message.user("q")])).text == "ok"
    assert seen["content"] == big and not seen["has_inline"]


def test_claude_system_prompt_size_error_tells_the_loop_not_to_compact():
    big = "R" * (claude_bridge.SYSTEM_PROMPT_ARGV_MAX_BYTES + 10)
    old = ClaudeCliProvider(runner=FakeCli(help_text=CLAUDE_HELP_OLD))
    with pytest.raises(ProviderError) as ei:
        old.complete(CompletionRequest([Message.system(big), Message.user("q")]))
    err = ei.value.error
    assert err.category.value == "truncated" and err.details["compactable"] is False and err.details["limit"] == claude_bridge.SYSTEM_PROMPT_ARGV_LIMIT.name
