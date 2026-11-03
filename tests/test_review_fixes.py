"""Regressions found by the adversarial review of the integrated harness: the text tool protocol, the permission
gates, the replay scope, the build lock and the CLI's failure paths."""
from __future__ import annotations

import json
import os

import pytest

from edgent.adapters.providers.mock import MockProvider
from edgent.application.app import App, AppConfig
from edgent.application.permissions import Permissions, shell_segments
from edgent.domain.capabilities import Capabilities
from edgent.domain.outline import Outline, Unit
from edgent.domain.types import Reply
from edgent.domain.workspace import Workspace
from edgent.tools.core import build_core_registry
from edgent.tools.text_protocol import parse_tool_calls, render_tool_catalog
from edgent.ui.cli import main

FENCED = '# T\n\n```python\nprint(1)\n```\n'


def test_text_protocol_keeps_fences_inside_json_strings():
    block = '```tool\n' + json.dumps({"name": "write_file", "arguments": {"path": "lesson.md", "content": FENCED}}) + '\n```\nDone.'
    clean, calls, errors = parse_tool_calls(block)
    assert errors == [] and clean == "Done."
    assert calls[0].name == "write_file" and calls[0].arguments["content"] == FENCED


def test_text_protocol_tolerates_crlf_same_line_json_and_a_missing_close():
    _, calls, errors = parse_tool_calls('before\r\n```tool\r\n{"name": "grep", "arguments": {"pattern": "x"}}\r\n```\r\nafter')
    assert [c.name for c in calls] == ["grep"] and errors == []
    _, calls, _ = parse_tool_calls('```tool {"name": "read_file", "arguments": {"path": "a"}}\n```')
    assert [c.name for c in calls] == ["read_file"]
    _, calls, errors = parse_tool_calls('```tool\n{"name": "read_file", "arguments": {"path": "a"}}')
    assert [c.name for c in calls] == ["read_file"] and errors == []
    _, calls, errors = parse_tool_calls('```tool\n[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {"k": "```tool"}}]\n```')
    assert [c.name for c in calls] == ["a", "b"] and calls[1].arguments == {"k": "```tool"}


def test_text_protocol_catalogue_shows_nested_object_shapes():
    text = render_tool_catalog([{"name": "t", "description": "d", "parameters": {"type": "object", "properties": {"opts": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "string"}}, "required": ["a"]}}, "required": ["opts"]}}])
    assert "opts: {a: integer, b?: string}" in text


def test_text_protocol_loop_writes_a_chapter_with_a_code_fence(tmp_path, home):
    ws = Workspace.init(tmp_path / "ws", git=False)
    provider = MockProvider(caps=Capabilities(native_tools=False, system_channel=False, json_schema=False))
    block = '```tool\n' + json.dumps({"name": "write_file", "arguments": {"path": "book/chapters/01.md", "content": FENCED}}) + '\n```'
    provider.push(Reply(text=block), Reply(text="Written."))
    app = App(ws, AppConfig(provider="mock", home=str(home)), provider=provider)
    _, result = app.chat("write it", permission_mode="accept-edits")
    assert result.stop_reason == "done" and result.tool_calls == 1 and result.errors == []
    assert ws.read_text("book/chapters/01.md") == FENCED


def test_bash_with_allow_network_is_judged_as_a_network_tool(workspace):
    reg = build_core_registry()
    bash = reg.get("bash")
    for mode in ("ask", "auto", "accept-edits"):
        perms = Permissions(mode, workspace=workspace)
        assert perms.decide(bash, {"command": "curl x", "allow_network": True}) == "ask", mode
        assert perms.decide(bash, {"command": "echo 1"}) == "allow", mode
    assert Permissions("bypass", workspace=workspace).decide(bash, {"command": "curl x", "allow_network": True}) == "allow"
    assert Permissions("auto", allow=["bash(curl *)"], workspace=workspace).decide(bash, {"command": "curl x", "allow_network": True}) == "allow"


def test_bash_deny_patterns_match_every_shell_segment(workspace):
    assert shell_segments("echo hi; rm -rf .") == ["echo hi", "rm -rf ."]
    assert shell_segments("a && b || c | d\n$(e) `f`") == ["a", "b", "c", "d", "e)", "f"]
    perms = Permissions("auto", deny=["bash(rm *)"], workspace=workspace)
    bash = build_core_registry().get("bash")
    for command in ("rm -rf .", "echo hi; rm -rf .", "true && rm x", "ls | rm -r y", "echo $(rm z)", "   rm q"):
        assert perms.decide(bash, {"command": command}) == "deny", command
    assert perms.decide(bash, {"command": "echo rm"}) == "allow"


def test_writes_into_the_control_plane_always_ask(workspace):
    reg = build_core_registry()
    write = reg.get("write_file")
    for mode in ("auto", "accept-edits"):
        perms = Permissions(mode, workspace=workspace)
        assert perms.decide(write, {"path": ".edgent/settings.json", "content": "{}"}) == "ask", mode
        assert perms.decide(write, {"path": ".edgent/agents/evil.md", "content": "x"}) == "ask", mode
        assert perms.decide(write, {"path": ".edgent/skills/x/SKILL.md", "content": "x"}) == "ask", mode
        assert perms.decide(write, {"path": "book/chapters/01.md", "content": "x"}) == "allow", mode
        assert perms.decide(write, {"path": ".edgent/cache/scratch.txt", "content": "x"}) == "allow", mode
    assert Permissions("plan", workspace=workspace).decide(write, {"path": ".edgent/settings.json", "content": "{}"}) == "deny"
    assert Permissions("auto", allow=["write_file(.edgent/*)"], workspace=workspace).decide(write, {"path": ".edgent/settings.json", "content": "{}"}) == "allow"


def test_replay_never_crosses_turns_even_when_a_server_reuses_call_ids(tmp_path, home):
    ws = Workspace.init(tmp_path / "ws", git=False)
    provider = MockProvider()
    first = Reply(text="", tool_calls=[], finish_reason="tool_calls")
    from edgent.domain.types import ToolCall

    first.tool_calls = [ToolCall(id="call_0", name="write_file", arguments={"path": "a.txt", "content": "v1"})]
    second = Reply(text="", tool_calls=[ToolCall(id="call_0", name="write_file", arguments={"path": "a.txt", "content": "v1"})], finish_reason="tool_calls")
    provider.push(first, Reply(text="done 1"), second, Reply(text="done 2"))
    app = App(ws, AppConfig(provider="mock", home=str(home)), provider=provider)
    session, _ = app.chat("write", permission_mode="accept-edits")
    (ws.root / "a.txt").write_text("changed outside", encoding="utf-8")
    app.chat("write again", session_id=session.session_id, permission_mode="accept-edits")
    assert (ws.root / "a.txt").read_text(encoding="utf-8") == "v1"  # the second turn's identical call ran again
    tool_messages = [m for m in app.load_session(session.session_id).messages if m.role == "tool"]
    assert len(tool_messages) == 2 and all('"replayed": true' not in m.text() for m in tool_messages)


def test_build_lock_refuses_a_second_live_build_and_takes_over_a_dead_one(tmp_path, home):
    ws = Workspace.init(tmp_path / "ws", git=False)
    outline = Outline(ws.path("outline"))
    outline.add(Unit(id="u1", kind="chapter", title="One", objective="o", path="book/chapters/01.md"))
    outline.save()
    app = App(ws, AppConfig(provider="mock", home=str(home), critic=False), provider=MockProvider().push(Reply(text="x")))
    lock = ws.path("state") / "build.lock"
    lock.write_text(json.dumps({"pid": os.getppid()}), encoding="utf-8")  # a live owner that is not this process
    with pytest.raises(RuntimeError, match="another build"):
        app.build_job().run(max_units=1)
    lock.write_text(json.dumps({"pid": 999_999_999}), encoding="utf-8")  # a dead owner is taken over
    report = app.build_job().run(max_units=1)
    assert report.units_run == ["u1"] and not lock.exists()


def test_cli_failure_paths_are_messages_not_tracebacks(tmp_path, home, capsys):
    ws = tmp_path / "ws"
    assert main(["init", str(ws), "--no-git"]) == 0
    capsys.readouterr()
    assert main(["--workspace", str(ws), "--provider", "mock", "build", "a book"]) == 2  # the mock proposes no units
    assert "planning failed" in capsys.readouterr().err
    assert main(["--workspace", str(ws), "--provider", "mock", "build", "--unblock", "nope"]) == 2
    assert "cannot unblock" in capsys.readouterr().err
    assert main(["--workspace", str(ws), "--provider", "mock", "trace", "no-such-session"]) == 2
    assert "no trace" in capsys.readouterr().err
    Workspace(ws).write_json("book/outline.json", {"version": 1, "title": "t", "persona": "p", "meta": {}, "units": [{"id": "u1", "kind": "chapter", "title": "One", "objective": "o", "status": "pending", "depends_on": [], "path": "book/chapters/01.md", "checks": [], "attempts": 0, "open_obligations": [], "history": [], "meta": {}}]})
    assert main(["--workspace", str(ws), "--provider", "mock", "build", "--plan-only"]) == 0  # shows the plan, runs nothing
    out = capsys.readouterr()
    assert "u1" in out.err and "units_run" not in out.out


def test_run_without_a_prompt_allows_workspace_writes_by_default(tmp_path, home, capsys, monkeypatch):
    ws = tmp_path / "ws"
    assert main(["init", str(ws), "--no-git"]) == 0
    captured: dict[str, object] = {}
    real_chat = App.chat

    def spy(self, text, session_id=None, parts=None, extra_context=None, **kw):
        captured.update(kw)
        return real_chat(self, text, session_id=session_id, parts=parts, extra_context=extra_context, **kw)

    monkeypatch.setattr(App, "chat", spy)
    assert main(["--workspace", str(ws), "--provider", "mock", "run", "hello"]) == 0
    assert captured.get("permission_mode") == "accept-edits"
    captured.clear()
    assert main(["--workspace", str(ws), "--provider", "mock", "--mode", "plan", "run", "hello"]) == 0
    assert "permission_mode" not in captured
