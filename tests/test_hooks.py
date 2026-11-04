from __future__ import annotations

import json
import textwrap

import pytest

from edgent.application.hooks import EVENTS, HookOutcome, HookRunner, merge_outcomes
from edgent.domain.errors import ConfigurationError
from edgent.observability import Tracer, TraceReader


def write_hook(workspace, name: str, body: str) -> str:
    rel = f".edgent/hooks/{name}.py"
    workspace.write_text(rel, textwrap.dedent(body))
    return f"python3 {rel}"


def test_python_hooks_and_merge(workspace):
    runner = HookRunner({}, workspace)
    assert not runner.has("PreToolUse") and runner.run("PreToolUse", {"tool_name": "bash"}) == HookOutcome()
    seen = []
    runner.register("PreToolUse", lambda p: seen.append(p) or HookOutcome(message="first", updated_arguments={"a": 1}))
    runner.register("PreToolUse", lambda p: {"allow": False, "message": "second says no", "updated_arguments": {"a": 2}})
    runner.register("PreToolUse", lambda p: None)
    runner.register("PreToolUse", lambda p: 1 / 0)  # a broken hook is logged and ignored
    assert runner.has("PreToolUse")
    out = runner.run("PreToolUse", {"tool_name": "bash", "arguments": {"command": "ls"}})
    assert not out.allow and out.message == "first\nsecond says no" and out.updated_arguments == {"a": 2}
    assert seen[0]["event"] == "PreToolUse" and seen[0]["workspace"] == str(workspace.root) and seen[0]["tool_name"] == "bash"
    assert merge_outcomes([]) == HookOutcome() and merge_outcomes([HookOutcome(), HookOutcome(updated_arguments={"x": 1})]).updated_arguments == {"x": 1}
    assert HookOutcome(False, "m").to_dict() == {"allow": False, "message": "m", "updated_arguments": None}
    with pytest.raises(ConfigurationError):
        runner.register("NotAnEvent", lambda p: None)
    assert "PreToolUse" in runner.describe() and len(runner.describe()["PreToolUse"]) == 4


def test_command_hook_exit_zero_updates_arguments(workspace):
    cmd = write_hook(
        workspace,
        "rewrite",
        """
        import json, sys
        payload = json.load(sys.stdin)
        args = dict(payload["arguments"]); args["command"] = args["command"] + " --safe"
        print(json.dumps({"updated_arguments": args, "message": "rewrote " + payload["tool_name"] + " in " + payload["event"]}))
        """,
    )
    runner = HookRunner({"hooks": {"PreToolUse": [{"matcher": "bash|write_file", "command": cmd, "timeout_s": 20}]}}, workspace)
    out = runner.run("PreToolUse", {"tool_name": "bash", "arguments": {"command": "ls"}, "session_id": "s"})
    assert out.allow and out.updated_arguments == {"command": "ls --safe"} and out.message == "rewrote bash in PreToolUse"
    # the matcher is a full regex on the tool name
    assert runner.run("PreToolUse", {"tool_name": "read_file", "arguments": {}}) == HookOutcome()
    assert runner.run("PreToolUse", {"tool_name": "bash_extra", "arguments": {}}) == HookOutcome()


def test_command_hook_exit_two_blocks(workspace):
    cmd = write_hook(workspace, "block", "import sys\nsys.stderr.write('chapters are frozen during review')\nsys.exit(2)\n")
    tracer = Tracer(workspace.path("traces"), "s1")
    runner = HookRunner({"hooks": {"PreToolUse": [{"matcher": "*", "command": cmd}]}}, workspace, tracer=tracer)
    out = runner.run("PreToolUse", {"tool_name": "write_file", "arguments": {"path": "book/chapters/01.md"}})
    assert not out.allow and out.message == "chapters are frozen during review" and out.updated_arguments is None
    events = [e for e in TraceReader(tracer.path) if e.kind == "hook"]
    assert len(events) == 1 and events[0].data["returncode"] == 2 and events[0].data["allow"] is False and events[0].data["tool_name"] == "write_file"


def test_command_hook_other_exit_and_timeout_continue(workspace):
    crash = write_hook(workspace, "crash", "import sys\nsys.stderr.write('boom')\nsys.exit(1)\n")
    slow = write_hook(workspace, "slow", "import time\ntime.sleep(5)\n")
    tracer = Tracer(workspace.path("traces"), "s2")
    runner = HookRunner({"hooks": {"PostToolUse": [{"command": crash}, {"command": slow, "timeout_s": 0.5}]}}, workspace, tracer=tracer)
    out = runner.run("PostToolUse", {"tool_name": "bash", "arguments": {}})
    assert out.allow and "timed out after 0.5 s" in out.message and out.updated_arguments is None
    data = [e.data for e in TraceReader(tracer.path) if e.kind == "hook"]
    assert [d["returncode"] for d in data] == [1, -9] and data[1]["timed_out"] is True and data[0]["stderr"] == "boom"


def test_command_hook_runs_sandboxed_in_workspace(workspace, monkeypatch):
    monkeypatch.setenv("AWS_SECRET", "hunter2")  # present in the harness process; the hook must not see it
    cmd = write_hook(workspace, "env", "import os, json\nprint(json.dumps({'message': os.getcwd() + '|' + str('AWS_SECRET' in os.environ)}))\n")
    runner = HookRunner({"hooks": {"SessionStart": [{"command": cmd}]}}, workspace)
    out = runner.run("SessionStart", {"session_id": "s"})
    assert out.message == f"{workspace.root}|False"


def test_hook_settings_validation(workspace):
    for bad in (
        {"hooks": ["x"]},
        {"hooks": {"PreToolUser": [{"command": "true"}]}},
        {"hooks": {"PreToolUse": {"command": "true"}}},
        {"hooks": {"PreToolUse": [{"matcher": "bash"}]}},
        {"hooks": {"PreToolUse": [{"command": ""}]}},
        {"hooks": {"PreToolUse": [{"command": "true", "matcher": "("}]}},
        {"hooks": {"PreToolUse": [{"command": "true", "timeout_s": 0}]}},
        {"hooks": {"PreToolUse": [{"command": "true", "timeout_s": "30"}]}},
    ):
        with pytest.raises(ConfigurationError):
            HookRunner(bad, workspace)
    runner = HookRunner({"hooks": {e: [{"command": "true"}] for e in EVENTS}}, workspace)
    assert all(runner.has(e) for e in EVENTS)
    assert json.loads(json.dumps(runner.describe()))["Stop"][0]["matcher"] == "*"
