from __future__ import annotations

import pytest

from edgent.application.permissions import Pattern, Permissions, subject_of
from edgent.domain.errors import ConfigurationError
from edgent.tools.registry import ToolSpec, obj


def spec(name: str, permission: str) -> ToolSpec:
    return ToolSpec(name=name, description="t", parameters=obj({}), fn=lambda c, a: None, permission=permission)


READ, COMPUTE, WRITE = spec("read_file", "read"), spec("bash", "compute"), spec("write_file", "write")
NET, SPAWN, ASK, DEL = spec("web_fetch", "network"), spec("agent", "spawn"), spec("ask_user", "interactive"), spec("delete_file", "delete")


def test_subjects():
    assert subject_of("bash", {"command": "manim render x.py", "timeout_s": 5}) == "manim render x.py"
    assert subject_of("Bash", {"command": "ls"}) == "ls"
    assert subject_of("read_file", {"path": "book/a.md"}) == "book/a.md"
    assert subject_of("grep", {"pattern": "damping"}) == "damping" and subject_of("glob", {"pattern": "**/*.md"}) == "**/*.md"
    assert subject_of("web_fetch", {"url": "https://x.y"}) == "https://x.y" and subject_of("web_search", {"query": "q"}) == "q"
    assert subject_of("agent", {"agent": "librarian", "task": "t"}) == "librarian"
    assert subject_of("bash", {}) == ""
    assert subject_of("custom", {"b": 1, "a": "x"}) == '{"a":"x","b":1}'


def test_pattern_forms():
    assert Pattern.parse("bash").matches("bash", "anything") and Pattern.parse("BASH").matches("Bash", "x")
    assert Pattern.parse("bash(*)").matches("bash", "rm -rf /") and Pattern.parse("bash( * )").kind == "any"
    manim = Pattern.parse("bash(manim *)")
    assert manim.kind == "prefix" and manim.matches("bash", "manim render scene.py") and manim.matches("bash", "manim")
    assert not manim.matches("bash", "manimx") and not manim.matches("bash", "Manim render") and not manim.matches("python", "manim x")
    exact = Pattern.parse("read_file(STUDY.md)")
    assert exact.kind == "exact" and exact.matches("READ_FILE", "STUDY.md") and not exact.matches("read_file", "study.md") and not exact.matches("read_file", "STUDY.md2")
    assert Pattern.parse("bash(npm run test:*)").matches("bash", "npm run test:unit")
    assert str(manim) == "bash(manim *)" and str(Pattern.parse("bash")) == "bash" and str(exact) == "read_file(STUDY.md)"
    with pytest.raises(ConfigurationError):
        Pattern.parse("bash(unclosed")
    with pytest.raises(ConfigurationError):
        Pattern.parse("")


def test_mode_ask(workspace):
    p = Permissions("ask", workspace=workspace)
    assert p.decide(READ, {"path": "a"}) == "allow" and p.decide(COMPUTE, {"command": "ls"}) == "allow"
    assert p.decide(WRITE, {"path": "book/a.md"}) == "ask" and p.decide(SPAWN, {}) == "ask" and p.decide(ASK, {}) == "ask"
    assert p.decide(NET, {"url": "u"}) == "ask" and p.decide(DEL, {"path": "a"}) == "ask"


def test_mode_accept_edits(workspace):
    p = Permissions("accept-edits", workspace=workspace)
    assert p.decide(READ, {"path": "a"}) == "allow" and p.decide(COMPUTE, {}) == "allow"
    assert p.decide(WRITE, {"path": "book/a.md"}) == "allow" and p.decide(WRITE, {"path": "../outside.md"}) == "ask"
    assert p.decide(WRITE, {"path": "/etc/passwd"}) == "ask" and p.decide(WRITE, {"path": "~/x"}) == "ask"
    assert p.decide(SPAWN, {}) == "ask" and p.decide(ASK, {}) == "ask" and p.decide(NET, {}) == "ask" and p.decide(DEL, {}) == "ask"


def test_mode_auto(workspace):
    p = Permissions("auto", workspace=workspace)
    assert p.decide(WRITE, {"path": "book/a.md"}) == "allow" and p.decide(SPAWN, {"agent": "x"}) == "allow" and p.decide(ASK, {}) == "allow"
    assert p.decide(COMPUTE, {"command": "python x.py"}) == "allow"
    assert p.decide(WRITE, {"path": "../outside.md"}) == "ask" and p.decide(READ, {"paths": ["a.md", "/etc/hosts"]}) == "ask"
    assert p.decide(READ, {"out_path": "media/x.svg", "pattern": "x"}) == "allow"
    assert p.decide(NET, {"url": "https://a"}) == "ask" and p.decide(DEL, {"path": "a"}) == "ask"
    no_ws = Permissions("auto")
    assert no_ws.decide(WRITE, {"path": "a"}) == "ask" and no_ws.decide(COMPUTE, {"command": "ls"}) == "allow"


def test_mode_plan_and_bypass(workspace):
    plan = Permissions("plan", workspace=workspace)
    assert plan.decide(READ, {}) == "allow"
    for s in (COMPUTE, WRITE, NET, SPAWN, ASK, DEL):
        assert plan.decide(s, {"path": "a"}) == "deny"
    assert Permissions("plan", allow=["write_file(book/*)"], workspace=workspace).decide(WRITE, {"path": "book/a.md"}) == "allow"
    bypass = Permissions("bypass", workspace=workspace)
    for s in (READ, COMPUTE, WRITE, NET, SPAWN, ASK, DEL):
        assert bypass.decide(s, {"path": "../x"}) == "allow"
    assert bypass.decide(DEL, {"path": "a"}) == "allow"
    assert Permissions("bypass", deny=["delete_file"]).decide(DEL, {"path": "a"}) == "deny"


def test_allow_deny_precedence_and_always_ask(workspace):
    p = Permissions("ask", allow=["bash(manim *)", "web_fetch(https://docs.python.org/*)", "write_file"], deny=["bash(manim --dangerous*)", "write_file(STUDY.md)"], workspace=workspace)
    assert p.decide(COMPUTE, {"command": "manim render a.py"}) == "allow"
    assert p.decide(COMPUTE, {"command": "manim --dangerous thing"}) == "deny"
    assert p.decide(COMPUTE, {"command": "rm -rf"}) == "allow"  # compute is allowed in ask mode
    assert p.decide(NET, {"url": "https://docs.python.org/3/"}) == "allow" and p.decide(NET, {"url": "https://evil"}) == "ask"
    assert p.decide(WRITE, {"path": "book/a.md"}) == "allow" and p.decide(WRITE, {"path": "STUDY.md"}) == "deny"
    custom = Permissions("auto", workspace=workspace, always_ask={"delete"})
    assert custom.decide(NET, {"url": "u"}) == "ask" and custom.decide(DEL, {"path": "a"}) == "ask"


def test_from_settings_describe_and_errors(workspace):
    settings = {"mode": "accept-edits", "permissions": {"allow": ["Bash(manim *)"], "deny": ["Bash(rm *)"], "always_ask": ["network", "delete", "spawn"]}}
    p = Permissions.from_settings(settings, workspace)
    assert p.mode == "accept-edits" and p.decide(COMPUTE, {"command": "manim x"}) == "allow" and p.decide(COMPUTE, {"command": "rm -rf /"}) == "deny"
    assert p.decide(SPAWN, {}) == "ask"
    d = p.describe()
    assert d == {"mode": "accept-edits", "allow": ["bash(manim *)"], "deny": ["bash(rm *)"], "always_ask": ["delete", "network", "spawn"], "workspace": str(workspace.root)}
    assert Permissions.from_settings({}, workspace).mode == "ask" and Permissions.from_settings({}).describe()["workspace"] is None
    assert p.with_mode("plan").mode == "plan" and p.with_mode("plan").decide(COMPUTE, {"command": "manim x"}) == "allow"
    with pytest.raises(ConfigurationError):
        Permissions("yolo")
    with pytest.raises(ConfigurationError):
        Permissions.from_settings({"permissions": {"allow": "bash"}})
    with pytest.raises(ConfigurationError):
        Permissions.from_settings({"mode": 3})
    with pytest.raises(ConfigurationError):
        Permissions.from_settings({"permissions": ["bash"]})
