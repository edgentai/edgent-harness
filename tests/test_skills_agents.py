from __future__ import annotations

import threading
from pathlib import Path

import pytest

from edgent.application.budgets import Budget, BudgetTracker
from edgent.application.loop import AgentLoop, TurnResult
from edgent.application.sessions import Session
from edgent.application.skills import INDEX_HEADER, Skill, SkillIndex, builtin_skills_dir
from edgent.application.subagents import AgentDefinition, AgentIndex, SubagentResult, SubagentRunner, builtin_agents_dir
from edgent.domain.errors import ConfigurationError, ToolFailure
from edgent.domain.types import Usage
from edgent.observability import Tracer, TraceReader
from edgent.tools.registry import ToolRegistry, obj


def make_skill(root: Path, name: str, description: str, body: str = "Body text.", files: dict[str, str] | None = None) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\nversion: 2\n---\n{body}\n")
    for rel, text in (files or {}).items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return d


def test_skill_index_precedence_body_and_references(workspace, home):
    ws_dir, user_dir, builtin = SkillIndex.default_dirs(workspace, home)
    assert ws_dir == workspace.root / ".edgent" / "skills" and user_dir == home / "skills" and builtin == builtin_skills_dir()
    make_skill(ws_dir, "explaining", "Workspace version of explaining.", "# Explaining (workspace)\nFind the one idea.", {"references/checklist.md": "- curse of knowledge", "scripts/measure.py": "print(1)", "references/.hidden": "x"})
    make_skill(user_dir, "explaining", "User version.", "user body")
    make_skill(user_dir, "diagramming", "When a picture earns its place.")
    (user_dir / "not-a-skill").mkdir()
    (user_dir / "not-a-skill" / "README.md").write_text("no SKILL.md here")
    idx = SkillIndex().load([ws_dir, user_dir], ["workspace", "user"])
    assert idx.names() == ["diagramming", "explaining"] and len(idx) == 2 and "explaining" in idx
    sk = idx.get("explaining")
    assert sk.source == "workspace" and sk.description == "Workspace version of explaining." and sk.metadata["version"] == 2
    body = sk.body()
    assert body.startswith("# Explaining (workspace)") and "name: explaining" not in body
    assert "- references/checklist.md" in body and "- scripts/measure.py" in body and ".hidden" not in body
    assert sk.reference("checklist.md") == "- curse of knowledge" and sk.reference("references/checklist.md") == "- curse of knowledge"
    assert sk.reference("scripts/measure.py") == "print(1)"
    with pytest.raises(FileNotFoundError):
        sk.reference("../SKILL.md")
    with pytest.raises(FileNotFoundError):
        sk.reference("missing.md")
    assert idx.get("diagramming").files() == [] and "## Files in this skill" not in idx.load_body("diagramming")
    with pytest.raises(KeyError) as ei:
        idx.load_body("nope")
    assert "diagramming, explaining" in str(ei.value)
    text = idx.index_text()
    assert text.startswith(INDEX_HEADER) and "- explaining: Workspace version of explaining." in text and "- diagramming: When a picture earns its place." in text
    assert idx.index_text() == idx.index_text()  # byte-identical across calls: it is part of the cache prefix
    short = idx.index_text(max_chars=len(INDEX_HEADER) + 50)
    assert "1 more skills" in short and short.count("\n") == 2
    found = SkillIndex.discover(workspace, home)  # the built-in skills join in; the workspace copy still wins by name
    assert SkillIndex().index_text() == "" and {"diagramming", "explaining"} <= set(found.names()) and found.get("explaining").source == "workspace"
    assert all(found.get(n).source == "builtin" for n in found.names() if n not in ("diagramming", "explaining"))
    assert Skill("x", "d", ws_dir / "explaining", "workspace").to_dict()["source"] == "workspace"


AGENT_MD = """---
name: librarian
description: Finds passages in sources/text/ that answer a question. Returns quotes with locators.
tools: grep, glob, read_file
disallowedTools: [write_file]
model: fast
effort: low
skills:
  - explaining
maxTurns: 15
background: false
permissionMode: plan
memory: project
---
You find evidence. Search sources/text/ and return at most eight passages.
"""


def test_agent_definition_parsing_and_index(workspace, home):
    d = AgentDefinition.from_markdown(AGENT_MD, "workspace", Path("librarian.md"))
    assert d.name == "librarian" and d.tools == ["grep", "glob", "read_file"] and d.disallowed_tools == ["write_file"]
    assert d.model == "fast" and d.effort == "low" and d.skills == ["explaining"] and d.max_turns == 15 and d.background is False
    assert d.permission_mode == "plan" and d.memory == "project" and d.system_prompt.startswith("You find evidence.") and d.source == "workspace"
    assert d.to_dict()["max_turns"] == 15 and d.to_dict()["path"] == "librarian.md"
    inherit = AgentDefinition.from_markdown("---\nname: author\ndescription: Drafts one unit.\ntools: *\n---\nDraft.")
    assert inherit.tools is None and inherit.max_turns is None and inherit.disallowed_tools == []
    assert AgentDefinition.from_markdown("---\ntools: [a, b]\n---\nx", default_name="fallback").name == "fallback"
    for bad in ("---\ndescription: no name\n---\nx", "---\nname: x\nmaxTurns: zero\n---\n", "---\nname: x\neffort: max\n---\n", "---\nname: x\npermissionMode: yolo\n---\n", "---\nname: x\nmaxTurns: 0\n---\n"):
        with pytest.raises(ConfigurationError):
            AgentDefinition.from_markdown(bad)

    ws_dir, user_dir, builtin = AgentIndex.default_dirs(workspace, home)
    assert ws_dir == workspace.root / ".edgent" / "agents" and user_dir == home / "agents" and builtin == builtin_agents_dir()
    ws_dir.mkdir(parents=True)
    user_dir.mkdir(parents=True)
    (ws_dir / "librarian.md").write_text(AGENT_MD)
    (user_dir / "librarian.md").write_text("---\nname: librarian\ndescription: User copy.\n---\nx")
    (user_dir / "critic.md").write_text("---\nname: critic\ndescription: Reviews a rendered unit against a rubric.\n---\nReview.")
    idx = AgentIndex().load([ws_dir, user_dir], ["workspace", "user"])
    assert idx.names() == ["critic", "librarian"] and idx.get("librarian").source == "workspace" and idx.get("nope") is None and len(idx) == 2
    text = idx.index_text()
    assert text.startswith("## Subagents (delegate with the agent tool)") and "- critic: Reviews a rendered unit against a rubric." in text
    assert "1 more subagents" in idx.index_text(max_chars=100) and AgentIndex().index_text() == ""
    found = AgentIndex.discover(workspace, home)  # the built-in agents join in; the workspace librarian still wins by name
    assert {"critic", "librarian"} <= set(found.names()) and found.get("librarian").source == "workspace" and found.get("critic").source == "user"
    assert all(found.get(n).source == "builtin" for n in found.names() if n not in ("critic", "librarian"))
    r = SubagentResult("librarian", "s", "done", "found it", {"k": 1}, Usage(5, 2), 1, [])
    assert r.to_dict() == {"name": "librarian", "session_id": "s", "status": "done", "summary": "found it", "structured": {"k": 1}, "usage": Usage(5, 2).to_dict(), "turns": 1, "errors": []}


class FakeChild:
    """Stands in for a child AgentLoop: records what the factory got and returns a scripted turn."""

    def __init__(self, kw, text="FOUND: p.12", stop="done", hold: threading.Event | None = None):
        self.kw = kw
        self.text = text
        self.stop = stop
        self.hold = hold

    def run_turn(self, task, parts=None, extra_context=None):
        if self.hold is not None:
            self.hold.wait(5)
        if self.text == "raise":
            raise RuntimeError("child exploded")
        return TurnResult(text=self.text, tool_calls=1, iterations=2, usage=Usage(30, 10, cost_usd=0.002), stop_reason=self.stop, errors=["e1"])


def make_parent(workspace, provider, allowlist=None):
    reg = ToolRegistry()
    for name in ("read_file", "grep", "glob", "write_file", "agent", "ask_user"):
        reg.register(__import__("edgent.tools.registry", fromlist=["ToolSpec"]).ToolSpec(name=name, description=name, parameters=obj({}), fn=lambda c, a: {}))
    session = Session.create(workspace, mode="chat", session_id="parent")
    tracer = Tracer(workspace.path("traces"), "parent")
    skills = SkillIndex()
    make_skill(workspace.root / ".edgent" / "skills", "explaining", "How to explain.", "Find the one idea.")
    skills.load([workspace.root / ".edgent" / "skills"], ["workspace"])
    return AgentLoop(provider, reg, workspace, session, system_prompt="parent", allowlist=allowlist, tracer=tracer, skills=skills, budget=BudgetTracker(Budget(max_iterations=7)))


def test_subagent_runner_spawn_and_isolation(workspace, provider):
    parent = make_parent(workspace, provider)
    captured = {}

    def factory(**kw):
        captured.update(kw)
        return FakeChild(kw, text='{"finding": "basis", "locators": ["book:p.12"]}')

    runner = SubagentRunner(factory, max_depth=2, max_concurrent=2)
    definition = AgentDefinition.from_markdown(AGENT_MD)
    result = runner.spawn(definition, "find the definition of basis", parent, depth=0)
    assert result.status == "done" and result.structured == {"finding": "basis", "locators": ["book:p.12"]} and result.turns == 2 and result.errors == ["e1"]
    assert result.usage.prompt_tokens == 30 and result.name == "librarian"
    child = Session.load(workspace, result.session_id)
    assert child.mode == "subagent" and child.parent_id == "parent" and child.title == "librarian" and child.meta["agent"] == "librarian" and child.meta["depth"] == 1
    assert parent.session.meta["children"] == [result.session_id] and Session.load(workspace, "parent").meta["children"] == [result.session_id]
    assert captured["session"].session_id == result.session_id and captured["role"] == "librarian" and captured["effort"] == "low"
    assert captured["max_iterations"] == 15 and captured["depth"] == 1 and captured["permission_mode"] == "plan" and captured["model"] == "fast"
    assert captured["allowlist"] == {"grep", "glob", "read_file"}
    assert captured["system_prompt"].startswith("You find evidence.") and "# Skill: explaining" in captured["system_prompt"] and "Find the one idea." in captured["system_prompt"]
    assert parent.budget.cost_usd == pytest.approx(0.002) and parent.budget.total_tokens == 40 and parent.budget.iterations == 0
    kinds = [(e.kind, e.data.get("phase")) for e in TraceReader(parent.tracer.path) if e.kind == "subagent"]
    assert kinds == [("subagent", "start"), ("subagent", "stop")]
    stop = [e.data for e in TraceReader(parent.tracer.path) if e.kind == "subagent"][-1]
    assert stop["usage"]["prompt_tokens"] == 30 and stop["status"] == "done"


def test_subagent_allowlist_depth_and_errors(workspace, provider):
    parent = make_parent(workspace, provider, allowlist={"read_file", "grep", "agent", "ask_user", "write_file"})
    runner = SubagentRunner(lambda **kw: FakeChild(kw), max_depth=2)
    inherit = AgentDefinition(name="author", description="d", tools=None, disallowed_tools=["write_file"], system_prompt="Draft.")
    # inherits the parent's allowlist minus disallowed, minus ask_user; keeps agent at depth 0 -> child depth 1 < max_depth
    assert runner.child_allowlist(inherit, parent, depth=0) == {"read_file", "grep", "agent"}
    # at the deepest level the child loses the agent tool
    assert runner.child_allowlist(inherit, parent, depth=1) == {"read_file", "grep"}
    missing_skill = AgentDefinition(name="x", skills=["nope"], system_prompt="p")
    r = runner.spawn(missing_skill, "t", parent, depth=0)
    assert r.status == "done" and any("unknown skill 'nope'" in e for e in r.errors)
    assert runner.spawn(AgentDefinition(name="m", system_prompt="p"), "t", parent, depth=1).status == "done"
    with pytest.raises(ToolFailure) as ei:
        runner.spawn(inherit, "t", parent, depth=2)
    assert ei.value.error.category.value == "permission_denied" and "max subagent depth" in ei.value.error.message
    assert runner.spawn(AgentDefinition(name="t", system_prompt="p"), "t", parent, depth=0)  # a 'done' child
    for text, stop, status in (("x", "max_iterations", "max_turns"), ("x", "budget", "budget"), ("x", "cancelled", "error"), ("raise", "done", "error")):
        res = SubagentRunner(lambda t=text, s=stop, **kw: FakeChild(kw, text=t, stop=s), max_depth=2).spawn(AgentDefinition(name="z", system_prompt="p"), "t", parent, depth=0)
        assert res.status == status
        if text == "raise":
            assert res.errors == ["RuntimeError: child exploded"] and res.summary == ""
    assert len(parent.session.meta["children"]) == 7


def test_builtin_skills_and_agents_are_consistent():
    skills = SkillIndex().load([builtin_skills_dir()], ["builtin"])
    agents = AgentIndex().load([builtin_agents_dir()], ["builtin"])
    assert skills.names() and agents.names()
    for name in skills.names():
        sk = skills.get(name)
        assert sk.description and sk.body().strip() and not sk.body().startswith("---"), name
    for name in agents.names():
        d = agents.get(name)
        assert d.description and d.system_prompt and all(s in skills for s in d.skills), (name, d.skills)


def test_nested_spawn_does_not_hold_the_parents_slot(workspace, provider, monkeypatch):
    import edgent.application.subagents as subagents_module

    monkeypatch.setattr(subagents_module, "SUBAGENT_QUEUE_TIMEOUT_S", 2.0)  # a regression fails fast instead of hanging
    parent = make_parent(workspace, provider)
    runner = SubagentRunner(None, max_depth=3, max_concurrent=1)

    class Nesting(FakeChild):
        def run_turn(self, task, parts=None, extra_context=None):
            depth = self.kw["depth"]
            if depth < 2:  # the child delegates while its parent's slot is still held
                inner = runner.spawn(AgentDefinition(name=f"d{depth}", system_prompt="p"), "t", parent, depth=depth)
                assert inner.status == "done", inner.errors
            return TurnResult(text=f"done at depth {depth}", stop_reason="done")

    runner.loop_factory = lambda **kw: Nesting(kw)
    result = runner.spawn(AgentDefinition(name="top", system_prompt="p"), "t", parent, depth=0)
    assert result.status == "done" and result.summary == "done at depth 1" and result.errors == []
    assert runner.peak_active == 2 and runner.active == 0 and len(parent.session.meta["children"]) == 2


def test_subagent_concurrency_cap(workspace, provider):
    parent = make_parent(workspace, provider)
    hold = threading.Event()
    runner = SubagentRunner(lambda **kw: FakeChild(kw, hold=hold), max_depth=2, max_concurrent=2)
    definition = AgentDefinition(name="w", system_prompt="p")
    threads = [threading.Thread(target=runner.spawn, args=(definition, "t", parent, 0)) for _ in range(4)]
    for t in threads:
        t.start()
    for _ in range(50):
        if runner.active == 2:
            break
        threading.Event().wait(0.02)
    assert runner.active == 2
    hold.set()
    for t in threads:
        t.join(5)
    assert runner.active == 0 and runner.peak_active == 2 and len(parent.session.meta["children"]) == 4
