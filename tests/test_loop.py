from __future__ import annotations

import json
from typing import Any

import pytest

from edgent.adapters.providers.base import ProviderError
from edgent.adapters.providers.mock import MockProvider
from edgent.application.budgets import Budget, BudgetExhausted, BudgetTracker
from edgent.application.compaction import CompactionLimits, Compactor
from edgent.application.hooks import HookOutcome, HookRunner
from edgent.application.loop import MAX_ITERATIONS_NOTE, AgentLoop, TurnResult
from edgent.application.permissions import Permissions
from edgent.application.sessions import Session
from edgent.application.skills import SkillIndex
from edgent.application.subagents import AgentIndex
from edgent.domain.capabilities import Capabilities
from edgent.domain.errors import ConfigurationError, ErrorCategory
from edgent.domain.types import Message, Reply, ToolCall, Usage
from edgent.domain.utils import FakeClock
from edgent.observability import Metrics, Tracer, TraceReader
from edgent.tools.registry import S, ToolRegistry, obj, opt

SYSTEM = "You are Edgent. Be brief."


def make_registry(workspace) -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool("read_file", "Read a workspace file.", obj({"path": opt(S, "Workspace-relative path.")}, ["path"]), permission="read")
    def read_file(ctx, a):
        return {"path": a["path"], "text": ctx.workspace.read_text(a["path"]), "untrusted": True}

    @reg.tool("write_file", "Write a workspace file.", obj({"path": S, "text": S}, ["path", "text"]), permission="write", side_effects=True, idempotent=False)
    def write_file(ctx, a):
        ctx.workspace.write_text(a["path"], a["text"])
        return {"written": a["path"]}

    @reg.tool("bash", "Run a command (fake).", obj({"command": S}, ["command"]), permission="compute")
    def bash(ctx, a):
        return {"stdout": f"ran {a['command']}"}

    @reg.tool("big", "Return a big output.", obj({}))
    def big(ctx, a):
        return {"data": "y" * 9000}

    @reg.tool("attach", "Attach an image.", obj({}), attaches_media=True)
    def attach(ctx, a):
        p = ctx.workspace.write_text("media/diagrams/x.png", "notapng")
        return {"_attach": [{"type": "image", "path": str(p), "text": "[image: media/diagrams/x.png]"}], "ok": True}

    @reg.tool("agent", "Delegate to a subagent.", obj({"agent": S, "task": S}, ["agent", "task"]), permission="spawn")
    def agent(ctx, a):
        return ctx.spawn(a["agent"], a["task"])

    @reg.tool("ask_user", "Ask the user.", obj({"question": S}, ["question"]), permission="interactive")
    def ask_user(ctx, a):
        return {"answer": ctx.ask_user(a["question"]) if ctx.ask_user else None}

    return reg


def make_loop(workspace, provider, session_id="s1", **kw) -> AgentLoop:
    reg = kw.pop("registry", None) or make_registry(workspace)
    session = Session.create(workspace, mode="chat", session_id=session_id)
    kw.setdefault("tracer", Tracer(workspace.path("traces"), session_id))
    return AgentLoop(provider, reg, workspace, session, system_prompt=SYSTEM, **kw)


def test_three_step_tool_conversation(workspace):
    workspace.write_text("sources/text/book.md", "A basis is a linearly independent spanning set.")
    provider = MockProvider()
    provider.push(
        MockProvider.tool_reply("read_file", {"path": "sources/text/book.md"}, text="Let me look."),
        MockProvider.tool_reply("write_file", {"path": "book/chapters/01.md", "text": "# Basis\n"}),
        "A basis is a minimal spanning set (book.md).",
    )
    events = []
    metrics = Metrics()
    loop = make_loop(workspace, provider, permissions=Permissions("auto", workspace=workspace), metrics=metrics, on_event=events.append)
    result = loop.run_turn("What is a basis? Write the chapter.")
    assert result.stop_reason == "done" and result.text == "A basis is a minimal spanning set (book.md)." and result.tool_calls == 2 and result.iterations == 3
    assert result.errors == [] and result.usage.prompt_tokens > 0 and result.to_dict()["stop_reason"] == "done"
    roles = [m.role for m in loop.session.messages]
    assert roles == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert loop.session.messages[2].name == "read_file" and "linearly independent" in loop.session.messages[2].text()
    assert workspace.read_text("book/chapters/01.md") == "# Basis\n"
    # the request the model saw: stable system prompt with the role marker, orientation, then the conversation
    req = provider.calls[-1]
    assert req.metadata == {"role": "main", "session_id": "s1"} and req.messages[0].text().startswith("[edgent-role: main]\nYou are Edgent.")
    assert [m.role for m in req.messages[:3]] == ["system", "system", "system"] and any(t["function"]["name"] == "read_file" for t in req.tools)
    assert provider.calls[0].messages[0].text() == req.messages[0].text()  # cache prefix is byte-identical
    assert [e["type"] for e in events] == ["model_call", "tool_call", "model_call", "tool_call", "model_call"]
    assert metrics.counters["model.calls"] == 3 and metrics.counters["tool.calls"] == 2
    kinds = [e.kind for e in TraceReader(loop.tracer.path)]
    assert kinds.count("model_call") == 3 and kinds.count("tool_call") == 2
    assert loop.session.usage_total.prompt_tokens == result.usage.prompt_tokens and Session.load(workspace, "s1").seq == 6
    assert loop.tool_context().session is loop.session and loop.tool_context().spawn == loop.spawn


def test_parallel_tool_calls_keep_order(workspace):
    provider = MockProvider()
    provider.push(MockProvider.tools_reply([("bash", {"command": "one"}), ("bash", {"command": "two"}), ("bash", {"command": "three"})]), "done")
    loop = make_loop(workspace, provider)
    result = loop.run_turn("run three things")
    assert result.tool_calls == 3 and result.stop_reason == "done"
    calls = loop.session.messages[1].tool_calls
    tools = loop.session.messages[2:5]
    assert [m.tool_call_id for m in tools] == [c.id for c in calls] and [json.loads(m.text())["stdout"] for m in tools] == ["ran one", "ran two", "ran three"]


def test_denied_permission_produces_error_and_loop_continues(workspace):
    provider = MockProvider()
    provider.push(MockProvider.tool_reply("write_file", {"path": "book/x.md", "text": "x"}), "I could not write; plan mode is read-only.")
    loop = make_loop(workspace, provider, permissions=Permissions("plan", workspace=workspace))
    result = loop.run_turn("write something")
    assert result.stop_reason == "done" and result.text.startswith("I could not write")
    err = json.loads(loop.session.messages[2].text())["error"]
    assert err["category"] == "permission_denied" and "denied by the permission policy" in err["message"] and result.errors and "permission_denied" in result.errors[0]
    assert not workspace.file_exists("book/x.md")
    # asking with no way to ask is also a typed error, not a crash
    provider.push(MockProvider.tool_reply("write_file", {"path": "book/y.md", "text": "y"}), "ok")
    loop2 = make_loop(workspace, provider, session_id="s2", permissions=Permissions("ask", workspace=workspace))
    loop2.run_turn("write")
    assert "no way to ask" in json.loads(loop2.session.messages[2].text())["error"]["message"]
    provider.push(MockProvider.tool_reply("write_file", {"path": "book/z.md", "text": "z"}), "ok")
    loop3 = make_loop(workspace, provider, session_id="s3", permissions=Permissions("ask", workspace=workspace), ask_permission=lambda spec, args: True)
    loop3.run_turn("write")
    assert workspace.read_text("book/z.md") == "z"


def test_pre_tool_use_command_hook_blocks(workspace):
    workspace.write_text(".edgent/hooks/block.py", "import sys, json\np = json.load(sys.stdin)\nsys.stderr.write('chapters are frozen: ' + p['arguments']['path'])\nsys.exit(2)\n")
    settings = {"hooks": {"PreToolUse": [{"matcher": "write_file", "command": "python3 .edgent/hooks/block.py", "timeout_s": 20}], "Stop": [{"command": "python3 -c \"print('bye')\""}]}}
    provider = MockProvider()
    provider.push(MockProvider.tools_reply([("write_file", {"path": "book/chapters/01.md", "text": "x"}), ("bash", {"command": "ls"})]), "understood")
    hooks = HookRunner(settings, workspace)
    stops = []
    hooks.register("Stop", lambda p: stops.append(p))
    starts = []
    hooks.register("SessionStart", lambda p: starts.append(p) or HookOutcome(message="remember: cite pages"))
    loop = make_loop(workspace, provider, hooks=hooks, permissions=Permissions("auto", workspace=workspace))
    result = loop.run_turn("write and list")
    assert result.stop_reason == "done" and result.tool_calls == 2
    blocked, ran = loop.session.messages[2], loop.session.messages[3]
    assert json.loads(blocked.text())["error"]["category"] == "permission_denied" and "chapters are frozen: book/chapters/01.md" in blocked.text()
    assert json.loads(ran.text())["stdout"] == "ran ls" and not workspace.file_exists("book/chapters/01.md")
    assert stops and stops[0]["stop_reason"] == "done" and starts[0]["source"] == "startup" and loop.session.meta["hook_messages"] == ["remember: cite pages"]
    assert "remember: cite pages" in provider.calls[0].messages[2].text()
    # a hook may rewrite arguments; the tool message says so
    hooks2 = HookRunner({}, workspace)
    hooks2.register("PreToolUse", lambda p: HookOutcome(updated_arguments={"command": p["arguments"]["command"] + " --safe"}) if p["tool_name"] == "bash" else None)
    hooks2.register("PostToolUse", lambda p: HookOutcome(message="checked"))
    provider.push(MockProvider.tool_reply("bash", {"command": "rm"}), "ok")
    loop2 = make_loop(workspace, provider, session_id="s2", hooks=hooks2)
    loop2.run_turn("x")
    m = loop2.session.messages[2]
    assert '"stdout": "ran rm --safe"' in m.text().replace("\n", "") or "ran rm --safe" in m.text()
    assert m.meta["arguments_updated_by_hook"] and m.text().endswith("[hook] checked")


def test_budget_max_iterations_and_exhaustion(workspace):
    provider = MockProvider(default=lambda req: MockProvider.tool_reply("bash", {"command": "again"}))
    loop = make_loop(workspace, provider, budget=BudgetTracker(Budget(max_iterations=3)))
    result = loop.run_turn("loop forever")
    assert result.stop_reason == "max_iterations" and result.iterations == 3 and result.tool_calls == 3 and result.text == MAX_ITERATIONS_NOTE
    assert loop.session.messages[-1].text() == MAX_ITERATIONS_NOTE and len(provider.calls) == 3
    # tool-call budget: exhaustion ends the turn with the limit message, never a silent stop
    loop2 = make_loop(workspace, provider, session_id="s2", budget=BudgetTracker(Budget(max_tool_calls=2)))
    result = loop2.run_turn("loop forever")
    assert result.stop_reason == "budget" and "max tool calls exceeded: observed 2 tool calls per turn" in result.text and "progress.md" in result.text
    assert result.errors and "max tool calls" in result.errors[0] and loop2.session.messages[-1].role == "assistant"
    # wall clock, with an injected clock
    clock = FakeClock()
    tracker = BudgetTracker(Budget(max_wall_s=10), clock=clock)
    tracker.start_turn()
    tracker.check("iteration")
    clock.advance(11)
    with pytest.raises(BudgetExhausted) as ei:
        tracker.check("iteration")
    assert ei.value.error.category is ErrorCategory.resource_constraint and ei.value.error.details["limit"] == "max wall s" and "EDGENT_MAX_WALL_S" in ei.value.error.remediation
    # the second turn starts fresh
    loop3 = make_loop(workspace, provider, session_id="s3", budget=BudgetTracker(Budget(max_iterations=2)))
    assert loop3.run_turn("a").stop_reason == "max_iterations" and loop3.run_turn("b").stop_reason == "max_iterations" and loop3.budget.turns == 2


def test_budget_stop_mid_call_leaves_no_dangling_tool_call(workspace):
    provider = MockProvider(default=lambda req: MockProvider.tool_reply("bash", {"command": "again"}))
    loop = make_loop(workspace, provider, budget=BudgetTracker(Budget(max_tool_calls=1)))
    result = loop.run_turn("loop")
    assert result.stop_reason == "budget" and "max tool calls exceeded" in result.text
    msgs = loop.session.messages
    # the second reply's call gets an 'interrupted' result before the harness note, so the conversation stays valid
    assert [m.role for m in msgs] == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    err = json.loads(msgs[4].text())["error"]
    assert err["code"] == "interrupted" and err["category"] == "cancelled" and msgs[4].tool_call_id == msgs[3].tool_calls[0].id and msgs[4].meta["interrupted"]
    assert loop.session.dangling_tool_calls() == [] and Session.load(workspace, "s1").dangling_tool_calls() == []
    provider.push("fine")
    assert loop.run_turn("continue").text == "fine"
    sent = provider.calls[-1].non_system()
    for i, m in enumerate(sent):  # what the provider saw: every tool call answered right after it
        if m.role == "assistant" and m.tool_calls:
            assert {t.tool_call_id for t in sent[i + 1 : i + 1 + len(m.tool_calls)]} == {tc.id for tc in m.tool_calls}


def test_resume_after_crash_repairs_dangling_tool_call(workspace):
    provider = MockProvider()
    provider.push(MockProvider.tool_reply("bash", {"command": "ls"}), "ok")
    loop = make_loop(workspace, provider)
    loop.run_turn("one")
    s = loop.session
    s.append(Message.user("two"))  # a crash right after the model asked for a tool: the call is persisted, its result never was
    s.append(Message.assistant("", [ToolCall("dangling", "bash", {"command": "pwd"})]))
    s.save()
    provider.push("resumed")
    resumed = AgentLoop.resume(workspace, "s1", provider=provider, registry=make_registry(workspace), system_prompt=SYSTEM)
    result = resumed.run_turn("three")
    assert result.stop_reason == "done" and result.text == "resumed"
    roles = [m.role for m in resumed.session.messages]
    assert roles == ["user", "assistant", "tool", "assistant", "user", "assistant", "tool", "user", "assistant"]
    repaired = resumed.session.messages[6]
    assert repaired.tool_call_id == "dangling" and json.loads(repaired.text())["error"]["code"] == "interrupted" and repaired.meta["seq"] == 7
    assert [m.role for m in provider.calls[-1].non_system()] == roles[:-1]


def test_turn_text_is_this_turns_text(workspace):
    provider = MockProvider()
    provider.push("first answer", Reply(text=""))
    loop = make_loop(workspace, provider)
    assert loop.run_turn("a").text == "first answer"
    result = loop.run_turn("b")  # an empty reply ends the turn with no text, never with a stale answer from an earlier turn
    assert result.stop_reason == "done" and result.text == ""


def test_schema_compaction_follows_the_byte_ceiling(workspace):
    roomy = MockProvider()
    roomy.push("ok")
    make_loop(workspace, roomy).run_turn("x")
    full = {t["function"]["name"]: t["function"] for t in roomy.calls[0].tools}
    assert full["read_file"]["parameters"]["properties"]["path"]["description"] == "Workspace-relative path."
    tight = MockProvider(caps=Capabilities(max_prompt_bytes=200_000))
    tight.push("ok")
    make_loop(workspace, tight, session_id="s2").run_turn("x")
    compact = {t["function"]["name"]: t["function"] for t in tight.calls[0].tools}
    assert "description" not in compact["read_file"]["parameters"]["properties"]["path"]


def test_spawn_budget_is_a_typed_tool_error(workspace, home):
    agents_dir = workspace.root / ".edgent" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "librarian.md").write_text(LIBRARIAN)
    provider = MockProvider()
    provider.on_role("librarian", lambda req: Reply(text="found"))
    main_calls: list[Any] = []
    provider.on_role("main", lambda req: main_calls.append(req) or (MockProvider.tool_reply("agent", {"agent": "librarian", "task": "go"}) if len(main_calls) < 3 else Reply(text="ok")))
    loop = make_loop(workspace, provider, agents=AgentIndex.discover(workspace, home), permissions=Permissions("auto", workspace=workspace), budget=BudgetTracker(Budget(max_spawns=1)))
    result = loop.run_turn("delegate twice")
    assert result.stop_reason == "done" and json.loads(loop.session.messages[2].text())["summary"] == "found"
    err = json.loads(loop.session.messages[4].text())["error"]
    assert err["category"] == "resource_constraint" and err["code"] == "limit:max spawns" and "progress.md" in err["remediation"]
    assert len(loop.session.meta["children"]) == 1


def test_budget_configuration():
    b = Budget.from_env(environ={"EDGENT_MAX_ITERATIONS": "5", "EDGENT_MAX_COST_USD": "1.5"})
    assert b.max_iterations == 5 and b.max_cost_usd == 1.5 and b.max_tool_calls == 300 and b.sources["max_iterations"] == "setting" and b.sources["max_tool_calls"] == "default"
    assert b.limit("max_iterations").source == "setting" and b.limit("max_iterations").setting_name == "EDGENT_MAX_ITERATIONS" and "model calls per turn" in b.limit("max_iterations").describe()
    for env in ({"EDGENT_MAX_ITERATIONS": "0"}, {"EDGENT_MAX_WALL_S": "abc"}, {"EDGENT_MAX_SPAWNS": "5000"}):
        with pytest.raises(ConfigurationError):
            Budget.from_env(environ=env)
    s = Budget.from_settings({"budget": {"max_iterations": 9, "max_wall_s": 12.5}, "max_depth": 1}, base=b)
    assert s.max_iterations == 9 and s.max_wall_s == 12.5 and s.max_depth == 1 and s.max_cost_usd == 1.5 and s.sources["max_depth"] == "setting"
    for bad in ({"budget": {"max_iterations": "9"}}, {"budget": {"max_iterations": 0}}, {"max_depth": 99}, {"budget": {"max_cost_usd": True}}):
        with pytest.raises(ConfigurationError):
            Budget.from_settings(bad)
    t = BudgetTracker(Budget(max_cost_usd=1.0, max_total_tokens=100, max_spawns=1))
    t.start_turn()
    t.charge_model(Usage(50, 10, cost_usd=0.4))
    t.charge_tools(3)
    t.charge_spawn()
    assert t.remaining()["cost_usd"] == pytest.approx(0.6) and t.remaining()["total_tokens"] == 40 and t.remaining()["spawns"] == 0 and t.to_dict()["used"]["tool_calls"] == 3
    with pytest.raises(BudgetExhausted):
        t.check("spawn")
    t.check("iteration")
    t.charge_usage(Usage(40, 0, cost_usd=0.6))
    with pytest.raises(BudgetExhausted) as ei:
        t.check("tool")
    assert "max cost usd" in ei.value.error.message
    assert BudgetTracker().remaining()["cost_usd"] is None


def test_provider_retry_then_success(workspace):
    provider = MockProvider()
    attempts = {"n": 0}

    def flaky(req):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ProviderError("overloaded", status=529, retryable=True)
        return Reply(text="finally")

    provider.push(flaky, flaky, flaky)
    slept = []
    loop = make_loop(workspace, provider, sleep=slept.append)
    result = loop.run_turn("hello")
    assert result.stop_reason == "done" and result.text == "finally" and slept == [1.0, 2.0] and result.iterations == 1
    assert [e.kind for e in TraceReader(loop.tracer.path)].count("retry") == 2
    # three failures in a row exhaust the retries
    provider.push(flaky, flaky, flaky, flaky)
    attempts["n"] = -10
    loop2 = make_loop(workspace, provider, session_id="s2", sleep=lambda s: None)
    assert loop2.run_turn("hello").stop_reason == "error"


def test_non_retryable_error_ends_turn_and_persists(workspace):
    provider = MockProvider(fail_on_roles={"main"})
    loop = make_loop(workspace, provider)
    result = loop.run_turn("hello")
    assert result.stop_reason == "error" and "mock failure for role main" in result.text and result.errors == ["mock failure for role main"]
    saved = Session.load(workspace, "s1")
    assert [m.role for m in saved.messages] == ["user", "assistant"] and "mock failure" in saved.messages[1].text()
    assert any(e.kind == "error" for e in TraceReader(loop.tracer.path))
    # an unexpected exception inside the loop is reported the same way, with the session persisted
    broken = MockProvider()
    broken.push(lambda req: (_ for _ in ()).throw(RuntimeError("adapter bug")))
    loop2 = make_loop(workspace, broken, session_id="s2")
    result = loop2.run_turn("x")
    assert result.stop_reason == "error" and "RuntimeError: adapter bug" in result.text and Session.load(workspace, "s2").seq == 2


def test_compaction_triggers_under_pressure_and_transcript_is_complete(workspace):
    workspace.write_text("sources/text/book.md", "word " * 400)
    provider = MockProvider(caps=Capabilities(context_tokens=600, prompt_caching=True))
    provider.push(*([MockProvider.tool_reply("read_file", {"path": "sources/text/book.md"})] * 6), "Summary: the book repeats the word 'word'.")
    compactor = Compactor(None, CompactionLimits(keep_recent_messages=2, block_size=4, clear_tool_result_over_chars=200))
    events = []
    loop = make_loop(workspace, provider, compactor=compactor, on_event=events.append)
    result = loop.run_turn("read the book six times, then summarise")
    assert result.stop_reason == "done" and result.text.startswith("Summary:") and result.tool_calls == 6
    s = loop.session
    assert s.notes and all(n["level"] == 0 for n in s.notes) and len(s.messages) < 14
    assert any(m.meta.get("cleared") for m in Session.load(workspace, "s1").messages) or len(s.messages) <= 6
    assert len(s.transcript_path.read_text().splitlines()) == 14  # user + 6 x (assistant, tool) + final assistant
    assert any(e.kind == "compaction" and e.data["reason"] == "context pressure" for e in TraceReader(loop.tracer.path))
    assert any(e["type"] == "compaction" for e in events)
    assert "<session-notes>" in provider.calls[-1].system_text() and "Asked: read the book six times" in provider.calls[-1].system_text()
    assert s.recall("word")  # verbatim recall still works over the whole transcript


def test_provider_prompt_refusal_forces_compaction(workspace):
    provider = MockProvider()
    refusals = {"n": 0}

    def refuse_once(req):
        refusals["n"] += 1
        raise ProviderError("prompt too large", retryable=False, category=ErrorCategory.truncated)

    provider.push(*([MockProvider.tool_reply("bash", {"command": "echo " + "x" * 300})] * 4), refuse_once, "done after compaction")
    compactor = Compactor(None, CompactionLimits(keep_recent_messages=2, block_size=2, clear_tool_result_over_chars=100))
    loop = make_loop(workspace, provider, compactor=compactor)
    result = loop.run_turn("go")
    assert result.stop_reason == "done" and result.text == "done after compaction" and refusals["n"] == 1 and result.iterations == 5
    forced = [e.data for e in TraceReader(loop.tracer.path) if e.kind == "compaction" and e.data["forced"]]
    assert forced and forced[0]["reason"] == "provider refused the prompt size" and forced[0]["notes_added"] >= 1 and loop.session.notes
    # a second refusal in a row is not retried again
    provider.push(refuse_once, refuse_once, "unreachable")
    loop3 = make_loop(workspace, provider, session_id="s3", compactor=compactor)
    assert loop3.run_turn("go").stop_reason == "error" and refusals["n"] == 3
    # when even forced compaction cannot fit, the turn ends with the limit message
    tiny = MockProvider(caps=Capabilities(max_prompt_bytes=200))
    loop2 = make_loop(workspace, tiny, session_id="s2")
    result = loop2.run_turn("hello " * 100)
    assert result.stop_reason == "error" and "prompt size exceeded" in result.text and Session.load(workspace, "s2").messages[-1].role == "assistant"


def test_kill_and_resume(workspace):
    provider = MockProvider()
    provider.push(MockProvider.tool_reply("bash", {"command": "ls"}), "first turn done")
    loop = make_loop(workspace, provider)
    loop.run_turn("turn one")
    assert loop.session.seq == 4
    del loop
    provider.push("second turn done")
    resumed = AgentLoop.resume(workspace, "s1", provider=provider, registry=make_registry(workspace), system_prompt=SYSTEM)
    assert len(resumed.session.messages) == 4 and resumed.session.messages[0].text() == "turn one"
    result = resumed.run_turn("turn two")
    assert result.text == "second turn done" and resumed.session.seq == 6
    saved = Session.load(workspace, "s1")
    assert [m.role for m in saved.messages] == ["user", "assistant", "tool", "assistant", "user", "assistant"]
    assert [m.meta["seq"] for m in saved.messages] == [1, 2, 3, 4, 5, 6] and len(saved.transcript_path.read_text().splitlines()) == 6
    # the resumed request still carried the whole conversation
    assert [m.role for m in provider.calls[-1].non_system()] == ["user", "assistant", "tool", "assistant", "user"]


LIBRARIAN = """---
name: librarian
description: Finds passages. Returns quotes with locators.
tools: read_file, bash
maxTurns: 3
---
Find evidence and return only the quotes.
"""


def test_spawn_subagent_returns_summary_only(workspace, home):
    agents_dir = workspace.root / ".edgent" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "librarian.md").write_text(LIBRARIAN)
    provider = MockProvider()
    child_requests = []

    def librarian(req):
        child_requests.append(req)
        if len(child_requests) == 1:
            return MockProvider.tool_reply("bash", {"command": "grep basis"})
        return Reply(text="QUOTE: 'a basis is...' (book:p.12)")

    provider.on_role("librarian", librarian)
    parent_requests = []

    def main(req):
        parent_requests.append(req)
        if len(parent_requests) == 1:
            return MockProvider.tool_reply("agent", {"agent": "librarian", "task": "find the definition of basis"})
        return Reply(text="The book defines a basis at p.12.")

    provider.on_role("main", main)
    loop = make_loop(workspace, provider, agents=AgentIndex.discover(workspace, home), skills=SkillIndex(), permissions=Permissions("auto", workspace=workspace))
    result = loop.run_turn("where is basis defined?")
    assert result.stop_reason == "done" and result.text == "The book defines a basis at p.12."
    tool_msg = json.loads(loop.session.messages[2].text())
    assert tool_msg["status"] == "done" and tool_msg["summary"] == "QUOTE: 'a basis is...' (book:p.12)" and tool_msg["turns"] == 2 and tool_msg["name"] == "librarian"
    assert "grep basis" not in loop.session.messages[2].text()  # the child's exploration stays out of the parent
    child_id = tool_msg["session_id"]
    assert loop.session.meta["children"] == [child_id]
    child = Session.load(workspace, child_id)
    assert child.mode == "subagent" and child.parent_id == "s1" and [m.role for m in child.messages] == ["user", "assistant", "tool", "assistant"]
    assert child.messages[0].text() == "find the definition of basis"
    first = child_requests[0]
    assert first.metadata["role"] == "librarian" and first.messages[0].text().startswith("[edgent-role: librarian]\nFind evidence")
    assert {t["function"]["name"] for t in first.tools} == {"read_file", "bash"} and len(first.non_system()) == 1
    assert loop.budget.spawns == 1 and loop.budget.total_tokens > result.usage.total_tokens
    kinds = [(e.kind, e.data.get("phase")) for e in TraceReader(loop.tracer.path) if e.kind == "subagent"]
    assert kinds == [("subagent", "start"), ("subagent", "stop")]
    # unknown agents and the depth limit are typed errors the model can read
    provider.push(MockProvider.tool_reply("agent", {"agent": "nobody", "task": "x"}), "ok")
    loop.run_turn("again")
    err = json.loads(loop.session.messages[-2].text())["error"]
    assert err["category"] == "not_found" and "librarian" in err["message"]
    provider.push(MockProvider.tool_reply("agent", {"agent": "librarian", "task": "x"}), "ok")
    deep = make_loop(workspace, provider, session_id="deep", agents=AgentIndex.discover(workspace, home), depth=2, permissions=Permissions("auto", workspace=workspace))
    deep.run_turn("go")
    err = json.loads(deep.session.messages[2].text())["error"]
    assert err["category"] == "permission_denied" and "max subagent depth" in err["message"] and "children" not in deep.session.meta


def test_text_protocol_provider_round_trip(workspace):
    provider = MockProvider(caps=Capabilities(native_tools=False, system_channel=True, context_tokens=100_000))
    provider.push('Reading.\n```tool\n{"name": "bash", "arguments": {"command": "ls"}}\n```', "```tool\nnot json\n```", "All done.")
    loop = make_loop(workspace, provider)
    result = loop.run_turn("list files")
    assert result.stop_reason == "done" and result.text == "All done." and result.tool_calls == 1 and result.iterations == 3
    msgs = loop.session.messages
    assert msgs[1].text() == "Reading." and msgs[1].tool_calls[0].name == "bash" and msgs[2].role == "tool" and "ran ls" in msgs[2].text()
    assert msgs[4].role == "user" and msgs[4].text().startswith("[harness] Your tool block could not be parsed") and any("parse error" in e for e in result.errors)
    second = provider.calls[1]
    assert second.tools == [] and "TOOL RESULT bash" in second.messages[-1].text() and "```tool" in second.messages[-2].text() and "Tool calls (text protocol)" in second.system_text()


def test_image_attachments_follow_capabilities(workspace):
    with_images = MockProvider(caps=Capabilities(images=True))
    with_images.push(MockProvider.tool_reply("attach", {}), "I see a diagram.")
    loop = make_loop(workspace, with_images)
    result = loop.run_turn("show me")
    tool_msg = loop.session.messages[2]
    assert tool_msg.media() and tool_msg.media()[0].type == "image" and tool_msg.media()[0].path.endswith("media/diagrams/x.png")
    assert result.attachments and result.attachments[0].type == "image" and result.to_dict()["attachments"][0]["type"] == "image"
    assert with_images.calls[-1].messages[-1].media()  # the provider received the image part
    no_images = MockProvider(caps=Capabilities(images=False))
    no_images.push(MockProvider.tool_reply("attach", {}), "I read a stand-in.")
    loop2 = make_loop(workspace, no_images, session_id="s2")
    loop2.run_turn("show me")
    tool_msg = loop2.session.messages[2]
    assert not tool_msg.media() and "[image: media/diagrams/x.png]" in tool_msg.plain_text() and len(tool_msg.content) == 2


def test_big_output_pointer_and_cancel(workspace):
    provider = MockProvider()
    provider.push(MockProvider.tool_reply("big", {}), "ok")
    loop = make_loop(workspace, provider)
    loop.run_turn("big")
    m = loop.session.messages[2]
    assert m.meta["persisted_path"].startswith(".edgent/tool-results/") and json.loads(m.text())["compacted"] is True
    provider.push(MockProvider.tool_reply("bash", {"command": "ls"}), "never used")
    holder = {}
    loop2 = make_loop(workspace, provider, session_id="s2", on_event=lambda e: holder["loop"].cancel() if e["type"] == "tool_call" else None)
    holder["loop"] = loop2
    result = loop2.run_turn("x")
    assert result.stop_reason == "cancelled" and result.tool_calls == 1 and len(provider.calls) == 3 and provider.script == [Reply(text="never used")]
    assert Session.load(workspace, "s2").seq == 3
    assert TurnResult("t").to_dict()["usage"]["prompt_tokens"] == 0
