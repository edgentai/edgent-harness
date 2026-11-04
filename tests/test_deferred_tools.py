"""Deferred tools: registered, named in a short index, and loaded on demand with tool_search (offline, MockProvider)."""
from __future__ import annotations

from typing import Any

import pytest

from edgent.adapters.providers.mock import MockProvider
from edgent.application.context import ContextAssembler
from edgent.application.loop import AgentLoop
from edgent.application.permissions import Permissions
from edgent.application.sessions import Session
from edgent.application.subagents import AgentDefinition, AgentIndex, SubagentRunner
from edgent.observability import Tracer, TraceReader
from edgent.tools import core
from edgent.tools.core import build_core_registry, tool_search_spec
from edgent.tools.registry import (
    ACTIVATED_TOOLS_KEY,
    DEFERRED_INDEX_HEADER,
    TOOL_SEARCH_NAME,
    IdempotencyStore,
    S,
    ToolContext,
    ToolRegistry,
    ToolSpec,
    obj,
    opt,
)

FAKE = ("mcp__fake__alpha", "mcp__fake__beta", "mcp__fake__danger_delete")
DESCRIPTIONS = {
    "mcp__fake__alpha": "Alpha fetches one page as Markdown. It also follows redirects and strips scripts.",
    "mcp__fake__beta": "Beta takes a screenshot of the current page. Returns a PNG path.",
    "mcp__fake__danger_delete": "Danger deletes every record in the account. Irreversible.",
}


def fake_spec(name: str, permission: str = "read", origin: str = "connector:fake") -> ToolSpec:
    def run(ctx: ToolContext, a: dict[str, Any]) -> dict[str, Any]:
        return {"tool": name, "x": a.get("x", ""), "untrusted": True}

    schema = obj({"x": opt(S, "An argument the fake connector echoes back.")})
    return ToolSpec(name=name, description=DESCRIPTIONS.get(name, f"{name} does a thing."), parameters=schema, fn=run,
                    permission=permission, deferred=True, origin=origin, group="custom")


def make_registry(workspace: Any) -> ToolRegistry:
    reg = ToolRegistry(idempotency=IdempotencyStore(workspace.path("state") / "state.db"))
    build_core_registry(reg, include={"read_file", "agent"})
    for name in FAKE:
        reg.register(fake_spec(name, permission="write" if "danger" in name else "read"))
    return reg


def tool_names(request: Any) -> list[str]:
    return sorted(t["function"]["name"] for t in request.tools)


def make_loop(workspace: Any, provider: MockProvider, reg: ToolRegistry, session_id: str = "s1", **kw: Any) -> AgentLoop:
    session = Session.create(workspace, mode="chat", session_id=session_id)
    kw.setdefault("permissions", Permissions("bypass", workspace=workspace))
    kw.setdefault("tracer", Tracer(workspace.path("traces"), session_id))
    return AgentLoop(provider, reg, workspace, session, system_prompt="You are Edgent.", **kw)


# ----------------------------------------------------------------------------- registry contract
def test_registry_defers_schemas_and_indexes_names(workspace):
    reg = make_registry(workspace)
    assert TOOL_SEARCH_NAME in reg.names()  # the first deferred tool brought tool_search with it
    assert reg.deferred_names() == sorted(FAKE) and reg.deferred_names({"mcp__fake__beta", "read_file"}) == ["mcp__fake__beta"]
    sent = [s["function"]["name"] for s in reg.openai_schemas()]
    assert not set(FAKE) & set(sent) and {"read_file", "agent", TOOL_SEARCH_NAME} <= set(sent)
    loaded = [s["function"]["name"] for s in reg.openai_schemas(activated={"mcp__fake__alpha"})]
    assert "mcp__fake__alpha" in loaded and "mcp__fake__beta" not in loaded
    assert reg.openai_schemas(allowlist={"read_file"}, activated={"mcp__fake__alpha"})[0]["function"]["name"] == "read_file"
    assert len(reg.openai_schemas(allowlist={"read_file"}, activated={"mcp__fake__alpha"})) == 1
    assert {t["name"] for t in reg.mcp_manifest()["tools"]} >= set(FAKE)  # MCP hosts do their own deferral

    index = reg.deferred_index()
    lines = index.splitlines()
    assert lines[0] == DEFERRED_INDEX_HEADER and "### connector:fake" in lines
    assert "- mcp__fake__alpha: Alpha fetches one page as Markdown." in lines  # the first sentence only
    assert "redirects" not in index and len(index) < 1000
    narrowed = reg.deferred_index(allowlist={"mcp__fake__beta"}, activated=set())
    assert "mcp__fake__beta" in narrowed and "mcp__fake__alpha" not in narrowed
    assert "mcp__fake__alpha" not in reg.deferred_index(activated={"mcp__fake__alpha"})
    assert reg.deferred_index(activated=set(FAKE)) == "" and ToolRegistry().deferred_index() == ""


def test_deferred_tool_still_runs_unactivated_and_can_be_unregistered(workspace, ctx):
    reg = make_registry(workspace)
    res = reg.call("mcp__fake__alpha", {"x": "1"}, ctx)
    assert res.ok and res.output["tool"] == "mcp__fake__alpha"
    bad = reg.call("mcp__fake__alpha", {"y": 1}, ctx)
    assert not bad.ok and bad.error.code == "invalid_input"  # still validated against its schema
    assert reg.unregister("mcp__fake__alpha") is True and reg.unregister("mcp__fake__alpha") is False
    assert "mcp__fake__alpha" not in reg.names() and "mcp__fake__alpha" not in reg.deferred_index()
    other = ToolRegistry()
    other.include(reg)
    assert TOOL_SEARCH_NAME in other.names() and "mcp__fake__beta" in other.deferred_names()


def test_index_stays_under_its_cap(workspace):
    reg = ToolRegistry()
    for server in ("firecrawl", "playwright", "posthog"):
        for i in range(40):
            reg.register(fake_spec(f"mcp__{server}__tool_{i:02d}", origin=f"connector:{server}"))
    full = reg.deferred_index()
    assert len(full) <= 12_000 and full.count("\n- ") == 120  # over the cap with descriptions: names only, every tool
    tiny = reg.deferred_index(max_chars=600)
    assert len(tiny) <= 600 and tiny.splitlines()[-1].startswith("- … ") and "more deferred tools" in tiny
    listed = sum(1 for line in tiny.splitlines() if line.startswith("- mcp__"))
    assert f"- … {120 - listed} more" in tiny
    long = ToolSpec(name="mcp__x__y", description="word " * 80, parameters=obj({}), fn=lambda c, a: {}, deferred=True)
    assert len(long.first_sentence()) <= 100 and long.first_sentence().endswith("…")


def test_stable_prompt_puts_the_tools_index_after_the_agents_index(workspace):
    s = Session.create(workspace, session_id="ctx")
    a = ContextAssembler()
    msgs = a.assemble(s, workspace, "SYSTEM", "## Skills\n- a: b", "## Subagents\n- c: d", tools_index="## Deferred tools\n- e")
    text = msgs[0].text()
    assert text.index("## Skills") < text.index("## Subagents") < text.index("## Deferred tools")
    req = a.build_request(s, workspace, system_prompt="P", tool_schemas=[], effort="low", max_tokens=1, tools_index="## Deferred tools\n- e")
    assert req.messages[0].text().endswith("## Deferred tools\n- e")
    assert ContextAssembler.stable_prompt("P", "", "", "") == "P"


# ----------------------------------------------------------------------------- tool_search
def search_ctx(ctx: ToolContext, reg: ToolRegistry, session: Any = None, allowlist: set[str] | None = None) -> ToolContext:
    ctx.extra.update({"registry": reg, "allowlist": allowlist})
    ctx.session = session
    return ctx


def test_tool_search_select_records_activation(workspace, ctx):
    reg = make_registry(workspace)
    session = Session.create(workspace, session_id="ts")
    search_ctx(ctx, reg, session)
    res = reg.call(TOOL_SEARCH_NAME, {"query": "select:mcp__fake__alpha, nope ,mcp__fake__alpha,read_file"}, ctx)
    assert res.ok, res.error
    out = res.output
    assert [t["name"] for t in out["tools"]] == ["mcp__fake__alpha", "read_file"] and out["not_found"] == ["nope"]
    alpha = out["tools"][0]
    assert alpha["origin"] == "connector:fake" and alpha["parameters"]["properties"]["x"] == {"type": "string"}  # compact
    assert alpha["description"].startswith("Alpha fetches") and set(alpha) == {"name", "description", "parameters", "origin"}
    assert out["activated"] == ["mcp__fake__alpha"] and out["remaining_deferred"] == 2  # read_file was never deferred
    assert Session.load(workspace, "ts").meta[ACTIVATED_TOOLS_KEY] == ["mcp__fake__alpha"]  # persisted before returning
    again = reg.call(TOOL_SEARCH_NAME, {"query": "SELECT:MCP__FAKE__BETA"}, ctx).output  # case-insensitive fallback
    assert again["activated"] == ["mcp__fake__beta"] and again["remaining_deferred"] == 1
    assert Session.load(workspace, "ts").meta[ACTIVATED_TOOLS_KEY] == ["mcp__fake__alpha", "mcp__fake__beta"]


def test_tool_search_keywords_rank_deferred_first(workspace, ctx):
    reg = make_registry(workspace)

    @reg.tool("snap_note", "Write a note about a screenshot you took.", obj({}))
    def snap_note(c, a):
        return {}

    search_ctx(ctx, reg, Session.create(workspace, session_id="kw"))
    out = reg.call(TOOL_SEARCH_NAME, {"query": "Screenshot"}, ctx).output
    assert [t["name"] for t in out["tools"]] == ["mcp__fake__beta", "snap_note"] and out["activated"] == ["mcp__fake__beta"]
    both = reg.call(TOOL_SEARCH_NAME, {"query": "page redirects"}, ctx).output
    assert [t["name"] for t in both["tools"]] == ["mcp__fake__alpha"] and "note" not in both  # every word must match
    named = reg.call(TOOL_SEARCH_NAME, {"query": "fake", "max_results": 2}, ctx).output
    assert len(named["tools"]) == 2 and all(t["name"].startswith("mcp__fake__") for t in named["tools"])
    partial = reg.call(TOOL_SEARCH_NAME, {"query": "screenshot zebra"}, ctx).output
    assert "note" in partial and partial["tools"][0]["name"] == "mcp__fake__beta"
    none = reg.call(TOOL_SEARCH_NAME, {"query": "zebra"}, ctx).output
    assert none["tools"] == [] and none["activated"] == [] and "hint" in none
    assert TOOL_SEARCH_NAME not in [t["name"] for t in reg.call(TOOL_SEARCH_NAME, {"query": "tool"}, ctx).output["tools"]]
    bad = reg.call(TOOL_SEARCH_NAME, {"query": "x", "max_results": 50}, ctx)
    assert not bad.ok and bad.error.code == "invalid_input"


def test_tool_search_respects_the_allowlist_and_context(workspace, ctx):
    reg = make_registry(workspace)
    search_ctx(ctx, reg, Session.create(workspace, session_id="al"), allowlist={TOOL_SEARCH_NAME, "mcp__fake__alpha"})
    out = reg.call(TOOL_SEARCH_NAME, {"query": "select:mcp__fake__alpha,mcp__fake__beta"}, ctx).output
    assert [t["name"] for t in out["tools"]] == ["mcp__fake__alpha"] and out["not_found"] == ["mcp__fake__beta"]
    assert out["remaining_deferred"] == 0
    assert reg.call(TOOL_SEARCH_NAME, {"query": "screenshot"}, ctx).output["tools"] == []
    # host mode: no session, so the schemas come back but nothing is recorded
    search_ctx(ctx, reg, None)
    host = reg.call(TOOL_SEARCH_NAME, {"query": "select:mcp__fake__beta"}, ctx).output
    assert host["tools"][0]["name"] == "mcp__fake__beta" and host["activated"] == [] and "session_note" in host
    assert host["tools"][0]["parameters"]["properties"]["x"]["description"]  # the only copy of the schema: kept whole
    ctx.extra.pop("registry")
    missing = reg.call(TOOL_SEARCH_NAME, {"query": "select:mcp__fake__beta"}, ctx)
    assert not missing.ok and missing.error.code == "unavailable" and missing.error.remediation
    search_ctx(ctx, reg, None)
    empty = reg.call(TOOL_SEARCH_NAME, {"query": "select: , "}, ctx)
    assert not empty.ok and empty.error.code == "invalid_input"
    many = ",".join(f"t{i}" for i in range(25))
    over = reg.call(TOOL_SEARCH_NAME, {"query": f"select:{many}"}, ctx).error
    assert over.code.startswith("limit") and "20" in over.message and over.remediation


def test_tool_search_contract():
    spec = tool_search_spec()
    assert spec.name == TOOL_SEARCH_NAME and spec.permission == "read" and spec.group == "core" and not spec.deferred
    assert spec.parameters["additionalProperties"] is False and spec.parameters["required"] == ["query"]
    assert spec.parameters["properties"]["max_results"]["minimum"] == 1 and spec.parameters["properties"]["max_results"]["maximum"] == 20
    assert spec.when_not_to_use and "Deferred tools" in spec.description and "select:" in spec.description
    assert TOOL_SEARCH_NAME not in build_core_registry().names()  # the 17 general tools stay as they are
    assert build_core_registry(include={TOOL_SEARCH_NAME}).names() == [TOOL_SEARCH_NAME]
    assert TOOL_SEARCH_NAME not in core.CORE_TOOL_NAMES and TOOL_SEARCH_NAME in core.OPTIONAL_CORE_TOOL_NAMES


# ----------------------------------------------------------------------------- the loop
def test_loop_sends_deferred_schemas_only_after_tool_search(workspace):
    reg = make_registry(workspace)
    provider = MockProvider()
    provider.push(
        MockProvider.tool_reply(TOOL_SEARCH_NAME, {"query": "select:mcp__fake__alpha,mcp__fake__beta"}),
        MockProvider.tool_reply("mcp__fake__alpha", {"x": "page"}),
        "Fetched the page.",
    )
    loop = make_loop(workspace, provider, reg, session_id="loop1")
    turn = loop.run_turn("fetch the page")
    assert turn.stop_reason == "done" and not turn.errors, turn.errors
    first, second, third = provider.calls
    assert tool_names(first) == sorted(["agent", "read_file", TOOL_SEARCH_NAME])
    system = first.messages[0].text()
    assert DEFERRED_INDEX_HEADER in system and all(f"- {n}:" in system for n in FAKE)
    assert tool_names(second) == sorted(["agent", "read_file", TOOL_SEARCH_NAME, "mcp__fake__alpha", "mcp__fake__beta"])
    later = second.messages[0].text()
    assert "- mcp__fake__danger_delete:" in later and "mcp__fake__alpha" not in later and "mcp__fake__beta" not in later
    result = [m for m in third.messages if m.role == "tool"][-1]
    assert '"tool":"mcp__fake__alpha"' in result.text().replace(" ", "")
    # the activation is session state: it survives a reload and a resumed loop sends the same schemas
    assert Session.load(workspace, "loop1").meta[ACTIVATED_TOOLS_KEY] == ["mcp__fake__alpha", "mcp__fake__beta"]
    provider.push("Still here.")
    resumed = AgentLoop.resume(workspace, "loop1", provider=provider, registry=reg, system_prompt="You are Edgent.", permissions=Permissions("bypass"))
    resumed.run_turn("again")
    assert {"mcp__fake__alpha", "mcp__fake__beta"} <= set(tool_names(provider.calls[-1])) and "mcp__fake__danger_delete" not in tool_names(provider.calls[-1])
    # activating everything removes the index section entirely
    provider.push(MockProvider.tool_reply(TOOL_SEARCH_NAME, {"query": "select:mcp__fake__danger_delete"}), "ok")
    resumed.run_turn("load the rest")
    assert DEFERRED_INDEX_HEADER not in provider.calls[-1].messages[0].text() and "mcp__fake__danger_delete" in tool_names(provider.calls[-1])


def test_loop_without_tool_search_shows_no_index(workspace):
    reg = make_registry(workspace)
    provider = MockProvider()
    provider.push("hello")
    loop = make_loop(workspace, provider, reg, session_id="loop2", allowlist={"read_file", "mcp__fake__alpha"})
    loop.run_turn("hi")
    req = provider.calls[0]
    assert tool_names(req) == ["read_file"] and DEFERRED_INDEX_HEADER not in req.messages[0].text()
    ctx = loop.tool_context()
    assert ctx.extra["registry"] is reg and ctx.extra["allowlist"] == {"read_file", "mcp__fake__alpha"}


# ----------------------------------------------------------------------------- subagents
def test_subagent_wildcard_tools_and_its_own_activations(workspace):
    reg = make_registry(workspace)
    agents = AgentIndex()
    agents.add(AgentDefinition(name="scout", description="Uses the fake connector.", tools=["mcp__fake__*", TOOL_SEARCH_NAME], system_prompt="You scout."))
    provider = MockProvider()
    provider.push(
        MockProvider.tool_reply(TOOL_SEARCH_NAME, {"query": "select:mcp__fake__alpha"}),  # parent activates alpha
        MockProvider.tool_reply("agent", {"agent": "scout", "task": "take a screenshot"}),
        MockProvider.tool_reply(TOOL_SEARCH_NAME, {"query": "select:mcp__fake__beta"}),  # child activates beta
        MockProvider.tool_reply("mcp__fake__beta", {"x": "shot"}),
        "Screenshot taken.",  # child's report
        "The scout took the screenshot.",  # parent's answer
    )
    loop = make_loop(workspace, provider, reg, session_id="parent", agents=agents)
    turn = loop.run_turn("screenshot please")
    assert turn.stop_reason == "done" and not turn.errors, turn.errors
    child = [r for r in provider.calls if r.metadata.get("role") == "scout"]
    parent = [r for r in provider.calls if r.metadata.get("role") == "main"]
    assert len(child) == 3 and len(parent) == 3
    assert tool_names(child[0]) == [TOOL_SEARCH_NAME]  # the parent's activation of alpha is not inherited
    assert all(f"- {n}:" in child[0].messages[0].text() for n in FAKE)
    assert tool_names(child[1]) == ["mcp__fake__beta", TOOL_SEARCH_NAME]
    assert "mcp__fake__alpha" in tool_names(parent[-1]) and "mcp__fake__beta" not in tool_names(parent[-1])  # nor shared back
    start = next(e.data for e in TraceReader(loop.tracer.path) if e.kind == "subagent" and e.data.get("phase") == "start")
    assert start["allowlist"] == sorted([*FAKE, TOOL_SEARCH_NAME])
    child_id = Session.load(workspace, "parent").meta["children"][0]
    assert Session.load(workspace, child_id).meta[ACTIVATED_TOOLS_KEY] == ["mcp__fake__beta"]


def test_child_allowlist_expands_patterns_against_the_parent_registry(workspace):
    reg = make_registry(workspace)
    provider = MockProvider()
    parent = make_loop(workspace, provider, reg, session_id="p2")
    runner = SubagentRunner(lambda **kw: None, max_depth=2)

    def allow(tools: list[str] | None, disallowed: list[str] | None = None) -> set[str]:
        return runner.child_allowlist(AgentDefinition(name="x", tools=tools, disallowed_tools=disallowed or []), parent, depth=0)

    assert allow(["mcp__fake__*", TOOL_SEARCH_NAME]) == {*FAKE, TOOL_SEARCH_NAME}
    assert allow(["mcp__fake", "read_file"]) == {*FAKE, "read_file"}  # a bare server name means all its tools
    assert allow(["mcp__absent__*", "mcp__absent", "read_file"]) == {"read_file"}  # an absent connector adds nothing
    assert allow(["mcp__fake__*"], ["mcp__fake__danger*"]) == {"mcp__fake__alpha", "mcp__fake__beta"}
    assert allow(None, ["mcp__fake"]) == {"read_file", "agent", TOOL_SEARCH_NAME}
    text = "---\nname: scout\ntools: [mcp__fake__*, tool_search]\ndisallowedTools:\n  - mcp__fake__danger*\n---\nScout."
    parsed = AgentDefinition.from_markdown(text)
    assert parsed.tools == ["mcp__fake__*", "tool_search"] and parsed.disallowed_tools == ["mcp__fake__danger*"]
    assert AgentDefinition.from_markdown("---\nname: s\ntools: mcp__fake, read_file\n---\nS.").tools == ["mcp__fake", "read_file"]


# ----------------------------------------------------------------------------- permissions
@pytest.mark.parametrize(
    ("deny", "denied"),
    [
        (["mcp__fake__danger*"], {"mcp__fake__danger_delete"}),
        (["mcp__fake"], set(FAKE)),
        (["mcp__*"], set(FAKE)),
        (["mcp__fake__alpha"], {"mcp__fake__alpha"}),
        (["MCP__FAKE__BETA"], {"mcp__fake__beta"}),
    ],
)
def test_wildcard_deny_patterns_through_registry_call(workspace, ctx, deny, denied):
    reg = make_registry(workspace)
    ctx.permissions = Permissions("bypass", deny=deny, workspace=workspace)
    for name in FAKE:
        res = reg.call(name, {"x": "1"}, ctx)
        assert res.ok is (name not in denied), (deny, name, res.error)
        if name in denied:
            assert res.error.code == "permission_denied"
    assert ctx.permissions.decide(reg.get("read_file"), {"path": "STUDY.md"}) == "allow"  # built-in tools are untouched


def test_wildcard_allow_patterns_through_registry_call(workspace, ctx):
    reg = make_registry(workspace)
    reg.register(fake_spec("mcp__other__post", permission="network", origin="connector:other"))
    reg.register(fake_spec("mcp__fake__net", permission="network"))
    ctx.permissions = Permissions("ask", allow=["mcp__fake__*"], workspace=workspace)
    ctx.ask_permission = None
    assert reg.call("mcp__fake__net", {}, ctx).ok  # allow beats the network always-ask
    assert reg.call("mcp__fake__danger_delete", {}, ctx).ok
    other = reg.call("mcp__other__post", {}, ctx)
    assert not other.ok and other.error.code == "permission_denied" and "needs approval" in other.error.message
