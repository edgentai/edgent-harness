"""Connectors: preset resolution, registration as Edgent tools, permissions, deferral, caching, calls, the CLI."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from edgent.adapters.mcp_client import McpClient
from edgent.adapters.providers.mock import MockProvider
from edgent.application import connectors as conn
from edgent.application.app import App, AppConfig
from edgent.application.connectors import (
    ConnectorManager,
    edgent_tool_name,
    load_configs,
    looks_like_secret,
    normalise_schema,
)
from edgent.domain.errors import ToolFailure
from edgent.domain.types import Reply, ToolCall
from edgent.domain.workspace import Workspace
from edgent.tools.registry import AllowAll, ToolContext, ToolRegistry
from edgent.ui.cli import main

FAKE = str(Path(__file__).parent / "fixtures" / "fake_mcp_stdio.py")


def fake_entry(*extra: str, **kw) -> dict:
    return {"command": sys.executable, "args": ["-I", FAKE, *extra], **kw}


@pytest.fixture
def ws(tmp_path):
    return Workspace.init(tmp_path / "ws", git=False)


def manager(ws, connectors: dict, environ: dict | None = None, **kw) -> tuple[ConnectorManager, ToolRegistry]:
    reg = ToolRegistry()
    m = ConnectorManager(ws, {"connectors": connectors}, home=str(ws.root.parent / "home"), environ=environ or {}, **kw)
    m.attach(reg)
    return m, reg


def ctx_for(ws, reg) -> ToolContext:
    return ToolContext(workspace=ws, session_id="s", permissions=AllowAll(), extra={"registry": reg})


# --------------------------------------------------------------------------- configuration
def test_presets_resolve_from_the_environment_and_say_what_is_missing(ws):
    env = {"PERPLEXITY_API_KEY": "pplx-abcdef123456", "POSTHOG_API_KEY": "phx_abcdef123456"}
    cfgs = {c.name: c for c in load_configs({"connectors": {n: {"preset": n} for n in conn.PRESETS}}, env, str(ws.root))}
    p = cfgs["perplexity"]
    assert p.ready and p.transport == "http" and p.spec.url == "https://api.perplexity.ai/mcp"
    assert p.spec.headers["Authorization"] == "Bearer pplx-abcdef123456" and "pplx-abcdef123456" in p.secrets
    assert "pplx-abcdef123456" not in json.dumps(p.public()) and p.public()["headers"] == ["Authorization"]
    f = cfgs["firecrawl"]
    assert f.ready and "Authorization" not in f.spec.headers and f.spec.headers["X-Firecrawl-Origin"] == "edgent"  # keyless free tier
    ph = cfgs["posthog"]
    assert ph.spec.headers["x-posthog-read-only"] == "true" and ph.spec.headers["x-posthog-mcp-mode"] == "cli"
    assert "x-posthog-project-id" not in ph.spec.headers  # optional and unset: left out
    assert not cfgs["composio"].ready and cfgs["composio"].missing == ["COMPOSIO_CONSUMER_KEY"]
    pw = cfgs["playwright"]
    assert pw.transport == "stdio" and pw.spec.command == "npx" and "@playwright/mcp@0.0.84" in pw.spec.args
    assert f"--output-dir={ws.root}/media/browser" in pw.spec.args and "--headless" in pw.spec.args and "--isolated" in pw.spec.args


def test_preset_variants_and_overrides(ws):
    env = {"PERPLEXITY_API_KEY": "pplx-abcdef123456", "COMPOSIO_API_KEY": "ck_123456789", "COMPOSIO_MCP_URL": "https://backend.composio.dev/v3/mcp/abc?user_id=me"}
    cfgs = {c.name: c for c in load_configs({"connectors": {
        "perplexity": {"preset": "perplexity", "variant": "local"},
        "composio": {"preset": "composio", "variant": "session"},
        "search": {"preset": "firecrawl", "variant": "search", "timeout_s": 30},
        "bad": {"preset": "firecrawl", "variant": "nope"},
    }}, env, str(ws.root))}
    local = cfgs["perplexity"]
    assert local.transport == "stdio" and local.spec.env["PERPLEXITY_API_KEY"] == "pplx-abcdef123456" and local.spec.args[0] == "-y"
    session = cfgs["composio"]
    assert session.ready and session.spec.headers == {"x-api-key": "ck_123456789"} and "user_id=***" in session.public()["url"]
    assert cfgs["search"].spec.url.endswith("/v2/mcp-search") and cfgs["search"].timeout_s == 30
    assert "no variant" in cfgs["bad"].problem


def test_custom_entries_claude_style_mcp_servers_and_validation(ws):
    env = {"MY_TOKEN": "tok-1234567890"}
    cfgs = {c.name: c for c in load_configs({
        "connectors": {"docs": {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer ${MY_TOKEN}", "X-Region": "${REGION:-eu}"}},
                       "Bad Name": {"url": "https://x"}, "empty": {}},
        "mcpServers": {"edgent": {"command": "edgent", "args": ["mcp"]}, "local": {"command": "node", "args": ["server.js"], "env": {"K": "${MY_TOKEN}"}},
                       "remote": {"type": "http", "url": "https://r.example/mcp"}},
    }, env, str(ws.root))}
    assert "edgent" not in cfgs  # Edgent's own registration for hosts is never loaded as a connector
    assert cfgs["docs"].spec.headers == {"Authorization": "Bearer tok-1234567890", "X-Region": "eu"} and not cfgs["docs"].trust_annotations
    assert cfgs["local"].spec.env == {"K": "tok-1234567890"} and cfgs["local"].source == "mcpServers"
    assert cfgs["remote"].transport == "http"
    assert "lower-case" in cfgs["bad name"].problem and "needs a url" in cfgs["empty"].problem
    missing = load_configs({"connectors": {"x": {"url": "https://x.example/${HOST_PART}/mcp"}}}, {}, str(ws.root))[0]
    assert missing.missing == ["HOST_PART"] and not missing.ready


def test_playwright_gets_the_known_browser_and_no_sandbox_only_as_root(ws, monkeypatch):
    import edgent.adapters.toolchain as toolchain

    monkeypatch.setattr(toolchain, "find_chromium", lambda: "/opt/chromium/chrome")
    monkeypatch.setattr(conn.os, "geteuid", lambda: 0, raising=False)
    pw = load_configs({"connectors": {"playwright": {}}}, {}, str(ws.root))[0]
    assert pw.spec.args[-3:] == ["--browser=chromium", "--executable-path=/opt/chromium/chrome", "--no-sandbox"]
    monkeypatch.setattr(conn.os, "geteuid", lambda: 1000, raising=False)
    pw = load_configs({"connectors": {"playwright": {"extra_args": ["--caps", "vision"]}}}, {}, str(ws.root))[0]
    assert "--no-sandbox" not in pw.spec.args and pw.spec.args[-1] == "--executable-path=/opt/chromium/chrome" and "vision" in pw.spec.args
    pw = load_configs({"connectors": {"playwright": {"extra_args": ["--browser=firefox"]}}}, {}, str(ws.root))[0]
    assert not any(a.startswith("--executable-path") for a in pw.spec.args)  # the user chose a browser


def test_tool_names_and_schemas_are_provider_safe():
    assert edgent_tool_name("perplexity", "perplexity_search") == "mcp__perplexity__perplexity_search"
    long = edgent_tool_name("composio", "COMPOSIO_" + "X" * 80)
    assert len(long) <= 64 and long.startswith("mcp__composio__COMPOSIO_")
    assert edgent_tool_name("x", "a.b c") == "mcp__x__a_b_c"
    schema = normalise_schema({"$schema": "http://json-schema.org/draft-07/schema#", "type": "object", "$defs": {"P": {"type": "string", "description": "p"}},
                               "properties": {"p": {"$ref": "#/$defs/P"}, "n": {"type": "integer"}}, "required": ["p"]})
    assert schema == {"type": "object", "properties": {"p": {"type": "string", "description": "p"}, "n": {"type": "integer"}}, "required": ["p"]}
    assert normalise_schema(None) == {"type": "object", "properties": {}}
    zod = normalise_schema({"type": "object", "properties": {"scale": {"type": "string", "default": "css"}, "q": {"type": "string"}}, "required": ["scale", "q"]})
    assert zod["required"] == ["q"]  # a defaulted field is optional: the server fills it in
    assert looks_like_secret("Bearer sk-abcdef1234567890") and not looks_like_secret("Bearer ${KEY}") and not looks_like_secret("edgent")


# --------------------------------------------------------------------------- registration and calls
def test_tools_are_listed_registered_cached_and_called(ws):
    m, reg = manager(ws, {"fake": fake_entry()})
    assert reg.names() == []  # nothing cached yet: nothing started
    assert m.ensure()["fake"].startswith("ok")
    names = set(m.tool_names("fake"))
    assert {"mcp__fake__echo", "mcp__fake__add", "mcp__fake__image", "mcp__fake__fail"} <= names
    spec = reg.get("mcp__fake__echo")
    assert spec.origin == "connector:fake" and spec.group == "connector" and spec.permission == "network" and spec.side_effects  # not trusted
    assert "[fake connector" in spec.description and "untrusted" not in spec.parameters
    ctx = ctx_for(ws, reg)
    r = reg.call("mcp__fake__echo", {"text": "hi"}, ctx)
    assert r.ok and r.output["text"] == "hi" and r.output["untrusted"] is True and r.output["connector"] == "fake"
    r = reg.call("mcp__fake__add", {"a": 1, "b": 2}, ctx)
    assert r.output["structured"] == {"sum": 3} and r.output["text"] == ""  # the JSON text duplicate is dropped
    r = reg.call("mcp__fake__image", {}, ctx)
    assert r.ok and r.attachments and r.attachments[0].type == "image" and Path(r.attachments[0].path).exists()
    assert r.output["media"][0]["path"].startswith(".edgent/cache/connectors/fake/media/")
    r = reg.call("mcp__fake__fail", {}, ctx)
    assert not r.ok and r.error.code == "connector_tool_error" and r.error.remediation
    r = reg.call("mcp__fake__resource", {}, ctx)
    assert r.ok and r.output["links"] and "[resource" in r.output["text"]
    cache = json.loads(m._cache_path("fake").read_text())
    assert ws.root not in m._cache_path("fake").parents  # outside the workspace the model can write to
    assert [t["name"] for t in cache["tools"]][:2] == ["echo", "add"]
    m.close_all()

    started: list[object] = []

    def spy(spec, options):
        started.append(spec)
        return McpClient(spec, options)

    m2, reg2 = manager(ws, {"fake": fake_entry()}, client_factory=spy)
    assert "mcp__fake__echo" in reg2.names() and started == []  # registered from the cache, no process started
    assert reg2.call("mcp__fake__echo", {"text": "lazy"}, ctx_for(ws, reg2)).output["text"] == "lazy" and len(started) == 1
    m2.close_all()


def test_trusted_annotations_map_to_permissions_and_include_exclude_filter(ws):
    m, reg = manager(ws, {"fake": fake_entry(trust_annotations=True, exclude=["slow", "roots_*"])})
    m.ensure()
    assert reg.get("mcp__fake__echo").permission == "read" and not reg.get("mcp__fake__echo").side_effects
    assert reg.get("mcp__fake__fail").permission == "delete"
    assert reg.get("mcp__fake__image").permission == "network"
    assert reg.get("mcp__fake__slow") is None and reg.get("mcp__fake__roots_probe") is None
    m.close_all()
    m, reg = manager(ws, {"only": fake_entry(include=["echo"], permission="write")})
    m.ensure()
    assert m.tool_names("only") == ["mcp__only__echo"] and reg.get("mcp__only__echo").permission == "write"
    m.close_all()


def test_many_tools_are_deferred_unless_exposed(ws):
    m, reg = manager(ws, {"big": fake_entry("--extra-tools", "12")})
    m.ensure()
    assert reg.get("mcp__big__echo").deferred and "tool_search" in reg.names()
    assert not any(s["function"]["name"].startswith("mcp__big__") for s in reg.openai_schemas())
    assert "mcp__big__echo" in reg.deferred_index()
    m.close_all()
    m, reg = manager(ws, {"big": fake_entry("--extra-tools", "12", expose="all")})
    m.ensure()
    assert not reg.get("mcp__big__echo").deferred
    m.close_all()


def test_not_ready_connectors_explain_themselves(ws):
    m, reg = manager(ws, {"perplexity": {"preset": "perplexity"}})
    assert reg.names() == [] and m.ensure() == {}
    status = m.status()[0]
    assert status["missing"] == ["PERPLEXITY_API_KEY"] and "console.perplexity.ai" in status["setup"]
    with pytest.raises(ToolFailure) as e:
        m.test("perplexity")
    assert e.value.error.code == "unavailable" and "PERPLEXITY_API_KEY" in e.value.error.message
    with pytest.raises(ToolFailure):
        m.get("nope")


def test_secrets_never_reach_the_cache_status_or_errors(ws):
    secret = "sk-live-0123456789abcdef"
    m, _ = manager(ws, {"fake": fake_entry("--require-env", "MISSING_ONE", env={"API_KEY": "${MY_KEY}"})}, environ={"MY_KEY": secret})
    results = m.ensure()
    assert results["fake"].startswith("error") and secret not in results["fake"]
    assert secret not in json.dumps(m.status())
    m.close_all()
    m, _ = manager(ws, {"fake": fake_entry(env={"API_KEY": "${MY_KEY}"})}, environ={"MY_KEY": secret})
    m.ensure()
    assert secret not in m._cache_path("fake").read_text()
    m.close_all()


# --------------------------------------------------------------------------- through the app and the loop
def test_the_model_loads_a_deferred_connector_tool_and_calls_it(tmp_path, home):
    ws = Workspace.init(tmp_path / "ws", git=False)
    provider = MockProvider().push(
        Reply(text="", tool_calls=[ToolCall.new("tool_search", {"query": "select:mcp__big__echo"})], finish_reason="tool_calls"),
        Reply(text="", tool_calls=[ToolCall.new("mcp__big__echo", {"text": "from the model"})], finish_reason="tool_calls"),
        Reply(text="The connector echoed it."),
    )
    settings = {"connectors": {"big": fake_entry("--extra-tools", "12")}, "permissions": {"allow": ["mcp__big"]}}
    app = App(ws, AppConfig(provider="mock", home=str(home), settings_overrides=settings), provider=provider)
    try:
        session, result = app.chat("echo something through the connector")
        assert result.stop_reason == "done" and result.tool_calls == 2 and not result.errors, result.errors
        first, last = provider.calls[0], provider.calls[-1]
        assert not any(t["function"]["name"] == "mcp__big__echo" for t in first.tools) and "mcp__big__echo" in first.system_text()
        assert any(t["function"]["name"] == "mcp__big__echo" for t in last.tools)
        tool_texts = [m.text() for m in app.load_session(session.session_id).messages if m.role == "tool"]
        assert "from the model" in tool_texts[-1]
        assert app.doctor()["connectors"][0]["tools"] >= 7
        assert "mcp__big__echo" not in app.build_allow()  # untrusted connector: not read-only, never auto-allowed in builds
    finally:
        app.close()


def test_connector_tools_ask_by_default(tmp_path, home):
    ws = Workspace.init(tmp_path / "ws", git=False)
    provider = MockProvider().push(Reply(text="", tool_calls=[ToolCall.new("mcp__fake__echo", {"text": "x"})], finish_reason="tool_calls"), Reply(text="denied"))
    app = App(ws, AppConfig(provider="mock", home=str(home), settings_overrides={"connectors": {"fake": fake_entry()}}), provider=provider)
    try:
        session, result = app.chat("call it", permission_mode="auto")
        tool_msg = [m.text() for m in app.load_session(session.session_id).messages if m.role == "tool"][0]
        assert "permission_denied" in tool_msg  # network class: asks, and nobody can answer here
    finally:
        app.close()


def test_cli_add_test_tools_list_remove(tmp_path, home, capsys):
    ws = tmp_path / "ws"
    assert main(["init", str(ws), "--no-git"]) == 0
    base = ["--workspace", str(ws), "--provider", "mock", "connectors"]
    capsys.readouterr()
    assert main([*base, "presets"]) == 0
    out = capsys.readouterr().out
    assert all(name in out for name in conn.PRESETS)
    assert main([*base, "add", "fake", "--command", sys.executable, "--allow", "--", "-I", FAKE]) == 0
    out = capsys.readouterr().out
    assert "added connector fake" in out and "tools" in out and "mcp__fake" in out
    settings = json.loads((ws / ".edgent" / "settings.json").read_text())
    assert settings["connectors"]["fake"]["command"] == sys.executable and "mcp__fake" in settings["permissions"]["allow"]
    assert main([*base, "tools", "fake"]) == 0
    assert "mcp__fake__echo" in capsys.readouterr().out
    assert main([*base, "list", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["name"] == "fake" and rows[0]["tools"] >= 7
    assert main([*base]) == 0 and "ready" in capsys.readouterr().out
    assert main([*base, "test", "fake"]) == 0 and "connected to" in capsys.readouterr().out
    assert main([*base, "add", "perplexity", "--no-test"]) == 0
    assert "needs PERPLEXITY_API_KEY" in capsys.readouterr().out
    assert main([*base, "add", "x", "--url", "https://x.example/mcp", "--header", "Authorization=Bearer sk-abcdef1234567890", "--no-test"]) == 0
    assert "looks like a literal key" in capsys.readouterr().err
    assert main([*base, "add", "nothing"]) == 2
    assert main([*base, "add", "y", "--arg=-I", "--no-test", "--", "x"]) == 2  # arguments without a command
    assert main([*base, "remove", "fake"]) == 0
    settings = json.loads((ws / ".edgent" / "settings.json").read_text())
    assert "fake" not in settings["connectors"] and "mcp__fake" not in settings["permissions"]["allow"]
    assert main(["--workspace", str(ws), "--provider", "mock", "doctor"]) == 0
    assert "connector perplexity" in capsys.readouterr().out
