"""Regressions for the adversarial review of the connector feature: one test per confirmed finding."""
from __future__ import annotations

import http.server
import json
import os
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from edgent.adapters import mcp_client
from edgent.adapters.mcp_client import ClientOptions, HttpSpec, McpClient, McpError, StdioSpec, _session_lost
from edgent.adapters.providers.mock import MockProvider
from edgent.application.app import App, AppConfig
from edgent.application.connectors import ConnectorManager, ConnectorState, load_configs, normalise_schema
from edgent.application.hosts import plan_skills
from edgent.domain.errors import ToolFailure
from edgent.domain.workspace import Workspace
from edgent.tools.registry import AllowAll, ToolContext, ToolRegistry, validate_args
from tests.fixtures.fake_mcp_http import FakeHttpMcpServer

FAKE = str(Path(__file__).parent / "fixtures" / "fake_mcp_stdio.py")


def fake_entry(*extra: str, **kw) -> dict:
    return {"command": sys.executable, "args": ["-I", FAKE, *extra], **kw}


@pytest.fixture
def ws(tmp_path):
    return Workspace.init(tmp_path / "ws", git=False)


def manager(ws, connectors, environ=None, home=None):
    reg = ToolRegistry()
    m = ConnectorManager(ws, {"connectors": connectors}, home=str(home or ws.root.parent / "home"), environ=environ or {})
    m.attach(reg)
    return m, reg


def script(tmp_path: Path, body: str) -> str:
    path = tmp_path / f"hostile_{abs(hash(body)) % 10_000}.py"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return str(path)


INIT_OK = '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-11-25","capabilities":{"tools":{}},"serverInfo":{"name":"h","version":"0"}}}'


# --------------------------------------------------------------------------- security
def test_the_tool_cache_lives_outside_the_workspace_and_cannot_be_dated_into_the_future(ws, tmp_path):
    home = tmp_path / "home"
    m, reg = manager(ws, {"fake": fake_entry(trust_annotations=True)}, home=home)
    m.ensure()
    path = m._cache_path("fake")
    assert home in path.parents and ws.root not in path.parents  # write_file cannot reach it: paths escape the workspace
    assert not (ws.path("cache") / "connectors" / "fake.json").exists()
    data = json.loads(path.read_text())
    data["fetched_at"] = time.time() + 10 * 365 * 24 * 3600
    path.write_text(json.dumps(data))
    m.close_all()
    m2, reg2 = manager(ws, {"fake": fake_entry(trust_annotations=True)}, home=home)
    assert reg2.names() == []  # a cache from the future is ignored, so the tools come from a live listing
    m2.close_all()


def test_secrets_in_a_stdio_command_line_are_redacted_when_the_process_dies(ws):
    secret = "sk-live-command-line-123456"
    m, reg = manager(ws, {"fake": fake_entry("--crash-after", "2", "--token=${MY_TOKEN}")}, environ={"MY_TOKEN": secret})
    m.ensure()
    ctx = ToolContext(workspace=ws, session_id="s", permissions=AllowAll(), extra={"registry": reg})
    r = reg.call("mcp__fake__echo", {"text": "x"}, ctx)
    assert not r.ok and secret not in json.dumps(r.error.to_dict())
    assert secret not in json.dumps(m.status())
    m.close_all()


def test_a_key_in_the_url_path_is_redacted_everywhere(ws):
    secret = "fc-path-key-0123456789"
    cfg = load_configs({"connectors": {"x": {"url": "http://127.0.0.1:9/${KEY}/mcp"}}}, {"KEY": secret}, str(ws.root))[0]
    assert secret not in json.dumps(cfg.public())
    c = McpClient(cfg.spec, ClientOptions(connect_timeout_s=2, secrets=cfg.secrets))
    with pytest.raises(McpError) as e:
        c.connect()
    assert secret not in e.value.message


class _Redirect(http.server.BaseHTTPRequestHandler):
    seen: list[dict] = []

    def do_POST(self):  # noqa: N802
        self.send_response(302)
        self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/elsewhere")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        _Redirect.seen.append(dict(self.headers))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *a):
        pass


def test_a_redirect_is_refused_and_the_key_never_follows_it():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Redirect)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        c = McpClient(HttpSpec(f"http://127.0.0.1:{srv.server_port}/mcp", {"Authorization": "Bearer key-do-not-forward"}), ClientOptions(connect_timeout_s=3))
        with pytest.raises(McpError) as e:
            c.connect()
        assert e.value.code == "configuration_invalid" and "redirects" in e.value.message
        assert _Redirect.seen == []
    finally:
        srv.shutdown()


def test_large_schemas_shrink_without_losing_a_parameter_and_titles_are_capped(ws):
    props = {f"p{i}": {"type": "string", "description": "d" * 300, "enum": [f"v{j}" for j in range(60)]} for i in range(40)}
    raw = {"type": "object", "properties": props, "required": ["p0", "p1"], "additionalProperties": False}
    out = normalise_schema(raw)
    assert len(json.dumps(out)) <= 12_000 and set(out["properties"]) == set(props) and out["required"] == ["p0", "p1"]
    assert validate_args(out, {"p0": "v59", "p1": "anything"}) == []  # nothing a server accepts is refused here
    nested = {"type": "object", "properties": {"q": {"type": "object", "properties": {f"k{i}": {"type": "string", "description": "x" * 500} for i in range(60)}, "required": ["k0"], "additionalProperties": False}}}
    small = normalise_schema(nested)
    assert len(json.dumps(small)) <= 12_000 and validate_args(small, {"q": {"k0": "a", "k5": "b"}}) == []
    huge = normalise_schema({"type": "object", "properties": {f"n{i}": {"type": "string"} for i in range(2000)}, "required": ["n0"]})
    assert len(json.dumps(huge)) <= 12_000 and validate_args(huge, {"n0": "x", "n1999": "y"}) == []

    m, reg = manager(ws, {"fake": fake_entry()})
    state = m.get("fake")
    state.tools = [{"name": "t", "title": "T" * 5000, "description": "short", "inputSchema": {"type": "object"}}]
    m._register(state)
    assert len(reg.get("mcp__fake__t").description) < 1600


def test_stderr_lines_without_newlines_stay_bounded(tmp_path):
    path = script(tmp_path, f"""
        import sys, json
        sys.stderr.write("x" * 3_000_000); sys.stderr.flush()
        for line in sys.stdin:
            msg = json.loads(line)
            if msg.get("method") == "initialize":
                print('{INIT_OK}' % msg["id"], flush=True)
            elif "id" in msg:
                print(json.dumps({{"jsonrpc": "2.0", "id": msg["id"], "result": {{"tools": []}}}}), flush=True)
        """)
    c = McpClient(StdioSpec(sys.executable, ["-I", path]), ClientOptions(connect_timeout_s=10))
    try:
        c.connect()
        assert c.list_tools() == []
        assert len(c._transport.stderr_tail(10_000_000)) < 20_000  # one bounded line, not three megabytes
    finally:
        c.close()


# --------------------------------------------------------------------------- protocol
def test_parallel_calls_on_an_expired_session_all_recover():
    with FakeHttpMcpServer(era="legacy") as server:
        c = McpClient(HttpSpec(server.url), ClientOptions(connect_timeout_s=10, request_timeout_s=10))
        c.connect()
        c.call_tool("echo", {"text": "warm"})
        server.expire_sessions()  # the server forgets the session while six calls are about to use it
        results, errors = [], []

        def call(i):
            try:
                results.append(c.call_tool("echo", {"text": str(i)})["content"][0]["text"])
            except McpError as e:
                errors.append(e.message)

        threads = [threading.Thread(target=call, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(15)
        assert not errors and sorted(results) == [str(i) for i in range(6)]
        assert server.rpc_methods().count("initialize") <= 3  # one re-initialisation, not one per thread (a straggler may add one)
        c.close()


def test_a_forgotten_session_answered_with_400_counts_as_expired():
    assert _session_lost(400, {"code": -32000, "message": "Bad Request: No valid session ID provided"}, "")
    assert _session_lost(404, None, "")
    assert not _session_lost(400, {"code": -32602, "message": "bad params"}, "")


def test_modern_era_mirrors_x_mcp_header_parameters_and_offers_no_roots(tmp_path):
    with FakeHttpMcpServer(era="modern", x_mcp_header=True) as server:
        c = McpClient(HttpSpec(server.url), ClientOptions(connect_timeout_s=10, roots=[str(tmp_path)]))
        c.connect()
        c.list_tools()
        assert c.call_tool("echo", {"text": "héllo"})["content"][0]["text"] == "héllo"
        call = [r for r in server.requests if r.rpc_method == "tools/call"][-1]
        assert call.header("Mcp-Param-Text") and "roots" not in call.body["params"]["_meta"]["io.modelcontextprotocol/clientCapabilities"]
        c.close()


class _Stall(http.server.BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "1000")
        self.end_headers()
        self.wfile.write(b'{"jsonrpc"')
        self.wfile.flush()
        time.sleep(3)

    def log_message(self, *a):
        pass


def test_a_stalled_http_body_is_a_typed_timeout():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Stall)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        c = McpClient(HttpSpec(f"http://127.0.0.1:{srv.server_port}/mcp"), ClientOptions(connect_timeout_s=1))
        with pytest.raises(McpError) as e:
            c.connect()
        assert e.value.code in ("timeout", "unavailable") and e.value.category.value in ("timeout", "unavailable")
    finally:
        srv.shutdown()


def test_a_legacy_http_timeout_cancels_the_request_on_the_server():
    with FakeHttpMcpServer(era="legacy") as server:
        c = McpClient(HttpSpec(server.url), ClientOptions(connect_timeout_s=10))
        c.connect()
        with pytest.raises(McpError) as e:
            c.call_tool("slow", {"seconds": 5}, timeout_s=0.5)
        assert e.value.code == "timeout"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not server.cancelled:
            time.sleep(0.05)
        assert server.cancelled and "notifications/cancelled" in server.rpc_methods()
        c.close()


class _Polling(http.server.BaseHTTPRequestHandler):
    """2025-11-25 polling: the POST stream ends after a priming event with an id; the answer comes on GET with Last-Event-ID."""

    calls: list[str] = []

    def do_POST(self):  # noqa: N802
        msg = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        if "id" not in msg:
            self.send_response(202)
            self.end_headers()
            return
        if msg["method"] == "initialize":
            body = INIT_OK % msg["id"]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Mcp-Session-Id", "s1")
            self.end_headers()
            self.wfile.write(body.encode())
            return
        _Polling.calls.append(msg["method"])
        self.server.pending = msg["id"]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b"id: ev-1\nretry: 50\ndata:\n\n")
        self.wfile.flush()

    def do_GET(self):  # noqa: N802
        assert self.headers.get("Last-Event-ID") == "ev-1"
        body = json.dumps({"jsonrpc": "2.0", "id": self.server.pending, "result": {"content": [{"type": "text", "text": "resumed"}]}})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(f"id: ev-2\ndata: {body}\n\n".encode())

    def log_message(self, *a):
        pass


def test_a_legacy_stream_closed_after_a_priming_event_is_resumed_not_retried():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Polling)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        c = McpClient(HttpSpec(f"http://127.0.0.1:{srv.server_port}/mcp"), ClientOptions(connect_timeout_s=5, request_timeout_s=5))
        assert c.call_tool("echo", {"text": "x"})["content"][0]["text"] == "resumed"
        assert _Polling.calls == ["tools/call"]  # resumed with GET, never sent twice
    finally:
        srv.shutdown()


def test_a_message_with_an_invalid_id_does_not_kill_the_reader(tmp_path):
    path = script(tmp_path, f"""
        import sys, json
        for line in sys.stdin:
            msg = json.loads(line)
            if msg.get("method") == "initialize":
                print('{INIT_OK}' % msg["id"], flush=True)
            elif "id" in msg:
                print(json.dumps({{"jsonrpc": "2.0", "id": [1, 2], "result": {{}}}}), flush=True)
                print(json.dumps({{"jsonrpc": "2.0", "id": msg["id"], "result": {{"tools": [{{"name": "a"}}]}}}}), flush=True)
        """)
    c = McpClient(StdioSpec(sys.executable, ["-I", path]), ClientOptions(connect_timeout_s=10, request_timeout_s=5))
    try:
        assert [t["name"] for t in c.list_tools()] == ["a"] and [t["name"] for t in c.list_tools()] == ["a"]
    finally:
        c.close()


def test_an_oversize_message_fails_its_request_at_once(monkeypatch):
    monkeypatch.setattr(mcp_client, "MAX_MESSAGE_BYTES", 4000)
    c = McpClient(StdioSpec(sys.executable, ["-I", FAKE]), ClientOptions(connect_timeout_s=10, request_timeout_s=30))
    try:
        c.connect()
        t0 = time.monotonic()
        with pytest.raises(McpError) as e:
            c.call_tool("echo", {"text": "y" * 10_000})
        assert e.value.code == "truncated" and time.monotonic() - t0 < 10
        assert c.call_tool("echo", {"text": "small"})["content"][0]["text"] == "small"
    finally:
        c.close()


def test_initialize_is_never_cancelled(tmp_path):
    record = tmp_path / "in.log"
    path = script(tmp_path, f"""
        import sys
        with open({str(record)!r}, "a") as out:
            for line in sys.stdin:
                out.write(line); out.flush()
        """)
    c = McpClient(StdioSpec(sys.executable, ["-I", path]), ClientOptions(connect_timeout_s=1))
    with pytest.raises(McpError):
        c.connect()
    time.sleep(0.2)
    assert "notifications/cancelled" not in record.read_text()


def test_pagination_cut_short_is_reported(monkeypatch):
    monkeypatch.setattr(mcp_client, "MAX_LIST_PAGES", 2)
    c = McpClient(StdioSpec(sys.executable, ["-I", FAKE, "--extra-tools", "40", "--page-size", "5"]), ClientOptions(connect_timeout_s=10))
    try:
        c.list_tools()
        assert c.tools_partial and c.describe()["tools_partial"]
    finally:
        c.close()


# --------------------------------------------------------------------------- integration
def test_build_sessions_list_connectors_before_choosing_what_they_may_use(tmp_path, home):
    ws = Workspace.init(tmp_path / "ws", git=False)
    app = App(ws, AppConfig(provider="mock", home=str(home), settings_overrides={"connectors": {"fake": fake_entry(trust_annotations=True)}}), provider=MockProvider())
    try:
        assert "mcp__fake__echo" in app.build_allow() and "mcp__fake__fail" not in app.build_allow()
    finally:
        app.close()


def test_a_failed_listing_is_retried_after_the_pause(ws, monkeypatch):
    import edgent.application.connectors as conn

    m, reg = manager(ws, {"fake": fake_entry("--require-env", "READY")})
    assert m.ensure()["fake"].startswith("error")
    m.ensure_once()
    assert reg.names() == []  # inside the pause: not retried
    m.get("fake").config.spec.env["READY"] = "1"
    monkeypatch.setattr(conn, "LIST_RETRY_AFTER_S", 0.0)
    m.ensure_once()
    assert "mcp__fake__echo" in reg.names()
    m.close_all()


def test_hosts_do_not_get_the_connectors_skill(tmp_path):
    ws = Workspace.init(tmp_path / "ws", git=False)
    targets = [str(c.path) for c in plan_skills("claude", ws)]
    assert targets and not any("/connectors/" in t for t in targets)


def test_connector_names_cannot_fake_another_connectors_prefix_and_collisions_are_reported(ws):
    bad = {c.name: c for c in load_configs({"connectors": {"a__b": {"url": "https://x/mcp"}, "a_": {"url": "https://x/mcp"}, "a-b": {"url": "https://x/mcp"}}}, {}, str(ws.root))}
    assert bad["a__b"].problem and bad["a_"].problem and not bad["a-b"].problem
    m, reg = manager(ws, {"fake": fake_entry()})
    state: ConnectorState = m.get("fake")
    state.tools = [{"name": "files.read", "inputSchema": {"type": "object"}}, {"name": "files_read", "inputSchema": {"type": "object"}}]
    names = m._register(state)
    assert len(names) == 2 and len(set(names)) == 2 and state.warnings


def test_a_request_never_carries_more_than_128_tool_schemas(tmp_path, home):
    from edgent.domain.types import Reply
    from edgent.tools.registry import ToolSpec, obj

    ws = Workspace.init(tmp_path / "ws", git=False)
    provider = MockProvider().push(Reply(text="ok"))
    settings = {"connectors": {}}
    app = App(ws, AppConfig(provider="mock", home=str(home), settings_overrides=settings), provider=provider)
    try:
        for i in range(150):
            app.registry.register(ToolSpec(name=f"mcp__many__t{i}", description="x", parameters=obj({}), fn=lambda c, a: {}, origin="connector:many", permission="network"))
        app.chat("hi")
        tools = provider.calls[-1].tools
        names = {t["function"]["name"] for t in tools}
        assert len(tools) == 128 and "read_file" in names and "grep" in names
    finally:
        app.close()


def test_failures_from_the_manager_are_redacted_and_typed(ws):
    secret = "sk-manager-0123456789"
    m, reg = manager(ws, {"x": {"url": "http://127.0.0.1:9/mcp", "headers": {"Authorization": "Bearer ${K}"}}}, environ={"K": secret})
    with pytest.raises(ToolFailure) as e:
        m.test("x")
    assert e.value.error.code == "unavailable" and secret not in json.dumps(e.value.error.to_dict()) and secret not in json.dumps(m.status())
    assert os.environ.get("K") is None
