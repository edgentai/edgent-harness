"""The MCP client against the fake servers in tests/fixtures: both eras, both transports, failures."""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

from edgent.adapters.mcp_client import ClientOptions, HttpSpec, McpClient, McpError, StdioSpec, header_value, redact_url

FAKE = str(Path(__file__).parent / "fixtures" / "fake_mcp_stdio.py")


def stdio(*args: str, env: dict[str, str] | None = None, **opts) -> McpClient:
    return McpClient(StdioSpec(sys.executable, ["-I", FAKE, *args], env=env or {}), ClientOptions(connect_timeout_s=20, request_timeout_s=20, **opts))


@pytest.fixture
def client():
    c = stdio()
    yield c
    c.close()


def test_legacy_handshake_tools_and_calls(client):
    info = client.connect()
    assert info["era"] == "legacy" and info["protocol_version"] == "2025-11-25" and info["server_info"].get("name")
    tools = {t["name"]: t for t in client.list_tools()}
    assert {"echo", "add", "image", "fail", "slow", "resource"} <= set(tools)
    assert tools["echo"]["annotations"]["readOnlyHint"] is True
    r = client.call_tool("echo", {"text": "héllo\nworld"})
    assert r["content"][0]["text"] == "héllo\nworld"
    r = client.call_tool("add", {"a": 2, "b": 3})
    assert r["structuredContent"] == {"sum": 5}
    assert client.call_tool("fail", {})["isError"] is True
    img = client.call_tool("image", {})
    assert any(p["type"] == "image" and p["mimeType"] == "image/png" for p in img["content"])


def test_pagination_collects_every_page():
    c = stdio("--extra-tools", "35", "--page-size", "7")
    try:
        names = [t["name"] for t in c.list_tools()]  # list_tools connects on demand
        assert len(names) == len(set(names)) and sum(n.startswith("filler_") for n in names) == 35
    finally:
        c.close()


def test_noise_on_stdout_and_a_stderr_flood_do_not_break_the_connection():
    events = []
    c = stdio("--noise", "--stderr-flood-kb", "256", "--stderr-chatter", on_event=lambda k, d: events.append(k))
    try:
        c.connect()
        assert c.call_tool("echo", {"text": "ok"})["content"][0]["text"] == "ok"
        assert "noise" in events
    finally:
        c.close()


def test_concurrent_calls_interleave_and_a_slow_call_does_not_block_others(client):
    client.connect()
    results: dict[str, str] = {}

    def slow():
        results["slow"] = client.call_tool("slow", {"seconds": 1.5})["content"][0]["text"]

    t = threading.Thread(target=slow)
    t.start()
    t0 = time.monotonic()
    texts = [client.call_tool("echo", {"text": str(i)})["content"][0]["text"] for i in range(5)]
    assert texts == [str(i) for i in range(5)] and time.monotonic() - t0 < 1.4
    t.join(10)
    assert results["slow"].startswith("slept")


def test_timeout_cancels_the_request_and_the_client_stays_usable(tmp_path):
    rec = tmp_path / "rec.jsonl"
    c = stdio("--record", str(rec))
    try:
        c.connect()
        with pytest.raises(McpError) as e:
            c.call_tool("slow", {"seconds": 5}, timeout_s=0.5)
        assert e.value.code == "timeout" and e.value.retryable
        assert c.call_tool("echo", {"text": "still here"})["content"][0]["text"] == "still here"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            lines = [json.loads(x) for x in rec.read_text().splitlines() if x.strip()]
            if any(x.get("dir") == "in" and x["msg"].get("method") == "notifications/cancelled" for x in lines):
                break
            time.sleep(0.05)
        else:
            pytest.fail("notifications/cancelled was never sent")
    finally:
        c.close()


def test_roots_are_offered_to_a_server_that_asks(tmp_path):
    c = stdio(roots=[str(tmp_path)])
    try:
        out = c.call_tool("roots_probe", {})
        assert tmp_path.resolve().as_uri() in json.dumps(out)
    finally:
        c.close()


def test_env_is_passed_and_other_variables_are_not(monkeypatch):
    monkeypatch.setenv("UNRELATED_SECRET_KEY", "do-not-pass-me")
    c = stdio("--require-env", "CONNECTOR_KEY=abc123", env={"CONNECTOR_KEY": "abc123"})
    try:
        c.connect()
    finally:
        c.close()
    c2 = stdio("--require-env", "UNRELATED_SECRET_KEY")
    with pytest.raises(McpError) as e:
        c2.connect()
    assert e.value.code == "unavailable" and "UNRELATED_SECRET_KEY" in e.value.message  # the fake says what it missed, via stderr


def test_a_crashed_server_is_restarted_on_the_next_request():
    c = stdio("--crash-after", "3")  # initialize, tools/list, echo; the 4th request kills it
    try:
        c.connect()
        c.list_tools()
        c.call_tool("echo", {"text": "1"})
        with pytest.raises(McpError) as e:
            c.call_tool("echo", {"text": "2"})
        assert e.value.code == "unavailable" and e.value.retryable
        assert c.call_tool("echo", {"text": "3"})["content"][0]["text"] == "3"  # a fresh process
    finally:
        c.close()


def test_missing_command_is_a_typed_unavailable_error():
    c = McpClient(StdioSpec("definitely-not-a-command-xyz", []), ClientOptions(connect_timeout_s=5))
    with pytest.raises(McpError) as e:
        c.connect()
    assert e.value.code == "unavailable" and "not found" in e.value.message and e.value.remediation


def test_npx_hint_when_node_is_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(McpError) as e:
        McpClient(StdioSpec("npx", ["-y", "x"]), ClientOptions(connect_timeout_s=5)).connect()
    assert "Node.js" in e.value.remediation


def test_modern_only_server_is_spoken_to_statelessly(tmp_path):
    rec = tmp_path / "rec.jsonl"
    c = stdio("--era", "modern", "--record", str(rec))
    try:
        info = c.connect()
        assert info["era"] == "modern" and info["protocol_version"] == "2026-07-28"
        tools = c.list_tools()
        assert any(t["name"] == "echo" for t in tools) and c.tools_ttl_s is not None
        assert c.call_tool("echo", {"text": "m"})["content"][0]["text"] == "m"
        sent = [json.loads(x)["msg"] for x in rec.read_text().splitlines() if '"dir": "in"' in x or '"dir":"in"' in x]
        calls = [m for m in sent if m.get("method") == "tools/call"]
        assert calls and calls[0]["params"]["_meta"]["io.modelcontextprotocol/protocolVersion"] == "2026-07-28"
    finally:
        c.close()


def test_dual_era_server_is_kept_legacy(client):
    c = stdio("--era", "dual")
    try:
        assert c.connect()["era"] == "legacy"
        assert c.call_tool("echo", {"text": "d"})["content"][0]["text"] == "d"
    finally:
        c.close()


def test_secrets_are_redacted_from_errors():
    c = stdio("--require-env", "NEVER_SET_VAR", "--stderr-chatter", env={"API_TOKEN": "sk-very-secret-value-123"}, secrets=["sk-very-secret-value-123"])
    with pytest.raises(McpError) as e:
        c.connect()
    assert "sk-very-secret-value-123" not in e.value.message


def test_header_values_and_url_redaction():
    assert header_value("get_weather") == "get_weather"
    assert header_value("météo").startswith("=?base64?") and header_value(" x ").endswith("?=")
    assert redact_url("https://user:pw@mcp.example.com:8443/v2/mcp?key=abc&x=1") == "https://mcp.example.com:8443/v2/mcp?key=***&x=***"
    with pytest.raises(McpError) as e:
        McpClient(HttpSpec("ftp://example.com/mcp")).connect()
    assert e.value.code == "configuration_invalid"


# --------------------------------------------------------------------------- Streamable HTTP
from tests.fixtures.fake_mcp_http import FakeHttpMcpServer  # noqa: E402


def http(server: FakeHttpMcpServer, headers: dict[str, str] | None = None, **opts) -> McpClient:
    return McpClient(HttpSpec(server.url, headers or {}), ClientOptions(connect_timeout_s=10, request_timeout_s=10, **opts))


@pytest.mark.parametrize("sse", [False, True])
def test_http_legacy_session_json_and_sse(sse, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")  # a proxy that does not exist: loopback must bypass it
    monkeypatch.setenv("NO_PROXY", "")
    with FakeHttpMcpServer(era="legacy", sse=sse, extra_tools=15, page_size=6) as server:
        c = http(server)
        info = c.connect()
        assert info["era"] == "legacy" and server.session_ids()
        names = [t["name"] for t in c.list_tools()]
        assert len(names) == len(set(names)) and "echo" in names and "filler_14" in names
        assert c.call_tool("echo", {"text": "ü"})["content"][0]["text"] == "ü"
        posts = [r for r in server.requests if r.http_method == "POST" and r.rpc_method == "tools/call"]
        assert posts[0].header("Mcp-Session-Id") and posts[0].header("MCP-Protocol-Version") == info["protocol_version"]
        assert "text/event-stream" in posts[0].header("Accept")
        c.close()
        assert server.ended_sessions  # DELETE on close


def test_http_sse_with_crlf_progress_and_a_server_request_mid_stream(tmp_path):
    events = []
    with FakeHttpMcpServer(era="legacy", sse=True, sse_crlf=True) as server:
        c = http(server, roots=[str(tmp_path)], on_event=lambda k, d: events.append(k))
        out = c.call_tool("roots_probe", {})
        assert tmp_path.resolve().as_uri() in json.dumps(out)
        assert "<response>" in server.rpc_methods()  # the client answered roots/list with a POST
        c.close()


def test_http_expired_session_is_reinitialised_once():
    with FakeHttpMcpServer(era="legacy", expire_session_after=2) as server:  # initialize and one call, then 404
        c = http(server)
        c.connect()
        c.call_tool("echo", {"text": "1"})
        assert c.call_tool("echo", {"text": "2"})["content"][0]["text"] == "2"  # 404 -> new session -> retried
        assert server.rpc_methods().count("initialize") == 2
        c.close()


def test_http_auth_error_names_the_header_and_never_leaks_the_key():
    with FakeHttpMcpServer(era="legacy", require_auth=("Authorization", "Bearer right-key-123456")) as server:
        c = http(server, {"Authorization": "Bearer wrong-key-987654"}, secrets=["wrong-key-987654"])
        with pytest.raises(McpError) as e:
            c.connect()
        assert e.value.code == "auth_required" and "401" in e.value.message and "wrong-key-987654" not in e.value.message
        assert "API key" in e.value.remediation
        ok = http(server, {"Authorization": "Bearer right-key-123456"})
        assert ok.connect()["era"] == "legacy"
        ok.close()


@pytest.mark.parametrize("sse", [False, True])
def test_http_modern_server_gets_stateless_requests_with_mirrored_headers(sse):
    with FakeHttpMcpServer(era="modern", sse=sse) as server:
        c = http(server)
        info = c.connect()
        assert info["era"] == "modern" and info["protocol_version"] == "2026-07-28"
        assert c.call_tool("echo", {"text": "météo"})["content"][0]["text"] == "météo"
        call = [r for r in server.requests if r.rpc_method == "tools/call"][-1]
        assert call.header("Mcp-Method") == "tools/call" and call.header("Mcp-Name") == "echo" and call.header("MCP-Protocol-Version") == "2026-07-28"
        assert call.header("Mcp-Session-Id") is None and c.list_tools()
        c.close()


def test_http_dual_server_stays_legacy_and_the_tool_error_flag_survives():
    with FakeHttpMcpServer(era="dual") as server:
        c = http(server)
        assert c.connect()["era"] == "legacy" and c.call_tool("fail", {})["isError"] is True
        c.close()


def test_http_unreachable_and_not_an_mcp_endpoint():
    c = McpClient(HttpSpec("http://127.0.0.1:9/mcp"), ClientOptions(connect_timeout_s=3))
    with pytest.raises(McpError) as e:
        c.connect()
    assert e.value.code == "unavailable" and e.value.retryable
    with FakeHttpMcpServer(era="legacy") as server:
        wrong = McpClient(HttpSpec(server.url.replace("/mcp", "/nope")), ClientOptions(connect_timeout_s=3))
        with pytest.raises(McpError) as e:
            wrong.connect()
        assert "404" in e.value.message
