"""The fake MCP servers in tests/fixtures, driven with raw JSON-RPC so their behaviour is pinned for client tests."""
from __future__ import annotations

import base64
import json
import os
import queue
import re
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib
from collections.abc import Callable, Iterator
from http.client import HTTPConnection
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.fake_mcp_http import FakeHttpMcpServer
from tests.fixtures.fake_mcp_stdio import (
    CRASH_EXIT_CODE,
    HEADER_MISMATCH,
    INVALID_PARAMS,
    INVALID_REQUEST,
    META_CLIENT_CAPABILITIES,
    META_CLIENT_INFO,
    META_PROTOCOL_VERSION,
    META_SERVER_INFO,
    METHOD_NOT_FOUND,
    NOISE_LINE,
    PARSE_ERROR,
    TOOLS_TTL_MS,
    UNSUPPORTED_PROTOCOL_VERSION,
)

STDIO_SCRIPT = Path(__file__).parent / "fixtures" / "fake_mcp_stdio.py"
WAIT_SECONDS = 5.0
QUIET_SECONDS = 0.4
MODERN = "2026-07-28"
ACCEPT = "application/json, text/event-stream"
CORE_TOOLS = ["echo", "add", "image", "fail", "slow", "resource", "big"]
LEGACY_TOOLS = CORE_TOOLS + ["roots_probe"]
ROOTS = [{"uri": "file:///home/user/project", "name": "Project"}]


def meta(version: str = MODERN, caps: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    return {META_PROTOCOL_VERSION: version, META_CLIENT_CAPABILITIES: caps or {},
            META_CLIENT_INFO: {"name": "raw-test", "version": "0"}, **extra}


def init_params(version: str = "2025-11-25", caps: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"protocolVersion": version, "capabilities": caps or {}, "clientInfo": {"name": "raw-test", "version": "0"}}


def png_is_valid(data: bytes) -> bool:
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return False
    pos, kinds = 8, []
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        kind, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length:pos + 12 + length])
        if zlib.crc32(kind + body) != crc:
            return False
        kinds.append(kind)
        pos += 12 + length
    return kinds[0] == b"IHDR" and kinds[-1] == b"IEND"


# -- stdio peer ----------------------------------------------------------------------------------------------------


class StdioPeer:
    """The client end of a fake stdio server: JSON lines in, JSON lines out, stderr drained on a thread."""

    def __init__(self, *args: str, env: dict[str, str] | None = None) -> None:
        self.proc = subprocess.Popen([sys.executable, "-I", str(STDIO_SCRIPT), *args], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        self._inbox: queue.Queue[Any] = queue.Queue()
        self._backlog: list[dict[str, Any]] = []
        self.raw_lines: list[str] = []
        self.stderr_lines: list[str] = []
        self.eof = False
        self._next_id = 0
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()

    def _pump_stdout(self) -> None:
        assert self.proc.stdout is not None
        for raw in iter(self.proc.stdout.readline, b""):
            text = raw.decode("utf-8").rstrip("\n")
            try:
                self._inbox.put(json.loads(text))
            except ValueError:
                self.raw_lines.append(text)
        self._inbox.put(None)

    def _pump_stderr(self) -> None:
        assert self.proc.stderr is not None
        for raw in iter(self.proc.stderr.readline, b""):
            self.stderr_lines.append(raw.decode("utf-8", "replace").rstrip("\n"))

    def send(self, message: dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(message).encode() + b"\n")
        self.proc.stdin.flush()

    def send_line(self, line: bytes) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + b"\n")
        self.proc.stdin.flush()

    def start(self, method: str, params: dict[str, Any] | None = None, msg_id: Any = None) -> Any:
        if msg_id is None:
            self._next_id += 1
            msg_id = self._next_id
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": msg_id, "method": method}
        if params is not None:
            message["params"] = params
        self.send(message)
        return msg_id

    def request(self, method: str, params: dict[str, Any] | None = None, msg_id: Any = None) -> dict[str, Any]:
        msg_id = self.start(method, params, msg_id)
        return self.response(msg_id)

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self.send(message)

    def wait_for(self, matches: Callable[[dict[str, Any]], bool], timeout: float = WAIT_SECONDS) -> dict[str, Any]:
        for i, message in enumerate(self._backlog):
            if matches(message):
                return self._backlog.pop(i)
        deadline = time.monotonic() + timeout
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                message = self._inbox.get(timeout=remaining)
            except queue.Empty:
                break
            if message is None:
                self.eof = True
                raise AssertionError("the fake server closed stdout")
            if matches(message):
                return message
            self._backlog.append(message)
        raise AssertionError(f"no matching message within {timeout} s; backlog={self._backlog}")

    def response(self, msg_id: Any, timeout: float = WAIT_SECONDS) -> dict[str, Any]:
        return self.wait_for(lambda m: m.get("id") == msg_id and "method" not in m, timeout)

    def assert_no_response(self, msg_id: Any, seconds: float = QUIET_SECONDS) -> None:
        with pytest.raises(AssertionError, match="no matching message"):
            self.response(msg_id, timeout=seconds)

    def handshake(self, version: str = "2025-11-25", caps: dict[str, Any] | None = None) -> dict[str, Any]:
        reply = self.request("initialize", init_params(version, caps))
        self.notify("notifications/initialized")
        return reply

    def close(self) -> int:
        if self.proc.stdin is not None and not self.proc.stdin.closed:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
        try:
            return self.proc.wait(WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            return self.proc.wait(WAIT_SECONDS)


@pytest.fixture
def stdio() -> Iterator[Callable[..., StdioPeer]]:
    peers: list[StdioPeer] = []

    def spawn(*args: str, env: dict[str, str] | None = None) -> StdioPeer:
        peer = StdioPeer(*args, env=env)
        peers.append(peer)
        return peer

    yield spawn
    for peer in peers:
        peer.close()


def list_all_tools(page: Callable[[str | None], dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    tools, cursor, pages = [], None, 0
    while True:
        result = page(cursor)
        pages += 1
        tools.extend(result["tools"])
        cursor = result.get("nextCursor")
        if cursor is None:
            return tools, pages


# -- stdio: legacy -------------------------------------------------------------------------------------------------


def test_stdio_legacy_handshake_negotiates_version_and_answers_ping(stdio: Callable[..., StdioPeer]) -> None:
    peer = stdio()
    early = peer.request("tools/list")
    assert early["error"]["code"] == INVALID_REQUEST and "initialize" in early["error"]["message"]
    assert peer.request("ping")["result"] == {}
    result = peer.handshake("2025-06-18")["result"]
    assert result["protocolVersion"] == "2025-06-18"
    assert result["capabilities"] == {"tools": {"listChanged": True}}
    assert result["serverInfo"]["name"] == "edgent-fake-mcp" and result["instructions"]
    assert "resultType" not in result
    other = stdio()
    assert other.request("initialize", init_params("2024-11-05"))["result"]["protocolVersion"] == "2025-11-25"
    probe = other.request("server/discover", {"_meta": meta()})
    assert probe["error"]["code"] == METHOD_NOT_FOUND  # what a dual-era client falls back to initialize on


def test_stdio_legacy_pagination_and_annotations(stdio: Callable[..., StdioPeer]) -> None:
    peer = stdio("--extra-tools", "12", "--page-size", "6")
    peer.handshake()
    tools, pages = list_all_tools(
        lambda cursor: peer.request("tools/list", {"cursor": cursor} if cursor else {})["result"])
    names = [t["name"] for t in tools]
    assert names == LEGACY_TOOLS + [f"filler_{i}" for i in range(12)] and pages == 4
    by_name = {t["name"]: t for t in tools}
    for name in ("echo", "add"):
        assert by_name[name]["annotations"] == {"readOnlyHint": True, "openWorldHint": False}
    assert by_name["fail"]["annotations"] == {"destructiveHint": True}
    assert all("annotations" not in by_name[n] for n in ("image", "slow", "resource", "big", "roots_probe"))
    assert by_name["add"]["outputSchema"]["properties"]["sum"]["type"] == "number"
    assert all(t["inputSchema"]["type"] == "object" for t in tools)
    bad = peer.request("tools/list", {"cursor": "not-a-cursor"})
    assert bad["error"]["code"] == INVALID_PARAMS


def test_stdio_legacy_tool_results(stdio: Callable[..., StdioPeer]) -> None:
    peer = stdio()
    peer.handshake()

    def call(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        return peer.request("tools/call", {"name": name, "arguments": arguments or {}})

    assert call("echo", {"text": "héllo world"})["result"]["content"] == [{"type": "text", "text": "héllo world"}]
    added = call("add", {"a": 2, "b": 3.5})["result"]
    assert added["structuredContent"] == {"sum": 5.5} and json.loads(added["content"][0]["text"]) == {"sum": 5.5}
    caption, image = call("image")["result"]["content"]
    assert caption["type"] == "text" and image["type"] == "image" and image["mimeType"] == "image/png"
    assert png_is_valid(base64.b64decode(image["data"]))
    failed = call("fail")["result"]
    assert failed["isError"] is True and "fail" in failed["content"][0]["text"]
    link, embedded = call("resource")["result"]["content"]
    assert link["type"] == "resource_link" and link["uri"] == embedded["resource"]["uri"]
    assert embedded["type"] == "resource" and embedded["resource"]["text"]
    big = call("big", {"chars": 123_457})["result"]["content"][0]["text"]
    assert len(big) == 123_457 and big.startswith("0123456789")
    assert call("echo", {})["result"]["isError"] is True
    assert call("nope")["error"]["code"] == INVALID_PARAMS
    assert call("filler_0")["error"]["code"] == INVALID_PARAMS  # no fillers without --extra-tools


def test_stdio_slow_does_not_block_and_honours_cancellation(stdio: Callable[..., StdioPeer], tmp_path: Path) -> None:
    record = tmp_path / "record.jsonl"
    peer = stdio("--record", str(record))
    peer.handshake()
    started = time.monotonic()
    slow_id = peer.start("tools/call", {"name": "slow", "arguments": {"seconds": 5}}, msg_id="slow-1")
    echoed = peer.request("tools/call", {"name": "echo", "arguments": {"text": "first"}})
    assert echoed["result"]["content"][0]["text"] == "first"
    peer.notify("notifications/cancelled", {"requestId": slow_id, "reason": "test gave up"})
    peer.assert_no_response(slow_id)
    assert peer.request("ping")["result"] == {}
    assert time.monotonic() - started < 3
    quick = peer.request("tools/call", {"name": "slow", "arguments": {"seconds": 0.05}})
    assert quick["result"]["content"][0]["text"] == "slept 0.05 s"
    events = [json.loads(line) for line in record.read_text().splitlines()]
    assert {"dir": "event", "event": "cancelled", "id": "slow-1"} in events
    assert any(e["dir"] == "in" and e["msg"].get("method") == "initialize" for e in events)


def test_stdio_progress_notifications_precede_the_result(stdio: Callable[..., StdioPeer]) -> None:
    peer = stdio()
    peer.handshake()
    msg_id = peer.start("tools/call", {"name": "slow", "arguments": {"seconds": 0.05}, "_meta": {"progressToken": "p1"}})
    first = peer.wait_for(lambda m: m.get("method") == "notifications/progress")
    second = peer.wait_for(lambda m: m.get("method") == "notifications/progress")
    assert first["params"]["progressToken"] == "p1" and first["params"]["progress"] < second["params"]["progress"]
    assert not peer._backlog
    assert peer.response(msg_id)["result"]["isError"] is False


def test_stdio_roots_round_trip(stdio: Callable[..., StdioPeer]) -> None:
    peer = stdio()
    peer.handshake(caps={"roots": {"listChanged": False}})
    call_id = peer.start("tools/call", {"name": "roots_probe", "arguments": {}})
    ask = peer.wait_for(lambda m: m.get("method") == "roots/list")
    assert "id" in ask and ask["id"] != call_id
    peer.send({"jsonrpc": "2.0", "id": ask["id"], "result": {"roots": ROOTS}})
    result = peer.response(call_id)["result"]
    assert result["structuredContent"] == {"roots": ROOTS} and json.loads(result["content"][0]["text"]) == {"roots": ROOTS}
    refused = peer.start("tools/call", {"name": "roots_probe", "arguments": {}})
    ask = peer.wait_for(lambda m: m.get("method") == "roots/list")
    peer.send({"jsonrpc": "2.0", "id": ask["id"], "error": {"code": -32601, "message": "no roots here"}})
    assert "no roots here" in peer.response(refused)["result"]["content"][0]["text"]

    no_caps = stdio()
    no_caps.handshake()
    reply = no_caps.request("tools/call", {"name": "roots_probe", "arguments": {}})["result"]
    assert reply["isError"] is True and "roots" in reply["content"][0]["text"]


# -- stdio: modern and dual ----------------------------------------------------------------------------------------


def test_stdio_modern_is_stateless_and_checks_meta(stdio: Callable[..., StdioPeer]) -> None:
    peer = stdio("--era", "modern")
    init = peer.request("initialize", init_params())
    assert init["error"]["code"] == METHOD_NOT_FOUND and MODERN in init["error"]["message"]
    discovered = peer.request("server/discover", {"_meta": meta()})["result"]
    assert discovered["resultType"] == "complete" and discovered["supportedVersions"] == [MODERN]
    assert discovered["capabilities"] == {"tools": {"listChanged": True}} and discovered["serverInfo"]["name"]
    missing = peer.request("tools/list", {})
    assert missing["error"]["code"] == INVALID_PARAMS
    no_caps = peer.request("tools/list", {"_meta": {META_PROTOCOL_VERSION: MODERN}})
    assert no_caps["error"]["code"] == INVALID_PARAMS
    old = peer.request("tools/list", {"_meta": meta("1900-01-01")})["error"]
    assert old["code"] == UNSUPPORTED_PROTOCOL_VERSION and old["data"] == {"supported": [MODERN], "requested": "1900-01-01"}
    listed = peer.request("tools/list", {"_meta": meta()})["result"]
    assert [t["name"] for t in listed["tools"]] == CORE_TOOLS
    assert listed["resultType"] == "complete" and listed["ttlMs"] == TOOLS_TTL_MS and listed["cacheScope"] == "public"
    echoed = peer.request("tools/call", {"name": "echo", "arguments": {"text": "hi"}, "_meta": meta()})["result"]
    assert echoed["resultType"] == "complete" and echoed["content"][0]["text"] == "hi"
    assert echoed["_meta"][META_SERVER_INFO]["name"] == "edgent-fake-mcp"
    assert peer.request("ping", {"_meta": meta()})["error"]["code"] == METHOD_NOT_FOUND
    roots = peer.request("tools/call", {"name": "roots_probe", "arguments": {}, "_meta": meta()})
    assert roots["error"]["code"] == INVALID_PARAMS


def test_stdio_modern_cancellation(stdio: Callable[..., StdioPeer]) -> None:
    peer = stdio("--era", "modern")
    slow_id = peer.start("tools/call", {"name": "slow", "arguments": {"seconds": 5}, "_meta": meta()})
    peer.notify("notifications/cancelled", {"requestId": slow_id})
    peer.assert_no_response(slow_id)
    assert peer.request("server/discover", {"_meta": meta()})["result"]["resultType"] == "complete"


def test_stdio_dual_answers_both_eras(stdio: Callable[..., StdioPeer]) -> None:
    peer = stdio("--era", "dual")
    discovered = peer.request("server/discover", {"_meta": meta()})["result"]
    assert discovered["supportedVersions"] == [MODERN, "2025-11-25", "2025-06-18"]
    assert peer.request("server/discover", {})["error"]["code"] == INVALID_PARAMS
    assert peer.request("tools/list")["error"]["code"] == INVALID_REQUEST  # legacy semantics until initialize
    assert peer.handshake()["result"]["protocolVersion"] == "2025-11-25"
    legacy = peer.request("tools/list")["result"]
    assert "roots_probe" in [t["name"] for t in legacy["tools"]] and "resultType" not in legacy
    modern = peer.request("tools/list", {"_meta": meta()})["result"]
    assert "roots_probe" not in [t["name"] for t in modern["tools"]] and modern["resultType"] == "complete"
    legacy_version = peer.request("tools/list", {"_meta": meta("2025-11-25")})["error"]
    assert legacy_version["code"] == UNSUPPORTED_PROTOCOL_VERSION and "initialize" in legacy_version["message"]


# -- stdio: process behaviour --------------------------------------------------------------------------------------


def test_stdio_noise_line_and_parse_errors(stdio: Callable[..., StdioPeer]) -> None:
    peer = stdio("--noise")
    assert peer.request("ping")["result"] == {}
    assert peer.raw_lines == [NOISE_LINE]
    peer.send_line(b"this is not json")
    assert peer.wait_for(lambda m: m.get("error", {}).get("code") == PARSE_ERROR)["id"] is None
    peer.send_line(b'[{"jsonrpc": "2.0", "id": 9, "method": "ping"}]')
    assert peer.wait_for(lambda m: "error" in m)["error"]["code"] == INVALID_REQUEST


def test_stdio_crash_after_n_requests(stdio: Callable[..., StdioPeer]) -> None:
    peer = stdio("--crash-after", "2")
    peer.handshake()
    assert peer.request("tools/list")["result"]["tools"]
    peer.start("tools/call", {"name": "echo", "arguments": {"text": "never answered"}})
    with pytest.raises(AssertionError, match="closed stdout"):
        peer.wait_for(lambda m: True)
    assert peer.proc.wait(WAIT_SECONDS) == CRASH_EXIT_CODE
    wait_until(lambda: any("crashing on purpose" in line for line in peer.stderr_lines))


def test_stdio_require_env(stdio: Callable[..., StdioPeer]) -> None:
    env = {k: v for k, v in os.environ.items() if k != "FAKE_MCP_TOKEN"}
    missing = stdio("--require-env", "FAKE_MCP_TOKEN", env=env)
    assert missing.proc.wait(WAIT_SECONDS) == 1
    wait_until(lambda: any("FAKE_MCP_TOKEN is not set" in line for line in missing.stderr_lines))
    wrong = stdio("--require-env", "FAKE_MCP_TOKEN=right", env={**env, "FAKE_MCP_TOKEN": "wrong"})
    assert wrong.proc.wait(WAIT_SECONDS) == 1
    present = stdio("--require-env", "FAKE_MCP_TOKEN=right", env={**env, "FAKE_MCP_TOKEN": "right"})
    assert present.request("ping")["result"] == {}


def test_stdio_stderr_chatter_and_flood_leave_stdout_clean(stdio: Callable[..., StdioPeer]) -> None:
    peer = stdio("--stderr-chatter", "--stderr-flood-kb", "128")
    peer.handshake()
    assert peer.request("tools/list")["result"]["tools"]
    assert peer.close() == 0
    wait_until(lambda: any("stdin closed" in line for line in peer.stderr_lines))
    assert peer.raw_lines == []
    assert any("tools/list" in line for line in peer.stderr_lines) and len(peer.stderr_lines) > 128


# -- HTTP helpers --------------------------------------------------------------------------------------------------

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never route 127.0.0.1 through a proxy


def http(url: str, body: Any = None, headers: dict[str, str] | None = None, method: str = "POST",
         raw: bytes | None = None, defaults: bool = True) -> tuple[int, dict[str, str], bytes]:
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    sent = {"Content-Type": "application/json", "Accept": ACCEPT} if defaults else {}
    sent.update(headers or {})
    request = urllib.request.Request(url, data=data, headers=sent, method=method)
    try:
        with _OPENER.open(request, timeout=WAIT_SECONDS) as resp:
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read()
    except urllib.error.HTTPError as err:
        return err.code, {k.lower(): v for k, v in err.headers.items()}, err.read()


def rpc(msg_id: Any, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        message["params"] = params
    return message


def modern_headers(method: str, name: str | None = None, version: str = MODERN) -> dict[str, str]:
    headers = {"MCP-Protocol-Version": version, "Mcp-Method": method}
    if name is not None:
        headers["Mcp-Name"] = name
    return headers


def modern_call(url: str, msg_id: Any, method: str, params: dict[str, Any] | None = None,
                version: str = MODERN, headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], Any]:
    params = dict(params or {})
    params["_meta"] = meta(version)
    sent = modern_headers(method, params.get("name"), version)
    sent.update(headers or {})
    status, got, body = http(url, rpc(msg_id, method, params), sent)
    return status, got, json.loads(body) if body else None


def parse_sse(text: str) -> tuple[list[dict[str, str]], int]:
    """Minimal WHATWG event-stream parser: returns every event block (even empty-data ones) and the comment count."""
    events: list[dict[str, str]] = []
    current: dict[str, list[str] | str] = {}
    comments = 0
    for line in re.split(r"\r\n|\n|\r", text):
        if line == "":
            if current:
                events.append({k: "\n".join(v) if isinstance(v, list) else v for k, v in current.items()})
                current = {}
            continue
        if line.startswith(":"):
            comments += 1
            continue
        name, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if name == "data":
            data = current.setdefault("data", [])
            assert isinstance(data, list)
            data.append(value)
        else:
            current[name] = value
    return events, comments


def sse_messages(text: str) -> list[dict[str, Any]]:
    return [json.loads(e["data"]) for e in parse_sse(text)[0] if e.get("data")]


@pytest.fixture
def fake_http() -> Iterator[Callable[..., FakeHttpMcpServer]]:
    servers: list[FakeHttpMcpServer] = []

    def start(era: str = "legacy", **options: Any) -> FakeHttpMcpServer:
        server = FakeHttpMcpServer(era, **options).start()
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.stop()


def legacy_init(server: FakeHttpMcpServer, caps: dict[str, Any] | None = None) -> str:
    status, headers, body = http(server.url, rpc(1, "initialize", init_params(caps=caps)))
    assert status == 200, body
    sid = headers["mcp-session-id"]
    status, _, _ = http(server.url, {"jsonrpc": "2.0", "method": "notifications/initialized"},
                        {"MCP-Session-Id": sid, "MCP-Protocol-Version": "2025-11-25"})
    assert status == 202
    return sid


def legacy_headers(sid: str) -> dict[str, str]:
    return {"MCP-Session-Id": sid, "MCP-Protocol-Version": "2025-11-25"}


def wait_until(condition: Callable[[], bool], timeout: float = WAIT_SECONDS) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.01)


# -- HTTP: legacy --------------------------------------------------------------------------------------------------


def test_http_legacy_session_lifecycle(fake_http: Callable[..., FakeHttpMcpServer]) -> None:
    server = fake_http()
    status, headers, body = http(server.url, rpc(1, "initialize", init_params("2025-06-18")))
    reply = json.loads(body)
    assert status == 200 and headers["content-type"] == "application/json"
    assert reply["result"]["protocolVersion"] == "2025-06-18"
    sid = headers["mcp-session-id"]
    assert re.fullmatch(r"[\x21-\x7e]+", sid)
    status, _, body = http(server.url, {"jsonrpc": "2.0", "method": "notifications/initialized"}, {"MCP-Session-Id": sid})
    assert status == 202 and body == b""
    status, _, body = http(server.url, rpc(2, "tools/list"))
    assert status == 400 and json.loads(body)["error"]["code"] == -32000
    version = {"MCP-Session-Id": sid, "MCP-Protocol-Version": "2025-06-18"}
    status, _, body = http(server.url, rpc(3, "tools/list"), version)
    assert status == 200 and [t["name"] for t in json.loads(body)["result"]["tools"]] == LEGACY_TOOLS
    status, _, _ = http(server.url, rpc(4, "tools/list"), {"MCP-Session-Id": sid, "MCP-Protocol-Version": "2025-11-25"})
    assert status == 400  # not the version this session negotiated
    status, _, body = http(server.url, rpc(5, "tools/call", {"name": "echo", "arguments": {"text": "x"}}), version)
    assert json.loads(body)["result"]["content"][0]["text"] == "x"
    assert http(server.url, method="GET", headers={"MCP-Session-Id": sid})[0] == 405
    assert http(server.url, method="DELETE", headers={"MCP-Session-Id": sid})[0] == 200
    assert server.ended_sessions == [sid]
    assert http(server.url, rpc(6, "tools/list"), version)[0] == 404
    assert http(server.url, rpc(7, "initialize", init_params()), {"MCP-Session-Id": sid})[0] == 400
    assert server.rpc_methods()[:3] == ["initialize", "notifications/initialized", "tools/list"]
    first = server.requests[0]
    assert first.http_method == "POST" and first.status == 200 and first.header("ACCEPT") == ACCEPT
    assert first.body["params"]["protocolVersion"] == "2025-06-18" and first.rpc_id == 1
    assert server.requests[3].session_id == sid


def test_http_legacy_session_expiry_forces_reinitialize(fake_http: Callable[..., FakeHttpMcpServer]) -> None:
    server = fake_http(expire_session_after=2)
    sid = legacy_init(server)
    assert http(server.url, rpc(2, "tools/list"), legacy_headers(sid))[0] == 200
    status, _, body = http(server.url, rpc(3, "tools/list"), legacy_headers(sid))
    assert status == 404 and json.loads(body)["error"]["code"] == -32001
    fresh = legacy_init(server)
    assert fresh != sid and server.session_ids() == [sid, fresh]
    assert http(server.url, rpc(4, "tools/list"), legacy_headers(fresh))[0] == 200
    server.expire_sessions()
    assert http(server.url, rpc(5, "tools/list"), legacy_headers(fresh))[0] == 404


def test_http_stateless_legacy_needs_no_session(fake_http: Callable[..., FakeHttpMcpServer]) -> None:
    server = fake_http(sessions=False)
    status, headers, _ = http(server.url, rpc(1, "initialize", init_params()))
    assert status == 200 and "mcp-session-id" not in headers
    status, _, body = http(server.url, rpc(2, "tools/list"))
    assert status == 200 and json.loads(body)["result"]["tools"]
    assert http(server.url, method="DELETE")[0] == 405


def test_http_auth_origin_and_content_negotiation(fake_http: Callable[..., FakeHttpMcpServer]) -> None:
    server = fake_http(require_auth=("Authorization", "Bearer s3cret"))
    status, headers, body = http(server.url, rpc(1, "initialize", init_params()))
    challenge = headers["www-authenticate"]
    assert status == 401 and challenge.startswith("Bearer ") and "resource_metadata=" in challenge
    assert "invalid_token" not in challenge and "id" not in json.loads(body)
    status, headers, _ = http(server.url, rpc(1, "initialize", init_params()), {"Authorization": "Bearer nope"})
    assert status == 401 and 'error="invalid_token"' in headers["www-authenticate"]
    good = {"Authorization": "Bearer s3cret"}
    assert http(server.url, rpc(1, "initialize", init_params()), good)[0] == 200
    assert http(server.url, rpc(1, "initialize", init_params()), {**good, "Origin": "http://evil.example"})[0] == 403
    assert http(server.url, rpc(1, "initialize", init_params()), {**good, "Origin": "http://localhost:3000"})[0] == 200
    assert http(server.url, rpc(1, "initialize", init_params()), {**good, "Accept": "application/json"})[0] == 406
    assert http(server.url, rpc(1, "ping"), {**good, "Content-Type": "text/plain"})[0] == 415
    status, _, body = http(server.url, raw=b"{not json", headers=good)
    assert status == 400 and json.loads(body)["error"]["code"] == PARSE_ERROR
    assert http(server.url, [rpc(1, "ping")], good)[0] == 400
    status, headers, body = http(server.base_url + "/elsewhere", rpc(1, "ping"), good)
    assert status == 404 and headers["content-type"].startswith("text/plain")


def test_http_legacy_sse_stream_shape(fake_http: Callable[..., FakeHttpMcpServer]) -> None:
    server = fake_http(sse=True)
    status, headers, body = http(server.url, rpc(1, "initialize", init_params()))
    assert status == 200 and headers["content-type"] == "text/event-stream" and headers["mcp-session-id"]
    events, comments = parse_sse(body.decode())
    assert events[0]["id"] and events[0]["data"] == "" and comments >= 1
    assert json.loads(events[-1]["data"])["result"]["protocolVersion"] == "2025-11-25"
    sid = headers["mcp-session-id"]
    assert http(server.url, {"jsonrpc": "2.0", "method": "notifications/initialized"}, legacy_headers(sid))[0] == 202
    params = {"name": "echo", "arguments": {"text": "streamed"}, "_meta": {"progressToken": 7}}
    status, headers, body = http(server.url, rpc(2, "tools/call", params), legacy_headers(sid))
    events, _ = parse_sse(body.decode())
    messages = [json.loads(e["data"]) for e in events if e["data"]]
    assert [m.get("method") for m in messages] == ["notifications/progress", "notifications/progress", None]
    assert all(m["params"]["progressToken"] == 7 for m in messages[:2])
    assert messages[-1]["result"]["content"][0]["text"] == "streamed"
    assert len({e["id"] for e in events}) == len(events) and all(e.get("event") == "message" for e in events[1:])
    status, _, body = http(server.url, rpc(3, "tools/call", {"name": "nope"}), legacy_headers(sid))
    assert status == 200 and json.loads(body)["error"]["code"] == INVALID_PARAMS  # refused before any stream opens


def test_http_sse_crlf_line_endings(fake_http: Callable[..., FakeHttpMcpServer]) -> None:
    server = fake_http("modern", sse=True, sse_crlf=True)
    status, headers, body = http(server.url, rpc(1, "server/discover", {"_meta": meta()}), modern_headers("server/discover"))
    text = body.decode()
    assert status == 200 and headers["content-type"] == "text/event-stream" and "\r\n\r\n" in text
    assert sse_messages(text)[-1]["result"]["supportedVersions"] == [MODERN]
    events, comments = parse_sse(text)
    assert comments >= 1 and all("id" not in e for e in events)  # 2026-07-28 streams carry no event ids


def test_http_legacy_sse_roots_round_trip(fake_http: Callable[..., FakeHttpMcpServer]) -> None:
    server = fake_http(sse=True)
    status, headers, body = http(server.url, rpc(1, "initialize", init_params(caps={"roots": {}})))
    sid = headers["mcp-session-id"]
    assert http(server.url, {"jsonrpc": "2.0", "method": "notifications/initialized"}, legacy_headers(sid))[0] == 202
    conn = HTTPConnection("127.0.0.1", server.port, timeout=WAIT_SECONDS)
    payload = json.dumps(rpc("probe", "tools/call", {"name": "roots_probe", "arguments": {}}))
    conn.request("POST", server.path, payload, {"Content-Type": "application/json", "Accept": ACCEPT,
                                                **legacy_headers(sid)})
    resp = conn.getresponse()
    assert resp.status == 200 and resp.getheader("Content-Type") == "text/event-stream"
    seen: list[str] = []
    ask: dict[str, Any] | None = None
    while ask is None:
        line = resp.readline().decode()
        assert line, "stream ended before roots/list arrived"
        seen.append(line)
        if line.startswith("data: "):
            message = json.loads(line[6:])
            ask = message if message.get("method") == "roots/list" else None
    answer = {"jsonrpc": "2.0", "id": ask["id"], "result": {"roots": ROOTS}}
    assert http(server.url, answer, legacy_headers(sid))[0] == 202
    rest = resp.read().decode()
    conn.close()
    final = sse_messages("".join(seen) + rest)[-1]
    assert final["id"] == "probe" and final["result"]["structuredContent"] == {"roots": ROOTS}
    assert server.rpc_methods()[-1] == "<response>"


def test_http_roots_probe_without_sse_explains_itself(fake_http: Callable[..., FakeHttpMcpServer]) -> None:
    server = fake_http()
    sid = legacy_init(server, caps={"roots": {}})
    status, _, body = http(server.url, rpc(2, "tools/call", {"name": "roots_probe", "arguments": {}}), legacy_headers(sid))
    result = json.loads(body)["result"]
    assert status == 200 and result["isError"] is True and "sse=True" in result["content"][0]["text"]


def test_http_legacy_cancel_notification_stops_slow(fake_http: Callable[..., FakeHttpMcpServer]) -> None:
    server = fake_http()
    sid = legacy_init(server)
    outcome: list[Any] = []

    def call_slow() -> None:
        try:
            outcome.append(http(server.url, rpc("s1", "tools/call", {"name": "slow", "arguments": {"seconds": 5}}),
                                legacy_headers(sid)))
        except Exception as exc:  # the fake closes the connection without answering a cancelled request
            outcome.append(exc)

    started = time.monotonic()
    worker = threading.Thread(target=call_slow)
    worker.start()
    wait_until(lambda: "s1" in server.running())
    cancel = {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": "s1"}}
    assert http(server.url, cancel, legacy_headers(sid))[0] == 202
    worker.join(WAIT_SECONDS)
    assert time.monotonic() - started < 3 and server.cancelled == ["s1"]
    assert isinstance(outcome[0], Exception)
    assert [r.status for r in server.requests if r.rpc_id == "s1"] == [None]


# -- HTTP: modern and dual -----------------------------------------------------------------------------------------


def test_http_modern_happy_path(fake_http: Callable[..., FakeHttpMcpServer]) -> None:
    server = fake_http("modern", extra_tools=15)
    status, headers, body = modern_call(server.url, 1, "server/discover", headers={"MCP-Session-Id": "stale"})
    assert status == 200 and body["result"]["supportedVersions"] == [MODERN] and "mcp-session-id" not in headers
    assert body["result"]["resultType"] == "complete" and body["result"]["_meta"][META_SERVER_INFO]["version"]

    def page(cursor: str | None) -> dict[str, Any]:
        status, _, reply = modern_call(server.url, 2, "tools/list", {"cursor": cursor} if cursor else {})
        assert status == 200
        assert reply["result"]["ttlMs"] == TOOLS_TTL_MS and reply["result"]["cacheScope"] == "public"
        return reply["result"]

    tools, pages = list_all_tools(page)
    assert [t["name"] for t in tools] == CORE_TOOLS + [f"filler_{i}" for i in range(15)] and pages == 3
    status, _, reply = modern_call(server.url, 3, "tools/call", {"name": "add", "arguments": {"a": 1, "b": 2}})
    assert reply["result"]["structuredContent"] == {"sum": 3} and reply["result"]["resultType"] == "complete"
    encoded = "=?base64?" + base64.b64encode(b"echo").decode() + "?="
    params = {"name": "echo", "arguments": {"text": "b64"}}
    status, _, reply = modern_call(server.url, 4, "tools/call", params, headers={"Mcp-Name": encoded})
    assert status == 200 and reply["result"]["content"][0]["text"] == "b64"
    status, _, reply = modern_call(server.url, 5, "tools/call", {"name": "filler_14", "arguments": {}})
    assert reply["result"]["content"][0]["text"] == "filler_14"
    assert http(server.url, {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 1}})[0] == 202
    assert http(server.url, method="GET")[0] == 405 and http(server.url, method="DELETE")[0] == 405
    assert server.requests[0].header("mcp-method") == "server/discover"


def test_http_modern_rejections(fake_http: Callable[..., FakeHttpMcpServer]) -> None:
    server = fake_http("modern")
    url = server.url

    def code(status: int, body: bytes) -> tuple[int, int]:
        return status, json.loads(body)["error"]["code"]

    status, _, body = http(url, rpc(1, "initialize", init_params()))
    assert code(status, body) == (400, METHOD_NOT_FOUND) and MODERN in json.loads(body)["error"]["message"]
    status, _, body = http(url, rpc(2, "tools/list", {}), modern_headers("tools/list"))
    assert code(status, body) == (400, INVALID_PARAMS)
    listing = rpc(3, "tools/list", {"_meta": meta()})
    assert code(*http(url, listing, {"Mcp-Method": "tools/list"})[::2]) == (400, HEADER_MISMATCH)
    status, _, body = http(url, listing, modern_headers("tools/list", version="2025-11-25"))
    assert code(status, body) == (400, HEADER_MISMATCH) and "MCP-Protocol-Version" in json.loads(body)["error"]["message"]
    assert code(*http(url, listing, modern_headers("tools/call"))[::2]) == (400, HEADER_MISMATCH)
    echo = rpc(4, "tools/call", {"name": "echo", "arguments": {"text": "x"}, "_meta": meta()})
    assert code(*http(url, echo, modern_headers("tools/call"))[::2]) == (400, HEADER_MISMATCH)
    assert code(*http(url, echo, modern_headers("tools/call", "add"))[::2]) == (400, HEADER_MISMATCH)
    status, _, body = modern_call(url, 5, "tools/list", version="1900-01-01")
    assert status == 400 and body["error"]["code"] == UNSUPPORTED_PROTOCOL_VERSION
    assert body["error"]["data"] == {"supported": [MODERN], "requested": "1900-01-01"}
    status, _, body = modern_call(url, 6, "ping")
    assert (status, body["error"]["code"]) == (404, METHOD_NOT_FOUND)
    status, _, body = modern_call(url, 7, "tools/call", {"name": "roots_probe", "arguments": {}})
    assert (status, body["error"]["code"]) == (200, INVALID_PARAMS)
    answer = {"jsonrpc": "2.0", "id": 1, "result": {}}
    assert http(url, answer)[0] == 400
    assert [r.status for r in server.requests] == [400] * 7 + [400, 404, 200, 400]


def test_http_modern_sse_cancel_by_closing_the_stream(fake_http: Callable[..., FakeHttpMcpServer]) -> None:
    server = fake_http("modern", sse=True)
    body = json.dumps(rpc("slow-9", "tools/call", {"name": "slow", "arguments": {"seconds": 5}, "_meta": meta()}))
    head = (f"POST {server.path} HTTP/1.1\r\nHost: 127.0.0.1:{server.port}\r\nContent-Type: application/json\r\n"
            f"Accept: {ACCEPT}\r\nMCP-Protocol-Version: {MODERN}\r\nMcp-Method: tools/call\r\nMcp-Name: slow\r\n"
            f"Content-Length: {len(body)}\r\n\r\n")
    started = time.monotonic()
    sock = socket.create_connection(("127.0.0.1", server.port), timeout=WAIT_SECONDS)
    sock.sendall(head.encode() + body.encode())
    received = b""
    while b"stream open" not in received:
        chunk = sock.recv(4096)
        assert chunk, "stream closed before it opened"
        received += chunk
    assert b"text/event-stream" in received
    sock.close()
    wait_until(lambda: server.cancelled == ["slow-9"])
    assert time.monotonic() - started < 3 and server.running() == []


def test_http_modern_json_cancel_by_disconnect(fake_http: Callable[..., FakeHttpMcpServer]) -> None:
    server = fake_http("modern")
    conn = HTTPConnection("127.0.0.1", server.port, timeout=WAIT_SECONDS)
    payload = json.dumps(rpc(11, "tools/call", {"name": "slow", "arguments": {"seconds": 5}, "_meta": meta()}))
    conn.request("POST", server.path, payload, {"Content-Type": "application/json", "Accept": ACCEPT,
                                                **modern_headers("tools/call", "slow")})
    wait_until(lambda: 11 in server.running())
    conn.close()
    wait_until(lambda: server.cancelled == [11])


def test_http_dual_serves_both_eras(fake_http: Callable[..., FakeHttpMcpServer]) -> None:
    server = fake_http("dual")
    status, headers, body = modern_call(server.url, 1, "tools/list")
    assert status == 200 and body["result"]["resultType"] == "complete" and "mcp-session-id" not in headers
    assert "roots_probe" not in [t["name"] for t in body["result"]["tools"]]
    status, _, raw = http(server.url, rpc(2, "tools/list"))
    assert status == 400 and json.loads(raw)["error"]["code"] == -32000  # not a modern error: a legacy fallback cue
    sid = legacy_init(server)
    status, _, raw = http(server.url, rpc(3, "tools/list"), legacy_headers(sid))
    legacy = json.loads(raw)["result"]
    assert "roots_probe" in [t["name"] for t in legacy["tools"]] and "resultType" not in legacy
    status, _, body = modern_call(server.url, 4, "server/discover")
    assert body["result"]["supportedVersions"] == [MODERN, "2025-11-25", "2025-06-18"]


def test_http_x_mcp_header_mirroring(fake_http: Callable[..., FakeHttpMcpServer]) -> None:
    server = fake_http("modern", x_mcp_header=True)
    _, _, listed = modern_call(server.url, 1, "tools/list")
    echo = next(t for t in listed["result"]["tools"] if t["name"] == "echo")
    assert echo["inputSchema"]["properties"]["text"]["x-mcp-header"] == "Text"
    params = {"name": "echo", "arguments": {"text": "Hello, 世界"}}
    status, _, body = modern_call(server.url, 2, "tools/call", params)
    assert status == 400 and body["error"]["code"] == HEADER_MISMATCH
    encoded = "=?base64?" + base64.b64encode("Hello, 世界".encode()).decode() + "?="
    status, _, body = modern_call(server.url, 3, "tools/call", params, headers={"Mcp-Param-Text": encoded})
    assert status == 200 and body["result"]["content"][0]["text"] == "Hello, 世界"
    plain = {"name": "echo", "arguments": {"text": "us-west1"}}
    assert modern_call(server.url, 4, "tools/call", plain, headers={"Mcp-Param-Text": "us-west1"})[0] == 200
    assert modern_call(server.url, 5, "tools/call", plain, headers={"Mcp-Param-Text": "eu"})[0] == 400


def test_fake_http_server_rejects_unknown_era() -> None:
    with pytest.raises(ValueError, match="era"):
        FakeHttpMcpServer("future")
    with pytest.raises(RuntimeError, match="start"):
        _ = FakeHttpMcpServer().url
