"""A fake MCP server over Streamable HTTP, for offline tests of an MCP client.

Use it in-process (it binds 127.0.0.1 on a free port and serves from daemon threads)::

    from tests.fixtures.fake_mcp_http import FakeHttpMcpServer

    with FakeHttpMcpServer(era="legacy", sse=True) as server:   # or: server = FakeHttpMcpServer(...).start()
        client = MyClient(server.url)                              # http://127.0.0.1:<port>/mcp
        ...
        assert server.rpc_methods()[:2] == ["initialize", "notifications/initialized"]

The protocol logic (tools, eras, pagination, error codes) is shared with the stdio fake; see the docstring of
``tests/fixtures/fake_mcp_stdio.py`` for the tool catalogue and the era rules. This module adds the HTTP binding.

Constructor options (all keyword-only except ``era``)
-----------------------------------------------------
``era="legacy"|"modern"|"dual"``
    ``legacy`` is 2025-11-25 / 2025-06-18 Streamable HTTP; ``modern`` is 2026-07-28 (no sessions, header mirroring);
    ``dual`` serves an ``initialize`` POST, a client response, or any POST with an ``MCP-Session-Id`` header as
    legacy; a ``server/discover`` POST, or one whose ``params._meta`` carries ``io.modelcontextprotocol/protocolVersion``,
    as modern; anything else as legacy (so a bare ``tools/list`` gets the legacy 400 / -32000 "missing
    MCP-Session-Id", which carries no modern error code: the cue a dual-era client falls back to ``initialize`` on).
``extra_tools=0``, ``page_size=10``
    As for the stdio fake: ``filler_0`` .. ``filler_{N-1}`` and the ``tools/list`` page size.
``require_auth=None`` or ``(header_name, expected_value)``, e.g. ``("Authorization", "Bearer s3cret")``
    Every request without that exact header value gets 401 with ``WWW-Authenticate: Bearer realm="edgent-fake-mcp",
    resource_metadata="<base>/.well-known/oauth-protected-resource"`` (plus ``error="invalid_token"`` when a wrong
    value was sent). ``tools/list`` then reports ``cacheScope: "private"`` in the modern era.
``sse=False``
    False: every request is answered with one ``application/json`` body. True: requests that pass validation are
    answered with ``text/event-stream`` (chunked): legacy streams open with an empty priming event that has an id
    (``id: <stream>:0`` / ``data:``), modern streams (2026-07-28 removed SSE event ids) open with a comment line;
    then a ``: keep-alive`` comment, then ``notifications/progress`` events when the request's
    ``params._meta.progressToken`` is set, then (legacy, ``roots_probe`` only) a server-to-client ``roots/list``
    request, then the response event; then the stream ends. Events are ``event: message`` plus ``data: <json>``.
    Validation failures (400/401/403/404/406/415) are always plain JSON. With ``sse=False`` the legacy server has no
    stream for server-to-client requests, so ``roots_probe`` answers ``isError: true`` saying so.
``sse_crlf=False``
    True ends SSE lines with CRLF instead of LF (as sse-starlette does), to prove the client's parser handles both.
``sessions=True`` (legacy only)
    ``initialize`` mints an ``MCP-Session-Id`` response header; every later POST must carry it (400 without it, 404
    for an unknown, deleted or expired one); an ``initialize`` that carries one is 400. ``DELETE`` with the header
    ends the session (200; then 404). False is a stateless legacy server: no session ids, no initialize needed, and
    ``roots_probe`` skips its capability checks.
``expire_session_after=None`` or N
    A session serves N JSON-RPC requests (``initialize`` included; notifications and responses do not count); every
    later POST carrying that session id gets 404, so the client must re-initialize without the old id.
``x_mcp_header=False`` (modern only)
    True marks ``echo``'s ``text`` property with ``"x-mcp-header": "Text"`` and requires a matching
    ``Mcp-Param-Text`` header on ``tools/call echo`` (plain or ``=?base64?...?=``); mismatch is 400 / -32020.
``path="/mcp"``, ``instructions=...``
    The MCP endpoint path (any other path is a plain-text 404) and the instructions string.

Checks the fake always makes
----------------------------
* ``Origin`` present and not loopback (localhost, 127.0.0.1, ::1): 403 with a JSON-RPC error that has no id.
* POST ``Content-Type`` must be ``application/json`` (else 415); ``Accept`` must list both ``application/json`` and
  ``text/event-stream`` (else 406). A body that is not JSON is 400 / -32700; a batch (JSON array) is 400 / -32600.
* Notifications and client responses are answered 202 with no body. ``GET`` is 405 (no standalone stream).
* Legacy: an ``MCP-Protocol-Version`` header, when present, must be the version this session negotiated (400 else).
* Modern: ``_meta`` must carry the protocol version and client capabilities (400 / -32602); ``MCP-Protocol-Version``
  must equal ``_meta``'s version, ``Mcp-Method`` the method, and ``Mcp-Name`` (``tools/call``) the tool name, plain
  or ``=?base64?...?=`` (missing or different: 400 / -32020 ``HeaderMismatch``); an unsupported version is 400 /
  -32022 with ``data: {supported, requested}``; an unknown method is 404 / -32601; ``initialize`` is 400 / -32601
  naming the supported versions; ``MCP-Session-Id`` is ignored and never minted; ``DELETE`` is 405.

Cancellation: legacy honours ``notifications/cancelled`` (posted with the session id) and then sends no response
(the SSE stream ends, a JSON request's connection is closed). Modern treats the client closing the connection or
stream as cancellation. Either way the request id is appended to ``server.cancelled``.

What tests can inspect
----------------------
``server.requests``: one ``RecordedRequest`` per HTTP request, in arrival order, with ``http_method``, ``path``,
``headers`` (names lower-cased; ``.header(name)`` is case-insensitive), ``body`` (parsed JSON or None), ``raw_body``,
``status`` (None while running, or when nothing was sent because the request was cancelled), ``session_id``, and
the properties ``rpc_method`` and ``rpc_id``. ``server.rpc_methods()`` lists the JSON-RPC methods seen
(responses appear as ``"<response>"``). ``server.running()`` lists the ids executing now; ``server.cancelled``,
``server.session_ids()``, ``server.ended_sessions``, ``server.expire_sessions()`` (forces 404 for every live
session) and ``server.clear()`` round it off.

Client note: this machine routes outbound HTTP through a proxy set in the environment. A client (or test) that
talks to the fake must not send 127.0.0.1 through it: with urllib, build the opener with
``urllib.request.ProxyHandler({})``; with http.client there is nothing to do.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import json
import select
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

try:
    from .fake_mcp_stdio import (
        CANCEL_POLL_SECONDS,
        DEFAULT_INSTRUCTIONS,
        DEFAULT_PAGE_SIZE_TOOLS,
        ERAS,
        HEADER_MISMATCH,
        INVALID_REQUEST,
        LEGACY_VERSIONS,
        META_PROTOCOL_VERSION,
        PARSE_ERROR,
        Call,
        Channel,
        ChannelUnsupported,
        FakeMcpCore,
        LegacyState,
        RpcError,
        _Pending,
        classify,
        error_response,
        params_of,
    )
except ImportError:  # run as a script: the sibling module is on sys.path instead of the tests package
    from fake_mcp_stdio import (  # type: ignore[no-redef]
        CANCEL_POLL_SECONDS,
        DEFAULT_INSTRUCTIONS,
        DEFAULT_PAGE_SIZE_TOOLS,
        ERAS,
        HEADER_MISMATCH,
        INVALID_REQUEST,
        LEGACY_VERSIONS,
        META_PROTOCOL_VERSION,
        PARSE_ERROR,
        Call,
        Channel,
        ChannelUnsupported,
        FakeMcpCore,
        LegacyState,
        RpcError,
        _Pending,
        classify,
        error_response,
        params_of,
    )

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
NAMED_METHODS = {"tools/call": "name", "prompts/get": "name", "resources/read": "uri"}
LEGACY_BAD_REQUEST = -32000  # what the reference SDK answers for session problems (legacy, implementation-defined)
LEGACY_SESSION_NOT_FOUND = -32001
STOP_JOIN_SECONDS = 2.0
SERVE_POLL_SECONDS = 0.05
MAX_BODY_BYTES = 10_000_000
BASE64_PREFIX, BASE64_SUFFIX = "=?base64?", "?="


@dataclass
class RecordedRequest:
    """One HTTP request as the fake saw it."""

    http_method: str
    path: str
    headers: dict[str, str]
    body: Any
    raw_body: str
    status: int | None = None
    session_id: str | None = None

    def header(self, name: str) -> str | None:
        return self.headers.get(name.lower())

    @property
    def rpc_method(self) -> str | None:
        if not isinstance(self.body, dict):
            return None
        if "method" in self.body:
            return self.body["method"]
        return "<response>" if "id" in self.body else None

    @property
    def rpc_id(self) -> Any:
        return self.body.get("id") if isinstance(self.body, dict) else None


@dataclass
class HttpSession:
    """A legacy session: what initialize negotiated, how many requests it served, what is in flight."""

    id: str
    state: LegacyState
    requests: int = 0
    ended: bool = False
    inflight: dict[str, threading.Event] = field(default_factory=dict)


def _decode_header_value(value: str) -> str:
    if value.startswith(BASE64_PREFIX) and value.endswith(BASE64_SUFFIX) and len(value) >= 11:
        inner = value[len(BASE64_PREFIX):-len(BASE64_SUFFIX)]
        try:
            return base64.b64decode(inner, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError) as exc:
            raise ValueError("invalid base64 sentinel value") from exc
    return value


def _header_form(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return value if isinstance(value, str) else str(value)


def _peer_closed(sock: socket.socket) -> bool:
    """True when the client has closed its end (EOF or reset), without consuming pipelined bytes."""
    try:
        readable, _, _ = select.select([sock], [], [], 0)
        return bool(readable) and sock.recv(1, socket.MSG_PEEK) == b""
    except (OSError, ValueError):
        return True


class _SseStream:
    """A chunked ``text/event-stream`` response for one request."""

    def __init__(self, handler: BaseHTTPRequestHandler, legacy: bool, crlf: bool) -> None:
        self._h = handler
        self._legacy = legacy
        self._eol = "\r\n" if crlf else "\n"
        self._stream_id = uuid.uuid4().hex[:8]
        self._events = 0
        self._lock = threading.Lock()
        self.broken = False

    def start(self, extra_headers: dict[str, str]) -> None:
        h = self._h
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.send_header("Cache-Control", "no-cache")
        h.send_header("X-Accel-Buffering", "no")
        h.send_header("Transfer-Encoding", "chunked")
        h.send_header("Connection", "close")  # the stream ends the connection; say so, so clients do not reuse it
        for name, value in extra_headers.items():
            h.send_header(name, value)
        h.end_headers()
        eol = self._eol
        self._write(f"id: {self._next_id()}{eol}data:{eol}{eol}" if self._legacy else f": stream open{eol}{eol}")
        self._write(f": keep-alive{eol}{eol}")

    def _next_id(self) -> str:
        event_id = f"{self._stream_id}:{self._events}"
        self._events += 1
        return event_id

    def event(self, message: dict[str, Any]) -> None:
        eol = self._eol
        data = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            event_id = f"id: {self._next_id()}{eol}" if self._legacy else ""
        self._write(f"event: message{eol}{event_id}data: {data}{eol}{eol}")

    def finish(self) -> None:
        self._write_raw(b"0\r\n\r\n")

    def _write(self, text: str) -> None:
        payload = text.encode("utf-8")
        self._write_raw(f"{len(payload):X}\r\n".encode() + payload + b"\r\n")

    def _write_raw(self, data: bytes) -> None:
        with self._lock:
            if self.broken:
                return
            try:
                self._h.wfile.write(data)
                self._h.wfile.flush()
            except OSError:
                self.broken = True


class _HttpChannel(Channel):
    def __init__(self, fake: FakeHttpMcpServer, handler: BaseHTTPRequestHandler, call: Call,
                 cancel: threading.Event, stream: _SseStream | None) -> None:
        self._fake, self._h, self._call, self._cancel, self._stream = fake, handler, call, cancel, stream

    def notify(self, message: dict[str, Any]) -> None:
        if self._stream is not None:
            self._stream.event(message)

    def cancelled(self) -> bool:
        if self._cancel.is_set() or self._fake.stopping or (self._stream is not None and self._stream.broken):
            return True
        # 2026-07-28: closing the response stream is the cancellation signal. Legacy: a disconnect is not a cancel.
        return self._call.modern and _peer_closed(self._h.connection)

    def ask_client(self, method: str, params: dict[str, Any] | None, timeout_s: float) -> dict[str, Any] | None:
        if self._call.modern:
            raise ChannelUnsupported("2026-07-28 servers never send requests to the client")
        if self._stream is None:
            raise ChannelUnsupported("this fake runs with sse=False, so it has no stream to send roots/list on; "
                                     "start it with sse=True")
        return self._fake._ask_client(self._stream, method, params, timeout_s, self.cancelled)


class _HttpServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    allow_reuse_address = True
    fake: FakeHttpMcpServer

    def handle_error(self, request: Any, client_address: Any) -> None:
        return  # a client that hangs up mid-stream is expected here; the default prints a traceback to stderr


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "edgent-fake-mcp/1.0"
    server: _HttpServer

    def log_message(self, format: str, *args: Any) -> None:
        return  # the default access log writes to stderr; tests read server.requests instead

    def do_POST(self) -> None:
        self.server.fake._on_post(self)

    def do_GET(self) -> None:
        self.server.fake._on_other(self, "GET")

    def do_DELETE(self) -> None:
        self.server.fake._on_other(self, "DELETE")


class FakeHttpMcpServer:
    """Streamable HTTP MCP test double on 127.0.0.1; see the module docstring for every option and check."""

    def __init__(self, era: str = "legacy", *, extra_tools: int = 0, page_size: int = DEFAULT_PAGE_SIZE_TOOLS,
                 require_auth: tuple[str, str] | None = None, sse: bool = False, sse_crlf: bool = False,
                 sessions: bool = True, expire_session_after: int | None = None, x_mcp_header: bool = False,
                 path: str = "/mcp", instructions: str = DEFAULT_INSTRUCTIONS) -> None:
        if era not in ERAS:
            raise ValueError(f"era must be one of {ERAS}, got {era!r}")
        self.era = era
        self.core = FakeMcpCore(era, extra_tools=extra_tools, page_size=page_size, instructions=instructions,
                                cache_scope="private" if require_auth else "public", x_mcp_header=x_mcp_header)
        self.require_auth = require_auth
        self.sse = sse
        self.sse_crlf = sse_crlf
        self.sessions = sessions
        self.expire_session_after = expire_session_after
        self.path = path
        self.requests: list[RecordedRequest] = []
        self.cancelled: list[Any] = []
        self.ended_sessions: list[str] = []
        self._sessions: dict[str, HttpSession] = {}
        self._stateless = HttpSession("", LegacyState(strict=False))
        self._modern_state = LegacyState(strict=False)
        self._pending: dict[str, _Pending] = {}
        self._running: list[Any] = []
        self._server_ids = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._httpd: _HttpServer | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle -------------------------------------------------------------------------------------------------

    def start(self) -> FakeHttpMcpServer:
        if self._httpd is not None:
            return self
        httpd = _HttpServer(("127.0.0.1", 0), _Handler)
        httpd.fake = self
        self._httpd = httpd
        self._thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": SERVE_POLL_SECONDS},
                                        name="fake-mcp-http", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(STOP_JOIN_SECONDS)
        self._httpd, self._thread = None, None

    def __enter__(self) -> FakeHttpMcpServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    @property
    def port(self) -> int:
        if self._httpd is None:
            raise RuntimeError("FakeHttpMcpServer is not started; call start() first")
        return int(self._httpd.server_address[1])

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def url(self) -> str:
        return self.base_url + self.path

    # -- inspection ------------------------------------------------------------------------------------------------

    def rpc_methods(self) -> list[str]:
        with self._lock:
            return [r.rpc_method for r in self.requests if r.rpc_method]

    def running(self) -> list[Any]:
        """Ids of the JSON-RPC requests executing right now (wait on this before cancelling one)."""
        with self._lock:
            return list(self._running)

    def session_ids(self) -> list[str]:
        with self._lock:
            return list(self._sessions)

    def expire_sessions(self) -> None:
        with self._lock:
            for session in self._sessions.values():
                session.ended = True

    def clear(self) -> None:
        with self._lock:
            self.requests.clear()
            self.cancelled.clear()

    # -- plumbing --------------------------------------------------------------------------------------------------

    def _record(self, h: BaseHTTPRequestHandler, method: str, headers: dict[str, str], raw: str) -> RecordedRequest:
        try:
            body = json.loads(raw) if raw else None
        except ValueError:
            body = None
        rec = RecordedRequest(method, h.path, headers, body, raw)
        with self._lock:
            self.requests.append(rec)
        return rec

    def _reply(self, h: BaseHTTPRequestHandler, rec: RecordedRequest, status: int, body: Any = None,
               headers: dict[str, str] | None = None, text: str | None = None) -> None:
        if text is not None:
            payload, ctype = text.encode("utf-8"), "text/plain; charset=utf-8"
        else:
            payload, ctype = (b"" if body is None else json.dumps(body).encode("utf-8")), "application/json"
        rec.status = status  # before the bytes leave, so a client that saw the reply also sees the record
        h.send_response(status)
        if payload:
            h.send_header("Content-Type", ctype)
        h.send_header("Content-Length", str(len(payload)))
        for name, value in (headers or {}).items():
            h.send_header(name, value)
        h.end_headers()
        try:
            if payload:
                h.wfile.write(payload)
            h.wfile.flush()
        except OSError:
            h.close_connection = True

    def _error(self, h: BaseHTTPRequestHandler, rec: RecordedRequest, status: int, code: int, message: str,
               msg_id: Any = None, data: Any = None, headers: dict[str, str] | None = None) -> None:
        body = error_response(msg_id, RpcError(code, message, data))
        if msg_id is None:
            del body["id"]  # transport-level refusals carry no id (the request may not even be JSON-RPC)
        self._reply(h, rec, status, body, headers)

    @staticmethod
    def _read_body(h: BaseHTTPRequestHandler, headers: dict[str, str]) -> str | None:
        if "chunked" in headers.get("transfer-encoding", "").lower():
            chunks: list[bytes] = []
            while (size := int(h.rfile.readline().split(b";")[0].strip() or b"0", 16)) > 0:
                chunks.append(h.rfile.read(size))
                h.rfile.readline()
            while h.rfile.readline() not in (b"\r\n", b"\n", b""):
                pass  # trailers
            return b"".join(chunks).decode("utf-8", "replace")
        length = headers.get("content-length")
        if length is None or not length.isdigit() or int(length) > MAX_BODY_BYTES:
            return None
        return h.rfile.read(int(length)).decode("utf-8", "replace")

    def _gate(self, h: BaseHTTPRequestHandler, rec: RecordedRequest, headers: dict[str, str]) -> bool:
        """Path, Origin and auth checks shared by every HTTP method; True when a refusal was already sent."""
        if urlsplit(h.path).path != self.path:
            self._reply(h, rec, 404, text=f"Not Found: the MCP endpoint is {self.path}")
            return True
        origin = headers.get("origin")
        if origin is not None and urlsplit(origin).hostname not in LOOPBACK_HOSTS:
            self._error(h, rec, 403, LEGACY_BAD_REQUEST, f"Forbidden: Origin {origin!r} is not a loopback origin")
            return True
        if self.require_auth is not None:
            name, expected = self.require_auth
            got = headers.get(name.lower())
            if got != expected:
                challenge = (f'Bearer realm="edgent-fake-mcp", '
                             f'resource_metadata="{self.base_url}/.well-known/oauth-protected-resource"')
                if got is not None:
                    challenge += ', error="invalid_token"'
                self._error(h, rec, 401, LEGACY_BAD_REQUEST, f"Unauthorized: send the {name} header",
                            headers={"WWW-Authenticate": challenge})
                return True
        return False

    def _on_other(self, h: BaseHTTPRequestHandler, method: str) -> None:
        headers = {k.lower(): v for k, v in h.headers.items()}
        raw = self._read_body(h, headers) or ""
        rec = self._record(h, method, headers, raw)
        if self._gate(h, rec, headers):
            return
        session_capable = self.era != "modern" and self.sessions
        if method == "GET" or not session_capable:
            allow = "POST, DELETE" if session_capable else "POST"
            self._error(h, rec, 405, LEGACY_BAD_REQUEST, f"Method Not Allowed: {method}", headers={"Allow": allow})
            return
        sid = headers.get("mcp-session-id")
        if sid is None:
            self._error(h, rec, 400, LEGACY_BAD_REQUEST, "Bad Request: DELETE needs an MCP-Session-Id header")
            return
        with self._lock:
            session = self._sessions.get(sid)
            known = session is not None and not session.ended and not self._expired(session)
            if known and session is not None:
                session.ended = True
                self.ended_sessions.append(sid)
        rec.session_id = sid
        if not known:
            self._error(h, rec, 404, LEGACY_SESSION_NOT_FOUND, "Session not found")
            return
        self._reply(h, rec, 200)

    def _expired(self, session: HttpSession) -> bool:
        return self.expire_session_after is not None and session.requests >= self.expire_session_after

    def _on_post(self, h: BaseHTTPRequestHandler) -> None:
        headers = {k.lower(): v for k, v in h.headers.items()}
        raw = self._read_body(h, headers)
        rec = self._record(h, "POST", headers, raw or "")
        if raw is None:
            h.close_connection = True
            self._error(h, rec, 411, INVALID_REQUEST, "Length Required: send Content-Length")
            return
        if self._gate(h, rec, headers):
            return
        if headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
            self._error(h, rec, 415, LEGACY_BAD_REQUEST, "Unsupported Media Type: Content-Type must be application/json")
            return
        accept = headers.get("accept", "").lower()
        if "application/json" not in accept or "text/event-stream" not in accept:
            self._error(h, rec, 406, LEGACY_BAD_REQUEST,
                        "Not Acceptable: Accept must list both application/json and text/event-stream")
            return
        if rec.body is None:
            self._error(h, rec, 400, PARSE_ERROR, "Parse error: the body is not JSON")
            return
        kind = classify(rec.body)
        if kind == "invalid":
            reason = "batches are not supported" if isinstance(rec.body, list) else "not a JSON-RPC 2.0 message"
            self._error(h, rec, 400, INVALID_REQUEST, f"Invalid Request: {reason}")
            return
        legacy_shaped = kind == "response" or rec.body.get("method") == "initialize" or "mcp-session-id" in headers
        if self.era == "modern" or (self.era == "dual" and not legacy_shaped and self.core.is_modern(rec.body)):
            self._post_modern(h, rec, headers, rec.body, kind)
        else:
            self._post_legacy(h, rec, headers, rec.body, kind)

    # -- legacy ----------------------------------------------------------------------------------------------------

    def _legacy_session(self, h: BaseHTTPRequestHandler, rec: RecordedRequest, headers: dict[str, str],
                        msg: dict[str, Any], kind: str) -> tuple[HttpSession, dict[str, str]] | None:
        """Find or mint the session for a legacy POST; None when a refusal was already sent."""
        if not self.sessions:
            version = headers.get("mcp-protocol-version")
            if version is not None and version not in LEGACY_VERSIONS:
                self._error(h, rec, 400, LEGACY_BAD_REQUEST, f"Bad Request: unsupported MCP-Protocol-Version {version}")
                return None
            return self._stateless, {}
        sid = headers.get("mcp-session-id")
        if kind == "request" and msg.get("method") == "initialize":
            if sid is not None:
                self._error(h, rec, 400, INVALID_REQUEST, "Bad Request: initialize must not carry MCP-Session-Id; "
                            "start a new session without it", msg.get("id"))
                return None
            session = HttpSession(uuid.uuid4().hex, LegacyState())
            with self._lock:
                self._sessions[session.id] = session
            return session, {"MCP-Session-Id": session.id}
        if sid is None:
            self._error(h, rec, 400, LEGACY_BAD_REQUEST, "Bad Request: missing MCP-Session-Id header; "
                        "send initialize first", msg.get("id"))
            return None
        with self._lock:
            known = self._sessions.get(sid)
            live = known is not None and not known.ended and not self._expired(known)
        if not live or known is None:
            self._error(h, rec, 404, LEGACY_SESSION_NOT_FOUND, "Session not found", msg.get("id"))
            return None
        version = headers.get("mcp-protocol-version")
        if version is not None and version != known.state.protocol_version:
            self._error(h, rec, 400, LEGACY_BAD_REQUEST, f"Bad Request: MCP-Protocol-Version {version} is not the "
                        f"version this session negotiated ({known.state.protocol_version})", msg.get("id"))
            return None
        return known, {}

    def _post_legacy(self, h: BaseHTTPRequestHandler, rec: RecordedRequest, headers: dict[str, str],
                     msg: dict[str, Any], kind: str) -> None:
        found = self._legacy_session(h, rec, headers, msg, kind)
        if found is None:
            return
        session, extra = found
        rec.session_id = session.id or None
        if kind == "notification":
            self._legacy_notification(session, msg)
            self._reply(h, rec, 202)
            return
        if kind == "response":
            with self._lock:
                pending = self._pending.get(json.dumps(msg.get("id")))
            if pending is not None:
                pending.reply = msg
                pending.done.set()
            self._reply(h, rec, 202)
            return
        with self._lock:
            session.requests += 1
        try:
            call = self.core.prepare(msg, session.state)
        except RpcError as err:
            self._reply(h, rec, err.http_status or 200, error_response(msg.get("id"), err), extra)
            return
        self._run(h, rec, call, session, extra)

    def _legacy_notification(self, session: HttpSession, msg: dict[str, Any]) -> None:
        params = params_of(msg)
        if msg["method"] == "notifications/initialized":
            session.state.ready = True
        elif msg["method"] == "notifications/cancelled":
            with self._lock:
                event = session.inflight.get(json.dumps(params.get("requestId")))
            if event is not None:
                event.set()

    # -- modern ----------------------------------------------------------------------------------------------------

    def _post_modern(self, h: BaseHTTPRequestHandler, rec: RecordedRequest, headers: dict[str, str],
                     msg: dict[str, Any], kind: str) -> None:
        if kind == "response":
            self._error(h, rec, 400, INVALID_REQUEST, "Invalid Request: 2026-07-28 clients never send responses")
            return
        if kind == "notification":
            self._reply(h, rec, 202)  # nothing to act on: closing the stream is how 2026-07-28 cancels over HTTP
            return
        params = params_of(msg)
        try:
            if msg.get("method") == "initialize":
                raise self.core.modern_initialize_error(params)
            self.core.check_modern_meta(params)
            problem = self._header_problem(msg, params, headers)
            if problem:
                raise RpcError(HEADER_MISMATCH, problem, http_status=400)
            call = self.core.prepare(msg, self._modern_state)
        except RpcError as err:
            self._reply(h, rec, err.http_status or 200, error_response(msg.get("id"), err))
            return
        self._run(h, rec, call, None, {})

    def _header_problem(self, msg: dict[str, Any], params: dict[str, Any], headers: dict[str, str]) -> str | None:
        method = msg["method"]
        checks: list[tuple[str, Any]] = [("MCP-Protocol-Version", params["_meta"][META_PROTOCOL_VERSION]),
                                         ("Mcp-Method", method)]
        if method in NAMED_METHODS:
            checks.append(("Mcp-Name", params.get(NAMED_METHODS[method])))
        if self.core.x_mcp_header and method == "tools/call" and params.get("name") == "echo":
            raw_args = params.get("arguments")
            args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
            if args.get("text") is not None:
                checks.append(("Mcp-Param-Text", args["text"]))
            elif "mcp-param-text" in headers:
                return "Header mismatch: Mcp-Param-Text is present but arguments.text is absent"
        for name, expected in checks:
            got = headers.get(name.lower())
            if got is None:
                return f"Header mismatch: required header {name} is missing"
            try:
                value = _decode_header_value(got) if name.lower().startswith(("mcp-name", "mcp-param-")) else got
            except ValueError:
                return f"Header mismatch: {name} carries an invalid =?base64?...?= value"
            if value != _header_form(expected):
                return f"Header mismatch: {name} header value {got!r} does not match body value {expected!r}"
        return None

    # -- execution -------------------------------------------------------------------------------------------------

    def _run(self, h: BaseHTTPRequestHandler, rec: RecordedRequest, call: Call, session: HttpSession | None,
             extra: dict[str, str]) -> None:
        cancel = threading.Event()
        key = json.dumps(call.id)
        with self._lock:
            self._running.append(call.id)
            if session is not None:
                session.inflight[key] = cancel
        state = session.state if session is not None else self._modern_state
        try:
            if self.sse:
                stream = _SseStream(h, legacy=not call.modern, crlf=self.sse_crlf)
                rec.status = 200
                stream.start(extra)
                response = self.core.execute(call, state, _HttpChannel(self, h, call, cancel, stream))
                if response is not None:
                    stream.event(response)
                stream.finish()
                h.close_connection = True
            else:
                response = self.core.execute(call, state, _HttpChannel(self, h, call, cancel, None))
                if response is not None:
                    self._reply(h, rec, 200, response, extra)
                else:
                    h.close_connection = True  # a cancelled request gets no response at all
        finally:
            with self._lock:
                self._running.remove(call.id)
                if session is not None:
                    session.inflight.pop(key, None)
        if response is None:
            with self._lock:
                self.cancelled.append(call.id)

    def _ask_client(self, stream: _SseStream, method: str, params: dict[str, Any] | None, timeout_s: float,
                    cancelled: Any) -> dict[str, Any] | None:
        with self._lock:
            self._server_ids += 1
            request_id = f"fake-srv-{self._server_ids}"
            pending = self._pending[json.dumps(request_id)] = _Pending()
        request: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            request["params"] = params
        stream.event(request)
        deadline = time.monotonic() + timeout_s
        try:
            while not pending.done.wait(CANCEL_POLL_SECONDS):
                if cancelled():
                    return None
                if time.monotonic() > deadline:
                    raise TimeoutError(method)
            return pending.reply
        finally:
            with self._lock:
                self._pending.pop(json.dumps(request_id), None)


def main(argv: list[str] | None = None) -> int:
    """Serve until interrupted, for poking at by hand: ``python3 tests/fixtures/fake_mcp_http.py --era dual --sse``."""
    parser = argparse.ArgumentParser(description="Fake Streamable HTTP MCP server on 127.0.0.1.")
    parser.add_argument("--era", choices=ERAS, default="legacy")
    parser.add_argument("--sse", action="store_true")
    parser.add_argument("--extra-tools", type=int, default=0)
    args = parser.parse_args(argv)
    server = FakeHttpMcpServer(args.era, sse=args.sse, extra_tools=args.extra_tools).start()
    print(server.url, flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
