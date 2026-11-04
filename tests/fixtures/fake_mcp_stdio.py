"""A fake MCP server over stdio, and the protocol core the HTTP fake shares, for offline tests of an MCP client.

Run it the way a client launches a real server (``-I`` keeps the interpreter isolated; this file imports only the
standard library, so it needs nothing from the repository)::

    python3 -I tests/fixtures/fake_mcp_stdio.py [options]

It speaks newline-delimited JSON-RPC 2.0 on stdin/stdout, one message per line, and flushes every line. Each request
is served on its own thread, so ``slow`` never blocks ``echo``; ``initialize`` and every notification are handled
in order on the reader thread, so a client that pipelines ``initialize`` and ``notifications/initialized`` is safe.
When stdin reaches end-of-file the server waits up to ``STDIN_EOF_GRACE_SECONDS`` for in-flight answers, then exits 0.

Options
-------
``--era legacy|modern|dual`` (default ``legacy``)
    ``legacy`` speaks 2025-11-25 and 2025-06-18: ``initialize`` (the server echoes the client's protocolVersion when it
    supports it, otherwise answers ``2025-11-25``), ``notifications/initialized``, ``ping``, ``tools/list`` (cursor
    pagination), ``tools/call``, ``notifications/cancelled``, and one server-to-client request (``roots/list``, sent
    when the ``roots_probe`` tool is called). Any request but ``initialize``/``ping`` before ``initialize`` is answered
    with -32600; ``server/discover`` is an unknown method (-32601), which is the signal a dual-era client probes for.
    ``modern`` speaks 2026-07-28 (stateless): no handshake; ``initialize`` is answered with -32601 whose message names
    the supported versions; ``server/discover`` returns ``{supportedVersions, capabilities, serverInfo, instructions,
    resultType: "complete"}``; every request must carry ``params._meta["io.modelcontextprotocol/protocolVersion"]``
    and ``params._meta["io.modelcontextprotocol/clientCapabilities"]`` (else -32602); an unsupported version is
    -32022 ``UnsupportedProtocolVersion`` with ``data: {supported, requested}``; every result carries
    ``resultType: "complete"`` and ``_meta["io.modelcontextprotocol/serverInfo"]``; ``tools/list`` also carries
    ``ttlMs`` and ``cacheScope``; ``ping`` no longer exists (-32601). ``roots_probe`` is not offered.
    ``dual`` answers both: ``initialize`` starts legacy semantics for this process; a request whose ``_meta`` carries
    the protocol version (or any ``server/discover``) is served statelessly as modern, before or after ``initialize``.
``--extra-tools N`` (default 0)
    Append N tools named ``filler_0`` .. ``filler_{N-1}`` (no arguments; each answers with its own name).
``--page-size N`` (default ``DEFAULT_PAGE_SIZE_TOOLS`` = 10 tools per page)
    ``tools/list`` page size; a page that is not the last carries an opaque ``nextCursor``. A bad cursor is -32602.
``--noise``
    Print one non-JSON line on stdout before anything else, the way ``npx`` prints install chatter.
``--crash-after N``
    Answer the first N requests (notifications do not count), then exit with code ``CRASH_EXIT_CODE`` (3) the moment
    request N+1 arrives, without answering it. Simulates a server that dies mid-call.
``--stderr-chatter``
    Log a banner and one line per message to stderr (stdout stays pure protocol).
``--stderr-flood-kb N``
    Write N KiB to stderr before reading stdin. With N above the pipe buffer (64 KiB on Linux) a client that never
    drains stderr deadlocks the server; a correct client does not notice.
``--require-env NAME`` or ``--require-env NAME=VALUE`` (repeatable)
    Exit with code 1 and a message on stderr unless NAME is set (to VALUE, when given). Proves env passing.
``--record PATH``
    Append one JSON line per event to PATH: ``{"dir": "in", "msg": ...}`` for each message read, ``{"dir": "out",
    "msg": ...}`` for each message written, ``{"dir": "event", "event": "cancelled", "id": ...}`` when a request is
    abandoned because of ``notifications/cancelled``.

Tools (both eras unless noted; listing order is fixed)
------------------------------------------------------
``echo {text}``           one text part with ``text``. Annotations ``readOnlyHint: true, openWorldHint: false``.
``add {a, b}``            text ``{"sum": a+b}`` plus ``structuredContent {"sum": a+b}``; has an ``outputSchema``.
                          Annotations ``readOnlyHint: true, openWorldHint: false``.
``image {}``              one text part and one image part (a valid 1x1 PNG, base64, ``mimeType: image/png``).
``fail {}``               ``isError: true`` with a text explanation. Annotation ``destructiveHint: true``.
``slow {seconds}``        sleeps (at most ``SLOW_MAX_SECONDS``) then answers ``slept N s``; aborts without answering
                          when the request is cancelled.
``resource {}``           a ``resource_link`` and an embedded ``resource`` with text.
``big {chars}``           one text part of exactly ``chars`` characters (default ``BIG_DEFAULT_CHARS``, at most
                          ``BIG_MAX_CHARS``), the digits ``0123456789`` repeated, so truncation is detectable.
``roots_probe {}``        legacy only: sends ``roots/list`` to the client, waits up to ``CLIENT_REPLY_TIMEOUT_SECONDS``
                          for its answer and returns the roots as text and as ``structuredContent {"roots": [...]}``.
                          ``isError: true`` when the client did not declare the ``roots`` capability, has not sent
                          ``notifications/initialized``, answered with an error, or did not answer in time.
``filler_i {}``           see ``--extra-tools``.

Every ``tools/call`` whose ``params._meta.progressToken`` is set emits two ``notifications/progress`` (0 of 1 before
the tool runs, 1 of 1 after it finishes) before the response. Bad arguments are a tool execution error
(``isError: true``); an unknown tool name is the protocol error -32602.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, BinaryIO

SERVER_INFO: dict[str, str] = {"name": "edgent-fake-mcp", "title": "Edgent fake MCP server", "version": "1.0.0"}
LEGACY_VERSIONS: tuple[str, ...] = ("2025-11-25", "2025-06-18")
DEFAULT_LEGACY_VERSION = "2025-11-25"
MODERN_VERSIONS: tuple[str, ...] = ("2026-07-28",)
ERAS: tuple[str, ...] = ("legacy", "modern", "dual")
DEFAULT_INSTRUCTIONS = "Fake MCP server for Edgent tests: echo, add, image, fail, slow, resource, big tools."

META_PROTOCOL_VERSION = "io.modelcontextprotocol/protocolVersion"
META_CLIENT_CAPABILITIES = "io.modelcontextprotocol/clientCapabilities"
META_CLIENT_INFO = "io.modelcontextprotocol/clientInfo"
META_SERVER_INFO = "io.modelcontextprotocol/serverInfo"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
HEADER_MISMATCH = -32020
MISSING_REQUIRED_CLIENT_CAPABILITY = -32021
UNSUPPORTED_PROTOCOL_VERSION = -32022

DEFAULT_PAGE_SIZE_TOOLS = 10
TOOLS_TTL_MS = 300_000
SLOW_MAX_SECONDS = 30.0
BIG_DEFAULT_CHARS = 20_000
BIG_MAX_CHARS = 5_000_000
CLIENT_REPLY_TIMEOUT_SECONDS = 10.0
CANCEL_POLL_SECONDS = 0.02
STDIN_EOF_GRACE_SECONDS = 2.0
CRASH_EXIT_CODE = 3
ENV_MISSING_EXIT_CODE = 1
NOISE_LINE = "npm warn exec The following package was not found and will be installed: edgent-fake-mcp@1.0.0"

PNG_1X1_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
RESOURCE_URI = "fake://docs/readme.txt"
RESOURCE_TEXT = "Fake readme: this text came from an embedded resource."

LEGACY_METHODS = frozenset({"initialize", "ping", "tools/list", "tools/call"})
MODERN_METHODS = frozenset({"server/discover", "tools/list", "tools/call"})


class RpcError(Exception):
    """A JSON-RPC error; ``http_status`` is the status the HTTP fake answers with (None means 200)."""

    def __init__(self, code: int, message: str, data: Any = None, http_status: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data
        self.http_status = http_status

    def to_error(self) -> dict[str, Any]:
        err: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            err["data"] = self.data
        return err


class ChannelUnsupported(Exception):
    """The transport cannot carry what the tool needs (a server-to-client request on a plain JSON response)."""


def error_response(msg_id: Any, err: RpcError) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": err.to_error()}


@dataclass
class LegacyState:
    """What one legacy connection (a stdio process or an HTTP session) negotiated. ``strict=False`` is stateless."""

    initialized: bool = False
    ready: bool = False  # notifications/initialized received
    protocol_version: str | None = None
    client_capabilities: dict[str, Any] = field(default_factory=dict)
    client_info: dict[str, Any] = field(default_factory=dict)
    strict: bool = True


@dataclass
class Call:
    """A validated request, ready to execute."""

    id: Any
    method: str
    params: dict[str, Any]
    modern: bool
    version: str | None = None

    @property
    def progress_token(self) -> Any:
        meta = self.params.get("_meta")
        return meta.get("progressToken") if isinstance(meta, dict) else None


class Channel:
    """What a tool may do on the wire while it runs. Subclasses override what their transport can carry."""

    def notify(self, message: dict[str, Any]) -> None:
        """Send a notification related to the request (dropped when the transport cannot carry it)."""

    def ask_client(self, method: str, params: dict[str, Any] | None, timeout_s: float) -> dict[str, Any] | None:
        """Send a server-to-client request; return the client's response message, or None when cancelled."""
        raise ChannelUnsupported("this transport cannot send requests to the client")

    def cancelled(self) -> bool:
        return False


def encode_cursor(offset: int) -> str:
    return base64.urlsafe_b64encode(f"offset:{offset}".encode()).decode().rstrip("=")


def decode_cursor(cursor: Any) -> int:
    if not isinstance(cursor, str):
        raise RpcError(INVALID_PARAMS, "Invalid cursor: must be a string from a previous nextCursor")
    try:
        text = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode()
        prefix, _, number = text.partition(":")
        offset = int(number)
    except (ValueError, binascii.Error, UnicodeDecodeError):
        raise RpcError(INVALID_PARAMS, f"Invalid cursor {cursor!r}: pass back the nextCursor value unchanged") from None
    if prefix != "offset" or offset < 0:
        raise RpcError(INVALID_PARAMS, f"Invalid cursor {cursor!r}: pass back the nextCursor value unchanged")
    return offset


def _text(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _tool_error(text: str) -> dict[str, Any]:
    return {"content": [_text(text)], "isError": True}


def _no_args() -> dict[str, Any]:
    return {"type": "object", "properties": {}, "additionalProperties": False}


READ_ONLY_CLOSED = {"readOnlyHint": True, "openWorldHint": False}


def _core_tool_defs(legacy: bool, x_mcp_header: bool) -> list[dict[str, Any]]:
    text_prop: dict[str, Any] = {"type": "string", "description": "the text to send back unchanged"}
    if x_mcp_header and not legacy:
        text_prop["x-mcp-header"] = "Text"
    tools: list[dict[str, Any]] = [
        {"name": "echo", "title": "Echo", "description": "Answer with the given text unchanged.",
         "inputSchema": {"type": "object", "properties": {"text": text_prop}, "required": ["text"],
                         "additionalProperties": False},
         "annotations": dict(READ_ONLY_CLOSED)},
        {"name": "add", "title": "Add", "description": "Add two numbers; the sum comes back as structured content.",
         "inputSchema": {"type": "object", "properties": {"a": {"type": "number", "description": "first addend"},
                                                          "b": {"type": "number", "description": "second addend"}},
                         "required": ["a", "b"], "additionalProperties": False},
         "outputSchema": {"type": "object", "properties": {"sum": {"type": "number"}}, "required": ["sum"]},
         "annotations": dict(READ_ONLY_CLOSED)},
        {"name": "image", "description": "Answer with a caption and a 1x1 PNG image.", "inputSchema": _no_args()},
        {"name": "fail", "description": "Always fails with a tool execution error (isError true).",
         "inputSchema": _no_args(), "annotations": {"destructiveHint": True}},
        {"name": "slow", "description": "Sleep for the given number of seconds, then answer; honours cancellation.",
         "inputSchema": {"type": "object", "properties": {"seconds": {
             "type": "number", "minimum": 0, "maximum": SLOW_MAX_SECONDS, "description": "how long to sleep, seconds"}},
             "required": ["seconds"], "additionalProperties": False}},
        {"name": "resource", "description": "Answer with a resource link and an embedded text resource.",
         "inputSchema": _no_args()},
        {"name": "big", "description": "Answer with a long text of exactly `chars` characters.",
         "inputSchema": {"type": "object", "properties": {"chars": {
             "type": "integer", "minimum": 0, "maximum": BIG_MAX_CHARS, "description": "length of the text, chars"}},
             "additionalProperties": False}},
    ]
    if legacy:
        tools.append({"name": "roots_probe", "description": "Ask the client for its roots (roots/list) and report them.",
                      "inputSchema": _no_args()})
    return tools


class FakeMcpCore:
    """Transport-free protocol logic shared by the stdio and HTTP fakes. Thread-safe: holds no per-request state."""

    def __init__(self, era: str = "legacy", *, extra_tools: int = 0, page_size: int = DEFAULT_PAGE_SIZE_TOOLS,
                 instructions: str = DEFAULT_INSTRUCTIONS, cache_scope: str = "public",
                 x_mcp_header: bool = False) -> None:
        if era not in ERAS:
            raise ValueError(f"era must be one of {ERAS}, got {era!r}")
        if page_size < 1 or extra_tools < 0:
            raise ValueError("page_size must be >= 1 and extra_tools >= 0")
        self.era = era
        self.extra_tools = extra_tools
        self.page_size = page_size
        self.instructions = instructions
        self.cache_scope = cache_scope
        self.x_mcp_header = x_mcp_header

    # -- catalogue -------------------------------------------------------------------------------------------------

    def tools(self, legacy: bool) -> list[dict[str, Any]]:
        fillers = [{"name": f"filler_{i}", "description": f"Filler tool {i} for pagination tests; answers its name.",
                    "inputSchema": _no_args()} for i in range(self.extra_tools)]
        return _core_tool_defs(legacy, self.x_mcp_header) + fillers

    def supported_versions(self) -> list[str]:
        if self.era == "legacy":
            return list(LEGACY_VERSIONS)
        if self.era == "modern":
            return list(MODERN_VERSIONS)
        return list(MODERN_VERSIONS) + list(LEGACY_VERSIONS)

    # -- validation ------------------------------------------------------------------------------------------------

    def is_modern(self, msg: dict[str, Any]) -> bool:
        """Which era serves this message: modern-only and legacy-only are fixed; dual looks at the request."""
        if self.era != "dual":
            return self.era == "modern"
        method = msg.get("method")
        if method == "initialize":
            return False
        params = msg.get("params")
        meta = params.get("_meta") if isinstance(params, dict) else None
        return method == "server/discover" or (isinstance(meta, dict) and META_PROTOCOL_VERSION in meta)

    def modern_initialize_error(self, params: dict[str, Any]) -> RpcError:
        versions = ", ".join(self.supported_versions())
        return RpcError(METHOD_NOT_FOUND,
                        f"initialize is not supported: this server speaks only stateless MCP (no handshake). "
                        f"Supported protocol versions: {versions}. Send requests with "
                        f"params._meta[{META_PROTOCOL_VERSION!r}] set to one of them.",
                        {"supported": self.supported_versions(), "requested": params.get("protocolVersion")},
                        http_status=400)

    def check_modern_meta(self, params: dict[str, Any]) -> str:
        """Return the requested version or raise -32602 (HTTP 400) when a required ``_meta`` field is missing."""
        meta = params.get("_meta")
        missing = [k for k in (META_PROTOCOL_VERSION, META_CLIENT_CAPABILITIES)
                   if not isinstance(meta, dict) or k not in meta]
        if missing:
            raise RpcError(INVALID_PARAMS, f"Invalid params: every request must carry params._meta fields {missing}",
                           {"missing": missing}, http_status=400)
        assert isinstance(meta, dict)
        version, caps = meta[META_PROTOCOL_VERSION], meta[META_CLIENT_CAPABILITIES]
        if not isinstance(version, str) or not isinstance(caps, dict):
            raise RpcError(INVALID_PARAMS, f"Invalid params: {META_PROTOCOL_VERSION} must be a string and "
                           f"{META_CLIENT_CAPABILITIES} an object", http_status=400)
        return version

    def _check_modern_version(self, version: str) -> None:
        if version in MODERN_VERSIONS:
            return
        hint = (" (that version uses the initialize handshake)" if version in LEGACY_VERSIONS else "")
        raise RpcError(UNSUPPORTED_PROTOCOL_VERSION, f"Unsupported protocol version {version!r}{hint}",
                       {"supported": self.supported_versions(), "requested": version}, http_status=400)

    def prepare(self, msg: dict[str, Any], state: LegacyState) -> Call:
        """Validate a request and pick its era; raise ``RpcError`` for anything that is not executable."""
        method, msg_id = msg.get("method"), msg.get("id")
        if not isinstance(method, str):
            raise RpcError(INVALID_REQUEST, "Invalid Request: method must be a string", http_status=400)
        params = msg.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise RpcError(INVALID_PARAMS, "Invalid params: params must be an object", http_status=400)
        if method == "initialize" and self.era == "modern":
            raise self.modern_initialize_error(params)
        modern = self.is_modern(msg)
        version: str | None = None
        if modern:
            version = self.check_modern_meta(params)
            self._check_modern_version(version)
            if method not in MODERN_METHODS:
                raise RpcError(METHOD_NOT_FOUND, f"Method not found: {method}", http_status=404)
        else:
            if method not in LEGACY_METHODS:
                raise RpcError(METHOD_NOT_FOUND, f"Method not found: {method}")
            if method not in ("initialize", "ping") and state.strict and not state.initialized:
                raise RpcError(INVALID_REQUEST, "Server not initialized: send initialize first")
        call = Call(msg_id, method, params, modern, version)
        if method == "tools/call":
            name = params.get("name")
            if not isinstance(name, str) or name not in {t["name"] for t in self.tools(legacy=not modern)}:
                raise RpcError(INVALID_PARAMS, f"Unknown tool: {name}")
            if not isinstance(params.get("arguments", {}), dict):
                raise RpcError(INVALID_PARAMS, "Invalid params: arguments must be an object")
        if method == "tools/list" and "cursor" in params:
            decode_cursor(params["cursor"])
        return call

    # -- execution -------------------------------------------------------------------------------------------------

    def handle(self, msg: dict[str, Any], state: LegacyState, channel: Channel) -> dict[str, Any] | None:
        """Prepare and execute one request; protocol errors become error responses. None means: send nothing."""
        try:
            return self.execute(self.prepare(msg, state), state, channel)
        except RpcError as err:
            return error_response(msg.get("id"), err)

    def execute(self, call: Call, state: LegacyState, channel: Channel) -> dict[str, Any] | None:
        try:
            result = self._dispatch(call, state, channel)
        except RpcError as err:
            return error_response(call.id, err)
        except Exception as exc:  # a fake must answer, not die: report the bug in-band
            return error_response(call.id, RpcError(INTERNAL_ERROR, f"Internal error in fake: {exc!r}"))
        if result is None:
            return None
        if call.modern:
            result = {"resultType": "complete", **result, "_meta": {META_SERVER_INFO: dict(SERVER_INFO)}}
        return {"jsonrpc": "2.0", "id": call.id, "result": result}

    def _dispatch(self, call: Call, state: LegacyState, channel: Channel) -> dict[str, Any] | None:
        if call.method == "initialize":
            return self._initialize(call.params, state)
        if call.method == "ping":
            return {}
        if call.method == "server/discover":
            return {"supportedVersions": self.supported_versions(), "capabilities": {"tools": {"listChanged": True}},
                    "serverInfo": dict(SERVER_INFO), "instructions": self.instructions}
        if call.method == "tools/list":
            return self._list_tools(call)
        return self._call_with_progress(call, state, channel)

    def _initialize(self, params: dict[str, Any], state: LegacyState) -> dict[str, Any]:
        requested = params.get("protocolVersion")
        chosen = requested if requested in LEGACY_VERSIONS else DEFAULT_LEGACY_VERSION
        caps = params.get("capabilities")
        info = params.get("clientInfo")
        state.initialized, state.ready, state.protocol_version = True, False, chosen
        state.client_capabilities = caps if isinstance(caps, dict) else {}
        state.client_info = info if isinstance(info, dict) else {}
        return {"protocolVersion": chosen, "capabilities": {"tools": {"listChanged": True}},
                "serverInfo": dict(SERVER_INFO), "instructions": self.instructions}

    def _list_tools(self, call: Call) -> dict[str, Any]:
        tools = self.tools(legacy=not call.modern)
        offset = decode_cursor(call.params["cursor"]) if "cursor" in call.params else 0
        if offset > len(tools):
            raise RpcError(INVALID_PARAMS, "Invalid cursor: it points past the end of the tool list")
        page = tools[offset:offset + self.page_size]
        result: dict[str, Any] = {"tools": page}
        if offset + self.page_size < len(tools):
            result["nextCursor"] = encode_cursor(offset + self.page_size)
        if call.modern:
            result["ttlMs"] = TOOLS_TTL_MS
            result["cacheScope"] = self.cache_scope
        return result

    def _call_with_progress(self, call: Call, state: LegacyState, channel: Channel) -> dict[str, Any] | None:
        token = call.progress_token
        if token is not None:
            channel.notify(_progress(token, 0, "started"))
        result = self._call_tool(call, state, channel)
        if result is not None and token is not None and not channel.cancelled():
            channel.notify(_progress(token, 1, "done"))
        return result

    def _call_tool(self, call: Call, state: LegacyState, channel: Channel) -> dict[str, Any] | None:
        name: str = call.params["name"]
        args: dict[str, Any] = call.params.get("arguments") or {}
        if name.startswith("filler_"):
            return {"content": [_text(name)], "isError": False}
        handler = getattr(self, f"_tool_{name}")
        return handler(args, state, channel)

    def _tool_echo(self, args: dict[str, Any], state: LegacyState, channel: Channel) -> dict[str, Any]:
        if not isinstance(args.get("text"), str):
            return _tool_error("Invalid arguments for echo: 'text' is required and must be a string.")
        return {"content": [_text(args["text"])], "isError": False}

    def _tool_add(self, args: dict[str, Any], state: LegacyState, channel: Channel) -> dict[str, Any]:
        a, b = args.get("a"), args.get("b")
        if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in (a, b)):
            return _tool_error("Invalid arguments for add: 'a' and 'b' are required and must be numbers.")
        total = a + b  # type: ignore[operator]
        return {"content": [_text(json.dumps({"sum": total}))], "structuredContent": {"sum": total}, "isError": False}

    def _tool_image(self, args: dict[str, Any], state: LegacyState, channel: Channel) -> dict[str, Any]:
        return {"content": [_text("A 1x1 PNG follows."),
                            {"type": "image", "data": PNG_1X1_B64, "mimeType": "image/png"}], "isError": False}

    def _tool_fail(self, args: dict[str, Any], state: LegacyState, channel: Channel) -> dict[str, Any]:
        return _tool_error("The fail tool always fails: this is a deliberate tool execution error, not a crash.")

    def _tool_slow(self, args: dict[str, Any], state: LegacyState, channel: Channel) -> dict[str, Any] | None:
        seconds = args.get("seconds")
        if not isinstance(seconds, int | float) or isinstance(seconds, bool) or not 0 <= seconds <= SLOW_MAX_SECONDS:
            return _tool_error(f"Invalid arguments for slow: 'seconds' must be a number from 0 to {SLOW_MAX_SECONDS}.")
        deadline = time.monotonic() + seconds
        while (remaining := deadline - time.monotonic()) > 0:
            if channel.cancelled():
                return None
            time.sleep(min(CANCEL_POLL_SECONDS, remaining))
        if channel.cancelled():
            return None
        return {"content": [_text(f"slept {seconds} s")], "isError": False}

    def _tool_resource(self, args: dict[str, Any], state: LegacyState, channel: Channel) -> dict[str, Any]:
        link = {"type": "resource_link", "uri": RESOURCE_URI, "name": "readme.txt",
                "description": "The fake server's readme", "mimeType": "text/plain"}
        embedded = {"type": "resource", "resource": {"uri": RESOURCE_URI, "mimeType": "text/plain",
                                                     "text": RESOURCE_TEXT}}
        return {"content": [link, embedded], "isError": False}

    def _tool_big(self, args: dict[str, Any], state: LegacyState, channel: Channel) -> dict[str, Any]:
        chars = args.get("chars", BIG_DEFAULT_CHARS)
        if not isinstance(chars, int) or isinstance(chars, bool) or not 0 <= chars <= BIG_MAX_CHARS:
            return _tool_error(f"Invalid arguments for big: 'chars' must be an integer from 0 to {BIG_MAX_CHARS}.")
        return {"content": [_text(("0123456789" * (chars // 10 + 1))[:chars])], "isError": False}

    def _tool_roots_probe(self, args: dict[str, Any], state: LegacyState, channel: Channel) -> dict[str, Any] | None:
        if state.strict and "roots" not in state.client_capabilities:
            return _tool_error("roots_probe: the client did not declare the 'roots' capability in initialize.")
        if state.strict and not state.ready:
            return _tool_error("roots_probe: the client has not sent notifications/initialized yet.")
        try:
            reply = channel.ask_client("roots/list", None, CLIENT_REPLY_TIMEOUT_SECONDS)
        except ChannelUnsupported as exc:
            return _tool_error(f"roots_probe: {exc}")
        except TimeoutError:
            return _tool_error(f"roots_probe: the client did not answer roots/list within "
                               f"{CLIENT_REPLY_TIMEOUT_SECONDS} s.")
        if reply is None:
            return None
        if "error" in reply:
            err = reply["error"] if isinstance(reply["error"], dict) else {}
            return _tool_error(f"roots_probe: the client answered roots/list with error {err.get('code')}: "
                               f"{err.get('message')}")
        result = reply.get("result")
        roots = result.get("roots") if isinstance(result, dict) else None
        if not isinstance(roots, list):
            return _tool_error("roots_probe: the client's roots/list result has no 'roots' array.")
        return {"content": [_text(json.dumps({"roots": roots}))], "structuredContent": {"roots": roots},
                "isError": False}


def _progress(token: Any, progress: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "method": "notifications/progress",
            "params": {"progressToken": token, "progress": progress, "total": 1, "message": message}}


def params_of(msg: dict[str, Any]) -> dict[str, Any]:
    """The message's params when they are an object, else an empty dict."""
    params = msg.get("params")
    return params if isinstance(params, dict) else {}


def classify(msg: Any) -> str:
    """'request', 'notification', 'response' or 'invalid' for one decoded JSON-RPC message."""
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
        return "invalid"
    if "method" in msg:
        return "request" if "id" in msg else "notification"
    if "id" in msg and ("result" in msg or "error" in msg):
        return "response"
    return "invalid"


# -- stdio transport -----------------------------------------------------------------------------------------------


@dataclass
class _Pending:
    done: threading.Event = field(default_factory=threading.Event)
    reply: dict[str, Any] | None = None


class _StdioChannel(Channel):
    def __init__(self, server: StdioServer, cancel: threading.Event) -> None:
        self._server = server
        self._cancel = cancel

    def notify(self, message: dict[str, Any]) -> None:
        if not self._cancel.is_set():
            self._server.send(message)

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def ask_client(self, method: str, params: dict[str, Any] | None, timeout_s: float) -> dict[str, Any] | None:
        return self._server.ask_client(method, params, timeout_s, self._cancel)


class StdioServer:
    """Reads newline-delimited JSON-RPC from ``stdin`` and writes it to ``stdout``; one thread per request."""

    def __init__(self, core: FakeMcpCore, stdin: BinaryIO, stdout: BinaryIO, stderr: Any, *,
                 crash_after: int | None = None, chatter: bool = False, record: BinaryIO | None = None) -> None:
        self.core = core
        self.state = LegacyState()
        self._stdin, self._stdout, self._stderr = stdin, stdout, stderr
        self._crash_after = crash_after
        self._chatter = chatter
        self._record = record
        self._write_lock = threading.Lock()
        self._lock = threading.Lock()
        self._inflight: dict[str, threading.Event] = {}
        self._pending: dict[str, _Pending] = {}
        self._threads: list[threading.Thread] = []
        self._requests_seen = 0
        self._server_ids = 0

    def log(self, text: str) -> None:
        if self._chatter:
            self._stderr.write(f"fake-mcp[stdio]: {text}\n")
            self._stderr.flush()

    def _write_record(self, entry: dict[str, Any]) -> None:
        if self._record is not None:
            with self._write_lock:
                self._record.write(json.dumps(entry, ensure_ascii=False).encode("utf-8") + b"\n")
                self._record.flush()

    def send(self, message: dict[str, Any]) -> None:
        line = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        with self._write_lock:
            try:
                self._stdout.write(line)
                self._stdout.flush()
            except (BrokenPipeError, ValueError, OSError):
                return  # the client went away; nothing left to tell it
        self._write_record({"dir": "out", "msg": message})
        self.log(f"-> {message.get('method') or 'response'} id={message.get('id')}")

    def ask_client(self, method: str, params: dict[str, Any] | None, timeout_s: float,
                   cancel: threading.Event) -> dict[str, Any] | None:
        with self._lock:
            self._server_ids += 1
            request_id = f"fake-srv-{self._server_ids}"
            pending = self._pending[json.dumps(request_id)] = _Pending()
        request: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            request["params"] = params
        self.send(request)
        deadline = time.monotonic() + timeout_s
        try:
            while not pending.done.wait(CANCEL_POLL_SECONDS):
                if cancel.is_set():
                    return None
                if time.monotonic() > deadline:
                    raise TimeoutError(method)
            return pending.reply
        finally:
            with self._lock:
                self._pending.pop(json.dumps(request_id), None)

    def run(self) -> int:
        for raw in iter(self._stdin.readline, b""):
            line = raw.strip()
            if line:
                self._receive(line)
        self.log("stdin closed; exiting")
        deadline = time.monotonic() + STDIN_EOF_GRACE_SECONDS
        for thread in list(self._threads):
            thread.join(max(0.0, deadline - time.monotonic()))
        return 0

    def _receive(self, line: bytes) -> None:
        try:
            msg = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            self.log(f"<- unparseable line {line[:80]!r}")
            self.send(error_response(None, RpcError(PARSE_ERROR, "Parse error: the line is not valid JSON")))
            return
        self._write_record({"dir": "in", "msg": msg})
        kind = classify(msg)
        self.log(f"<- {kind} {msg.get('method', '') if isinstance(msg, dict) else ''} "
                 f"id={msg.get('id') if isinstance(msg, dict) else None}")
        if kind == "invalid":
            msg_id = msg.get("id") if isinstance(msg, dict) else None
            reason = "batches are not supported" if isinstance(msg, list) else "not a JSON-RPC 2.0 message"
            self.send(error_response(msg_id, RpcError(INVALID_REQUEST, f"Invalid Request: {reason}")))
        elif kind == "response":
            self._route_response(msg)
        elif kind == "notification":
            self._notification(msg)
        else:
            self._request(msg)

    def _route_response(self, msg: dict[str, Any]) -> None:
        with self._lock:
            pending = self._pending.get(json.dumps(msg.get("id")))
        if pending is not None:
            pending.reply = msg
            pending.done.set()

    def _notification(self, msg: dict[str, Any]) -> None:
        method = msg["method"]
        params = params_of(msg)
        if method == "notifications/initialized":
            self.state.ready = True
        elif method == "notifications/cancelled":
            key = json.dumps(params.get("requestId"))
            with self._lock:
                event = self._inflight.get(key)
            if event is not None and not event.is_set():
                event.set()
                self._write_record({"dir": "event", "event": "cancelled", "id": params.get("requestId")})
                self.log(f"cancelled request {params.get('requestId')} ({params.get('reason', 'no reason')})")

    def _request(self, msg: dict[str, Any]) -> None:
        self._requests_seen += 1
        if self._crash_after is not None and self._requests_seen > self._crash_after:
            self._stderr.write(f"fake-mcp[stdio]: crashing on purpose after {self._crash_after} requests\n")
            self._stderr.flush()
            os._exit(CRASH_EXIT_CODE)
        if msg.get("method") == "initialize":
            self._serve(msg)  # in order, so a pipelined notifications/initialized cannot overtake it
            return
        thread = threading.Thread(target=self._serve, args=(msg,), daemon=True)
        self._threads = [t for t in self._threads if t.is_alive()] + [thread]
        thread.start()

    def _serve(self, msg: dict[str, Any]) -> None:
        key = json.dumps(msg.get("id"))
        cancel = threading.Event()
        with self._lock:
            self._inflight[key] = cancel
        try:
            response = self.core.handle(msg, self.state, _StdioChannel(self, cancel))
        finally:
            with self._lock:
                self._inflight.pop(key, None)
        if response is not None and not cancel.is_set():
            self.send(response)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fake MCP server over stdio for offline client tests.")
    parser.add_argument("--era", choices=ERAS, default="legacy")
    parser.add_argument("--extra-tools", type=int, default=0)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE_TOOLS)
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--crash-after", type=int, default=None)
    parser.add_argument("--stderr-chatter", action="store_true")
    parser.add_argument("--stderr-flood-kb", type=int, default=0)
    parser.add_argument("--require-env", action="append", default=[])
    parser.add_argument("--record", default=None)
    return parser.parse_args(argv)


def _missing_env(requirements: list[str]) -> str | None:
    for requirement in requirements:
        name, sep, expected = requirement.partition("=")
        value = os.environ.get(name)
        if value is None:
            return f"required environment variable {name} is not set"
        if sep and value != expected:
            return f"environment variable {name} does not have the expected value"
    return None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    problem = _missing_env(args.require_env)
    if problem:
        sys.stderr.write(f"fake-mcp[stdio]: {problem}; refusing to start\n")
        sys.stderr.flush()
        return ENV_MISSING_EXIT_CODE
    if args.stderr_flood_kb > 0:
        sys.stderr.write(("x" * 1023 + "\n") * args.stderr_flood_kb)
        sys.stderr.flush()
    if args.noise:
        sys.stdout.buffer.write(NOISE_LINE.encode() + b"\n")
        sys.stdout.buffer.flush()
    core = FakeMcpCore(args.era, extra_tools=args.extra_tools, page_size=args.page_size)
    record = open(args.record, "ab") if args.record else None  # stays open for the life of the process
    server = StdioServer(core, sys.stdin.buffer, sys.stdout.buffer, sys.stderr, crash_after=args.crash_after,
                         chatter=args.stderr_chatter, record=record)
    server.log(f"started: era={args.era} tools={len(core.tools(legacy=args.era != 'modern'))} pid={os.getpid()}")
    return server.run()


if __name__ == "__main__":
    sys.exit(main())
