"""A Model Context Protocol client over stdio and Streamable HTTP, standard library only.

Edgent uses it to reach connectors: MCP servers run by a third party (Perplexity, Firecrawl, PostHog,
Composio) or launched locally (Playwright). The client speaks both protocol eras:

- legacy revisions (2025-11-25 and earlier) open with an ``initialize`` handshake and, over HTTP, may
  hand out an ``MCP-Session-Id``;
- the stateless 2026-07-28 revision drops the handshake and carries the protocol version and client
  capabilities in every request's ``_meta`` (plus ``Mcp-Method``/``Mcp-Name`` headers over HTTP).

Every SDK the official servers are built on still speaks the legacy era, so the client opens with
``initialize`` and moves to the modern era only when a server rejects it in a way that names a modern
version. The deprecated 2024-11-05 HTTP+SSE transport is not implemented (the specification tells new
clients not to adopt it). OAuth is not implemented either: every server here accepts a static key in a
header, and a 401 is reported with the header to set.

Untrusted by design: a server's tool descriptions, annotations and results are data from a third party.
The caller (``application.connectors``) decides what the model may see and what a call is allowed to do.
"""
from __future__ import annotations

import base64
import http.client
import itertools
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import __version__
from ..domain.errors import ErrorCategory, ToolFailure

LEGACY_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
MODERN_VERSIONS = ("2026-07-28",)
PREFERRED_LEGACY_VERSION = LEGACY_VERSIONS[0]

CONNECT_TIMEOUT_S = 60.0  # initialize or discover, including an npx download on first use (s)
REQUEST_TIMEOUT_S = 120.0  # default per request (s); connectors raise it for slow tools
SHUTDOWN_GRACE_S = 3.0  # how long a stdio server gets to exit after stdin closes, then after SIGTERM (s)
MAX_MESSAGE_BYTES = 32 * 1024 * 1024  # one JSON-RPC message, stdio line or HTTP body (bytes)
MAX_LIST_PAGES = 50  # tools/list pages followed before giving up (pages)
STDERR_TAIL_LINES = 200  # server stderr lines kept for error messages (lines)
ERROR_TAIL_CHARS = 1500  # stderr shown in an error message (chars)
DELETE_TIMEOUT_S = 5.0  # ending an HTTP session on close (s)
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})
CANCEL_TIMEOUT_S = 5.0  # posting notifications/cancelled for an abandoned HTTP request (s)
SSE_RESUMES_MAX = 5  # reconnections with Last-Event-ID after a legacy server closes a stream early (count)
SSE_DEFAULT_RETRY_MS = 1000  # wait before reconnecting when the server sent no retry field (ms)
SSE_RETRY_CAP_S = 10.0  # the longest wait a server's retry field can impose (s)
STDERR_LINE_MAX_BYTES = 8192  # one stderr line kept (bytes); the rest of a longer line is dropped
READ_BUFFER_BYTES = 64 * 1024  # buffered pipe reads instead of one syscall per byte (bytes)

METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
HEADER_MISMATCH = -32020
MISSING_CLIENT_CAPABILITY = -32021
UNSUPPORTED_PROTOCOL_VERSION = -32022
MODERN_ERROR_CODES = frozenset({HEADER_MISMATCH, MISSING_CLIENT_CAPABILITY, UNSUPPORTED_PROTOCOL_VERSION})

# What a stdio server inherits from Edgent's environment by default: enough to run node/npx behind a proxy,
# nothing else. A Perplexity server has no business seeing GEMINI_API_KEY. ``inherit_env`` widens it.
STDIO_PASSTHROUGH_ENV = (
    "PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TMPDIR", "TEMP", "TMP", "SHELL",
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "APPDATA", "LOCALAPPDATA", "USERPROFILE", "PROGRAMDATA", "COMSPEC", "PATHEXT",
    "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy", "ALL_PROXY", "all_proxy",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "NODE_EXTRA_CA_CERTS", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
    "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "NPM_CONFIG_CACHE", "npm_config_cache", "PLAYWRIGHT_BROWSERS_PATH",
    "DISPLAY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR",
)


class McpError(Exception):
    """A connector failure with the fields of Edgent's error contract."""

    def __init__(
        self,
        code: str,
        message: str,
        category: ErrorCategory = ErrorCategory.unavailable,
        remediation: str = "",
        retryable: bool = False,
        rpc_code: int | None = None,
        data: Any = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.category = category
        self.remediation = remediation
        self.retryable = retryable
        self.rpc_code = rpc_code
        self.data = data

    def to_failure(self, **details: Any) -> ToolFailure:
        if self.rpc_code is not None:
            details.setdefault("rpc_code", self.rpc_code)
        return ToolFailure(self.code, self.message, self.category, retryable=self.retryable, remediation=self.remediation, **details)


@dataclass
class StdioSpec:
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    inherit_env: bool = False  # pass Edgent's whole environment instead of STDIO_PASSTHROUGH_ENV

    def describe(self) -> str:
        return " ".join([self.command, *self.args])


@dataclass
class HttpSpec:
    url: str
    headers: dict[str, str] = field(default_factory=dict)

    def describe(self) -> str:
        return redact_url(self.url)


@dataclass
class ClientOptions:
    client_name: str = "edgent"
    client_version: str = __version__
    connect_timeout_s: float = CONNECT_TIMEOUT_S
    request_timeout_s: float = REQUEST_TIMEOUT_S
    roots: list[str] = field(default_factory=list)  # directories offered to servers that ask roots/list (legacy era)
    secrets: list[str] = field(default_factory=list)  # values replaced by *** in every message the client produces
    era: str = "auto"  # auto | legacy | modern
    on_event: Callable[[str, dict[str, Any]], None] | None = None  # (kind, data): log, progress, stderr, noise, list_changed


# --------------------------------------------------------------------------- helpers
def redact(text: str, secrets: list[str]) -> str:
    out = text
    for s in secrets:
        if s and len(s) >= 4:
            out = out.replace(s, "***")
    return out


def redact_url(url: str) -> str:
    """Drop userinfo and query values from a URL for display; keys often travel there."""
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return "<invalid url>"
    netloc = parts.hostname or ""
    if parts.port:
        netloc += f":{parts.port}"
    query = "&".join(f"{k}=***" for k, _ in urllib.parse.parse_qsl(parts.query, keep_blank_values=True))
    return urllib.parse.urlunsplit((parts.scheme, netloc, parts.path, query, ""))


def header_value(value: str) -> str:
    """A header-safe value; anything outside visible ASCII travels in the specification's base64 sentinel."""
    if value and value == value.strip() and all(0x20 <= ord(c) <= 0x7E for c in value):
        return value
    return "=?base64?" + base64.b64encode(value.encode("utf-8")).decode("ascii") + "?="


def file_uri(path: str) -> str:
    return Path(path).resolve().as_uri()


def _valid_id(value: Any) -> bool:
    """JSON-RPC ids are strings or numbers; anything else cannot key a pending request."""
    return isinstance(value, (str, int, float)) and not isinstance(value, bool)


def _rpc_error(err: Any, secrets: list[str]) -> McpError:
    if not isinstance(err, dict):
        return McpError("protocol_error", f"malformed JSON-RPC error: {str(err)[:200]}", ErrorCategory.execution_failure)
    code = err.get("code") if isinstance(err.get("code"), int) else None
    message = redact(str(err.get("message") or "server error"), secrets)[:1000]
    category = ErrorCategory.invalid_input if code in (INVALID_PARAMS, METHOD_NOT_FOUND) else ErrorCategory.execution_failure
    remediation = "Check the arguments against the tool's schema." if code == INVALID_PARAMS else "Read the server's message; change the request rather than repeating it."
    return McpError("server_error", message, category, remediation, rpc_code=code, data=err.get("data"))


def _is_modern_error(err: McpError) -> bool:
    if err.rpc_code in MODERN_ERROR_CODES:
        return True
    supported = err.data.get("supported") if isinstance(err.data, dict) else None
    named = " ".join(str(v) for v in supported) if isinstance(supported, list) else err.message
    return any(v in named for v in MODERN_VERSIONS)


# --------------------------------------------------------------------------- transports
class _Pending:
    __slots__ = ("event", "message")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.message: dict[str, Any] | None = None


class StdioTransport:
    """A server launched as a child process speaking newline-delimited JSON-RPC on stdin/stdout."""

    def __init__(self, spec: StdioSpec, options: ClientOptions, on_message: Callable[[dict[str, Any]], None]) -> None:
        self.spec = spec
        self.options = options
        self.on_message = on_message  # server requests and notifications
        self._proc: subprocess.Popen[bytes] | None = None
        self._pending: dict[Any, _Pending] = {}
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._stderr: deque[str] = deque(maxlen=STDERR_TAIL_LINES)
        self._closed = False
        self._reader_done = threading.Event()  # set when the stdout reader stops for any reason
        self.exit_code: int | None = None

    # -- lifecycle
    def start(self) -> None:
        env = dict(os.environ) if self.spec.inherit_env else {k: os.environ[k] for k in STDIO_PASSTHROUGH_ENV if k in os.environ}
        env.update({k: str(v) for k, v in self.spec.env.items()})
        exe = shutil.which(self.spec.command, path=env.get("PATH")) or (self.spec.command if os.path.isfile(self.spec.command) else None)
        if exe is None:
            hint = "Install Node.js 18 or newer (npx comes with it)." if self.spec.command in ("npx", "node", "npm") else f"Install {self.spec.command} or put it on PATH."
            raise McpError("unavailable", f"command {self.spec.command!r} was not found on PATH", ErrorCategory.unavailable, hint)
        try:
            self._proc = subprocess.Popen(  # noqa: S603 - the command comes from the user's connector settings
                [exe, *self.spec.args],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=self.spec.cwd or None, env=env, bufsize=READ_BUFFER_BYTES,  # buffered: readline must not cost a syscall per byte
                start_new_session=os.name == "posix",  # npx spawns node: the whole group is stopped on close
            )
        except OSError as e:
            raise McpError("unavailable", f"could not start {self.spec.command!r}: {e}", ErrorCategory.unavailable, "Check the connector's command and arguments.") from e
        self._reader_done.clear()
        threading.Thread(target=self._read_stdout, name="mcp-stdio-out", daemon=True).start()
        threading.Thread(target=self._read_stderr, name="mcp-stdio-err", daemon=True).start()

    def close(self) -> None:
        self._closed = True
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.wait(SHUTDOWN_GRACE_S)
        except (OSError, subprocess.TimeoutExpired):
            self._signal(proc, signal.SIGTERM)
            try:
                proc.wait(SHUTDOWN_GRACE_S)
            except subprocess.TimeoutExpired:
                self._signal(proc, signal.SIGKILL if hasattr(signal, "SIGKILL") else signal.SIGTERM)
                try:
                    proc.wait(SHUTDOWN_GRACE_S)
                except subprocess.TimeoutExpired:
                    pass
        self.exit_code = proc.returncode
        self._fail_pending("the connector process was closed")

    @staticmethod
    def _signal(proc: subprocess.Popen[bytes], sig: int) -> None:
        try:
            if os.name == "posix":
                os.killpg(proc.pid, sig)
            else:
                proc.terminate()
        except (OSError, ProcessLookupError):
            pass

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None and not self._closed and not self._reader_done.is_set()

    def stderr_tail(self, chars: int = ERROR_TAIL_CHARS) -> str:
        return redact("\n".join(self._stderr), self.options.secrets)[-chars:]

    # -- reading
    def _read_stdout(self) -> None:
        try:
            self._read_stdout_loop()
        finally:  # whatever ended the loop, nobody keeps waiting on this transport
            self._reader_done.set()
            self.exit_code = self._proc.poll() if self._proc is not None else None
            self._fail_pending("the connector process stopped answering" + (f" (exit code {self.exit_code})" if self.exit_code is not None else ""))

    def _read_stdout_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        out = self._proc.stdout
        while True:
            try:
                line = out.readline(MAX_MESSAGE_BYTES + 1)
            except (OSError, ValueError):
                break
            if not line:
                break
            if len(line) > MAX_MESSAGE_BYTES and not line.endswith(b"\n"):
                self._stderr.append(f"[edgent] dropped a stdout message over {MAX_MESSAGE_BYTES} bytes")
                while (rest := out.readline(MAX_MESSAGE_BYTES)) and not rest.endswith(b"\n"):
                    pass
                self._fail_pending(f"the connector sent a message over {MAX_MESSAGE_BYTES} bytes, which Edgent does not read", code="truncated")
                continue
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            if text[0] not in "{[":  # npx and some servers print banners on stdout; they are not messages
                self._stderr.append(f"[stdout] {text[:300]}")
                self._event("noise", {"line": text[:300]})
                continue
            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                self._stderr.append(f"[stdout, not JSON] {text[:300]}")
                continue
            for m in msg if isinstance(msg, list) else [msg]:
                if isinstance(m, dict):
                    try:
                        self._dispatch(m)
                    except Exception as e:  # noqa: BLE001 - one malformed message never stops the reader
                        self._stderr.append(f"[edgent] ignored a malformed message: {type(e).__name__}: {str(e)[:200]}")

    def _read_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        err = self._proc.stderr
        while True:
            try:
                raw = err.readline(STDERR_LINE_MAX_BYTES)
            except (OSError, ValueError):
                break
            if not raw:
                break
            if not raw.endswith(b"\n"):  # an over-long line (a progress bar): keep its start, drop the rest
                while (rest := err.readline(STDERR_LINE_MAX_BYTES)) and not rest.endswith(b"\n"):
                    pass
            text = raw.decode("utf-8", errors="replace").rstrip()
            if text:
                self._stderr.append(text[:1000])

    def _dispatch(self, msg: dict[str, Any]) -> None:
        if "id" in msg and not _valid_id(msg["id"]):
            self._stderr.append(f"[edgent] ignored a message with an invalid id: {str(msg.get('id'))[:80]}")
            return
        if "method" not in msg and "id" in msg:
            with self._lock:
                pending = self._pending.pop(msg["id"], None)
            if pending is not None:
                pending.message = msg
                pending.event.set()
            return
        if "method" in msg:
            if "id" in msg:  # a server request: answered on its own thread so a slow handler never blocks reading
                threading.Thread(target=self._answer, args=(msg,), name="mcp-stdio-req", daemon=True).start()
            else:
                self.on_message(msg)

    def _answer(self, msg: dict[str, Any]) -> None:
        self.on_message(msg)

    def _fail_pending(self, reason: str, code: str = "unavailable") -> None:
        with self._lock:
            waiting, self._pending = list(self._pending.values()), {}
        for p in waiting:
            p.message = {"_transport_error": reason, "_code": code}
            p.event.set()

    def _event(self, kind: str, data: dict[str, Any]) -> None:
        if self.options.on_event:
            try:
                self.options.on_event(kind, data)
            except Exception:  # noqa: BLE001 - a listener never breaks the connection
                pass

    # -- writing
    def write(self, msg: dict[str, Any]) -> None:
        if not self.alive or self._proc is None or self._proc.stdin is None:
            raise self._dead_error()
        data = (json.dumps(msg, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            with self._write_lock:
                self._proc.stdin.write(data)
                self._proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as e:
            raise self._dead_error() from e

    def request(self, msg: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        pending = _Pending()
        with self._lock:
            self._pending[msg["id"]] = pending
        try:
            self.write(msg)
        except McpError:
            with self._lock:
                self._pending.pop(msg["id"], None)
            raise
        if not pending.event.wait(timeout_s):
            with self._lock:
                self._pending.pop(msg["id"], None)
            if msg.get("method") != "initialize":  # the specification forbids cancelling initialize
                try:  # tell the server to stop working on it
                    self.write({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": msg["id"], "reason": f"timed out after {timeout_s:g} s"}})
                except McpError:
                    pass
            raise McpError("timeout", f"{msg.get('method')} got no answer within {timeout_s:g} s", ErrorCategory.timeout,
                           "Retry once with a narrower request, or raise the connector's timeout_s.", retryable=True)
        assert pending.message is not None
        if "_transport_error" in pending.message:
            if pending.message.get("_code") == "truncated":
                raise McpError("truncated", pending.message["_transport_error"], ErrorCategory.resource_constraint, "Ask the tool for less (a smaller page, fewer results).")
            raise self._dead_error(pending.message["_transport_error"])
        return pending.message

    def _dead_error(self, reason: str = "the connector process is not running") -> McpError:
        tail = self.stderr_tail()
        return McpError("unavailable", f"{reason} ({redact(self.spec.describe(), self.options.secrets)})" + (f"; its last output:\n{tail}" if tail else ""),
                        ErrorCategory.unavailable, "Run `edgent connectors test <name>` to see why it stopped; check its key and command.", retryable=True)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """An MCP endpoint never redirects a POST. Following one would re-send the API key headers to whatever host the
    Location names (urllib copies them), so a redirect is reported instead of followed."""

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        raise _Redirected(code, newurl)


class _Redirected(Exception):
    def __init__(self, code: int, location: str) -> None:
        super().__init__(location)
        self.code = code
        self.location = location


def _opener(url: str) -> Callable[..., Any]:
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    handlers: list[Any] = [_NoRedirect()]
    if host in LOOPBACK_HOSTS or host.endswith(".localhost"):
        handlers.append(urllib.request.ProxyHandler({}))  # a server on this machine never goes through HTTPS_PROXY
    return urllib.request.build_opener(*handlers).open


def _session_lost(status: int, rpc: dict[str, Any] | None, raw: str) -> bool:
    """A server that no longer knows the session: 404 per the spec, or the SDK's 400 'No valid session ID provided'."""
    if status == 404:
        return True
    text = (str(rpc.get("message", "")) if rpc else raw).lower()
    return status == 400 and "session" in text


class HttpTransport:
    """Streamable HTTP: one POST per message; the answer is a JSON body or an SSE stream ending in the response."""

    def __init__(self, spec: HttpSpec, options: ClientOptions, on_message: Callable[[dict[str, Any]], dict[str, Any] | None]) -> None:
        self.spec = spec
        self.options = options
        self.on_message = on_message
        self.session_id: str | None = None
        self.protocol_version: str | None = None
        self.modern = False
        self.tool_schemas: dict[str, dict[str, Any]] = {}  # name -> inputSchema, for modern x-mcp-header mirroring
        self._lock = threading.Lock()
        parts = urllib.parse.urlsplit(spec.url)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise McpError("configuration_invalid", f"connector URL must be http(s)://host/path; got {redact_url(spec.url)}",
                           ErrorCategory.configuration_invalid, "Fix the url in the connector settings.")
        self.display_url = redact(redact_url(spec.url), options.secrets)  # what any message may show
        self._open = _opener(spec.url)

    @property
    def alive(self) -> bool:
        return True

    def stderr_tail(self, chars: int = ERROR_TAIL_CHARS) -> str:
        return ""

    def start(self) -> None:
        return None

    def close(self) -> None:
        with self._lock:
            sid, self.session_id = self.session_id, None
        if sid and not self.modern:
            req = urllib.request.Request(self.spec.url, method="DELETE", headers={**self._base_headers(), "Mcp-Session-Id": sid})
            try:
                with self._open(req, timeout=DELETE_TIMEOUT_S):  # noqa: S310 - http(s) checked in __init__
                    pass
            except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException, _Redirected):
                pass  # 405 (sessions cannot be ended) or the network is gone: nothing to do

    def _base_headers(self) -> dict[str, str]:
        headers = {"User-Agent": f"edgent/{__version__}"}
        headers.update(self.spec.headers)
        return headers

    def _headers(self, msg: dict[str, Any]) -> tuple[dict[str, str], str | None]:
        """The request headers and the session id they carry (an expired session is recognised by the id that failed)."""
        headers = self._base_headers()
        headers.update({"Content-Type": "application/json", "Accept": "application/json, text/event-stream"})
        with self._lock:
            sid = self.session_id if not self.modern else None
            if sid:
                headers["Mcp-Session-Id"] = sid
            if self.protocol_version:
                headers["MCP-Protocol-Version"] = self.protocol_version
        if self.modern and "method" in msg:
            headers["Mcp-Method"] = header_value(str(msg["method"]))
            params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
            name = params.get("name") if msg["method"] in ("tools/call", "prompts/get") else params.get("uri") if msg["method"] == "resources/read" else None
            if isinstance(name, str):
                headers["Mcp-Name"] = header_value(name)
            if msg["method"] == "tools/call" and isinstance(name, str):
                headers.update(_param_headers(self.tool_schemas.get(name), params.get("arguments")))
        return headers, sid

    def write(self, msg: dict[str, Any]) -> None:
        """A notification or a response to a server request: 202 expected, nothing read back."""
        self._post(msg, timeout_s=min(self.options.request_timeout_s, CANCEL_TIMEOUT_S * 4), expect_response=False)

    def request(self, msg: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        try:
            result = self._post(msg, timeout_s=timeout_s, expect_response=True)
        except McpError as e:
            if e.code in ("timeout", "unavailable") and not self.modern and msg.get("method") != "initialize" and "id" in msg:
                self._cancel(msg["id"], e.message)  # a legacy server keeps working on a request whose connection dropped
            raise
        assert result is not None
        return result

    def _cancel(self, request_id: Any, reason: str) -> None:
        note = {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": request_id, "reason": reason[:200]}}

        def send() -> None:
            try:
                self._post(note, timeout_s=CANCEL_TIMEOUT_S, expect_response=False)
            except McpError:
                pass

        threading.Thread(target=send, name="mcp-http-cancel", daemon=True).start()

    def _post(self, msg: dict[str, Any], timeout_s: float, expect_response: bool) -> dict[str, Any] | None:
        body = json.dumps(msg, ensure_ascii=False).encode("utf-8")
        headers, sid = self._headers(msg)
        req = urllib.request.Request(self.spec.url, data=body, method="POST", headers=headers)
        deadline = time.monotonic() + timeout_s
        resp = self._send(req, msg, timeout_s, sid)
        with resp:
            if msg.get("method") == "initialize":
                new_sid = resp.headers.get("Mcp-Session-Id")
                if new_sid:
                    with self._lock:
                        self.session_id = new_sid.strip()
            if not expect_response or resp.status == 202:
                if expect_response:
                    raise McpError("protocol_error", f"{msg.get('method')} was accepted with 202 but no response was sent",
                                   ErrorCategory.execution_failure, "The server does not follow the MCP transport; report it to its maintainers.")
                return None
            ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype == "text/event-stream":
                return self._read_sse(resp, msg, deadline, sid)
            raw = self._read_body(resp, msg, timeout_s)
        if len(raw) > MAX_MESSAGE_BYTES:
            raise McpError("truncated", f"response over {MAX_MESSAGE_BYTES} bytes", ErrorCategory.resource_constraint, "Ask for less (a smaller page, fewer results).")
        try:
            data = json.loads(raw.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            raise McpError("protocol_error", f"{self.display_url} did not answer with JSON-RPC", ErrorCategory.execution_failure,
                           "Check that the connector URL is the MCP endpoint (it usually ends in /mcp).") from None
        for m in data if isinstance(data, list) else [data]:
            if isinstance(m, dict) and m.get("id") == msg.get("id") and "method" not in m:
                return m
        raise McpError("protocol_error", "the response did not answer the request", ErrorCategory.execution_failure, "Report it to the server's maintainers.")

    def _send(self, req: urllib.request.Request, msg: dict[str, Any], timeout_s: float, sid: str | None) -> Any:
        try:
            return self._open(req, timeout=timeout_s)  # noqa: S310 - http(s) checked in __init__
        except _Redirected as r:
            target = redact(redact_url(r.location), self.options.secrets)
            raise McpError("configuration_invalid", f"{self.display_url} redirects (HTTP {r.code}) to {target}; Edgent does not follow it, so no key is sent there",
                           ErrorCategory.configuration_invalid, "Update the connector's url to the address the redirect names, if you trust it.") from None
        except urllib.error.HTTPError as e:
            raise self._http_error(e, msg, sid) from None
        except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException) as e:
            raise self._network_error(e, msg, timeout_s) from None

    def _network_error(self, e: BaseException, msg: dict[str, Any], timeout_s: float) -> McpError:
        reason = getattr(e, "reason", e)
        if isinstance(reason, TimeoutError) or "timed out" in str(reason):
            return McpError("timeout", f"{msg.get('method', 'message')} got no answer within {timeout_s:g} s", ErrorCategory.timeout,
                            "Retry once with a narrower request, or raise the connector's timeout_s.", retryable=True)
        text = redact(str(reason), self.options.secrets)
        if "Tunnel connection failed" in text:  # urllib's words when HTTPS_PROXY refuses the CONNECT
            host = urllib.parse.urlsplit(self.spec.url).hostname or "the host"
            return McpError("unavailable", f"the HTTPS proxy refused the connection to {host} ({text.split(':', 1)[-1].strip()})",
                            ErrorCategory.unavailable, f"This network's proxy does not allow {host}; allow it in the network policy or run Edgent where it is reachable.")
        return McpError("unavailable", f"cannot reach {self.display_url}: {text}", ErrorCategory.unavailable, "Check the network, the proxy and the connector URL.", retryable=True)

    def _read_body(self, resp: Any, msg: dict[str, Any], timeout_s: float) -> bytes:
        try:
            return resp.read(MAX_MESSAGE_BYTES + 1)
        except (OSError, ValueError, http.client.HTTPException) as e:
            if isinstance(e, TimeoutError) or "timed out" in str(e):
                raise McpError("timeout", f"{msg.get('method')} answer stalled for {timeout_s:g} s", ErrorCategory.timeout,
                               "Retry once, or raise the connector's timeout_s.", retryable=True) from None
            raise McpError("unavailable", f"the answer to {msg.get('method')} was cut off: {type(e).__name__}", ErrorCategory.unavailable,
                           "Retry the call once.", retryable=msg.get("method") != "tools/call") from None

    def _read_sse(self, resp: Any, msg: dict[str, Any], deadline: float, sid: str | None) -> dict[str, Any]:
        """Read events until the response; a legacy server may close the connection after an event with an id and expect
        the client to come back with GET and Last-Event-ID (2025-11-25 polling), which is followed within the deadline."""
        last_id: str | None = None
        retry_ms = SSE_DEFAULT_RETRY_MS
        for attempt in range(SSE_RESUMES_MAX + 1):
            found, last_id, retry_ms = self._consume_stream(resp, msg, deadline, last_id, retry_ms)
            if found is not None:
                return found
            if self.modern or last_id is None or attempt == SSE_RESUMES_MAX:
                break
            time.sleep(min(retry_ms / 1000.0, max(0.0, deadline - time.monotonic()), SSE_RETRY_CAP_S))
            if time.monotonic() >= deadline:
                break
            headers = {**self._base_headers(), "Accept": "text/event-stream", "Last-Event-ID": last_id}
            if sid:
                headers["Mcp-Session-Id"] = sid
            if self.protocol_version:
                headers["MCP-Protocol-Version"] = self.protocol_version
            resp.close()
            resp = self._send(urllib.request.Request(self.spec.url, method="GET", headers=headers), msg, max(0.1, deadline - time.monotonic()), sid)
        side_effects = msg.get("method") == "tools/call"
        raise McpError("unavailable", f"the stream for {msg.get('method')} closed before the response" + ("; the tool may have run" if side_effects else ""),
                       ErrorCategory.unavailable, "Check the result in the service before calling again." if side_effects else "Retry once; the server may have dropped the connection.",
                       retryable=not side_effects)

    def _consume_stream(self, resp: Any, msg: dict[str, Any], deadline: float, last_id: str | None, retry_ms: int) -> tuple[dict[str, Any] | None, str | None, int]:
        data_lines: list[str] = []
        total = 0
        with resp:
            while True:
                if time.monotonic() > deadline:
                    raise McpError("timeout", f"{msg.get('method')} stream gave no response in time", ErrorCategory.timeout,
                                   "Retry once, or raise the connector's timeout_s.", retryable=True)
                try:
                    raw = resp.readline(MAX_MESSAGE_BYTES + 1)
                except (OSError, ValueError, http.client.HTTPException) as e:
                    if isinstance(e, TimeoutError) or "timed out" in str(e):
                        raise McpError("timeout", f"{msg.get('method')} stream went silent", ErrorCategory.timeout,
                                       "Retry once, or raise the connector's timeout_s.", retryable=True) from None
                    return None, last_id, retry_ms  # a cut connection: resumable when the server gave event ids
                if not raw:
                    break
                total += len(raw)
                if total > MAX_MESSAGE_BYTES:
                    raise McpError("truncated", f"response stream over {MAX_MESSAGE_BYTES} bytes", ErrorCategory.resource_constraint, "Ask for less.")
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if line == "":
                    if data_lines:
                        found = self._sse_event("\n".join(data_lines), msg)
                        data_lines = []
                        if found is not None:
                            return found, last_id, retry_ms
                    continue
                if line.startswith(":"):
                    continue  # a comment or keep-alive
                name, _, value = line.partition(":")
                value = value[1:] if value.startswith(" ") else value
                if name == "data":
                    data_lines.append(value)
                elif name == "id" and "\0" not in value:
                    last_id = value
                elif name == "retry" and value.isdigit():
                    retry_ms = int(value)
        if data_lines:
            found = self._sse_event("\n".join(data_lines), msg)
            if found is not None:
                return found, last_id, retry_ms
        return None, last_id, retry_ms

    def _sse_event(self, data: str, msg: dict[str, Any]) -> dict[str, Any] | None:
        if not data.strip():
            return None  # the priming event
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            return None
        for m in parsed if isinstance(parsed, list) else [parsed]:
            if not isinstance(m, dict):
                continue
            if "method" not in m and m.get("id") == msg.get("id"):
                return m
            if "method" in m:
                reply = self.on_message(m)
                if reply is not None and "id" in m:  # legacy servers may ask the client something mid-stream
                    threading.Thread(target=self._send_reply, args=(reply,), name="mcp-http-reply", daemon=True).start()
        return None

    def _send_reply(self, reply: dict[str, Any]) -> None:
        try:
            self.write(reply)
        except McpError as e:
            if self.options.on_event:
                self.options.on_event("log", {"level": "warning", "data": f"could not answer a server request: {e.message}"})

    def _http_error(self, e: urllib.error.HTTPError, msg: dict[str, Any], sid: str | None) -> McpError:
        try:
            raw = e.read(64 * 1024).decode("utf-8", errors="replace")
        except (OSError, http.client.HTTPException):
            raw = ""
        body: Any = None
        try:
            body = json.loads(raw) if raw.strip() else None
        except json.JSONDecodeError:
            body = None
        secrets = self.options.secrets
        rpc = body.get("error") if isinstance(body, dict) and isinstance(body.get("error"), dict) else None
        if e.code == 401 or (e.code == 403 and not (rpc and "origin" in str(rpc.get("message", "")).lower())):
            www = e.headers.get("WWW-Authenticate", "") if e.headers else ""
            said = redact(str(rpc.get("message")), secrets)[:200] if rpc else ""
            return McpError("auth_required", f"{self.display_url} refused the credentials (HTTP {e.code})" + (f": {said}" if said else "")
                            + (f"; it asks for: {redact(www, secrets)[:300]}" if www else ""),
                            ErrorCategory.permission_denied, "Set the connector's API key (see `edgent connectors list`); Edgent sends keys as headers and does not run OAuth.")
        if sid and not self.modern and msg.get("method") != "initialize" and _session_lost(e.code, rpc, raw):
            return McpError("session_expired", "the server no longer knows this session", ErrorCategory.unavailable, "Reconnect.", retryable=True, data={"session_id": sid})
        if rpc is not None:
            err = _rpc_error(rpc, secrets)
            err.data = err.data if err.data is not None else {"http_status": e.code}
            return err
        if e.code == 429:
            retry = e.headers.get("Retry-After", "") if e.headers else ""
            return McpError("rate_limited", f"rate limited by {self.display_url}" + (f"; retry after {retry} s" if retry else ""),
                            ErrorCategory.resource_constraint, "Wait before calling this connector again.", retryable=True)
        detail = redact(raw.strip(), secrets)[:300]
        retryable = e.code >= 500
        return McpError("server_error" if retryable else "protocol_error", f"HTTP {e.code} from {self.display_url} for {msg.get('method', 'message')}" + (f": {detail}" if detail else ""),
                        ErrorCategory.unavailable if retryable else ErrorCategory.execution_failure,
                        "Retry once later." if retryable else "Check the connector URL and headers.", retryable=retryable, data={"http_status": e.code})


def _param_headers(schema: dict[str, Any] | None, arguments: Any) -> dict[str, str]:
    """2026-07-28: top-level parameters marked ``x-mcp-header`` are mirrored as ``Mcp-Param-<Name>`` headers."""
    if not isinstance(schema, dict) or not isinstance(arguments, dict):
        return {}
    out: dict[str, str] = {}
    for key, prop in (schema.get("properties") or {}).items():
        name = prop.get("x-mcp-header") if isinstance(prop, dict) else None
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9-]+", name) or key not in arguments:
            continue
        value = arguments[key]
        if value is None or isinstance(value, (dict, list)):
            continue
        text = ("true" if value else "false") if isinstance(value, bool) else str(value)
        out[f"Mcp-Param-{name}"] = header_value(text)
    return out


# --------------------------------------------------------------------------- the client
class McpClient:
    """One connection to one MCP server. Thread-safe: calls from several tool threads interleave by request id."""

    def __init__(self, spec: StdioSpec | HttpSpec, options: ClientOptions | None = None) -> None:
        self.spec = spec
        self.options = options or ClientOptions()
        self.era: str | None = None  # legacy | modern, once connected
        self.protocol_version: str | None = None
        self.server_info: dict[str, Any] = {}
        self.capabilities: dict[str, Any] = {}
        self.instructions: str = ""
        self.tools_changed = False
        self.tools_ttl_s: float | None = None
        self.tools_partial = ""  # why the last tools/list stopped early, if it did
        self._generation = 0  # bumped on every connect: one expired session is re-opened once, not once per thread
        self._ids = itertools.count(1)
        self._id_lock = threading.Lock()
        self._connect_lock = threading.Lock()
        self._transport: StdioTransport | HttpTransport | None = None

    # -- lifecycle
    @property
    def connected(self) -> bool:
        return self._transport is not None and self.era is not None and self._transport.alive

    def connect(self) -> dict[str, Any]:
        """Start the transport and agree a protocol version; returns what the server said about itself."""
        with self._connect_lock:
            if self.connected:
                return self.describe()
            self._close_transport()
            self._generation += 1
            self._transport = StdioTransport(self.spec, self.options, self._on_stdio_message) if isinstance(self.spec, StdioSpec) else HttpTransport(self.spec, self.options, self._on_http_message)
            self._transport.start()
            try:
                if self.options.era == "modern":
                    self._discover()
                else:
                    try:
                        self._initialize()
                    except McpError as e:
                        if self.options.era == "legacy" or e.code not in ("server_error", "protocol_error") or not (_is_modern_error(e) or e.rpc_code in (METHOD_NOT_FOUND, INVALID_PARAMS)):
                            raise
                        self._discover(original=e)
            except BaseException:
                self._close_transport()
                raise
            return self.describe()

    def close(self) -> None:
        with self._connect_lock:
            self._close_transport()

    def _close_transport(self) -> None:
        transport, self._transport = self._transport, None
        self.era = None
        if transport is not None:
            transport.close()

    def describe(self) -> dict[str, Any]:
        out = {"era": self.era, "protocol_version": self.protocol_version, "server_info": dict(self.server_info),
               "capabilities": dict(self.capabilities), "instructions": self.instructions}
        if self.tools_partial:
            out["tools_partial"] = self.tools_partial
        return out

    def stderr_tail(self) -> str:
        return self._transport.stderr_tail() if self._transport is not None else ""

    def _client_capabilities(self) -> dict[str, Any]:
        # roots are offered to legacy servers only: 2026-07-28 deprecates them and would ask through an input request
        return {"roots": {"listChanged": False}} if self.options.roots and self.era != "modern" else {}

    def _client_info(self) -> dict[str, str]:
        return {"name": self.options.client_name, "version": self.options.client_version}

    def _initialize(self) -> None:
        assert self._transport is not None
        result = self._raw_request("initialize", {"protocolVersion": PREFERRED_LEGACY_VERSION, "capabilities": self._client_capabilities(), "clientInfo": self._client_info()},
                                   self.options.connect_timeout_s)
        version = str(result.get("protocolVersion") or "")
        if version not in LEGACY_VERSIONS:
            raise McpError("unsupported_protocol", f"the server answered with protocol version {version!r}; Edgent speaks {', '.join(LEGACY_VERSIONS + MODERN_VERSIONS)}",
                           ErrorCategory.unsupported, "Use a newer or older release of the server that speaks one of these versions.")
        self.era, self.protocol_version = "legacy", version
        self.server_info = result.get("serverInfo") if isinstance(result.get("serverInfo"), dict) else {}
        self.capabilities = result.get("capabilities") if isinstance(result.get("capabilities"), dict) else {}
        self.instructions = str(result.get("instructions") or "")
        if isinstance(self._transport, HttpTransport):
            self._transport.protocol_version = version
        self._notify("notifications/initialized", {})

    def _discover(self, original: McpError | None = None) -> None:
        assert self._transport is not None
        if isinstance(self._transport, HttpTransport):
            self._transport.modern = True
            self._transport.protocol_version = MODERN_VERSIONS[0]
        self.era, self.protocol_version = "modern", MODERN_VERSIONS[0]
        try:
            result = self._request("server/discover", {}, self.options.connect_timeout_s)
        except McpError as e:
            supported = e.data.get("supported") if isinstance(e.data, dict) else None
            usable = [v for v in MODERN_VERSIONS if isinstance(supported, list) and v in supported]
            if e.rpc_code == UNSUPPORTED_PROTOCOL_VERSION and not usable:
                raise McpError("unsupported_protocol", f"the server supports {supported}; Edgent speaks {', '.join(LEGACY_VERSIONS + MODERN_VERSIONS)}",
                               ErrorCategory.unsupported, "Use a release of the server that speaks one of these versions.") from None
            self.era = None
            raise (original or e) from None
        versions = result.get("supportedVersions") if isinstance(result.get("supportedVersions"), list) else []
        chosen = next((v for v in MODERN_VERSIONS if v in versions), None)
        if chosen is None:
            self.era = None
            raise McpError("unsupported_protocol", f"the server supports {versions}; Edgent speaks {', '.join(MODERN_VERSIONS)} statelessly", ErrorCategory.unsupported,
                           "Use a release of the server that speaks one of these versions.")
        self.protocol_version = chosen
        self.server_info = result.get("serverInfo") if isinstance(result.get("serverInfo"), dict) else {}
        self.capabilities = result.get("capabilities") if isinstance(result.get("capabilities"), dict) else {}
        self.instructions = str(result.get("instructions") or "")

    # -- requests
    def _next_id(self) -> int:
        with self._id_lock:
            return next(self._ids)

    def _meta(self) -> dict[str, Any]:
        return {"io.modelcontextprotocol/protocolVersion": self.protocol_version, "io.modelcontextprotocol/clientInfo": self._client_info(),
                "io.modelcontextprotocol/clientCapabilities": self._client_capabilities()}

    def _raw_request(self, method: str, params: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        if self._transport is None:
            raise McpError("unavailable", "not connected", ErrorCategory.unavailable, "Connect first.", retryable=True)
        msg = {"jsonrpc": "2.0", "id": self._next_id(), "method": method, "params": params}
        reply = self._transport.request(msg, timeout_s)
        if "error" in reply:
            raise _rpc_error(reply["error"], self.options.secrets)
        result = reply.get("result")
        if not isinstance(result, dict):
            raise McpError("protocol_error", f"{method} returned no result object", ErrorCategory.execution_failure, "Report it to the server's maintainers.")
        return result

    def _request(self, method: str, params: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        if self.era == "modern":
            params = {**params, "_meta": {**(params.get("_meta") or {}), **self._meta()}}
        result = self._raw_request(method, params, timeout_s)
        kind = result.get("resultType", "complete")
        if kind == "input_required":
            asks = result.get("inputRequests") if isinstance(result.get("inputRequests"), dict) else {}
            methods = sorted({str(v.get("method")) for v in asks.values() if isinstance(v, dict) and v.get("method")}) or ["an interaction"]
            raise McpError("input_required", f"the server needs input Edgent cannot give ({', '.join(methods)})",
                           ErrorCategory.unsupported, "Do the step in the service's own interface, then retry; Edgent answers no elicitation or sampling requests.")
        if kind != "complete":
            raise McpError("protocol_error", f"unknown resultType {kind!r}", ErrorCategory.execution_failure, "Report it to the server's maintainers.")
        return result

    def request(self, method: str, params: dict[str, Any] | None = None, timeout_s: float | None = None) -> dict[str, Any]:
        """One request on a connected client, within one deadline that also covers (re)connecting; a legacy HTTP
        session the server ended is re-opened once, by whichever thread sees it first."""
        timeout = self.options.request_timeout_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout
        if not self.connected:
            self.connect()
        generation = self._generation
        try:
            return self._request(method, params or {}, self._remaining(deadline, method))
        except McpError as e:
            if e.code != "session_expired":
                raise
            with self._connect_lock:
                if self._generation == generation:  # nobody has re-opened it yet
                    self._close_transport()
            self.connect()
            return self._request(method, params or {}, self._remaining(deadline, method))

    @staticmethod
    def _remaining(deadline: float, method: str) -> float:
        left = deadline - time.monotonic()
        if left <= 0:
            raise McpError("timeout", f"{method} ran out of time while connecting", ErrorCategory.timeout,
                           "Retry once; the connector was slow to start.", retryable=True)
        return left

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        if self._transport is not None:
            self._transport.write({"jsonrpc": "2.0", "method": method, "params": params})

    def list_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        seen: set[str] = set()
        cursors: set[str] = set()
        cursor: str | None = None
        self.tools_partial = ""
        for page in range(MAX_LIST_PAGES):
            result = self.request("tools/list", {"cursor": cursor} if cursor else {})
            ttl = result.get("ttlMs")
            if isinstance(ttl, (int, float)) and ttl > 0:
                self.tools_ttl_s = float(ttl) / 1000.0
            for t in result.get("tools") or []:
                if isinstance(t, dict) and isinstance(t.get("name"), str) and t["name"] not in seen:
                    seen.add(t["name"])
                    tools.append(t)
            nxt = result.get("nextCursor")
            if not nxt:
                break
            if str(nxt) in cursors:
                self.tools_partial = "the server repeated a page cursor"
                break
            if page == MAX_LIST_PAGES - 1:
                self.tools_partial = f"stopped after {MAX_LIST_PAGES} pages"
                break
            cursor = str(nxt)
            cursors.add(cursor)
        if self.tools_partial:
            self._event("log", {"level": "warning", "data": f"tool list may be incomplete: {self.tools_partial}"})
        if isinstance(self._transport, HttpTransport):
            self._transport.tool_schemas = {str(t["name"]): t.get("inputSchema") for t in tools if isinstance(t.get("inputSchema"), dict)}
        self.tools_changed = False
        return tools

    def call_tool(self, name: str, arguments: dict[str, Any], timeout_s: float | None = None) -> dict[str, Any]:
        return self.request("tools/call", {"name": name, "arguments": arguments}, timeout_s)

    # -- messages from the server
    def _on_stdio_message(self, msg: dict[str, Any]) -> None:
        reply = self._handle_server_message(msg)
        if reply is not None and self._transport is not None:
            try:
                self._transport.write(reply)
            except McpError:
                pass

    def _on_http_message(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        return self._handle_server_message(msg)

    def _handle_server_message(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        method = str(msg.get("method"))
        if "id" not in msg:
            if method == "notifications/tools/list_changed":
                self.tools_changed = True
                self._event("list_changed", {})
            elif method == "notifications/message":
                self._event("log", msg.get("params") if isinstance(msg.get("params"), dict) else {})
            elif method == "notifications/progress":
                self._event("progress", msg.get("params") if isinstance(msg.get("params"), dict) else {})
            return None
        if method == "ping":
            return {"jsonrpc": "2.0", "id": msg["id"], "result": {}}
        if method == "roots/list" and self.options.roots:
            roots = [{"uri": file_uri(r), "name": Path(r).name or r} for r in self.options.roots]
            return {"jsonrpc": "2.0", "id": msg["id"], "result": {"roots": roots}}
        return {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": METHOD_NOT_FOUND, "message": f"Edgent does not support {method}"}}

    def _event(self, kind: str, data: dict[str, Any]) -> None:
        if self.options.on_event:
            try:
                self.options.on_event(kind, data)
            except Exception:  # noqa: BLE001 - a listener never breaks the connection
                pass


def python_command() -> str:
    """The interpreter running Edgent, for connectors written in Python (and the test fakes)."""
    return sys.executable
