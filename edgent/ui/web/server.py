"""The local web UI server: a reader with the tutor beside it, the outline, the learner dashboard.

Standard-library ``http.server``, bound to localhost only, serving the static
app and a small JSON API over a ``Service`` facade. The facade is what the
integration layer implements over the agent loop; tests use a fake. Nothing
here is reachable from outside the machine, and every response carries a
content-security policy that forbids external scripts.
"""
from __future__ import annotations

import json
import mimetypes
import os
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qs, urlparse

STATIC_DIR = Path(__file__).parent / "static"
MAX_BODY_BYTES = 2_000_000  # named limit: request body size


class Service(Protocol):
    """What the UI needs from the application. All methods return JSON-serialisable data."""

    def workspace_info(self) -> dict[str, Any]: ...
    def list_sessions(self) -> list[dict[str, Any]]: ...
    def create_session(self, mode: str, title: str) -> dict[str, Any]: ...
    def session_messages(self, session_id: str, limit: int = 100) -> list[dict[str, Any]]: ...
    def send(self, session_id: str, text: str, screen: dict[str, Any] | None = None) -> dict[str, Any]: ...
    def outline(self) -> dict[str, Any]: ...
    def chapters(self) -> list[dict[str, Any]]: ...
    def chapter_html(self, path: str) -> dict[str, Any]: ...
    def sources(self) -> list[dict[str, Any]]: ...
    def locator(self, locator: str) -> dict[str, Any]: ...
    def learner(self) -> dict[str, Any]: ...
    def unknowns(self) -> dict[str, Any]: ...
    def build_status(self) -> dict[str, Any]: ...
    def start_build(self, goal: str) -> dict[str, Any]: ...
    def doctor(self) -> dict[str, Any]: ...
    def events(self, since: int = 0) -> list[dict[str, Any]]: ...


class _Handler(BaseHTTPRequestHandler):
    server_version = "edgent/0.1"
    service: Service
    token: str | None

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
        if os.environ.get("EDGENT_WEB_LOG"):
            super().log_message(fmt, *args)

    # ------------------------------------------------------------------ helpers
    def _send(self, status: int, body: bytes, ctype: str = "application/json; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, obj: Any) -> None:
        self._send(status, json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"))

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": {"code": status, "message": message}})

    def _authorised(self) -> bool:
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost", "[::1]", "::1", ""):
            return False
        if self.token:
            return self.headers.get("X-Edgent-Token") == self.token or parse_qs(urlparse(self.path).query).get("token", [None])[0] == self.token
        return True

    def _body(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY_BYTES:
            raise ValueError(f"request body over the {MAX_BODY_BYTES} byte limit")
        raw = self.rfile.read(n) if n else b""
        if not raw:
            return {}
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("body must be a JSON object")
        return data

    # ------------------------------------------------------------------ routing
    def do_GET(self) -> None:  # noqa: N802
        if not self._authorised():
            return self._error(403, "forbidden")
        url = urlparse(self.path)
        path = url.path
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path.startswith("/static/"):
                return self._static(path[len("/static/") :])
            if path == "/api/workspace":
                return self._json(200, self.service.workspace_info())
            if path == "/api/sessions":
                return self._json(200, self.service.list_sessions())
            if path.startswith("/api/sessions/") and path.endswith("/messages"):
                sid = path[len("/api/sessions/") : -len("/messages")]
                return self._json(200, self.service.session_messages(sid, int(q.get("limit", 100))))
            if path == "/api/outline":
                return self._json(200, self.service.outline())
            if path == "/api/chapters":
                return self._json(200, self.service.chapters())
            if path == "/api/chapter":
                return self._json(200, self.service.chapter_html(q.get("path", "")))
            if path == "/api/sources":
                return self._json(200, self.service.sources())
            if path == "/api/locator":
                return self._json(200, self.service.locator(q.get("locator", "")))
            if path == "/api/learner":
                return self._json(200, self.service.learner())
            if path == "/api/unknowns":
                return self._json(200, self.service.unknowns())
            if path == "/api/build":
                return self._json(200, self.service.build_status())
            if path == "/api/doctor":
                return self._json(200, self.service.doctor())
            if path == "/api/events":
                return self._json(200, self.service.events(int(q.get("since", 0))))
            return self._error(404, "not found")
        except FileNotFoundError as e:
            return self._error(404, str(e))
        except Exception as e:  # noqa: BLE001
            return self._error(500, f"{type(e).__name__}: {e}")

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorised():
            return self._error(403, "forbidden")
        path = urlparse(self.path).path
        try:
            body = self._body()
            if path == "/api/sessions":
                return self._json(201, self.service.create_session(str(body.get("mode", "chat")), str(body.get("title", ""))))
            if path.startswith("/api/sessions/") and path.endswith("/send"):
                sid = path[len("/api/sessions/") : -len("/send")]
                text = str(body.get("text", "")).strip()
                if not text:
                    return self._error(400, "text is required")
                return self._json(200, self.service.send(sid, text, body.get("screen")))
            if path == "/api/build/start":
                goal = str(body.get("goal", "")).strip()
                if not goal:
                    return self._error(400, "goal is required")
                return self._json(202, self.service.start_build(goal))
            return self._error(404, "not found")
        except (ValueError, json.JSONDecodeError) as e:
            return self._error(400, str(e))
        except FileNotFoundError as e:
            return self._error(404, str(e))
        except Exception as e:  # noqa: BLE001
            return self._error(500, f"{type(e).__name__}: {e}")

    def _static(self, name: str) -> None:
        p = (STATIC_DIR / name).resolve()
        if not str(p).startswith(str(STATIC_DIR.resolve())) or not p.is_file():
            return self._error(404, "not found")
        ctype = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        self._send(200, p.read_bytes(), ctype)


class LocalServer:
    """Serve on 127.0.0.1 only. ``token`` adds a shared secret for the API when set."""

    def __init__(self, service: Service, host: str = "127.0.0.1", port: int = 8765, token: str | None = None) -> None:
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("the local UI only binds to loopback; use a tunnel or reverse proxy you control if you need more")
        self.service = service
        self.host = host
        self.port = port
        self.token = token
        handler = type("EdgentHandler", (_Handler,), {"service": service, "token": token})
        self.httpd = ThreadingHTTPServer((host, port), handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/" + (f"?token={self.token}" if self.token else "")

    def start(self) -> LocalServer:
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="edgent-web", daemon=True)
        self._thread.start()
        return self

    def serve_forever(self) -> None:
        try:
            self.httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.httpd.server_close()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    @staticmethod
    def new_token() -> str:
        return secrets.token_urlsafe(16)
