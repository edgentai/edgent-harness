"""Local stdlib HTTP + real-time SSE streaming daemon for ``AgentLoop``.

Exposes a clean REST + Server-Sent Events API bound to ``127.0.0.1`` by default:
- ``GET  /healthz``
- ``GET  /v1/tools`` | ``/v1/skills`` | ``/v1/subagents`` | ``/v1/tasks``
- ``GET  /v1/sessions`` | ``GET /v1/sessions/{id}``
- ``POST /v1/sessions/{id}/messages`` (JSON or real-time ``text/event-stream`` SSE)
- ``POST /v1/sessions/{id}/review``   (JSON or SSE)
- ``POST /v1/sessions/{id}/cancel``
- ``POST /v1/sessions/{id}/undo``
- ``POST /v1/sessions/{id}/compact``
- ``GET  /v1/heartbeat`` | ``POST /v1/heartbeat/{job_id}/trigger``
"""
from __future__ import annotations

import json
import queue
import threading
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

from ..core.client import StreamEvent
from ..session.heartbeat import HeartbeatScheduler
from ..session.loop import AgentLoop


class HarnessServer:
    """Thread-safe local HTTP/SSE server wrapping an ``AgentLoop``."""

    def __init__(
        self,
        loop: AgentLoop,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        auth_token: str = "",
        heartbeat: Optional[HeartbeatScheduler] = None,
    ) -> None:
        self.loop = loop
        self.host = host
        self.port = port
        self.auth_token = auth_token
        self.heartbeat = heartbeat
        self._lock = threading.Lock()
        self._session_locks: dict[str, threading.Lock] = {}
        self._cancels: dict[str, threading.Event] = {}
        self._httpd = ThreadingHTTPServer((host, port), self._make_handler())
        self._httpd.daemon_threads = True
        self._thread: Optional[threading.Thread] = None

    @property
    def address(self) -> tuple[str, int]:
        addr = self._httpd.server_address
        return str(addr[0]), int(addr[1])

    @property
    def base_url(self) -> str:
        host, port = self.address
        return f"http://{host}:{port}"

    def _session_lock(self, session_id: str) -> threading.Lock:
        with self._lock:
            lk = self._session_locks.get(session_id)
            if lk is None:
                lk = threading.Lock()
                self._session_locks[session_id] = lk
            return lk

    def _register_cancel(self, session_id: str) -> threading.Event:
        ev = threading.Event()
        with self._lock:
            self._cancels[session_id] = ev
        return ev

    def _clear_cancel(self, session_id: str, ev: threading.Event) -> None:
        with self._lock:
            if self._cancels.get(session_id) is ev:
                self._cancels.pop(session_id, None)

    def cancel_session(self, session_id: str) -> bool:
        with self._lock:
            ev = self._cancels.get(session_id)
        if ev is not None:
            ev.set()
            return True
        return False

    def _make_handler(self):
        srv = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
                return

            def _check_auth(self) -> bool:
                if not srv.auth_token:
                    return True
                hdr = self.headers.get("Authorization") or ""
                if hdr == f"Bearer {srv.auth_token}":
                    return True
                self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return False

            def _read_json(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length") or "0")
                if length <= 0:
                    return {}
                raw = self.rfile.read(min(length, 2 * 1024 * 1024))
                try:
                    data = json.loads(raw.decode("utf-8"))
                    return data if isinstance(data, dict) else {}
                except json.JSONDecodeError:
                    return {}

            def _send_json(self, status: int, payload: Any) -> None:
                body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
                self.send_response(int(status))
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def _stream_turn(self, session_id: str, fn) -> None:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()

                q: queue.Queue[Optional[tuple[str, dict[str, Any]]]] = queue.Queue()

                def on_event(ev: StreamEvent) -> None:
                    q.put((ev.kind, ev.to_dict()))

                def worker() -> None:
                    cancel = srv._register_cancel(session_id)
                    try:
                        with srv._session_lock(session_id):
                            res = fn(on_event, cancel)
                        q.put(("result", res.to_dict()))
                    except Exception as e:  # noqa: BLE001
                        q.put(("error", {"error": f"{type(e).__name__}: {e}"}))
                    finally:
                        srv._clear_cancel(session_id, cancel)
                        q.put(None)

                t = threading.Thread(target=worker, daemon=True)
                t.start()
                while True:
                    item = q.get()
                    if item is None:
                        break
                    ev_name, payload = item
                    chunk = f"event: {ev_name}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n".encode("utf-8")
                    try:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    except OSError:
                        srv.cancel_session(session_id)
                        break
                t.join(timeout=2.0)

            def do_GET(self) -> None:
                if not self._check_auth():
                    return
                parsed = urllib.parse.urlparse(self.path)
                path = parsed.path.rstrip("/") or "/"
                if path == "/healthz":
                    self._send_json(HTTPStatus.OK, {
                        "ok": True,
                        "workspace": srv.loop.workspace.path,
                        "mode": srv.loop.policy.mode.value,
                        "tools": len(srv.loop.registry.names()),
                        "skills": srv.loop.skills.names(),
                        "sessions": len(srv.loop.sessions.list_sessions()),
                    })
                    return
                if path == "/v1/tools":
                    self._send_json(HTTPStatus.OK, {
                        "tools": [
                            {"name": s.name, "description": s.description, "permission": s.permission}
                            for s in srv.loop.registry.all()
                        ]
                    })
                    return
                if path == "/v1/skills":
                    self._send_json(HTTPStatus.OK, {
                        "skills": [{"name": s.name, "description": s.description} for s in srv.loop.skills.all()]
                    })
                    return
                if path == "/v1/subagents":
                    self._send_json(HTTPStatus.OK, {
                        "subagents": [
                            {"name": s.name, "description": s.description, "tools": s.tools}
                            for s in srv.loop.subagents.all()
                        ]
                    })
                    return
                if path == "/v1/tasks":
                    self._send_json(HTTPStatus.OK, {"tasks": srv.loop.list_tasks()})
                    return
                if path == "/v1/sessions":
                    self._send_json(HTTPStatus.OK, {"sessions": srv.loop.sessions.list_sessions()})
                    return
                if path == "/v1/heartbeat":
                    jobs = srv.heartbeat.list_jobs() if srv.heartbeat else []
                    self._send_json(HTTPStatus.OK, {"jobs": jobs})
                    return
                if path.startswith("/v1/sessions/"):
                    sid = urllib.parse.unquote(path[len("/v1/sessions/") :])
                    if "/" not in sid and sid:
                        st = srv.loop.sessions.load_state(sid)
                        self._send_json(HTTPStatus.OK, {
                            "state": st.to_dict(),
                            "notes": srv.loop.sessions.read_notes(sid),
                            "checkpoints": srv.loop.checkpoints.list_checkpoints(sid),
                        })
                        return
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"not found: {path}"})

            def do_POST(self) -> None:
                if not self._check_auth():
                    return
                parsed = urllib.parse.urlparse(self.path)
                path = parsed.path.rstrip("/")
                body = self._read_json()

                if path.startswith("/v1/heartbeat/") and path.endswith("/trigger"):
                    if not srv.heartbeat:
                        self._send_json(HTTPStatus.BAD_REQUEST, {"error": "heartbeat scheduler not enabled"})
                        return
                    job_id = urllib.parse.unquote(path[len("/v1/heartbeat/") : -len("/trigger")])
                    try:
                        rec = srv.heartbeat.trigger_now(job_id)
                        self._send_json(HTTPStatus.OK, rec)
                    except KeyError as e:
                        self._send_json(HTTPStatus.NOT_FOUND, {"error": str(e)})
                    return

                parts = [p for p in path.split("/") if p]
                if len(parts) == 4 and parts[0] == "v1" and parts[1] == "sessions":
                    sid = urllib.parse.unquote(parts[2])
                    action = parts[3]
                    want_stream = bool(body.get("stream")) or "text/event-stream" in (self.headers.get("Accept") or "")

                    if action == "messages":
                        msg = str(body.get("message") or "")
                        stage = body.get("stage")
                        if want_stream:
                            self._stream_turn(
                                sid,
                                lambda on_ev, cancel: srv.loop.send(sid, msg, stage=stage, on_event=on_ev, cancel_event=cancel),
                            )
                            return
                        cancel = srv._register_cancel(sid)
                        try:
                            with srv._session_lock(sid):
                                res = srv.loop.send(sid, msg, stage=stage, cancel_event=cancel)
                        finally:
                            srv._clear_cancel(sid, cancel)
                        self._send_json(HTTPStatus.OK, res.to_dict())
                        return

                    if action == "review":
                        approved = bool(body.get("approved", False))
                        feedback = str(body.get("feedback") or "")
                        stage = body.get("stage")
                        if want_stream:
                            self._stream_turn(
                                sid,
                                lambda on_ev, cancel: srv.loop.resume_review(
                                    sid, approved=approved, feedback=feedback, stage=stage, on_event=on_ev, cancel_event=cancel
                                ),
                            )
                            return
                        cancel = srv._register_cancel(sid)
                        try:
                            with srv._session_lock(sid):
                                res = srv.loop.resume_review(sid, approved=approved, feedback=feedback, stage=stage, cancel_event=cancel)
                        finally:
                            srv._clear_cancel(sid, cancel)
                        self._send_json(HTTPStatus.OK, res.to_dict())
                        return

                    if action == "cancel":
                        self._send_json(HTTPStatus.OK, {"session_id": sid, "cancelled": srv.cancel_session(sid)})
                        return

                    if action == "undo":
                        steps = int(body.get("steps", 1))
                        with srv._session_lock(sid):
                            res = srv.loop.undo(sid, steps=steps)
                        self._send_json(HTTPStatus.OK, res)
                        return

                    if action == "compact":
                        keep_recent = int(body.get("keep_recent", 6))
                        with srv._session_lock(sid):
                            summary = srv.loop.compact(sid, keep_recent=keep_recent)
                        self._send_json(HTTPStatus.OK, {"session_id": sid, "compacted": summary is not None, "summary": summary})
                        return

                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"not found: {path}"})

        return _Handler

    def start_background(self) -> tuple[str, int]:
        if self.heartbeat is not None:
            self.heartbeat.start()
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="edgent-http", daemon=True)
        self._thread.start()
        return self.address

    def serve_forever(self) -> None:
        if self.heartbeat is not None:
            self.heartbeat.start()
        try:
            self._httpd.serve_forever()
        finally:
            self.stop()

    def stop(self) -> None:
        if self.heartbeat is not None:
            self.heartbeat.stop()
        try:
            self._httpd.shutdown()
            self._httpd.server_close()
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
