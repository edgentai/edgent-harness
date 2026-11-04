from __future__ import annotations

import http.client
import json
import time

import pytest

from edgent.ui.web.server import LocalServer


class FakeService:
    def __init__(self):
        self.sent = []

    def workspace_info(self):
        return {"name": "ws"}

    def list_sessions(self):
        return [{"session_id": "s1", "mode": "chat", "title": "t"}]

    def create_session(self, mode, title):
        return {"session_id": "s2", "mode": mode, "title": title}

    def session_messages(self, session_id, limit=100):
        return [{"role": "user", "text": "hi"}]

    def send(self, session_id, text, screen=None):
        self.sent.append((session_id, text, screen))
        return {"text": "echo " + text, "stop_reason": "done"}

    def outline(self):
        return {"title": "O", "units": []}

    def chapters(self):
        return [{"path": "book/chapters/01.md", "title": "One"}]

    def chapter_html(self, path):
        if path != "book/chapters/01.md":
            raise FileNotFoundError(path)
        return {"path": path, "html": "<section data-section='a'><h1>One</h1></section>"}

    def sources(self):
        return []

    def locator(self, locator):
        return {"found": True, "text": "x"}

    def learner(self):
        return {"progress": {"streak_days": 1}}

    def unknowns(self):
        return {"markdown": "none"}

    def build_status(self):
        return {"units": 0}

    def start_build(self, goal):
        return {"started": True, "goal": goal}

    def doctor(self):
        return {"items": []}

    def events(self, since=0):
        return [{"seq": since + 1, "kind": "tool_call", "name": "grep"}]


@pytest.fixture
def server():
    svc = FakeService()
    srv = LocalServer(svc, port=0, token="secret").start()
    yield srv, svc
    srv.stop()


def req(srv, method, path, body=None, token="secret", host=None, headers=None, raw=None):
    c = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=30)
    hdrs = {"Content-Type": "application/json"}
    if token:
        hdrs["X-Edgent-Token"] = token
    if host:
        hdrs["Host"] = host
    hdrs.update(headers or {})
    payload = raw if raw is not None else (json.dumps(body) if body is not None else None)
    c.request(method, path, body=payload, headers=hdrs)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


def jreq(srv, method, path, body=None, **kw):
    status, data = req(srv, method, path, body, **kw)
    return status, json.loads(data)


def test_routes_and_security(server):
    srv, svc = server
    status, data = req(srv, "GET", "/")
    assert status == 200 and b"<title>Edgent</title>" in data
    status, data = req(srv, "GET", "/static/app.js")
    assert status == 200 and b"Edgent local UI" in data
    assert req(srv, "GET", "/static/../server.py")[0] == 404
    assert req(srv, "GET", "/api/workspace", token=None)[0] == 403
    assert req(srv, "GET", "/api/workspace", host="evil.example")[0] == 403
    status, data = req(srv, "GET", "/api/workspace")
    assert status == 200 and json.loads(data)["name"] == "ws"
    status, data = req(srv, "POST", "/api/sessions/s1/send", {"text": "hello", "screen": {"chapter": "c"}})
    assert status == 200 and json.loads(data)["text"] == "echo hello" and svc.sent[0][2] == {"chapter": "c"}
    assert req(srv, "POST", "/api/sessions/s1/send", {"text": "  "})[0] == 400
    assert req(srv, "GET", "/api/chapter?path=nope")[0] == 404
    status, data = req(srv, "GET", "/api/chapter?path=book/chapters/01.md")
    assert status == 200 and "<h1>One</h1>" in json.loads(data)["html"]
    status, data = req(srv, "POST", "/api/build/start", {"goal": "a book"})
    assert status == 202 and json.loads(data)["goal"] == "a book"
    status, data = req(srv, "GET", "/api/events?since=4")
    assert json.loads(data)[0]["seq"] == 5
    assert req(srv, "GET", "/api/nothing")[0] == 404
    assert req(srv, "POST", "/api/sessions", "not json")[0] == 400
    with pytest.raises(ValueError):
        LocalServer(svc, host="0.0.0.0", port=0)


def test_bracketed_ipv6_host_is_loopback(server):
    srv, _ = server
    assert req(srv, "GET", "/api/workspace", host=f"[::1]:{srv.port}")[0] == 200
    assert req(srv, "GET", "/api/workspace", host=f"localhost:{srv.port}")[0] == 200
    assert req(srv, "GET", "/api/workspace", host="[")[0] == 403  # malformed, not loopback
    assert req(srv, "GET", "/api/workspace", host="[::1]evil.example")[0] == 403


def test_state_changing_requests_must_come_from_our_origin():
    """With --no-token the Host header alone must not let a page on another origin drive the agent."""
    svc = FakeService()
    srv = LocalServer(svc, port=0, token=None).start()
    try:
        send = {"text": "hello"}
        own = f"http://127.0.0.1:{srv.port}"
        # same-origin and header-less requests (our page, curl) keep working
        assert req(srv, "POST", "/api/sessions/s1/send", send, token=None)[0] == 200
        assert req(srv, "POST", "/api/sessions/s1/send", send, token=None, headers={"Origin": own, "Sec-Fetch-Site": "same-origin"})[0] == 200
        assert req(srv, "POST", "/api/sessions/s1/send", send, token=None, headers={"Referer": own + "/?x=1"})[0] == 200
        assert req(srv, "POST", "/api/sessions/s1/send", send, token=None, host=f"localhost:{srv.port}", headers={"Origin": f"http://localhost:{srv.port}"})[0] == 200
        assert len(svc.sent) == 4
        # another origin, in any of the three signals, is refused before the body is read
        for headers in (
            {"Origin": "https://evil.example"},
            {"Origin": "null"},
            {"Origin": "http://["},
            {"Origin": f"http://127.0.0.1:{srv.port + 1}"},  # another local server's page
            {"Referer": "https://evil.example/page"},
            {"Sec-Fetch-Site": "cross-site"},
            {"Origin": own, "Sec-Fetch-Site": "cross-site"},
        ):
            status, data = jreq(srv, "POST", "/api/sessions/s1/send", send, token=None, headers=headers)
            assert status == 403 and "cross-site" in data["error"]["message"], headers
        # a no-preflight content type is refused too, so a simple cross-origin fetch cannot slip a JSON body through
        status, data = jreq(srv, "POST", "/api/sessions/s1/send", token=None, raw=json.dumps(send), headers={"Content-Type": "text/plain"})
        assert status == 400 and "application/json" in data["error"]["message"]
        assert len(svc.sent) == 4
        # reads stay open: without CORS headers a cross-origin page cannot see the response anyway
        assert req(srv, "GET", "/api/workspace", token=None, headers={"Origin": "https://evil.example"})[0] == 200
    finally:
        srv.stop()


def test_bad_query_values_and_unknown_mode_are_400(server):
    srv, _ = server
    status, data = jreq(srv, "GET", "/api/sessions/s1/messages?limit=abc")
    assert status == 400 and "limit" in data["error"]["message"]
    status, data = jreq(srv, "GET", "/api/events?since=-1")
    assert status == 400 and "since" in data["error"]["message"]
    assert jreq(srv, "GET", "/api/events?since=")[0] == 200  # empty means the default
    status, data = jreq(srv, "POST", "/api/sessions", {"mode": "weird"})
    assert status == 400 and "weird" in data["error"]["message"] and "chat" in data["error"]["message"]
    status, data = jreq(srv, "POST", "/api/sessions", {"mode": "build", "title": "B"})
    assert status == 201 and data["mode"] == "build"
    assert jreq(srv, "POST", "/api/sessions", {})[1]["mode"] == "chat"


def test_real_app_service_over_http(tmp_path, home):
    """The production seam: AppService over App with the mock provider, behind LocalServer, driven over HTTP."""
    from edgent.adapters.providers.mock import MockProvider
    from edgent.application.app import App, AppConfig
    from edgent.domain.types import Reply
    from edgent.domain.workspace import Workspace
    from edgent.ui.web.app_service import AppService

    ws = Workspace.init(tmp_path / "ws", git=False)
    ws.write_text("book/chapters/01-vectors.md", "---\ntitle: Vectors\n---\n# Vectors\n\nAn arrow [t:p.1] and ![d](x\"onerror=\"y).\n")
    provider = MockProvider().push(Reply(text="An arrow with length and direction."))
    app = App(ws, AppConfig(provider="mock", home=str(home)), provider=provider)
    srv = LocalServer(AppService(app), port=0, token="t").start()
    try:
        status, info = jreq(srv, "GET", "/api/workspace", token="t")
        assert status == 200 and info["provider"] == "mock" and info["root"] == str(ws.root)
        status, s = jreq(srv, "POST", "/api/sessions", {"mode": "chat", "title": "Study"}, token="t")
        assert status == 201 and s["session_id"].startswith("chat-")
        sid = s["session_id"]
        status, r = jreq(srv, "POST", f"/api/sessions/{sid}/send", {"text": "What is a vector?", "screen": {"chapter": "book/chapters/01-vectors.md"}}, token="t")
        assert status == 200 and r["text"].startswith("An arrow") and r["stop_reason"] == "done" and r["session_id"] == sid
        status, msgs = jreq(srv, "GET", f"/api/sessions/{sid}/messages?limit=10", token="t")
        assert status == 200 and [m["role"] for m in msgs] == ["user", "assistant"]
        assert jreq(srv, "GET", f"/api/sessions/{sid}/messages?limit=x", token="t")[0] == 400
        assert jreq(srv, "GET", "/api/sessions/nope/messages", token="t")[0] == 404
        assert jreq(srv, "GET", "/api/sessions/..%2Fx/messages", token="t")[0] in (400, 404)
        status, sessions = jreq(srv, "GET", "/api/sessions", token="t")
        assert status == 200 and sessions[0]["session_id"] == sid
        status, evs = jreq(srv, "GET", "/api/events?since=0", token="t")
        assert status == 200 and any(e["kind"] == "model_call" for e in evs)
        assert jreq(srv, "GET", "/api/events?since=abc", token="t")[0] == 400
        status, chapters = jreq(srv, "GET", "/api/chapters", token="t")
        assert status == 200 and chapters == [{"path": "book/chapters/01-vectors.md", "title": "Vectors"}]
        status, ch = jreq(srv, "GET", "/api/chapter?path=book%2Fchapters%2F01-vectors.md", token="t")
        assert status == 200 and "<h1" in ch["html"] and 'data-locator="t:p.1"' in ch["html"]
        assert 'src="x&quot;onerror=&quot;y"' in ch["html"] and 'onerror="' not in ch["html"]
        assert jreq(srv, "GET", "/api/chapter?path=..%2Fetc%2Fpasswd", token="t")[0] == 404
        for path in ("/api/outline", "/api/sources", "/api/learner", "/api/unknowns", "/api/doctor"):
            assert jreq(srv, "GET", path, token="t")[0] == 200, path
        status, b = jreq(srv, "GET", "/api/build", token="t")
        assert status == 200 and b["running"] is False
        status, started = jreq(srv, "POST", "/api/build/start", {"goal": "a short book"}, token="t")
        assert status == 202 and started["started"] is True
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            status, b = jreq(srv, "GET", "/api/build", token="t")
            assert status == 200
            if not b["running"]:
                break
            time.sleep(0.05)
        assert b["running"] is False and b["goal"] == "a short book" and ("report" in b or b.get("error"))
    finally:
        srv.stop()
