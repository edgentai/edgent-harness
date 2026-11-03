from __future__ import annotations

import http.client
import json

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


def req(srv, method, path, body=None, token="secret", host=None):
    c = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=5)
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Edgent-Token"] = token
    if host:
        headers["Host"] = host
    c.request(method, path, body=json.dumps(body) if body is not None else None, headers=headers)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


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
