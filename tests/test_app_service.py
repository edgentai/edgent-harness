"""The web UI's service facade over the composition root."""
from __future__ import annotations

import pytest

from edgent.adapters.providers.mock import MockProvider
from edgent.application.app import App, AppConfig
from edgent.domain.types import Reply
from edgent.domain.workspace import Workspace
from edgent.ui.web.app_service import AppService


def test_service_facade_round_trip(tmp_path, home):
    ws = Workspace.init(tmp_path / "ws", git=False)
    ws.write_text("book/chapters/01-vectors.md", "---\ntitle: Vectors\n---\n# Vectors\n\nA vector is an arrow.\n")
    provider = MockProvider().push(Reply(text="An arrow with length and direction."))
    app = App(ws, AppConfig(provider="mock", home=str(home)), provider=provider)
    svc = AppService(app)
    info = svc.workspace_info()
    assert info["provider"] == "mock" and info["root"] == str(ws.root) and info["warnings"] == []
    sid = svc.create_session("chat", "Study")["session_id"]
    r = svc.send(sid, "What is a vector?", screen={"chapter": "book/chapters/01-vectors.md", "section": "Vectors"})
    assert r["text"].startswith("An arrow") and r["session_id"] == sid and r["stop_reason"] == "done"
    assert [m["role"] for m in svc.session_messages(sid)] == ["user", "assistant"]
    assert svc.list_sessions()[0]["session_id"] == sid
    assert svc.chapters() == [{"path": "book/chapters/01-vectors.md", "title": "Vectors"}]
    html = svc.chapter_html("book/chapters/01-vectors.md")
    assert "<h1" in html["html"] and "arrow" in html["html"]
    with pytest.raises(FileNotFoundError):
        svc.chapter_html("../etc/passwd")
    assert svc.outline()["units"] == []
    assert svc.sources() == [] and svc.learner()["due"] == [] and svc.unknowns()["open"] == []
    assert svc.build_status()["running"] is False
    assert svc.doctor()["provider"]["name"] == "mock"
    assert any(e["kind"] == "model_call" for e in svc.events())
    assert svc.start_heartbeat() is False
    svc.stop_heartbeat()
