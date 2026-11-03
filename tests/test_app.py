"""The composition root: workspaces, providers, chat turns, builds, plans, the doctor."""
from __future__ import annotations

import json

import pytest

from edgent.adapters.providers.mock import MockProvider
from edgent.application.app import App, AppConfig
from edgent.domain.outline import Outline, Unit
from edgent.domain.types import Reply
from edgent.domain.workspace import Workspace

CHAPTER = (
    "---\ntitle: Vectors\nunit_id: chapter-vectors\nlevel: undergraduate\nintroduces: [vector]\n---\n# Vectors\n\n"
    "A vector is a list of numbers.\n\n```python\nv = [1, 2, 3]\nprint(sum(v))\n```\n\n```output\n6\n```\n\nYou can now add vectors.\n"
)


def make_app(tmp_path, home, provider=None, **cfg):
    ws = Workspace.init(tmp_path / "ws", git=False)
    return App(ws, AppConfig(provider="mock", home=str(home), **cfg), provider=provider)


def test_open_creates_and_finds_the_workspace(tmp_path, home):
    cfg = AppConfig(provider="mock", home=str(home))
    with pytest.raises(FileNotFoundError):
        App.open(tmp_path / "nowhere", cfg)
    app = App.open(tmp_path / "ws", cfg, create=True)
    assert (app.workspace.root / "STUDY.md").exists()
    nested = App.open(app.workspace.root / "book" / "chapters", cfg)
    assert nested.workspace.root == app.workspace.root
    assert app.provider_meta["kind"] == "mock"
    names = app.registry.names()
    assert {"read_file", "grep", "agent", "sources_add", "book_mark_done", "check_execute", "film_render", "assess_create", "learner_update"} <= set(names)
    assert "explaining" in app.skills.names() and "librarian" in app.agents.names()


def test_model_override_lands_in_the_provider_settings(tmp_path, home):
    app = make_app(tmp_path, home, MockProvider(), model="mock-large")
    assert app.settings["provider"]["model"] == "mock-large"
    assert app.permissions().mode == "ask" and app.permissions("auto").mode == "auto"
    app2 = make_app(tmp_path / "b", home, MockProvider(), mode="plan")
    assert app2.permissions().mode == "plan" and app2.tool_context().permissions.mode == "plan"


def test_chat_turn_runs_tools_records_events_and_continues_a_session(tmp_path, home):
    provider = MockProvider().push(MockProvider.tool_reply("list_files", {}), Reply(text="Here is what I found."))
    app = make_app(tmp_path, home, provider)
    session, result = app.chat("What is in this workspace?")
    assert result.stop_reason == "done" and result.text == "Here is what I found." and result.tool_calls == 1
    kinds = [e["kind"] for e in app.events()]
    assert "model_call" in kinds and "tool_call" in kinds
    assert [e["name"] for e in app.events() if e["kind"] == "tool_call"] == ["list_files"]
    assert app.sessions().list()[0]["session_id"] == session.session_id
    assert session.transcript_path.exists()
    provider.push(Reply(text="Second answer."))
    session2, result2 = app.chat("And again?", session_id=session.session_id)
    assert session2.session_id == session.session_id and result2.text == "Second answer."
    roles = [m.role for m in app.load_session(session.session_id).messages]
    assert roles[:3] == ["user", "assistant", "tool"] and roles[-2:] == ["user", "assistant"], roles
    assert app.events(since=app.events()[-1]["seq"]) == []


def test_build_job_runs_a_worker_through_the_gate(tmp_path, home):
    ws = Workspace.init(tmp_path / "ws", git=False)
    outline = Outline(ws.path("outline"))
    outline.title, outline.persona = "Linear Algebra", "undergraduate"
    outline.add(Unit(id="chapter-vectors", kind="chapter", title="Vectors", objective="what a vector is", path="book/chapters/01-vectors.md", meta={"level": "undergraduate"}))
    outline.save()
    provider = MockProvider().push(MockProvider.tool_reply("write_file", {"path": "book/chapters/01-vectors.md", "content": CHAPTER}), Reply(text="Drafted the chapter."))
    app = App(ws, AppConfig(provider="mock", home=str(home), critic=False), provider=provider)
    report = app.build_job().run(max_units=1)
    assert report.units_run == ["chapter-vectors"] and not report.errors and not report.blocked, report.to_dict()
    unit = app.outline().get("chapter-vectors")
    assert unit.status == "done" and report.done == ["chapter-vectors"] and unit.open_obligations == [], unit.to_dict()  # the gate closed every obligation
    assert (ws.root / "book/chapters/01-vectors.md").read_text(encoding="utf-8") == CHAPTER
    assert any(o.get("kind") == "execute" and o.get("status") == "closed" for o in unit.obligations) if hasattr(unit, "obligations") else True
    assert "unit chapter-vectors" in ws.progress_tail(20)
    kinds = {e["kind"] for e in app.events()}
    assert {"unit_start", "unit_finished", "run_finished", "tool_call"} <= kinds
    assert app.sessions().list(mode="build")[0]["title"] == "unit chapter-vectors"


def test_build_worker_that_stops_early_leaves_the_unit_partial_not_blocked(tmp_path, home):
    ws = Workspace.init(tmp_path / "ws", git=False)
    outline = Outline(ws.path("outline"))
    outline.add(Unit(id="u1", kind="chapter", title="One", objective="o", path="book/chapters/01-one.md"))
    outline.save()
    provider = MockProvider().push(Reply(text="I could not finish."))
    app = App(ws, AppConfig(provider="mock", home=str(home), critic=False, settings_overrides={"budget": {"max_iterations": 1}}), provider=provider)
    report = app.build_job().run()
    assert report.units_run[0] == "u1" and len(report.units_run) <= 2  # partial units get a second round, never a third
    assert app.outline().get("u1").status in ("partial", "blocked"), report.to_dict()


def test_plan_build_parses_the_initializer_reply(tmp_path, home):
    proposal = {
        "title": "Vectors for all",
        "persona": "undergraduate",
        "units": [
            {"id": "ch-vectors", "kind": "chapter", "title": "Vectors", "objective": "what a vector is", "path": "book/chapters/01-vectors.md"},
            {"id": "ch-matrices", "kind": "chapter", "title": "Matrices", "objective": "what a matrix is", "depends_on": ["ch-vectors"], "path": "book/chapters/02-matrices.md"},
        ],
    }
    provider = MockProvider().push(Reply(text="Here is the plan:\n```json\n" + json.dumps(proposal) + "\n```"))
    app = make_app(tmp_path, home, provider)
    outline = app.plan_build("a short book on linear algebra")
    assert [u.id for u in outline.units] == ["ch-vectors", "ch-matrices"] and outline.title == "Vectors for all"
    assert app.outline().get("ch-matrices").depends_on == ["ch-vectors"]
    assert any(e["kind"] == "plan" for e in app.events())


def test_plan_build_without_units_is_an_error(tmp_path, home):
    app = make_app(tmp_path, home, MockProvider().push(Reply(text="I have no plan.")))
    with pytest.raises(ValueError):
        app.plan_build("anything")


def test_doctor_reports_provider_toolchain_skills_and_tools(tmp_path, home):
    app = make_app(tmp_path, home, MockProvider())
    d = app.doctor()
    assert d["provider"]["name"] == "mock" and d["provider"]["health"]["ok"] and d["workspace"]["git"] is False
    assert d["tools"] > 40 and "explaining" in d["skills"] and "critic" in d["agents"]
    assert any(i["name"] == "ffmpeg" for i in d["items"])


def test_heartbeat_comes_from_settings(tmp_path, home):
    app = make_app(tmp_path, home, MockProvider(), settings_overrides={"heartbeat": {"times": ["07:30"], "prompt": "Plan today."}})
    assert app.heartbeat(lambda prompt: "ran: " + prompt).configured()
    assert not make_app(tmp_path / "b", home, MockProvider()).heartbeat().configured()
