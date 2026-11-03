"""Book tools through the registry: outline, start, brief, the done-gate, build, status and walkthrough."""
from __future__ import annotations

import json
from typing import Any

import pytest

from edgent.adapters.providers.mock import MockProvider
from edgent.domain.outline import Outline
from edgent.domain.unknowns import UnknownsLedger
from edgent.tools.checks import cite as cite_mod
from edgent.tools.checks.registry import build_checks_registry
from edgent.tools.edu import book as book_mod
from edgent.tools.edu.book import (
    BOOK_TOOL_NAMES,
    build_book_registry,
    mark_unit_done,
    propose_units,
    render_walkthrough,
)
from edgent.tools.registry import ToolContext, ToolRegistry, ToolSpec, obj
from tests.conftest import skip_unless
from tests.test_checks import BAD_CHAPTER, GOOD_CHAPTER, TEXTBOOK

UNITS: list[dict[str, Any]] = [
    {"id": "chapter-matrices", "kind": "chapter", "title": "Matrices", "objective": "multiply matrices", "depends_on": ["chapter-vectors"], "path": "book/chapters/02-matrices.md", "meta": {"level": "undergraduate"}},
    {"id": "chapter-vectors", "kind": "chapter", "title": "Vectors", "objective": "add vectors", "path": "book/chapters/01-vectors.md", "meta": {"level": "undergraduate", "introduces": ["vector"]}},
    {"id": "film-intro", "kind": "film", "title": "Why vectors", "objective": "motivate the idea"},
    {"id": "integration", "kind": "integration", "title": "Build the book", "depends_on": ["chapter-vectors", "chapter-matrices", "film-intro"]},
]
GOOD_MATRICES = GOOD_CHAPTER.replace("title: Vectors", "title: Matrices").replace("unit_id: chapter-vectors", "unit_id: chapter-matrices").replace("introduces: [vector, vector addition]", "introduces: [matrix]").replace("requires: []", "requires: [vector]")


@pytest.fixture
def tools(registry: ToolRegistry, ctx: ToolContext, workspace, monkeypatch: pytest.MonkeyPatch) -> ToolRegistry:
    monkeypatch.setattr(cite_mod, "external_resolver", lambda: None)
    workspace.write_text("sources/text/textbook.md", TEXTBOOK)
    workspace.write_json("sources/text/textbook.meta.json", {"id": "textbook", "kind": "pdf", "coverage": {"status": "partial", "detail": {"pages_total": 120, "pages_failed": [99]}}})
    build_checks_registry(registry)
    return build_book_registry(registry)


def ok(tools: ToolRegistry, ctx: ToolContext, name: str, **args: Any) -> dict[str, Any]:
    r = tools.call(name, args, ctx)
    assert r.ok, r.error
    return r.output


def err(tools: ToolRegistry, ctx: ToolContext, name: str, **args: Any) -> Any:
    r = tools.call(name, args, ctx)
    assert not r.ok, r.output
    assert r.error.remediation
    return r.error


def propose(tools: ToolRegistry, ctx: ToolContext) -> dict[str, Any]:
    return ok(tools, ctx, "book_outline_propose", title="Linear Algebra", persona="second-year undergraduate", units=UNITS)


# ----------------------------------------------------------------------------- registry and outline
def test_registry_contracts(tools: ToolRegistry) -> None:
    assert set(BOOK_TOOL_NAMES) <= set(tools.names())
    for name in BOOK_TOOL_NAMES:
        spec = tools.get(name)
        assert spec is not None and spec.group == "book" and spec.parameters["additionalProperties"] is False and len(spec.description) > 60
    assert tools.get("book_mark_done").side_effects is False and tools.get("book_mark_done").permission == "write"
    assert tools.get("book_outline_propose").side_effects and tools.get("book_outline_get").permission == "read"
    build_book_registry(tools)  # idempotent


def test_outline_propose_validates_orders_and_never_replaces(tools: ToolRegistry, ctx: ToolContext, workspace) -> None:
    bad = [
        {"kind": "poem", "title": "x"},
        {"kind": "chapter", "title": ""},
        {"kind": "chapter", "title": "Escape", "path": "../../etc/passwd"},
        {"kind": "chapter", "title": "Orphan", "depends_on": ["nope"]},
        {"id": "bad id!", "kind": "chapter", "title": "Bad id"},
        {"id": "chapter-vectors", "kind": "chapter", "title": "Dup"},
    ]
    out = ok(tools, ctx, "book_outline_propose", title="Linear Algebra", persona="second-year undergraduate", units=UNITS + bad)
    assert out["added"] == ["chapter-vectors", "chapter-matrices", "film-intro", "integration"]  # dependencies first
    reasons = [r["reason"] for r in out["rejected"]]
    assert len(reasons) == 6 and any("kind must be one of" in r for r in reasons) and any("escapes the workspace" in r for r in reasons)
    assert any("do not exist or form a cycle" in r for r in reasons) and any("duplicate id" in r for r in reasons) and any("non-empty string" in r for r in reasons)
    assert err(tools, ctx, "book_outline_propose", units=["not a dict"]).category.value == "invalid_input"
    assert out["problems"] == [] and out["title"] == "Linear Algebra" and out["commit"]
    o = Outline(workspace.path("outline"))
    assert [u.id for u in o.units] == out["added"] and o.persona == "second-year undergraduate" and o.get("chapter-matrices").depends_on == ["chapter-vectors"]
    assert "proposed 4 units" in workspace.progress_tail(5)
    again = ok(tools, ctx, "book_outline_propose", units=[{"id": "chapter-vectors", "kind": "chapter", "title": "Replaced?"}, {"kind": "chapter", "title": "Vectors"}])
    assert again["existing"] == ["chapter-vectors"] and again["added"] == ["chapter-vectors-2"] and o.get("chapter-vectors").title == "Vectors"
    assert "units proposed" in workspace.git_log(3)


def test_propose_units_rejects_cycles_inside_the_batch(workspace) -> None:
    o = Outline(workspace.path("outline"))
    res = propose_units(o, [{"id": "a", "kind": "chapter", "title": "A", "depends_on": ["b"]}, {"id": "b", "kind": "chapter", "title": "B", "depends_on": ["a"]}, {"id": "c", "kind": "chapter", "title": "C"}], workspace=workspace)
    assert res["added"] == ["c"] and {r["id"] for r in res["rejected"]} == {"a", "b"}


def test_outline_get_paginates(tools: ToolRegistry, ctx: ToolContext) -> None:
    propose(tools, ctx)
    page = ok(tools, ctx, "book_outline_get", offset=1, limit=2)
    assert [u["id"] for u in page["units"]] == ["chapter-matrices", "film-intro"] and page["next_offset"] == 3 and page["total"] == 4
    last = ok(tools, ctx, "book_outline_get", offset=3, limit=2)
    assert [u["id"] for u in last["units"]] == ["integration"] and last["next_offset"] is None
    assert page["summary"]["ready"] == ["chapter-vectors", "film-intro"]
    assert err(tools, ctx, "book_outline_get", limit=0).category.value == "invalid_input"


def test_outline_update_descriptive_fields_only(tools: ToolRegistry, ctx: ToolContext, workspace) -> None:
    propose(tools, ctx)
    out = ok(tools, ctx, "book_outline_update", unit_id="chapter-matrices", fields={"objective": "multiply and invert", "meta": {"estimated_words": 3000}, "checks": ["check_judge"]})
    unit = Outline(workspace.path("outline")).get("chapter-matrices")
    assert unit.objective == "multiply and invert" and unit.meta == {"level": "undergraduate", "estimated_words": 3000} and unit.checks == ["check_judge"]
    assert out["unit"]["id"] == "chapter-matrices"
    for fields in ({"status": "done"}, {"attempts": 9}, {"open_obligations": []}, {"id": "x"}, {}):
        assert err(tools, ctx, "book_outline_update", unit_id="chapter-matrices", fields=fields).category.value == "invalid_input"
    assert "cannot be edited" in err(tools, ctx, "book_outline_update", unit_id="chapter-matrices", fields={"status": "done"}).message
    assert err(tools, ctx, "book_outline_update", unit_id="chapter-matrices", fields={"path": "/etc/passwd"}).category.value == "invalid_input"
    assert err(tools, ctx, "book_outline_update", unit_id="chapter-matrices", fields={"depends_on": ["ghost"]}).category.value == "invalid_input"
    assert err(tools, ctx, "book_outline_update", unit_id="chapter-matrices", fields={"depends_on": ["chapter-matrices"]}).category.value == "invalid_input"
    assert err(tools, ctx, "book_outline_update", unit_id="nope", fields={"title": "x"}).category.value == "not_found"


def test_unit_start_needs_finished_dependencies(tools: ToolRegistry, ctx: ToolContext, workspace) -> None:
    propose(tools, ctx)
    e = err(tools, ctx, "book_unit_start", unit_id="chapter-matrices")
    assert e.category.value == "invalid_input" and "dependencies not finished" in e.message
    out = ok(tools, ctx, "book_unit_start", unit_id="chapter-vectors")
    assert out["unit"]["status"] == "in_progress" and out["unit"]["attempts"] == 1
    assert Outline(workspace.path("outline")).get("chapter-vectors").history[-1]["reason"] == "started by s1"


def test_unit_brief_orients_and_lists_obligations(tools: ToolRegistry, ctx: ToolContext, workspace) -> None:
    propose(tools, ctx)
    workspace.write_text("book/chapters/02-matrices.md", BAD_CHAPTER)
    brief = ok(tools, ctx, "book_unit_brief", unit_id="chapter-matrices", neighbours=1)["brief"]
    assert brief.startswith("# Build unit `chapter-matrices`: Matrices")
    assert "## Orientation" in brief and "Recent commits" in brief and "edgent init" in brief
    assert "`chapter-vectors` (chapter, pending, before): Vectors — add vectors" in brief and "`film-intro`" in brief and "`integration`" not in brief
    assert "A matrix is a rectangular array" not in brief  # neighbours and the unit itself are described, never pasted
    assert "- execute: code block 1 (line 12) runs (closer: check_execute)" in brief and "- cite: citation [ghost:p.1]" in brief
    assert "- judge:" in brief and "advisory" in brief
    assert "## Chapter file contract" in brief and "# edgent: continue" in brief
    assert "one unit" in brief.lower() and "book_mark_done" in brief and "## Persona" in brief and "second-year undergraduate" in brief
    film = ok(tools, ctx, "book_unit_brief", unit_id="film-intro")["brief"]
    assert "media/films/<id>/" in film and "- look:" in film


# ----------------------------------------------------------------------------- the done-gate
def test_mark_done_good_chapter_is_done_with_evidence_commit_and_progress(tools: ToolRegistry, ctx: ToolContext, workspace) -> None:
    propose(tools, ctx)
    workspace.write_text("book/chapters/01-vectors.md", GOOD_CHAPTER)
    out = ok(tools, ctx, "book_mark_done", unit_id="chapter-vectors")
    assert out["status"] == "done" and out["open_obligations"] == [] and out["unknown_ids"] == []
    rows = {r["id"]: r for r in out["ledger"]}
    assert rows["chapter-vectors:execute:1"]["status"] == "closed" and rows["chapter-vectors:cite:textbook:p.12"]["status"] == "closed"
    assert rows["chapter-vectors:judge"]["status"] == "open" and rows["chapter-vectors:judge"]["advisory"]
    assert out["summary"]["blocking_open"] == [] and len(out["evidence_ids"]) >= 5 and all(ctx.evidence.exists(e) for e in out["evidence_ids"])
    assert out["commit"] and "unit chapter-vectors: done" in workspace.git_log(3)
    assert "unit chapter-vectors: done" in workspace.progress_tail(5)
    unit = Outline(workspace.path("outline")).get("chapter-vectors")
    assert unit.status == "done" and unit.open_obligations == [] and unit.history[0]["reason"] == "started by book_mark_done"
    assert "6/7 obligations closed, 1 advisory not run" in unit.history[-1]["reason"]
    assert ok(tools, ctx, "book_outline_get")["summary"]["ready"] == ["chapter-matrices", "film-intro"]
    again = ok(tools, ctx, "book_mark_done", unit_id="chapter-vectors")  # re-verification keeps done
    assert again["status"] == "done" and Outline(workspace.path("outline")).get("chapter-vectors").attempts == 1


def test_mark_done_bad_chapter_is_partial_with_unknowns_then_resolves(tools: ToolRegistry, ctx: ToolContext, workspace) -> None:
    propose(tools, ctx)
    workspace.write_text("book/chapters/01-vectors.md", GOOD_CHAPTER)
    ok(tools, ctx, "book_mark_done", unit_id="chapter-vectors")
    workspace.write_text("book/chapters/02-matrices.md", BAD_CHAPTER)
    out = ok(tools, ctx, "book_mark_done", unit_id="chapter-matrices")
    assert out["status"] == "partial"
    kinds = sorted({o["kind"] for o in out["open_obligations"]})
    assert kinds == ["cite", "execute", "prereqs", "readability"]
    assert "chapter-matrices:execute:1" in out["summary"]["blocking_open"] and out["next"].startswith("fix what is open")
    unit = Outline(workspace.path("outline")).get("chapter-matrices")
    assert unit.status == "partial" and [o["kind"] for o in unit.open_obligations] == [o["kind"] for o in out["open_obligations"]]
    ledger = UnknownsLedger(workspace.path("unknowns"))
    opened = ledger.open(scope="chapter-matrices")
    assert {u.unknown_id for u in opened} == set(out["unknown_ids"]) and all(u.domain == "verification" for u in opened)
    probes = {u.recommended_probes[0]["operation"] for u in opened}
    assert probes == {"check_execute", "check_cite", "check_readability", "check_prereqs"}
    assert all(u.required_authority for u in opened) and any(u.severity == "high" for u in opened)
    assert "partial" in workspace.git_log(2) and "- open execute:" in workspace.progress_tail(8)
    # fix the chapter: everything closes, the unknowns are resolved with the closers' evidence
    workspace.write_text("book/chapters/02-matrices.md", GOOD_MATRICES)
    fixed = ok(tools, ctx, "book_mark_done", unit_id="chapter-matrices")
    assert fixed["status"] == "done" and fixed["open_obligations"] == []
    ledger = UnknownsLedger(workspace.path("unknowns"))
    assert ledger.open(scope="chapter-matrices") == []
    resolved = [u for u in ledger.by_scope("chapter-matrices") if u.status == "resolved"]
    assert len(resolved) == 3 and all(u.resolution["disposition"] == "verified" and u.resolution["evidence_ids"] for u in resolved)
    withdrawn = [u for u in ledger.by_scope("chapter-matrices") if u.status == "withdrawn"]
    assert len(withdrawn) == 3 and set(fixed["unknowns_withdrawn"]) == {u.unknown_id for u in withdrawn}  # the removed citations and block
    assert {u.question.split(": ")[0] for u in withdrawn} == {"chapter-matrices:execute:2", "chapter-matrices:cite:textbook:p.99", "chapter-matrices:cite:ghost:p.1"}
    unit = Outline(workspace.path("outline")).get("chapter-matrices")
    assert unit.status == "done" and unit.open_obligations == []


def test_mark_done_missing_file_unknown_unit_and_bad_path(tools: ToolRegistry, ctx: ToolContext, workspace) -> None:
    propose(tools, ctx)
    out = ok(tools, ctx, "book_mark_done", unit_id="chapter-vectors")
    assert out["status"] == "partial" and all("does not exist" in o["reason"] for o in out["open_obligations"])
    assert out["results"]["file"]["error"]["code"] == "not_found"
    ok(tools, ctx, "book_outline_propose", units=[{"id": "no-path", "kind": "chapter", "title": "No path"}, {"id": "brief", "kind": "research", "title": "Brief", "path": "knowledge/brief.md"}])
    out = ok(tools, ctx, "book_mark_done", unit_id="no-path")
    assert out["status"] == "partial" and all("has no path; set one with book_outline_update" in o["reason"] for o in out["open_obligations"])
    assert out["results"] == {"file": {"error": {"code": "invalid_input", "message": "unit no-path has no path"}}}
    out = ok(tools, ctx, "book_mark_done", unit_id="brief")  # a research brief whose file is missing is not done
    assert out["status"] == "partial" and [o["kind"] for o in out["open_obligations"]] == ["structure"] and "does not exist" in out["open_obligations"][0]["reason"]
    workspace.write_text("knowledge/brief.md", "# Brief\n\nThe textbook defines displacement [textbook:p.12].\n")
    out = ok(tools, ctx, "book_mark_done", unit_id="brief")
    assert out["status"] == "done" and {r["kind"]: r["status"] for r in out["ledger"]} == {"structure": "closed", "cite": "closed", "judge": "open"}
    assert err(tools, ctx, "book_mark_done", unit_id="ghost").category.value == "not_found"
    o = Outline(workspace.path("outline"))
    o.get("chapter-vectors").path = "../../outside.md"
    o.save()
    assert err(tools, ctx, "book_mark_done", unit_id="chapter-vectors").category.value == "invalid_input"


def test_mark_done_media_unit_without_closers_stays_partial_with_probes(tools: ToolRegistry, ctx: ToolContext, workspace) -> None:
    propose(tools, ctx)
    out = ok(tools, ctx, "book_mark_done", unit_id="film-intro")
    assert out["status"] == "partial" and [o["kind"] for o in out["open_obligations"]] == ["render", "measure", "look"]
    assert all("is not available in this registry" in o["reason"] for o in out["open_obligations"])
    ledger = UnknownsLedger(workspace.path("unknowns"))
    assert {u.recommended_probes[0]["operation"] for u in ledger.open(scope="film-intro")} == {"film_render", "film_look"}


def test_mark_done_reaches_external_closers_through_the_registry(tools: ToolRegistry, ctx: ToolContext, workspace) -> None:
    propose(tools, ctx)
    seen: list[tuple[str, dict[str, Any]]] = []

    def closer(name: str, authority: str = "controlled_replay"):
        def fn(_ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
            seen.append((name, args))
            return {"passed": True, "authority": authority, "evidence_id": f"ev_{name}"}

        return fn

    for name in ("film_render", "film_look"):
        tools.register(ToolSpec(name=name, description="fake", parameters=obj({"path": {"type": "string"}, "film_id": {"type": "string"}}), fn=closer(name), permission="compute"))
    ctx.extra["registry"] = tools
    out = ok(tools, ctx, "book_mark_done", unit_id="film-intro")
    assert out["status"] == "done" and [n for n, _ in seen] == ["film_render", "film_look"]  # one render call closes render and measure
    assert seen[0][1] == {"path": "media/films/film-intro", "film_id": "film-intro"}
    rows = {r["id"]: r for r in out["ledger"]}
    assert rows["film-intro:render"]["evidence_ids"] == ["ev_film_render"] and rows["film-intro:measure"]["evidence_ids"] == ["ev_film_render"]


def test_mark_done_closes_a_widget_from_its_measurement_sidecar(tools: ToolRegistry, ctx: ToolContext, workspace) -> None:
    """widget_run takes the HTML itself, so the gate cannot re-run it; it trusts the tool's sidecar while the file is unchanged."""
    import os
    import time

    ok(tools, ctx, "book_outline_propose", units=[{"id": "widget-slider", "kind": "widget", "title": "Slider"}])
    tools.register(ToolSpec(name="widget_run", description="fake", parameters=obj({"widget_id": {"type": "string"}, "html": {"type": "string"}}, ["widget_id", "html"]), fn=lambda c, a: {"ok": True}, permission="compute"))
    ctx.extra["registry"] = tools
    out = ok(tools, ctx, "book_mark_done", unit_id="widget-slider")
    assert out["status"] == "partial" and all("media/widgets/widget-slider.html does not exist yet" in o["reason"] for o in out["open_obligations"])
    workspace.write_text("media/widgets/widget-slider.html", "<html><body>slider</body></html>")
    out = ok(tools, ctx, "book_mark_done", unit_id="widget-slider")
    assert out["status"] == "partial" and "no measurement record" in out["open_obligations"][0]["reason"]
    workspace.write_json("media/widgets/widget-slider.html.defects.json", {"ok": False, "defects": [{"kind": "console_error", "severity": "error", "detail": "ReferenceError: x"}]})
    out = ok(tools, ctx, "book_mark_done", unit_id="widget-slider")
    assert out["status"] == "partial" and out["open_obligations"][0]["reason"] == "ReferenceError: x"
    workspace.write_json("media/widgets/widget-slider.html.defects.json", {"ok": True, "defects": []})
    out = ok(tools, ctx, "book_mark_done", unit_id="widget-slider")
    rows = {r["id"]: r for r in out["ledger"]}
    assert out["status"] == "done" and rows["widget-slider:render"]["status"] == rows["widget-slider:measure"]["status"] == "closed"
    assert out["results"]["widget_run media/widgets/widget-slider.html"]["via"] == "measurement sidecar"
    # an edit after the measurement invalidates it
    time.sleep(0.01)
    html = workspace.root / "media/widgets/widget-slider.html"
    html.write_text("<html><body>changed</body></html>", encoding="utf-8")
    side = workspace.root / "media/widgets/widget-slider.html.defects.json"
    os.utime(side, (html.stat().st_atime, html.stat().st_mtime - 10))
    out = ok(tools, ctx, "book_mark_done", unit_id="widget-slider")
    assert out["status"] == "partial" and "changed after it was last measured" in out["open_obligations"][0]["reason"]


@skip_unless("pandoc")
def test_integration_unit_builds_the_book_and_reports_unresolved_references(tools: ToolRegistry, ctx: ToolContext, workspace) -> None:
    propose(tools, ctx)
    workspace.write_text("book/chapters/01-vectors.md", GOOD_CHAPTER.replace("## A worked example", "{{film: film-intro}}\n\n## A worked example"))
    ok(tools, ctx, "book_mark_done", unit_id="chapter-vectors")
    ok(tools, ctx, "book_outline_update", unit_id="integration", fields={"depends_on": ["chapter-vectors"]})
    out = ok(tools, ctx, "book_mark_done", unit_id="integration")
    rows = {r["id"]: r for r in out["ledger"]}
    assert out["status"] == "partial" and rows["integration:render"]["status"] == "closed" and "built book/build/book.html" in rows["integration:render"]["reason"]
    assert rows["integration:structure"]["status"] == "open" and rows["integration:structure"]["reason"] == "unresolved references: film-intro"
    assert (workspace.root / "book/build/book.html").exists() and ctx.evidence.get(rows["integration:render"]["evidence_ids"][0]).operation == "book_build"


def test_mark_done_with_judge_runs_the_critic_and_refuses_weak_closers(tools: ToolRegistry, ctx: ToolContext, workspace, provider: MockProvider) -> None:
    propose(tools, ctx)
    workspace.write_text("book/chapters/01-vectors.md", GOOD_CHAPTER)
    provider.push(json.dumps({"score": 9, "defects": [], "summary": "clear"}))
    out = ok(tools, ctx, "book_mark_done", unit_id="chapter-vectors", judge=True)
    rows = {r["id"]: r for r in out["ledger"]}
    assert rows["chapter-vectors:judge"]["status"] == "closed" and len(provider.calls) == 1 and out["status"] == "done"
    # a model's opinion can never close an execution obligation
    weak = tools.get("check_execute")
    tools.register(ToolSpec(name="film_render", description="fake", parameters=obj({"film_id": {"type": "string"}}), fn=lambda c, a: {"passed": True, "authority": "model_inference"}, permission="compute"))
    ctx.extra["registry"] = tools
    assert weak is not None
    film = ok(tools, ctx, "book_mark_done", unit_id="film-intro")
    rows = {r["id"]: r for r in film["ledger"]}
    assert rows["film-intro:render"]["status"] == "open" and "model_inference < controlled_replay" in rows["film-intro:render"]["reason"]


def test_mark_unit_done_function_matches_the_tool_and_records_a_trace(ctx: ToolContext, tools: ToolRegistry, workspace) -> None:
    propose(tools, ctx)
    workspace.write_text("book/chapters/01-vectors.md", GOOD_CHAPTER)
    events: list[Any] = []

    class Tracer:
        def record(self, kind: str, data: dict[str, Any], node_id: str | None = None, parent_id: str | None = None) -> None:
            events.append((kind, data, node_id))

    ctx.tracer = Tracer()
    out = mark_unit_done(ctx, "chapter-vectors")
    assert out["status"] == "done"
    kinds = [k for k, _, _ in events]
    assert "check" in kinds and kinds[-1] == "unit" and events[-1][1]["status"] == "done"


# ----------------------------------------------------------------------------- build
def _done_chapters(tools: ToolRegistry, ctx: ToolContext, workspace) -> None:
    propose(tools, ctx)
    workspace.write_text("book/chapters/01-vectors.md", GOOD_CHAPTER)
    workspace.write_text("book/chapters/02-matrices.md", GOOD_MATRICES.replace("## A worked example", "{{film: film-intro}}\n\n## A worked example"))
    ok(tools, ctx, "book_mark_done", unit_id="chapter-vectors")


@skip_unless("pandoc")
def test_build_html_concatenates_done_chapters_in_order(tools: ToolRegistry, ctx: ToolContext, workspace) -> None:
    _done_chapters(tools, ctx, workspace)
    out = ok(tools, ctx, "book_build")
    assert out["format"] == "html" and out["output_path"] == "book/build/book.html" and out["chapters"][0]["unit_id"] == "chapter-vectors"
    assert out["words"] > 100 and out["pages_estimate"] >= 1 and out["unresolved_references"] == [] and out["passed"]
    assert {s["unit_id"] for s in out["skipped"]} == {"chapter-matrices"} and ctx.evidence.get(out["evidence_id"]).operation == "book_build"
    html = (workspace.root / "book/build/book.html").read_text(encoding="utf-8")
    assert "katex" in html and "Linear Algebra" in html and 'id="TOC"' in html
    md = (workspace.root / "book/build/book.md").read_text(encoding="utf-8")
    assert md.startswith('---\ntitle: "Linear Algebra"') and "# Contents" in md and "1. Vectors" in md
    drafts = ok(tools, ctx, "book_build", format="html", include_drafts=True)
    assert [c["unit_id"] for c in drafts["chapters"]] == ["chapter-vectors", "chapter-matrices"]
    assert drafts["unresolved_references"] == [{"kind": "film", "id": "film-intro", "chapter": "chapter-matrices", "expected": "media/films/film-intro"}] and not drafts["passed"]


@skip_unless("pandoc")
def test_build_epub_and_pdf_without_engine(tools: ToolRegistry, ctx: ToolContext, workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    _done_chapters(tools, ctx, workspace)
    out = ok(tools, ctx, "book_build", format="epub")
    assert out["output_path"] == "book/build/book.epub" and out["bytes"] > 0
    monkeypatch.setattr(book_mod, "_pdf_engine", lambda: None)
    e = err(tools, ctx, "book_build", format="pdf")
    assert e.category.value == "unavailable" and "pdf engine" in e.message and e.partial["markdown_path"] == "book/build/book.md"
    assert err(tools, ctx, "book_build", format="docx").category.value == "invalid_input"


def test_build_without_pandoc_writes_markdown_and_reports_unavailable(tools: ToolRegistry, ctx: ToolContext, workspace) -> None:
    _done_chapters(tools, ctx, workspace)
    ctx.toolchain.binaries["pandoc"] = None
    e = err(tools, ctx, "book_build")
    assert e.category.value == "unavailable" and "pandoc" in e.remediation
    assert e.partial["format"] == "md" and e.partial["chapters"] == 1 and (workspace.root / "book/build/book.md").exists()


def test_build_without_chapters_is_invalid_input(tools: ToolRegistry, ctx: ToolContext) -> None:
    propose(tools, ctx)
    e = err(tools, ctx, "book_build")
    assert e.category.value == "invalid_input" and "include_drafts" in e.remediation


# ----------------------------------------------------------------------------- status and walkthrough
def test_status_and_walkthrough(tools: ToolRegistry, ctx: ToolContext, workspace) -> None:
    propose(tools, ctx)
    workspace.write_text("book/chapters/01-vectors.md", GOOD_CHAPTER)
    workspace.write_text("book/chapters/02-matrices.md", BAD_CHAPTER)
    ok(tools, ctx, "book_mark_done", unit_id="chapter-vectors")
    ok(tools, ctx, "book_mark_done", unit_id="chapter-matrices")
    status = ok(tools, ctx, "book_status")
    assert status["by_status"] == {"done": 1, "partial": 1, "pending": 2} and status["ready"] == ["chapter-matrices", "film-intro"]
    assert status["partial"] == [{"id": "chapter-matrices", "open": ["execute", "execute", "cite", "cite", "readability", "prereqs"]}]
    assert status["unknowns_open"] == 6 and len(status["unknowns"]) == 5 and "unit chapter-matrices: partial" in status["progress_tail"]
    walk = ok(tools, ctx, "book_walkthrough")
    text = (workspace.root / walk["path"]).read_text(encoding="utf-8")
    assert walk["path"] == "book/build/WALKTHROUGH.md" and walk["markdown"] == text and walk["chars"] == len(text)
    assert text.startswith("# Walkthrough: Linear Algebra") and "## What was built" in text and "- `chapter-vectors` (chapter): Vectors" in text
    assert "## What is partial, and why" in text and "open execute: ZeroDivisionError" in text and "## Unknowns" in text and "## How to review" in text
    assert "`film-intro` (pending)" in text and render_walkthrough(workspace) == text


# ----------------------------------------------------------------------------- gate and outline edge cases found in review
def test_mark_done_refuses_a_unit_whose_dependencies_are_unfinished(tools: ToolRegistry, ctx: ToolContext, workspace) -> None:
    propose(tools, ctx)
    workspace.write_text("book/chapters/02-matrices.md", GOOD_MATRICES)
    e = err(tools, ctx, "book_mark_done", unit_id="chapter-matrices")
    assert e.category.value == "invalid_input" and "dependencies are not finished: ['chapter-vectors']" in e.message and "book_status" in e.remediation
    assert Outline(workspace.path("outline")).get("chapter-matrices").status == "pending"
    assert UnknownsLedger(workspace.path("unknowns")).all() == []  # refused before any check ran or anything was recorded
    assert "unit chapter-matrices" not in workspace.git_log(3)


def test_outline_update_rejects_cycles_and_badly_typed_fields(tools: ToolRegistry, ctx: ToolContext, workspace) -> None:
    propose(tools, ctx)
    e = err(tools, ctx, "book_outline_update", unit_id="chapter-vectors", fields={"depends_on": ["chapter-matrices"]})
    assert e.category.value == "invalid_input" and "cycle: chapter-vectors -> chapter-matrices -> chapter-vectors" in e.message
    e = err(tools, ctx, "book_outline_update", unit_id="chapter-vectors", fields={"depends_on": ["integration"]})
    assert "cycle" in e.message  # integration depends on chapter-matrices, which depends on chapter-vectors
    for fields in ({"depends_on": 5}, {"depends_on": ["chapter-matrices", 3]}, {"title": ""}, {"title": 7}, {"checks": "check_judge"}, {"meta": []}):
        assert err(tools, ctx, "book_outline_update", unit_id="film-intro", fields=fields).category.value == "invalid_input"
    o = Outline(workspace.path("outline"))
    assert o.validate() == [] and o.get("chapter-vectors").depends_on == [] and o.get("film-intro").title == "Why vectors"
    assert ok(tools, ctx, "book_outline_update", unit_id="film-intro", fields={"depends_on": ["chapter-vectors"]})["problems"] == []


def test_gate_readability_falls_back_when_the_persona_names_no_level(tools: ToolRegistry, ctx: ToolContext, workspace) -> None:
    units = [{"id": "chapter-vectors", "kind": "chapter", "title": "Vectors", "path": "book/chapters/01-vectors.md"}]
    ok(tools, ctx, "book_outline_propose", title="Maths again", persona="a curious adult returning to maths", units=units)
    workspace.write_text("book/chapters/01-vectors.md", GOOD_CHAPTER.replace("persona: second-year undergraduate\nlevel: undergraduate\n", ""))
    out = ok(tools, ctx, "book_mark_done", unit_id="chapter-vectors")
    row = next(r for r in out["ledger"] if r["kind"] == "readability")
    assert row["status"] == "closed" and "no level known" in row["reason"] and out["status"] == "done"


def test_mark_done_reaches_the_assessment_closer_without_a_registry_in_the_context(tools: ToolRegistry, ctx: ToolContext, workspace) -> None:
    ok(tools, ctx, "book_outline_propose", units=[{"id": "set-vectors", "kind": "exercise_set", "title": "Vector drills"}])
    assert "registry" not in ctx.extra
    out = ok(tools, ctx, "book_mark_done", unit_id="set-vectors")
    (keys,) = out["open_obligations"]
    assert out["status"] == "partial" and keys["kind"] == "keys" and keys["reason"].startswith("assess_items_check: not_found")
    assert "assessments/set-vectors/key.json" in keys["reason"]
    (u,) = UnknownsLedger(workspace.path("unknowns")).open(scope="set-vectors")
    assert u.recommended_probes[0]["operation"] == "assess_items_check"
