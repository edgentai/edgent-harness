"""A full BuildJob with a fake worker over three units with dependencies: parallel rounds, the real
done-gate, an interrupted run that resumes, worker errors, retries, and a critic round that creates a fix unit."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from edgent.application.jobs.briefs import CHAPTER_FILE_CONTRACT, critic_brief, initializer_brief, worker_brief
from edgent.application.jobs.build import BuildJob, BuildReport, UnitOutcome
from edgent.domain.outline import Outline, Unit
from edgent.domain.types import Usage
from edgent.domain.unknowns import UnknownsLedger
from edgent.tools.checks import cite as cite_mod
from tests.test_checks import GOOD_CHAPTER, TEXTBOOK

A, B, C = "chapter-vectors", "chapter-matrices", "chapter-spans"


def chapter(unit_id: str, title: str, introduces: str, requires: str = "") -> str:
    return (
        GOOD_CHAPTER.replace("title: Vectors", f"title: {title}")
        .replace("unit_id: chapter-vectors", f"unit_id: {unit_id}")
        .replace("introduces: [vector, vector addition]", f"introduces: [{introduces}]")
        .replace("requires: []", f"requires: [{requires}]")
    )


class FakeWorker:
    """Writes a passing chapter for each unit it is given; can fail or misbehave for chosen units."""

    def __init__(self, fail: set[str] | None = None, error_status: set[str] | None = None, bad: set[str] | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self.threads: set[str] = set()
        self.fail = set(fail or ())
        self.error_status = set(error_status or ())
        self.bad = set(bad or ())
        self._lock = threading.Lock()

    def __call__(self, unit: Unit, brief: str) -> UnitOutcome:
        with self._lock:
            self.calls.append((unit.id, brief))
            self.threads.add(threading.current_thread().name)
        if unit.id in self.fail:
            raise RuntimeError("provider exploded")
        if unit.id in self.error_status:
            return UnitOutcome(unit_id=unit.id, status="error", error="budget exhausted")
        target = unit.meta.get("fix_for") or unit.id
        spec = {A: ("Vectors", "vector", ""), B: ("Matrices", "matrix", "vector"), C: ("Spans", "span", "vector")}[target]
        text = chapter(target, *spec)
        if unit.id in self.bad:
            text = text.replace("print([a + b for a, b in zip(v, w)])", "print(1 / 0)")
        assert unit.path
        (Path(self._root) / unit.path).write_text(text, encoding="utf-8")  # type: ignore[attr-defined]
        return UnitOutcome(unit_id=unit.id, status="done", text=f"wrote {unit.path}", usage=Usage(prompt_tokens=100, completion_tokens=20))


@pytest.fixture
def book(workspace, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cite_mod, "external_resolver", lambda: None)
    workspace.write_text("sources/text/textbook.md", TEXTBOOK)
    o = Outline(workspace.path("outline"))
    o.title, o.persona = "Linear Algebra", "second-year undergraduate"
    o.add(Unit(id=A, kind="chapter", title="Vectors", objective="add vectors", path="book/chapters/01-vectors.md", meta={"level": "undergraduate"}))
    o.add(Unit(id=B, kind="chapter", title="Matrices", objective="multiply", depends_on=[A], path="book/chapters/02-matrices.md", meta={"level": "undergraduate"}))
    o.add(Unit(id=C, kind="chapter", title="Spans", objective="span", depends_on=[A], path="book/chapters/03-spans.md", meta={"level": "undergraduate"}))
    return workspace


def worker(workspace, **kw: Any) -> FakeWorker:
    w = FakeWorker(**kw)
    w._root = workspace.root  # type: ignore[attr-defined]
    return w


def statuses(workspace) -> dict[str, str]:
    return {u.id: u.status for u in Outline(workspace.path("outline")).units}


# ----------------------------------------------------------------------------- the happy path
def test_run_builds_three_units_in_dependency_order_with_the_real_gate(book) -> None:
    w = worker(book)
    events: list[dict[str, Any]] = []
    job = BuildJob(book, run_unit=w, max_parallel=2, on_event=events.append)
    report = job.run()
    assert isinstance(report, BuildReport) and report.ok
    assert report.units_run == [A, B, C] and report.done == [A, B, C] and report.partial == [] and report.blocked == []
    assert report.usage.prompt_tokens == 300 and report.duration_s > 0 and report.to_dict()["done"] == [A, B, C]
    assert statuses(book) == {A: "done", B: "done", C: "done"}
    assert [c[0] for c in w.calls] == [A, B, C] and len(w.threads) >= 1
    brief_b = w.calls[1][1]
    assert brief_b.startswith(f"# Build unit `{B}`: Matrices") and "`chapter-vectors` (chapter, done, before)" in brief_b and "book_mark_done" in brief_b
    log = book.git_log(10)
    assert f"unit {A}: done" in log and f"unit {C}: done" in log
    assert f"unit {B}: done" in book.progress_tail(30)
    kinds = [e["event"] for e in events]
    assert kinds[:2] == ["unit_start", "unit_finished"] and kinds[-1] == "run_finished" and kinds.count("unit_finished") == 3
    assert job.status()["finished"] and job.status()["by_status"] == {"done": 3}
    assert job.run().units_run == []  # nothing left to do


def test_interrupted_run_resumes_the_unit_left_in_progress(book) -> None:
    o = Outline(book.path("outline"))
    o.set_status(A, "in_progress", reason="first run")  # the first run died here
    w = worker(book)
    job = BuildJob(book, run_unit=w, max_parallel=1)
    report = job.resume()
    unit = Outline(book.path("outline")).get(A)
    assert [h["reason"] for h in unit.history][:2] == ["first run", "resumed after interruption"] and unit.history[1]["to"] == "partial"
    assert unit.status == "done" and unit.attempts == 2 and report.done == [A, B, C]
    assert "resumed after interruption: chapter-vectors" in book.read_text("progress.md")


def test_max_units_caps_a_run_and_a_later_run_continues(book) -> None:
    w = worker(book)
    job = BuildJob(book, run_unit=w, max_parallel=4)
    first = job.run(max_units=1)
    assert first.units_run == [A] and statuses(book) == {A: "done", B: "pending", C: "pending"}
    second = job.run(max_units=1)
    assert second.units_run == [B]
    assert job.run().units_run == [C] and job.status()["finished"]


# ----------------------------------------------------------------------------- failures and retries
def test_worker_exception_blocks_the_unit_records_an_unknown_and_unblock_retries(book) -> None:
    w = worker(book, fail={B})
    job = BuildJob(book, run_unit=w, max_parallel=2)
    report = job.run()
    assert report.done == [A, C] and report.blocked == [B] and report.errors == [{"unit_id": B, "error": "RuntimeError: provider exploded"}] and not report.ok
    assert statuses(book)[B] == "blocked" and Outline(book.path("outline")).get(B).history[-1]["reason"] == "worker error: RuntimeError: provider exploded"
    ledger = UnknownsLedger(book.path("unknowns"))
    (u,) = ledger.open(scope=B)
    assert u.domain == "build" and u.severity == "high" and u.recommended_probes[0]["operation"] == "book_unit_start"
    assert job.status()["blocked"] == [{"id": B, "reason": "worker error: RuntimeError: provider exploded"}]
    assert "## Blocked" in job.walkthrough() and (book.root / "book/build/WALKTHROUGH.md").exists()
    w.fail.clear()
    assert job.unblock(B).status == "pending"
    assert job.run().done == [B] and statuses(book) == {A: "done", B: "done", C: "done"}


def test_worker_error_status_and_stop_on_error(book) -> None:
    w = worker(book, error_status={A})
    job = BuildJob(book, run_unit=w)
    report = job.run(stop_on_error=True)
    assert report.blocked == [A] and report.errors[0]["error"] == "budget exhausted" and report.done == []
    assert statuses(book) == {A: "blocked", B: "pending", C: "pending"} and job.status()["ready"] == []


def test_partial_unit_is_retried_at_most_max_rounds_times_per_run(book) -> None:
    w = worker(book, bad={A})
    job = BuildJob(book, run_unit=w, max_rounds=2, max_parallel=1)
    report = job.run()
    assert report.units_run == [A, A] and report.partial == [A, A] and report.done == []
    unit = Outline(book.path("outline")).get(A)
    assert unit.status == "partial" and unit.attempts == 2 and unit.open_obligations[0]["kind"] == "execute"
    second_brief = w.calls[1][1]
    assert "open after the last attempt" in second_brief and "ZeroDivisionError" in second_brief
    assert statuses(book)[B] == "pending"  # dependants never start on a partial unit
    w.bad.clear()
    assert job.run().done == [A, B, C]


def test_injected_mark_done_and_gate_crash(book) -> None:
    seen: list[str] = []

    def gate(unit: Unit) -> dict[str, Any]:
        seen.append(unit.id)
        if unit.id == C:
            raise RuntimeError("gate broke")
        o = Outline(book.path("outline"))
        o.set_status(unit.id, "done", reason="fake gate")
        return {"status": "done", "ledger": [], "evidence_ids": []}

    job = BuildJob(book, run_unit=worker(book), mark_done=gate, max_parallel=2)
    report = job.run()
    assert seen == [A, B, C] and report.done == [A, B] and report.blocked == [C]
    assert report.errors[0]["error"].startswith("gate error: RuntimeError: gate broke")


# ----------------------------------------------------------------------------- critic round
def test_critic_round_creates_a_fix_unit_that_runs(book) -> None:
    reviews: list[str] = []

    def critic(unit: Unit) -> list[dict[str, Any]]:
        reviews.append(unit.id)
        if unit.id == A and reviews.count(A) == 1:
            return [{"location": "Definition", "severity": "high", "message": "component is never defined"}, {"location": "x", "severity": "low", "message": "polish"}]
        return []

    w = worker(book)
    events: list[dict[str, Any]] = []
    job = BuildJob(book, run_unit=w, critic=critic, max_rounds=2, max_parallel=2, on_event=events.append)
    report = job.run()
    fix_id = f"fix-{A}-1"
    assert report.fix_units == [fix_id] and report.units_run == [A, B, C, fix_id] and report.done == [A, B, C, fix_id]
    fix = Outline(book.path("outline")).get(fix_id)
    assert fix.kind == "section" and fix.depends_on == [A] and fix.path == "book/chapters/01-vectors.md" and fix.status == "done"
    assert fix.meta["fix_for"] == A and fix.meta["defects"][0]["message"] == "component is never defined" and fix.meta["level"] == "undergraduate"
    assert reviews == [A, B, C, A]  # the fix unit is reviewed as its target
    fix_brief = w.calls[3][1]
    assert "critic defects to fix" in fix_brief and "component is never defined" in fix_brief
    assert [e for e in events if e["event"] == "fix_unit"] == [{"event": "fix_unit", "unit_id": fix_id, "fix_for": A, "defects": 1}]
    assert f"fix unit {fix_id} created" in book.read_text("progress.md") and "critic: fix unit" in book.git_log(10)


def test_critic_rounds_are_bounded_and_record_an_unknown(book) -> None:
    def critic(unit: Unit) -> list[dict[str, Any]]:
        return [{"location": "all", "severity": "high", "message": "still wrong"}]

    job = BuildJob(book, run_unit=worker(book), critic=critic, max_rounds=2, max_parallel=3)
    report = job.run()
    fixes = sorted(u.id for u in Outline(book.path("outline")).units if u.meta.get("fix_for"))
    assert fixes == sorted([f"fix-{A}-1", f"fix-{A}-2", f"fix-{B}-1", f"fix-{B}-2", f"fix-{C}-1", f"fix-{C}-2"]) and set(report.fix_units) == set(fixes)
    ledger = UnknownsLedger(book.path("unknowns"))
    assert [u.question for u in ledger.open(scope=A)] == [f"{A}: critic still reports high defects after 2 fix rounds"]
    assert job.run().units_run == []  # fully bounded: a new run creates nothing more


def test_critic_errors_are_skipped(book) -> None:
    def critic(unit: Unit) -> list[dict[str, Any]]:
        raise ValueError("no provider")

    events: list[dict[str, Any]] = []
    job = BuildJob(book, run_unit=worker(book), critic=critic, on_event=events.append)
    assert job.run().done == [A, B, C] and sum(1 for e in events if e["event"] == "critic_error") == 3


# ----------------------------------------------------------------------------- plan and briefs
def test_plan_calls_propose_with_the_initializer_brief_and_records(workspace) -> None:
    workspace.write_text("sources/INDEX.md", "# Sources\n\n- textbook | pdf | Linear Algebra Done Right | coverage complete | sources/text/textbook.md\n")
    briefs: list[str] = []

    def propose(brief: str) -> list[dict[str, Any]]:
        briefs.append(brief)
        return [
            {"id": "chapter-vectors", "kind": "chapter", "title": "Vectors", "objective": "add", "path": "book/chapters/01-vectors.md"},
            {"id": "chapter-matrices", "kind": "chapter", "title": "Matrices", "depends_on": ["chapter-vectors"], "path": "book/chapters/02-matrices.md"},
            {"kind": "sonnet", "title": "bad"},
        ]

    job = BuildJob(workspace, run_unit=worker(workspace))
    outline = job.plan("a short book on linear algebra", propose)
    assert [u.id for u in outline.units] == ["chapter-vectors", "chapter-matrices"] and outline.title == "a short book on linear algebra"
    assert briefs[0].startswith("# Plan the outline") and "Linear Algebra Done Right" in briefs[0] and "book_outline_propose" in briefs[0]
    assert "plan: 2 units added, 0 kept, 1 rejected" in workspace.read_text("progress.md") and "rejected 2: kind must be one of" in workspace.read_text("progress.md")
    assert "plan: 2 units" in workspace.git_log(3)
    dict_outline = job.plan("again", lambda _b: {"title": "Linear Algebra", "persona": "undergraduate", "units": [{"id": "chapter-vectors", "kind": "chapter", "title": "Dup"}]})
    assert dict_outline.title == "Linear Algebra" and dict_outline.persona == "undergraduate" and len(dict_outline.units) == 2
    with pytest.raises(ValueError):
        BuildJob(workspace, run_unit=worker(workspace)).plan("x", lambda _b: "nonsense")  # type: ignore[arg-type,return-value]


def test_plan_with_an_empty_proposal_and_no_units_fails(workspace) -> None:
    job = BuildJob(workspace, run_unit=worker(workspace))
    with pytest.raises(ValueError, match="no units"):
        job.plan("x", lambda _b: [])


def test_briefs(book) -> None:
    o = Outline(book.path("outline"))
    unit = o.get(B)
    brief = worker_brief(book, o, unit, o.persona, neighbours=1)
    assert "## Your unit" in brief and f"- id: `{B}`" in brief and "- depends on: `chapter-vectors` (Vectors)" in brief
    assert "`chapter-vectors` (chapter, pending, before): Vectors — add vectors" in brief and "`chapter-spans` (chapter, pending, after): Spans — span" in brief
    assert CHAPTER_FILE_CONTRACT in brief and "## The rule" in brief and "## Persona" in brief
    init = initializer_brief("teach me eigenvalues", book, "")
    assert "Goal: teach me eigenvalues" in init and "no sources yet" in init and "Units that already exist" in init and "`chapter-matrices`" in init
    crit = critic_brief(unit, "book/build/02-matrices.html", "be strict")
    assert crit.startswith("[edgent-role: critic]") and "be strict" in crit and "book/build/02-matrices.html" in crit and '"score"' in crit


def test_unit_outcome_round_trip() -> None:
    o = UnitOutcome(unit_id="u", status="done", ledger=[{"id": "x"}], text="t", usage=Usage(1, 2), error=None)
    assert o.to_dict() == {"unit_id": "u", "status": "done", "ledger": [{"id": "x"}], "text": "t", "usage": Usage(1, 2).to_dict(), "error": None}
    assert UnitOutcome(unit_id="u", status="error").to_dict()["usage"] is None


# ----------------------------------------------------------------------------- scheduling and interruption found in review
def test_only_max_parallel_units_are_in_progress_at_once(book) -> None:
    class Watching(FakeWorker):
        seen: list[dict[str, str]] = []

        def __call__(self, unit: Unit, brief: str) -> UnitOutcome:
            self.seen.append(statuses(book))
            return super().__call__(unit, brief)

    w = Watching()
    w._root = book.root  # type: ignore[attr-defined]
    report = BuildJob(book, run_unit=w, max_parallel=1).run()
    assert report.done == [A, B, C]
    assert w.seen[1][B] == "in_progress" and w.seen[1][C] == "pending"  # C waits, still pending, while B is worked on
    assert w.seen[2][C] == "in_progress" and w.seen[2][B] == "done"


def test_an_interrupt_propagates_and_the_unit_resumes_next_run(book) -> None:
    class Abort(BaseException):
        pass

    class Interrupted(FakeWorker):
        def __call__(self, unit: Unit, brief: str) -> UnitOutcome:
            if unit.id == A and not self.calls:
                self.calls.append((unit.id, brief))
                raise Abort()
            return super().__call__(unit, brief)

    w = Interrupted()
    w._root = book.root  # type: ignore[attr-defined]
    job = BuildJob(book, run_unit=w, max_parallel=1)
    with pytest.raises(Abort):
        job.run()
    assert statuses(book)[A] == "in_progress" and UnknownsLedger(book.path("unknowns")).open(scope=A) == []  # not a worker failure
    report = job.resume()
    assert report.done == [A, B, C] and "resumed after interruption" in [h["reason"] for h in Outline(book.path("outline")).get(A).history]
