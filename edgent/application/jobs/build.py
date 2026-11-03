"""The long-running build job: plan, run units through the done-gate, critic rounds, resume.

Every status change goes through ``Outline.set_status`` and is on disk before
the next step, so a crash or a closed laptop resumes from ``outline.json``
and git: a unit left ``in_progress`` is reset to ``partial`` with a history
note and picked up again. Workers run in a thread pool over the ready set;
the gate, the outline and git are serialised behind one lock.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ...domain.errors import Limit
from ...domain.outline import FINISHED, Outline, OutlineError, Unit
from ...domain.types import Usage
from ...domain.unknowns import Unknown, UnknownsLedger
from .briefs import initializer_brief, worker_brief

MAX_PARALLEL_LIMIT = Limit("build parallel units", 8, "units", does_not_raise=("the provider's concurrency, which bounds it in practice",))
MAX_ROUNDS_LIMIT = Limit("build rounds per unit", 5, "rounds")
LOCK_FILE = "build.lock"  # under .edgent: the pid of the process running a build; a second build over the same workspace refuses
ERROR_TEXT_CHARS = 300
DEFAULT_CRITIC_RUBRIC = "motivation before definition; example density; notation discipline; each section ends with what the reader can now do; no unexplained jargon for the persona"


@dataclass
class UnitOutcome:
    unit_id: str
    status: str  # done | partial | blocked | error
    ledger: list[dict[str, Any]] = field(default_factory=list)
    text: str = ""
    usage: Usage | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"unit_id": self.unit_id, "status": self.status, "ledger": list(self.ledger), "text": self.text, "usage": self.usage.to_dict() if self.usage else None, "error": self.error}


@dataclass
class BuildReport:
    units_run: list[str] = field(default_factory=list)
    done: list[str] = field(default_factory=list)
    partial: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    duration_s: float = 0.0
    fix_units: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    @property
    def ok(self) -> bool:
        return not self.errors and not self.blocked

    def to_dict(self) -> dict[str, Any]:
        return {
            "units_run": list(self.units_run), "done": list(self.done), "partial": list(self.partial), "blocked": list(self.blocked),
            "errors": list(self.errors), "duration_s": round(self.duration_s, 3), "fix_units": list(self.fix_units), "usage": self.usage.to_dict(),
        }


class BuildJob:
    def __init__(
        self,
        workspace: Any,
        *,
        run_unit: Callable[[Unit, str], UnitOutcome],
        mark_done: Callable[[Unit], dict[str, Any]] | None = None,
        critic: Callable[[Unit], list[dict[str, Any]]] | None = None,
        max_rounds: int = 2,
        max_parallel: int = 2,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        tracer: Any = None,
        session_id: str = "build",
    ) -> None:
        self.workspace = workspace
        self.run_unit = run_unit
        self.mark_done = mark_done or self._default_mark_done
        self.critic = critic
        self.max_rounds = max(1, min(int(max_rounds), int(MAX_ROUNDS_LIMIT.value)))
        self.max_parallel = max(1, min(int(max_parallel), int(MAX_PARALLEL_LIMIT.value)))
        self.on_event = on_event
        self.tracer = tracer
        self.session_id = session_id
        self._toolchain: Any = None
        try:  # the book tools serialise outline.json and git behind this lock; the job joins it so workers' tool calls cannot race the gate
            from ...tools.edu.book import OUTLINE_LOCK

            self._lock: Any = OUTLINE_LOCK
        except Exception:  # noqa: BLE001
            self._lock = threading.RLock()
        self.outline = Outline(workspace.path("outline"))

    # ------------------------------------------------------------------ plumbing
    def _reload(self) -> Outline:
        self.outline = Outline(self.workspace.path("outline"))
        return self.outline

    def _event(self, event: str, **data: Any) -> None:
        payload = {"event": event, **data}
        if self.tracer is not None:
            try:
                self.tracer.record("unit", payload, node_id=data.get("unit_id"))
            except Exception:  # noqa: BLE001
                pass
        if self.on_event is not None:
            try:
                self.on_event(payload)
            except Exception:  # noqa: BLE001 - a listener never breaks the build
                pass

    def _commit(self, message: str) -> str | None:
        with self._lock:
            try:
                return self.workspace.commit(message)
            except Exception:  # noqa: BLE001
                return None

    def _default_mark_done(self, unit: Unit) -> dict[str, Any]:
        from ...adapters.toolchain import Toolchain
        from ...domain.evidence import EvidenceStore
        from ...tools.edu.book import mark_unit_done
        from ...tools.registry import AllowAll, ToolContext

        if self._toolchain is None:
            self._toolchain = Toolchain.detect()
        ctx = ToolContext(
            workspace=self.workspace, session_id=self.session_id, node_id=unit.id, role="build",
            evidence=EvidenceStore(self.workspace.path("evidence")), toolchain=self._toolchain, permissions=AllowAll(), tracer=self.tracer,
        )
        return mark_unit_done(ctx, unit.id)

    def _record_unknown(self, unit: Unit, question: str, probe: str, rationale: str) -> str:
        ledger = UnknownsLedger(self.workspace.path("unknowns"))
        u = Unknown(question=question, domain="build", scope=unit.id, severity="high", recommended_probes=[{"operation": probe, "rationale": rationale[:ERROR_TEXT_CHARS]}])
        if ledger.record(u) == "duplicate":
            existing = ledger.get(u.unknown_id)
            if existing is not None and existing.status not in ("open", "investigating"):
                ledger.set_status(u.unknown_id, "open", rationale[:ERROR_TEXT_CHARS])
        return u.unknown_id

    # ------------------------------------------------------------------ plan
    def plan(self, goal: str, propose: Callable[[str], list[dict[str, Any]] | dict[str, Any]]) -> Outline:
        """Ask ``propose`` for the unit list given the initializer brief; validate, save, commit, record."""
        from ...tools.edu.book import propose_units

        index = self.workspace.read_text("sources/INDEX.md") if self.workspace.file_exists("sources/INDEX.md") else ""
        brief = initializer_brief(goal, self.workspace, index)
        proposal = propose(brief)
        title = persona = None
        if isinstance(proposal, dict):
            title, persona = proposal.get("title"), proposal.get("persona")
            proposal = proposal.get("units") or []
        if not isinstance(proposal, list):
            raise ValueError("propose must return a list of unit dicts or an object with a 'units' list")
        with self._lock:
            outline = self._reload()
            if isinstance(title, str) and title.strip():
                outline.title = title.strip()
            elif not outline.units:  # a fresh plan is named by its goal until the initializer names the book
                outline.title = goal.strip()[:120]
            if isinstance(persona, str) and persona.strip():
                outline.persona = persona.strip()
            outline.save()
            res = propose_units(outline, proposal, workspace=self.workspace)
            problems = outline.validate()
            if not outline.units:
                reasons = "; ".join(f"{r.get('id') or r.get('index')}: {r['reason']}" for r in res["rejected"][:5])
                raise ValueError(f"the proposal produced no units ({reasons or 'empty proposal'})")
            summary = f"plan: {len(res['added'])} units added, {len(res['existing'])} kept, {len(res['rejected'])} rejected for goal: {goal.strip()[:200]}"
            if res["rejected"]:
                summary += "\n" + "\n".join(f"- rejected {r.get('id') or r.get('index')}: {r['reason']}" for r in res["rejected"][:10])
            if problems:
                summary += "\n" + "\n".join(f"- problem: {p}" for p in problems)
            self.workspace.append_progress(summary, self.session_id)
            sha = self._commit(f"plan: {len(outline.units)} units")
        self._event("plan", units=len(outline.units), added=res["added"], rejected=res["rejected"], problems=problems, commit=sha)
        return outline

    # ------------------------------------------------------------------ run
    def _recover_interrupted(self) -> list[str]:
        recovered = []
        with self._lock:
            outline = self._reload()
            for u in outline.by_status("in_progress"):
                outline.set_status(u.id, "partial", reason="resumed after interruption")
                recovered.append(u.id)
        if recovered:
            self.workspace.append_progress("resumed after interruption: " + ", ".join(recovered), self.session_id)
            self._event("resumed", units=recovered)
        return recovered

    def _start(self, unit_id: str) -> tuple[Unit, str]:
        with self._lock:
            outline = self._reload()
            unit = outline.set_status(unit_id, "in_progress", reason=f"started by {self.session_id}")
            brief = worker_brief(self.workspace, outline, unit, outline.persona)
        self._event("unit_start", unit_id=unit.id, attempt=unit.attempts)
        return unit, brief

    def _block(self, unit: Unit, reason: str, report: BuildReport, error: bool) -> None:
        reason = reason.strip()[:ERROR_TEXT_CHARS] or "no reason given"
        with self._lock:
            outline = self._reload()
            current = outline.get(unit.id)
            unknown_id = self._record_unknown(unit, f"{unit.id}: worker {'failed' if error else 'blocked'}: {reason}", "book_unit_start", "retry the unit after fixing the cause; blocked units are reset with BuildJob.unblock")
            if current.status != "blocked":
                try:
                    outline.set_status(unit.id, "blocked", reason=("worker error: " if error else "worker blocked: ") + reason)
                except OutlineError:
                    outline.set_status(unit.id, "in_progress", reason="reopened to block")
                    outline.set_status(unit.id, "blocked", reason=("worker error: " if error else "worker blocked: ") + reason)
            self.workspace.append_progress(f"unit {unit.id}: blocked ({reason}); unknown {unknown_id}", self.session_id)
            self._commit(f"unit {unit.id}: blocked")
        report.blocked.append(unit.id)
        if error:
            report.errors.append({"unit_id": unit.id, "error": reason})
        self._event("unit_blocked", unit_id=unit.id, reason=reason, error=error)

    def _finish(self, unit: Unit, outcome: UnitOutcome | Exception, report: BuildReport, newly_done: list[str]) -> None:
        report.units_run.append(unit.id)
        if isinstance(outcome, Exception):
            self._block(unit, f"{type(outcome).__name__}: {outcome}", report, error=True)
            return
        if outcome.usage is not None:
            report.usage = report.usage + outcome.usage
        if outcome.status == "error" or (outcome.error and outcome.status not in ("done", "partial")):
            self._block(unit, outcome.error or outcome.text or "worker reported an error", report, error=True)
            return
        if outcome.status == "blocked":
            self._block(unit, outcome.error or outcome.text or "worker reported the unit blocked", report, error=False)
            return
        try:
            with self._lock:
                result = self.mark_done(unit)
                outline = self._reload()
                status = str(result.get("status") or outline.get(unit.id).status)
                open_rows = result.get("open_obligations") or [o for o in result.get("ledger") or [] if o.get("status") == "open" and not o.get("advisory")]
                summary = f"unit {unit.id}: {status}; {len(result.get('ledger') or [])} obligations, {len(open_rows)} open"
                self.workspace.append_progress(summary + ("\n" + "\n".join(f"- open {o.get('kind')}: {str(o.get('reason', ''))[:160]}" for o in open_rows) if open_rows else ""), self.session_id)
                sha = self._commit(f"unit {unit.id}: {status}")
        except Exception as e:  # noqa: BLE001 - a crashed gate blocks the unit with the reason, never the job
            self._block(unit, f"gate error: {type(e).__name__}: {e}", report, error=True)
            return
        if status in FINISHED:
            report.done.append(unit.id)
            newly_done.append(unit.id)
        elif status == "blocked":
            report.blocked.append(unit.id)
        else:
            report.partial.append(unit.id)
        self._event("unit_finished", unit_id=unit.id, status=status, open=[o.get("id") for o in open_rows], commit=sha, evidence_ids=result.get("evidence_ids") or [])

    def _critic_round(self, newly_done: list[str], report: BuildReport, allow_fixes: bool = True) -> list[str]:
        """Review what just became done; create a fix unit per target with high defects while rounds remain."""
        if self.critic is None:
            return []
        created: list[str] = []
        seen: set[str] = set()
        for uid in newly_done:
            with self._lock:
                outline = self._reload()
                if not outline.has(uid):
                    continue
                unit = outline.get(uid)
                target_id = str(unit.meta.get("fix_for") or uid)
                if target_id in seen or not outline.has(target_id):
                    continue
                seen.add(target_id)
                target = outline.get(target_id)
            try:
                defects = [d for d in (self.critic(target) or []) if isinstance(d, dict)]
            except Exception as e:  # noqa: BLE001 - a failing critic is a skipped review, recorded, never a crash
                self._event("critic_error", unit_id=target_id, error=f"{type(e).__name__}: {e}"[:ERROR_TEXT_CHARS])
                continue
            high = [d for d in defects if d.get("severity") == "high"]
            self._event("critic", unit_id=target_id, defects=len(defects), high=len(high))
            if not high:
                continue
            with self._lock:
                outline = self._reload()
                existing = [u for u in outline.units if u.meta.get("fix_for") == target_id]
                if not allow_fixes or len(existing) >= self.max_rounds:
                    self._record_unknown(target, f"{target_id}: critic still reports high defects after {len(existing)} fix rounds", "check_judge", "; ".join(str(d.get("message", ""))[:120] for d in high[:3]))
                    self._event("critic_exhausted", unit_id=target_id, rounds=len(existing))
                    continue
                fix = Unit(
                    id=f"fix-{target_id}-{len(existing) + 1}", kind="section", title=f"Fix: {target.title}",
                    objective=f"resolve the critic's high-severity defects in {target_id}", depends_on=[target_id], path=target.path,
                    meta={"fix_for": target_id, "defects": high, **{k: target.meta[k] for k in ("level", "persona", "introduces", "requires") if k in target.meta}},
                )
                outline.add(fix)
                self.workspace.append_progress(f"critic: {len(high)} high defects in {target_id}; fix unit {fix.id} created", self.session_id)
                self._commit(f"critic: fix unit {fix.id}")
            created.append(fix.id)
            report.fix_units.append(fix.id)
            self._event("fix_unit", unit_id=fix.id, fix_for=target_id, defects=len(high))
        return created

    def _acquire_lock(self) -> Any:
        """Create .edgent/build.lock exclusively; a live owner means another build is running, a dead one is taken over."""
        path = self.workspace.path("state") / LOCK_FILE
        for _attempt in range(2):
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            except FileExistsError:
                try:
                    owner = json.loads(path.read_text(encoding="utf-8") or "{}")
                    pid = int(owner.get("pid", 0))
                except (OSError, ValueError, TypeError):
                    pid = 0
                if pid and pid != os.getpid() and _pid_alive(pid):
                    raise RuntimeError(f"another build (pid {pid}) is running on this workspace; wait for it or remove {path} if it is dead") from None
                path.unlink(missing_ok=True)  # stale: the owner is gone
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"pid": os.getpid(), "session_id": self.session_id, "started_at": time.time()}, fh)
            return path
        raise RuntimeError(f"could not take the build lock at {path}")

    def run(self, max_units: int | None = None, stop_on_error: bool = False) -> BuildReport:
        """Run ready units (in parallel when independent) through the gate until nothing is ready or a cap is hit."""
        lock = self._acquire_lock()
        try:
            return self._run(max_units, stop_on_error)
        finally:
            try:
                lock.unlink(missing_ok=True)
            except OSError:
                pass

    def _run(self, max_units: int | None, stop_on_error: bool) -> BuildReport:
        t0 = time.monotonic()
        report = BuildReport()
        self._recover_interrupted()
        attempts: dict[str, int] = {}
        newly_done: list[str] = []
        critic_rounds = 0
        while True:
            if max_units is not None and len(report.units_run) >= max_units:
                break
            with self._lock:
                ready = [u for u in self._reload().ready() if attempts.get(u.id, 0) < self.max_rounds]
            if not ready:
                if self.critic is None or not newly_done:
                    break
                created = self._critic_round(newly_done, report, allow_fixes=critic_rounds < self.max_rounds)
                newly_done = []
                critic_rounds += 1
                if not created:
                    break
                continue
            if max_units is not None:
                ready = ready[: max_units - len(report.units_run)]
            ready = ready[: self.max_parallel]  # a unit is in_progress only while a worker holds it; the rest wait, still pending
            started: list[tuple[Unit, str]] = []
            for u in ready:
                attempts[u.id] = attempts.get(u.id, 0) + 1
                try:
                    started.append(self._start(u.id))
                except OutlineError as e:
                    self._block(u, f"cannot start: {e}", report, error=True)
            if not started:
                continue
            outcomes: list[UnitOutcome | Exception] = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(started), self.max_parallel), thread_name_prefix="build") as pool:
                futures = [pool.submit(self.run_unit, unit, brief) for unit, brief in started]
                for fut in futures:
                    try:
                        outcomes.append(fut.result())
                    except Exception as e:  # noqa: BLE001 - the worker's failure is the unit's, recorded below; an interrupt
                        outcomes.append(e)  # propagates and leaves the unit in_progress for the next run to recover
            for (unit, _brief), outcome in zip(started, outcomes, strict=True):
                self._finish(unit, outcome, report, newly_done)
            if stop_on_error and report.errors:
                break
        report.duration_s = time.monotonic() - t0
        self._event("run_finished", **report.to_dict())
        return report

    def resume(self, max_units: int | None = None, stop_on_error: bool = False) -> BuildReport:
        return self.run(max_units=max_units, stop_on_error=stop_on_error)

    def unblock(self, unit_id: str, reason: str = "retry requested") -> Unit:
        with self._lock:
            outline = self._reload()
            return outline.set_status(unit_id, "pending", reason=reason)

    # ------------------------------------------------------------------ reads
    def status(self) -> dict[str, Any]:
        outline = self._reload()
        ledger = UnknownsLedger(self.workspace.path("unknowns"))
        return {
            "title": outline.title,
            "persona": outline.persona,
            "units": len(outline.units),
            "by_status": outline.summary()["by_status"],
            "ready": [u.id for u in outline.ready()],
            "in_progress": [u.id for u in outline.by_status("in_progress")],
            "partial": [{"id": u.id, "open": [o.get("kind") for o in u.open_obligations]} for u in outline.by_status("partial")],
            "blocked": [{"id": u.id, "reason": (u.history[-1].get("reason") if u.history else "")} for u in outline.by_status("blocked")],
            "fix_units": [u.id for u in outline.units if u.meta.get("fix_for")],
            "unknowns_open": len(ledger.open()),
            "finished": all(u.status in FINISHED for u in outline.units) if outline.units else False,
        }

    def walkthrough(self) -> str:
        from ...tools.edu.book import render_walkthrough

        text = render_walkthrough(self.workspace)
        try:
            self.workspace.write_text("book/build/WALKTHROUGH.md", text)
        except Exception:  # noqa: BLE001 - the text is still returned
            pass
        return text


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
