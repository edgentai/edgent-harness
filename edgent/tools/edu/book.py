"""Book tools: the outline, the worker brief, the done-gate, the build and the walkthrough.

``book_mark_done`` is where "done" becomes code: it derives the unit's
obligations, runs every closer it can reach, refuses to close an obligation
with a weaker verifier than it requires, turns what stays open into residual
unknowns with the probe that would close them, and only then moves the
status, commits and writes progress. The outline tools never remove a unit
and never replace one; statuses move only through ``Outline.set_status``.
"""
from __future__ import annotations

import math
import re
import shutil
import threading
from pathlib import Path
from typing import Any, get_args

from ...adapters.sandbox import run_subprocess
from ...adapters.toolchain import INSTALL_HINTS
from ...domain.errors import ErrorCategory, Limit, ToolFailure
from ...domain.evidence import Evidence
from ...domain.obligations import (
    CONTENT_UNIT_KINDS,
    TEXT_UNIT_KINDS,
    Obligation,
    blocking_obligations,
    derive_obligations,
    ledger_summary,
    placeholders,
)
from ...domain.outline import FINISHED, Outline, OutlineError, Unit, UnitKind
from ...domain.unknowns import TERMINAL, Unknown, UnknownsLedger
from ...domain.utils import ensure_dir, parse_front_matter, slugify
from ...domain.workspace import WorkspaceError
from ..checks.registry import close_with_checks, closer_registry, ledger_rows
from ..registry import B, I, S, ToolContext, ToolRegistry, ToolSpec, arr, obj, opt

BOOK_TOOL_NAMES: tuple[str, ...] = (
    "book_outline_get", "book_outline_propose", "book_outline_update", "book_unit_start", "book_unit_brief", "book_mark_done",
    "book_build", "book_status", "book_walkthrough",
)
UNIT_KINDS: tuple[str, ...] = tuple(get_args(UnitKind))
BUILD_FORMATS: tuple[str, ...] = ("html", "pdf", "epub")
EDITABLE_FIELDS: tuple[str, ...] = ("title", "objective", "depends_on", "path", "checks", "meta")
OUTLINE_PAGE_LIMIT = Limit("book_outline_get units per page", 200, "units", does_not_raise=("offset, which pages",))
PROPOSE_UNITS_LIMIT = Limit("book_outline_propose units", 500, "units")
BUILD_TIMEOUT_LIMIT = Limit("book_build pandoc timeout", 600, "s")
MARK_DONE_TIMEOUT_S = 1800.0
PANDOC_MEMORY_BYTES = 2 << 30
WORDS_PER_PAGE = 350  # a printed page of a textbook, for the page estimate
WALKTHROUGH_INLINE_CHARS = 6_000
MARK_DONE_INLINE_CHARS = 12_000
PROGRESS_TAIL_LINES = 12
STATUS_LIST_LIMIT = 20
PDF_ENGINES: tuple[str, ...] = ("pdflatex", "xelatex", "lualatex", "tectonic", "latexmk", "wkhtmltopdf", "weasyprint", "prince", "typst", "context", "pdfroff")
_HTML_PDF_ENGINES = frozenset({"wkhtmltopdf", "weasyprint", "prince"})
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,79}$")
_SEVERITY_BY_KIND = {"execute": "high", "cite": "high", "render": "high", "structure": "high", "keys": "high", "readability": "medium", "prereqs": "medium", "measure": "medium", "look": "medium", "judge": "low"}
_PLACEHOLDER_TARGETS = {"film": "media/films/{id}", "widget": "media/widgets/{id}.html", "exercises": "assessments/{id}"}

OUTLINE_LOCK = threading.RLock()  # one process-wide lock around load-mutate-save of outline.json and git commits


# ----------------------------------------------------------------------------- helpers
def _fail(code: str, message: str, category: ErrorCategory, remediation: str, **details: Any) -> ToolFailure:
    return ToolFailure(code, message, category, remediation=remediation, **details)


def _outline(ctx: ToolContext) -> Outline:
    return Outline(ctx.workspace.path("outline"))


def _unit(outline: Outline, unit_id: Any) -> Unit:
    if not isinstance(unit_id, str) or not outline.has(unit_id):
        known = ", ".join(u.id for u in outline.units[:STATUS_LIST_LIMIT]) or "none yet"
        raise _fail("not_found", f"unit {unit_id!r} is not in the outline", ErrorCategory.not_found, f"Use one of the outline's unit ids ({known}), or propose the unit with book_outline_propose.", unit_id=unit_id)
    return outline.get(unit_id)


def _commit(ws: Any, message: str) -> str | None:
    with OUTLINE_LOCK:
        try:
            return ws.commit(message)
        except Exception:  # noqa: BLE001 - a failed checkpoint never fails the work it records
            return None


def unit_summary(u: Unit) -> dict[str, Any]:
    d: dict[str, Any] = {"id": u.id, "kind": u.kind, "title": u.title, "objective": u.objective, "status": u.status, "depends_on": list(u.depends_on), "path": u.path, "attempts": u.attempts}
    if u.open_obligations:
        d["open_obligations"] = [{"id": o.get("id"), "kind": o.get("kind"), "reason": o.get("reason")} for o in u.open_obligations]
    if u.history:
        d["last_change"] = u.history[-1]
    if u.meta:
        d["meta_keys"] = sorted(u.meta)
    return d


def _check_path(ws: Any, path: Any) -> str | None:
    if path is None:
        return None
    if not isinstance(path, str) or not path.strip():
        return "path must be a non-empty workspace-relative string"
    if Path(path).is_absolute():
        return "path must be workspace-relative, not absolute"
    try:
        ws.resolve(path)
    except WorkspaceError as e:
        return str(e)
    return None


def _validate_entry(e: Any, ws: Any) -> str | None:
    if not isinstance(e, dict):
        return "each unit must be an object with kind and title"
    kind = e.get("kind")
    if kind not in UNIT_KINDS:
        return f"kind must be one of {', '.join(UNIT_KINDS)}; got {kind!r}"
    title = e.get("title")
    if not isinstance(title, str) or not title.strip():
        return "title must be a non-empty string"
    uid = e.get("id")
    if uid is not None and (not isinstance(uid, str) or not _ID_RE.match(uid)):
        return f"id must match {_ID_RE.pattern}; got {uid!r}"
    deps = e.get("depends_on")
    if deps is not None and (not isinstance(deps, list) or not all(isinstance(d, str) for d in deps)):
        return "depends_on must be a list of unit ids"
    if uid is not None and deps and uid in deps:
        return "a unit cannot depend on itself"
    if (problem := _check_path(ws, e.get("path"))) is not None:
        return problem
    if e.get("meta") is not None and not isinstance(e.get("meta"), dict):
        return "meta must be an object"
    if e.get("checks") is not None and (not isinstance(e.get("checks"), list) or not all(isinstance(c, str) for c in e["checks"])):
        return "checks must be a list of closer names"
    return None


def propose_units(outline: Outline, entries: list[Any], *, workspace: Any = None) -> dict[str, Any]:
    """Validate and add units that do not exist yet, dependencies first; never replaces. Returns added, existing, rejected."""
    added: list[str] = []
    rejected: list[dict[str, Any]] = []
    existing: list[str] = []
    pending: list[tuple[str, dict[str, Any]]] = []
    claimed: set[str] = set()
    for i, e in enumerate(entries):
        problem = _validate_entry(e, workspace) if workspace is not None else (_validate_entry(e, None) if not isinstance(e, dict) or e.get("path") is None else _validate_entry({**e, "path": None}, None))
        if problem:
            rejected.append({"index": i, "id": e.get("id") if isinstance(e, dict) else None, "reason": problem})
            continue
        uid = e.get("id")
        if uid is None:
            base = f"{e['kind']}-{slugify(e['title'])}"
            uid, n = base, 2
            while outline.has(uid) or uid in claimed:
                uid = f"{base}-{n}"
                n += 1
        if outline.has(uid):
            existing.append(uid)
            continue
        if uid in claimed:
            rejected.append({"index": i, "id": uid, "reason": "duplicate id in the proposal"})
            continue
        claimed.add(uid)
        pending.append((uid, e))
    while pending:  # reading order is kept wherever the dependencies allow it
        pick = next((i for i, (_uid, e) in enumerate(pending) if all(outline.has(str(d)) for d in e.get("depends_on") or [])), None)
        if pick is None:
            break
        uid, e = pending.pop(pick)
        deps = [str(d) for d in e.get("depends_on") or []]
        outline.add(Unit(id=uid, kind=e["kind"], title=e["title"].strip(), objective=str(e.get("objective") or "").strip(), depends_on=deps, path=e.get("path"), checks=list(e.get("checks") or []), meta=dict(e.get("meta") or {})))
        added.append(uid)
    for uid, e in pending:
        missing = [d for d in e.get("depends_on") or [] if not outline.has(d)]
        rejected.append({"id": uid, "reason": f"depends on units that do not exist or form a cycle: {missing}"})
    return {"added": added, "existing": existing, "rejected": rejected}


# ----------------------------------------------------------------------------- outline tools
def book_outline_get(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    outline = _outline(ctx)
    offset = max(0, int(args.get("offset") or 0))
    limit = max(1, min(int(args.get("limit") or 50), int(OUTLINE_PAGE_LIMIT.value)))
    units = outline.units
    page = [unit_summary(u) for u in units[offset : offset + limit]]
    return {"summary": outline.summary(), "persona": outline.persona, "units": page, "offset": offset, "limit": limit, "total": len(units), "next_offset": offset + limit if offset + limit < len(units) else None}


def book_outline_propose(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    entries = list(args.get("units") or [])
    if len(entries) > PROPOSE_UNITS_LIMIT.value:
        err = PROPOSE_UNITS_LIMIT.exceeded(len(entries), "Propose the units in batches.")
        raise ToolFailure(err.code, err.message, err.category, remediation=err.remediation, details=err.details)
    with OUTLINE_LOCK:
        outline = _outline(ctx)
        if isinstance(args.get("title"), str) and args["title"].strip():
            outline.title = args["title"].strip()
        if isinstance(args.get("persona"), str) and args["persona"].strip():
            outline.persona = args["persona"].strip()
        outline.save()
        res = propose_units(outline, entries, workspace=ctx.workspace)
        problems = outline.validate()
        sha = _commit(ctx.workspace, f"outline: {len(res['added'])} units proposed") if res["added"] else None
    if res["added"]:
        ctx.workspace.append_progress(f"outline: proposed {len(res['added'])} units ({', '.join(res['added'][:10])}{'…' if len(res['added']) > 10 else ''})", ctx.session_id)
    return {"summary": outline.summary(), "title": outline.title, "persona": outline.persona, **res, "problems": problems, "commit": sha}


def _would_cycle(outline: Outline, unit_id: str, deps: list[str]) -> list[str] | None:
    """The dependency path that leads from ``deps`` back to ``unit_id``, or None when the new edges keep the graph acyclic."""
    for start in deps:
        stack: list[tuple[str, list[str]]] = [(start, [unit_id, start])]
        seen: set[str] = set()
        while stack:
            cur, path = stack.pop()
            if cur == unit_id:
                return path
            if cur in seen or not outline.has(cur):
                continue
            seen.add(cur)
            stack.extend((d, path + [d]) for d in outline.get(cur).depends_on)
    return None


def _validate_update_fields(fields: dict[str, Any]) -> str | None:
    for key in ("title", "objective", "path"):
        if key in fields and not isinstance(fields[key], str):
            return f"{key} must be a string"
    if "title" in fields and not fields["title"].strip():
        return "title must be a non-empty string"
    for key in ("depends_on", "checks"):
        if key in fields and (not isinstance(fields[key], list) or not all(isinstance(x, str) and x for x in fields[key])):
            return f"{key} must be a list of ids"
    if "meta" in fields and not isinstance(fields["meta"], dict):
        return "meta must be an object"
    return None


def book_outline_update(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    fields = dict(args.get("fields") or {})
    bad = [k for k in fields if k not in EDITABLE_FIELDS]
    if bad:
        raise _fail("invalid_input", f"fields {bad} cannot be edited here; editable: {', '.join(EDITABLE_FIELDS)}", ErrorCategory.invalid_input, "Status moves only through book_unit_start and book_mark_done; ids and history are never edited.", fields=bad)
    if not fields:
        raise _fail("invalid_input", "fields is empty", ErrorCategory.invalid_input, f"Pass at least one of {', '.join(EDITABLE_FIELDS)}.")
    if (problem := _validate_update_fields(fields)) is not None:
        raise _fail("invalid_input", problem, ErrorCategory.invalid_input, "title, objective and path are strings; depends_on and checks are lists of ids; meta is an object.", fields=sorted(fields))
    if "path" in fields and (problem := _check_path(ctx.workspace, fields["path"])) is not None:
        raise _fail("invalid_input", problem, ErrorCategory.invalid_input, "Give a workspace-relative path such as book/chapters/02-matrices.md.")
    with OUTLINE_LOCK:
        outline = _outline(ctx)
        unit = _unit(outline, args.get("unit_id"))
        if "meta" in fields:
            fields["meta"] = {**unit.meta, **fields["meta"]}
        if "depends_on" in fields:
            if unit.id in fields["depends_on"]:
                raise _fail("invalid_input", "a unit cannot depend on itself", ErrorCategory.invalid_input, "Remove the unit's own id from depends_on.")
            cycle = _would_cycle(outline, unit.id, fields["depends_on"])
            if cycle is not None:
                raise _fail("invalid_input", "depends_on would create a cycle: " + " -> ".join(cycle), ErrorCategory.invalid_input, "A unit in a cycle can never become ready; drop one edge of the cycle (the later unit should depend on the earlier one, not both ways).", cycle=cycle)
        try:
            outline.update(unit.id, **fields)
        except OutlineError as e:
            raise _fail("invalid_input", str(e), ErrorCategory.invalid_input, "Use existing unit ids in depends_on; see book_outline_get.") from e
        problems = outline.validate()
    return {"unit": unit_summary(outline.get(unit.id)), "problems": problems}


def book_unit_start(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    with OUTLINE_LOCK:
        outline = _outline(ctx)
        unit = _unit(outline, args.get("unit_id"))
        try:
            outline.set_status(unit.id, "in_progress", reason=f"started by {ctx.session_id}")
        except OutlineError as e:
            raise _fail("invalid_input", str(e), ErrorCategory.invalid_input, "Finish the dependencies first (book_status lists what is ready), or pick a ready unit.", unit_id=unit.id) from e
    return {"unit": unit_summary(outline.get(unit.id)), "next": "call book_unit_brief for the brief, build the unit, then book_mark_done"}


def book_unit_brief(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    # The brief text lives with the build job; importing it lazily keeps the tool module free of the application layer at import time.
    from ...application.jobs.briefs import worker_brief

    outline = _outline(ctx)
    unit = _unit(outline, args.get("unit_id"))
    neighbours = max(0, min(int(args.get("neighbours", 2) or 0), 5))
    brief = worker_brief(ctx.workspace, outline, unit, outline.persona, neighbours=neighbours)
    return {"unit_id": unit.id, "brief": brief}


# ----------------------------------------------------------------------------- the done-gate
def _question(ob: Obligation) -> str:
    """The unknown's identity must survive edits to the chapter, so line numbers and output notes stay out of it."""
    if ob.kind == "execute":
        return f"{ob.id}: code block {ob.args.get('block')} runs as shown"
    return f"{ob.id}: {ob.description}"


def _withdraw_stale_unknowns(ledger: UnknownsLedger, scope: str, obligations: list[Obligation]) -> list[str]:
    current = {o.id for o in obligations}
    withdrawn = []
    for u in ledger.open(scope=scope):
        if u.domain == "verification" and u.question.split(": ", 1)[0] not in current:
            ledger.resolve(u.unknown_id, "withdrawn", "the content changed and no longer owes this obligation")
            withdrawn.append(u.unknown_id)
    return withdrawn


def _settle_unknown(ledger: UnknownsLedger, scope: str, ob: Obligation) -> str | None:
    """Mirror an obligation's state in the unknowns ledger; returns the unknown id when one is recorded or touched."""
    if ob.advisory and ob.status == "open" and ob.reason.startswith("advisory: not run"):
        return None  # an advisory check that was not asked for is not a gap in the work
    u = Unknown(question=_question(ob), domain="verification", scope=scope, severity=_SEVERITY_BY_KIND.get(ob.kind, "medium"),  # type: ignore[arg-type]
                required_authority=ob.required_authority, recommended_probes=[{"operation": ob.closer, "rationale": ob.reason[:300] or ob.description}])
    existing = ledger.get(u.unknown_id)
    if ob.status == "closed":
        if existing is None or existing.status in TERMINAL:
            return None
        if existing.status == "blocked":
            ledger.set_status(existing.unknown_id, "open", "closer became available")
        if ob.evidence_ids:
            ledger.resolve(existing.unknown_id, "verified", ob.reason or f"closed by {ob.closer}", ob.evidence_ids)
        else:
            ledger.resolve(existing.unknown_id, "withdrawn", f"closed by {ob.closer} without an evidence record")
        return existing.unknown_id
    if ob.status == "blocked":
        u.status = "blocked"
        u.severity = "low"
    if existing is None:
        ledger.record(u)
        return u.unknown_id
    if existing.status in TERMINAL:
        ledger.set_status(existing.unknown_id, "open", f"obligation open again: {ob.reason[:200]}")
    elif ob.status == "blocked" and existing.status == "open":
        ledger.set_status(existing.unknown_id, "blocked", ob.reason[:200])
    elif ob.status == "open" and existing.status == "blocked":
        ledger.set_status(existing.unknown_id, "open", f"closer ran and failed: {ob.reason[:200]}")
    return existing.unknown_id


def _move_status(outline: Outline, unit: Unit, target: str, reason: str, open_rows: list[dict[str, Any]]) -> Unit:
    current = unit.status
    if current in ("pending", "blocked"):
        outline.set_status(unit.id, "in_progress", reason="started by book_mark_done")
        current = "in_progress"
    if current in FINISHED and target == "partial":
        outline.set_status(unit.id, "in_progress", reason="reopened by book_mark_done: obligations open")
        current = "in_progress"
    final = current if (current == "reviewed" and target == "done") else target
    return outline.set_status(unit.id, final, reason=reason, open_obligations=open_rows)  # type: ignore[arg-type]


def _unfinished_dependencies(outline: Outline, unit: Unit) -> list[str]:
    return [d for d in unit.depends_on if not (outline.has(d) and outline.get(d).finished)]


def mark_unit_done(ctx: ToolContext, unit_id: str, *, judge: bool = False) -> dict[str, Any]:
    """Run the done-gate for one unit and persist the outcome; the function behind ``book_mark_done``."""
    with OUTLINE_LOCK:
        outline = _outline(ctx)
        unit = _unit(outline, unit_id)
        if unit.status in ("pending", "blocked") and (missing := _unfinished_dependencies(outline, unit)):
            raise _fail("invalid_input", f"unit {unit.id} cannot be marked: its dependencies are not finished: {missing}", ErrorCategory.invalid_input, "Finish the dependencies first (book_status lists what is ready), then call book_mark_done again; run_checks shows this unit's ledger meanwhile.", unit_id=unit.id, missing=missing)
        content: str | None = None
        if unit.path:
            problem = _check_path(ctx.workspace, unit.path)
            if problem:
                raise _fail("invalid_input", f"unit {unit.id} has an invalid path: {problem}", ErrorCategory.invalid_input, "Fix the path with book_outline_update.", unit_id=unit.id)
            p = ctx.workspace.resolve(unit.path)
            if unit.kind in CONTENT_UNIT_KINDS and p.is_file():  # a film or widget target is not text; its closers read it themselves
                content = p.read_text(encoding="utf-8", errors="replace")
        obligations = derive_obligations(unit, content, outline)
        if unit.path and content is None and unit.kind in TEXT_UNIT_KINDS:
            for ob in obligations:
                if ob.status == "open":
                    ob.reason = f"file {unit.path} does not exist"
            results: dict[str, Any] = {"file": {"error": {"code": "not_found", "message": f"no file at {unit.path}"}}}
        else:
            obligations, results = close_with_checks(ctx, obligations, registry=closer_registry(ctx), judge=judge)
        ledger = UnknownsLedger(ctx.workspace.path("unknowns"))
        unknown_ids = [uid for ob in obligations if (uid := _settle_unknown(ledger, unit.id, ob))]
        withdrawn = _withdraw_stale_unknowns(ledger, unit.id, obligations)
        open_obs = blocking_obligations(obligations)
        open_rows = [{"id": o.id, "kind": o.kind, "closer": o.closer, "reason": o.reason} for o in open_obs]
        status = "partial" if open_obs else "done"
        summary = ledger_summary(obligations)
        closed_n = summary["by_status"].get("closed", 0)
        reason = f"{closed_n}/{summary['total']} obligations closed"
        if summary["advisory_open"]:
            reason += f", {len(summary['advisory_open'])} advisory not run"
        if summary["blocked"]:
            reason += f", {len(summary['blocked'])} blocked (no executor)"
        if open_obs:
            reason += f"; open: {', '.join(o.kind for o in open_obs)}"
        outline = _outline(ctx)  # closers may have touched the file; the status move must see the latest outline
        try:
            unit = _move_status(outline, outline.get(unit.id), status, reason, open_rows)
        except OutlineError as e:
            raise _fail("invalid_input", f"unit {unit.id} verified ({reason}) but its status cannot move: {e}", ErrorCategory.invalid_input, "The ledger above stands; fix the outline state the message names (book_status shows it) and call book_mark_done again.", unit_id=unit.id, ledger=ledger_rows(obligations)) from e
        ctx.workspace.append_progress(f"unit {unit.id}: {unit.status} ({reason})" + ("\n" + "\n".join(f"- open {o.kind}: {o.reason[:160]}" for o in open_obs) if open_obs else ""), ctx.session_id)
        sha = _commit(ctx.workspace, f"unit {unit.id}: {unit.status}")
    evidence_ids = sorted({e for o in obligations for e in o.evidence_ids})
    tracer = getattr(ctx, "tracer", None)
    if tracer is not None:
        try:
            tracer.record("unit", {"unit_id": unit.id, "status": unit.status, "open": [o.id for o in open_obs], "evidence_ids": evidence_ids}, node_id=ctx.node_id or None)
        except Exception:  # noqa: BLE001
            pass
    return {
        "unit_id": unit.id,
        "status": unit.status,
        "summary": summary,
        "ledger": ledger_rows(obligations),
        "open_obligations": open_rows,
        "results": results,
        "evidence_ids": evidence_ids,
        "unknown_ids": unknown_ids,
        "unknowns_withdrawn": withdrawn,
        "commit": sha,
        "next": "the unit is done; stop" if unit.status in FINISHED else "fix what is open (the reasons say how), then call book_mark_done again",
    }


def book_mark_done(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    return mark_unit_done(ctx, str(args.get("unit_id")), judge=bool(args.get("judge")))


# ----------------------------------------------------------------------------- build
def _chapters_for_build(ctx: ToolContext, outline: Outline, include_drafts: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chapters: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for u in outline.units:
        if u.kind not in TEXT_UNIT_KINDS or u.meta.get("fix_for"):
            continue
        if not u.path or not u.path.endswith((".md", ".markdown")):
            skipped.append({"unit_id": u.id, "reason": "no Markdown path"})
            continue
        if not include_drafts and u.status not in FINISHED:
            skipped.append({"unit_id": u.id, "reason": f"status {u.status}"})
            continue
        if _check_path(ctx.workspace, u.path):
            skipped.append({"unit_id": u.id, "reason": "path escapes the workspace"})
            continue
        p = ctx.workspace.resolve(u.path)
        if not p.is_file():
            skipped.append({"unit_id": u.id, "reason": f"missing file {u.path}"})
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        front, body = parse_front_matter(text)
        chapters.append({"unit_id": u.id, "title": str(front.get("title") or u.title), "path": u.path, "status": u.status, "body": body, "words": len(body.split())})
    return chapters, skipped


def _unresolved_references(ctx: ToolContext, chapters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ch in chapters:
        for ph in placeholders(ch["body"]):
            target = _PLACEHOLDER_TARGETS[ph["kind"]].format(id=ph["id"])
            if not (ctx.workspace.root / target).exists():
                out.append({"kind": ph["kind"], "id": ph["id"], "chapter": ch["unit_id"], "expected": target})
    return out


def _yaml_quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _book_markdown(outline: Outline, chapters: list[dict[str, Any]]) -> str:
    title = outline.title or "Untitled"
    parts = ["---", f"title: {_yaml_quote(title)}"]
    if outline.persona:
        parts.append(f"subtitle: {_yaml_quote('for ' + outline.persona)}")
    parts += ["lang: en", "---", "", "# Contents", ""]
    parts += [f"{i}. {ch['title']}" for i, ch in enumerate(chapters, 1)]
    for ch in chapters:
        parts += ["", "", f'<a id="{slugify(ch["unit_id"])}"></a>', "", ch["body"].rstrip()]
    return "\n".join(parts) + "\n"


def _pdf_engine() -> str | None:
    return next((e for e in PDF_ENGINES if shutil.which(e)), None)


def book_build(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    fmt = str(args.get("format") or "html")
    include_drafts = bool(args.get("include_drafts"))
    outline = _outline(ctx)
    chapters, skipped = _chapters_for_build(ctx, outline, include_drafts)
    if not chapters:
        raise _fail("invalid_input", "no chapters to build" + (f"; skipped: {skipped[:5]}" if skipped else " (the outline has no chapter units with files)"), ErrorCategory.invalid_input, "Mark chapters done first, or pass include_drafts=true to build what exists.", skipped=skipped[:20])
    build_dir = Path(ensure_dir(ctx.workspace.path("build")))
    md_path = build_dir / "book.md"
    md_path.write_text(_book_markdown(outline, chapters), encoding="utf-8")
    md_rel = ctx.workspace.relpath(md_path)
    words = sum(ch["words"] for ch in chapters)
    unresolved = _unresolved_references(ctx, chapters)
    partial = {"format": "md", "markdown_path": md_rel, "chapters": len(chapters), "words": words, "pages_estimate": math.ceil(words / WORDS_PER_PAGE), "unresolved_references": unresolved, "skipped": skipped}
    pandoc = ctx.toolchain.path("pandoc") if ctx.toolchain is not None else shutil.which("pandoc")
    if not pandoc:
        raise ToolFailure("unavailable", f"pandoc is required for the {fmt} build and is not installed; book.md was written", ErrorCategory.unavailable, remediation=f"{INSTALL_HINTS['pandoc']} Until then the concatenated book is at {md_rel}.", partial=partial, toolchain="pandoc")
    out = build_dir / f"book.{fmt}"
    argv = [pandoc, "book.md", "--from", "markdown", "--toc", "--output", out.name]
    if fmt == "html":
        argv += ["--to", "html5", "--standalone", "--katex"]
    elif fmt == "epub":
        argv += ["--to", "epub3", "--mathml"]
    else:
        engine = _pdf_engine()
        if engine is None:
            raise ToolFailure("unavailable", "a PDF build needs a pdf engine (a TeX distribution, wkhtmltopdf, weasyprint, prince or typst) and none is installed; book.md was written", ErrorCategory.unavailable, remediation=f"{INSTALL_HINTS['latex']} Or build html, which needs nothing beyond pandoc.", partial=partial, toolchain="pdf-engine")
        argv += [f"--pdf-engine={engine}"] + (["--mathml"] if engine in _HTML_PDF_ENGINES else [])
    res = run_subprocess(argv, cwd=str(build_dir), timeout_s=BUILD_TIMEOUT_LIMIT.value, cpu_seconds=int(BUILD_TIMEOUT_LIMIT.value), memory_bytes=PANDOC_MEMORY_BYTES)
    warnings = [ln.strip() for ln in res.stderr.splitlines() if ln.startswith("[WARNING]")]
    if res.timed_out:
        raise ToolFailure("timeout", f"pandoc timed out after {BUILD_TIMEOUT_LIMIT.value:g} s; {BUILD_TIMEOUT_LIMIT.describe()}", ErrorCategory.timeout, remediation="Build fewer chapters (include_drafts=false) or split very large chapters.", partial=partial)
    if not res.ok or not out.exists():
        last = next((ln for ln in reversed(res.stderr.splitlines()) if ln.strip()), "no message")
        raise ToolFailure("execution_error", f"pandoc failed (exit {res.returncode}): {last[:300]}", ErrorCategory.execution_failure, remediation="Run check_render on the chapter named in the message, fix it, and build again.", partial=partial)
    out_rel = ctx.workspace.relpath(out)
    eid = None
    if ctx.evidence is not None:
        ev = Evidence(operation="book_build", parameters={"format": fmt, "include_drafts": include_drafts}, result={"output": out_rel, "chapters": [ch["unit_id"] for ch in chapters], "words": words, "unresolved_references": len(unresolved)}, confidence="observed", authority="controlled_replay", locations=[{"kind": "path", "ref": out_rel}])
        try:
            ctx.evidence.put(ev)
            eid = ev.evidence_id
        except Exception:  # noqa: BLE001
            eid = None
    return {
        "format": fmt,
        "output_path": out_rel,
        "markdown_path": md_rel,
        "bytes": out.stat().st_size,
        "chapters": [{k: ch[k] for k in ("unit_id", "title", "path", "status", "words")} for ch in chapters],
        "words": words,
        "pages_estimate": math.ceil(words / WORDS_PER_PAGE),
        "unresolved_references": unresolved,
        "passed": not unresolved,
        "skipped": skipped,
        "warnings": warnings[:20],
        "evidence_id": eid,
        "authority": "controlled_replay",
    }


# ----------------------------------------------------------------------------- status and walkthrough
def book_status(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    outline = _outline(ctx)
    ledger = UnknownsLedger(ctx.workspace.path("unknowns"))
    open_unknowns = ledger.open()
    blocked = [{"id": u.id, "reason": (u.history[-1].get("reason") if u.history else "")} for u in outline.by_status("blocked")]
    partial = [{"id": u.id, "open": [o.get("kind") for o in u.open_obligations]} for u in outline.by_status("partial")]
    return {
        "title": outline.title,
        "persona": outline.persona,
        "units": len(outline.units),
        "by_status": outline.summary()["by_status"],
        "ready": [u.id for u in outline.ready()][:STATUS_LIST_LIMIT],
        "in_progress": [u.id for u in outline.by_status("in_progress")][:STATUS_LIST_LIMIT],
        "partial": partial[:STATUS_LIST_LIMIT],
        "blocked": blocked[:STATUS_LIST_LIMIT],
        "unknowns_open": len(open_unknowns),
        "unknowns": [{"id": u.unknown_id, "severity": u.severity, "scope": u.scope, "question": u.question[:160]} for u in open_unknowns[:5]],
        "progress_tail": ctx.workspace.progress_tail(PROGRESS_TAIL_LINES),
    }


def render_walkthrough(workspace: Any) -> str:
    """What was built, what is partial and why, the unknowns ledger, and how to review it."""
    outline = Outline(workspace.path("outline"))
    ledger = UnknownsLedger(workspace.path("unknowns"))
    units = outline.units
    lines = [f"# Walkthrough: {outline.title or workspace.name}", ""]
    if outline.persona:
        lines += [f"Written for: {outline.persona}", ""]
    counts = outline.summary()["by_status"]
    lines += ["## Summary", "", f"{len(units)} units: " + (", ".join(f"{n} {s}" for s, n in sorted(counts.items())) or "none"), ""]
    done = [u for u in units if u.status in FINISHED]
    lines += ["## What was built", ""]
    lines += [f"- `{u.id}` ({u.kind}): {u.title}" + (f" — `{u.path}`" if u.path else "") + (" (reviewed)" if u.status == "reviewed" else "") for u in done] or ["_nothing verified yet_"]
    partial = [u for u in units if u.status == "partial"]
    lines += ["", "## What is partial, and why", ""]
    for u in partial:
        lines.append(f"- `{u.id}`: {u.title}")
        lines += [f"  - open {o.get('kind')}: {o.get('reason')} (closer: {o.get('closer')})" for o in u.open_obligations]
    if not partial:
        lines.append("_no partial units_")
    blocked = [u for u in units if u.status == "blocked"]
    lines += ["", "## Blocked", ""]
    lines += [f"- `{u.id}`: {(u.history[-1].get('reason') if u.history else 'no reason recorded')}" for u in blocked] or ["_nothing blocked_"]
    rest = [u for u in units if u.status in ("pending", "in_progress")]
    if rest:
        lines += ["", "## Not finished", "", ", ".join(f"`{u.id}` ({u.status})" for u in rest)]
    lines += ["", "## Unknowns", "", ledger.render_markdown().rstrip()]
    lines += [
        "", "## How to review", "",
        "1. Open `book/build/book.html` (run `book_build` if it is missing) and read the chapters in order.",
        "2. For each partial unit above, run the named closer on its file; the reason says what it found.",
        "3. Resolve, withdraw or mark out of scope each open unknown with a rationale; nothing is deleted.",
        "4. Anything marked unverified or 'absence is unknown' needs a source or a decision from you.",
        "", "## Build log (tail)", "", workspace.progress_tail(PROGRESS_TAIL_LINES) or "_empty_",
    ]
    return "\n".join(lines) + "\n"


def book_walkthrough(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    text = render_walkthrough(ctx.workspace)
    build_dir = Path(ensure_dir(ctx.workspace.path("build")))
    out = build_dir / "WALKTHROUGH.md"
    out.write_text(text, encoding="utf-8")
    rel = ctx.workspace.relpath(out)
    shown = text if len(text) <= WALKTHROUGH_INLINE_CHARS else text[:WALKTHROUGH_INLINE_CHARS] + f"\n…[{len(text) - WALKTHROUGH_INLINE_CHARS} more chars in {rel}]"
    return {"path": rel, "chars": len(text), "markdown": shown}


# ----------------------------------------------------------------------------- registry
_UNIT_ENTRY = obj(
    {
        "id": opt(S, "unit id (letters, digits, dot, dash, underscore); omitted means <kind>-<slug of title>"),
        "kind": opt(S, f"one of {', '.join(UNIT_KINDS)}; an entry with another kind is rejected on its own, the rest are added"),
        "title": opt(S, "short title"),
        "objective": opt(S, "what the reader can do after this unit, one sentence"),
        "depends_on": opt(arr(S), "ids of units that must be done first (earlier chapters, the figure a chapter embeds)"),
        "path": opt(S, "target file, workspace-relative, e.g. book/chapters/03-eigenvalues.md"),
        "checks": opt(arr(S), "closer names required beyond the kind's defaults"),
        "meta": opt({"type": "object"}, "brief, level, introduces, requires, estimated_words, film brief, ..."),
    },
    ["kind", "title"],
)


def build_book_registry(registry: ToolRegistry | None = None) -> ToolRegistry:
    reg = registry if registry is not None else ToolRegistry()
    specs = [
        ToolSpec(
            name="book_outline_get",
            description="Read the outline: the summary (counts by status, what is ready) and a page of units with status, dependencies, path and open obligations. Paginate with offset/limit when the book is large.",
            parameters=obj({"offset": opt(I, "index of the first unit to return (default 0)", minimum=0), "limit": opt(I, f"units per page (default 50, maximum {OUTLINE_PAGE_LIMIT.value:g})", minimum=1, maximum=int(OUTLINE_PAGE_LIMIT.value))}),
            fn=book_outline_get, permission="read", group="book",
            examples=[{"offset": 0, "limit": 50}],
        ),
        ToolSpec(
            name="book_outline_propose",
            description="Propose units for the outline: chapters, sections, figures, films, widgets, exercise sets, research and integration units, each with a kind, a title, an objective, dependencies and a target path. Units that already exist are kept as they are (never replaced); invalid entries are returned with the reason. Also sets the book title and persona.",
            parameters=obj({"title": opt(S, "book or course title"), "persona": opt(S, "who the reader is, e.g. 'second-year undergraduate, comfortable with calculus'"), "units": opt(arr(_UNIT_ENTRY), "the proposed units, in reading order")}, ["units"]),
            fn=book_outline_propose, permission="write", side_effects=True, group="book",
            when_not_to_use="changing an existing unit; use book_outline_update",
            examples=[{"title": "Linear Algebra", "persona": "undergraduate", "units": [{"kind": "chapter", "title": "Vectors", "objective": "add and scale vectors", "path": "book/chapters/01-vectors.md"}]}],
        ),
        ToolSpec(
            name="book_outline_update",
            description=f"Edit a unit's descriptive fields ({', '.join(EDITABLE_FIELDS)}); meta keys are merged. Status, id and history cannot be edited: status moves only through book_unit_start and book_mark_done.",
            parameters=obj({"unit_id": opt(S, "the unit to edit"), "fields": opt({"type": "object"}, "the fields to change")}, ["unit_id", "fields"]),
            fn=book_outline_update, permission="write", side_effects=True, group="book",
            examples=[{"unit_id": "chapter-vectors", "fields": {"objective": "add, scale and draw vectors", "meta": {"level": "undergraduate"}}}],
        ),
        ToolSpec(
            name="book_unit_start",
            description="Mark a unit in_progress before working on it. Fails with the reason when its dependencies are not done or the transition is not allowed.",
            parameters=obj({"unit_id": opt(S, "the unit to start")}, ["unit_id"]),
            fn=book_unit_start, permission="write", side_effects=True, group="book",
            examples=[{"unit_id": "chapter-vectors"}],
        ),
        ToolSpec(
            name="book_unit_brief",
            description="The worker brief for a unit: orientation (progress tail, git log), the unit and its neighbours, the obligations it must close, the chapter file contract, the persona, and the one rule (one unit, then book_mark_done, then stop).",
            parameters=obj({"unit_id": opt(S, "the unit"), "neighbours": opt(I, "how many units on each side to describe (default 2, maximum 5)", minimum=0, maximum=5)}, ["unit_id"]),
            fn=book_unit_brief, permission="read", group="book",
            examples=[{"unit_id": "chapter-vectors"}],
        ),
        ToolSpec(
            name="book_mark_done",
            description="The done-gate. Derives the unit's obligations from its kind and file (structure, render, one execution per python block, one resolution per citation, readability, prerequisites, keys for assessments, render/measure/look for media), runs every closer available here, and closes only what a verifier of sufficient authority passed. Open obligations become residual unknowns with the probe that would close them and the unit is recorded partial with the list; all closed means done. Commits and appends to progress.md. Call it when you believe the unit is finished; the result tells you exactly what is still open.",
            parameters=obj({"unit_id": opt(S, "the unit to verify and mark"), "judge": opt(B, "also run the advisory model critic (one model call); default false, the build job's critic round covers it")}, ["unit_id"]),
            fn=book_mark_done, permission="write", side_effects=False, idempotent=False, timeout_s=MARK_DONE_TIMEOUT_S, group="book", cost_class="moderate", max_inline_chars=MARK_DONE_INLINE_CHARS,
            when_not_to_use="you have not yet run the checks yourself; run_checks shows the same ledger without changing status",
            examples=[{"unit_id": "chapter-vectors"}],
        ),
        ToolSpec(
            name="book_build",
            description="Concatenate the chapters in outline order (done and reviewed units only, unless include_drafts) with a title page and table of contents into book/build/book.md and render it with pandoc: html (standalone, KaTeX, toc), epub, or pdf (needs a pdf engine; reported unavailable with the install hint when absent). Returns the output path, word and page counts, and the films, widgets and exercise sets referenced by chapters that do not exist yet.",
            parameters=obj({"format": opt(S, "html (default), pdf or epub", enum=list(BUILD_FORMATS)), "include_drafts": opt(B, "also include units that are not done (default false)")}),
            fn=book_build, permission="compute", side_effects=True, idempotent=False, timeout_s=BUILD_TIMEOUT_LIMIT.value + 30.0, group="book", cost_class="moderate",
            when_not_to_use="checking a single chapter; use check_render",
            examples=[{"format": "html"}],
        ),
        ToolSpec(
            name="book_status",
            description="Where the build stands: counts by status, ready units, units in progress, partial units with their open obligation kinds, blocked units with reasons, open unknowns, and the last progress entries.",
            parameters=obj({}),
            fn=book_status, permission="read", group="book",
        ),
        ToolSpec(
            name="book_walkthrough",
            description="Render the walkthrough for the user (what was built, what is partial and why, blocked units, the unknowns ledger, how to review) to book/build/WALKTHROUGH.md and return it.",
            parameters=obj({}),
            fn=book_walkthrough, permission="write", side_effects=False, group="book", max_inline_chars=WALKTHROUGH_INLINE_CHARS + 1_000,
        ),
    ]
    for spec in specs:
        if reg.get(spec.name) is None:
            reg.register(spec)
    return reg
