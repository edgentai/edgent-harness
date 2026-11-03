"""check_prereqs: no concept is used before the outline introduces it.

The outline (``book/outline.json``) and the chapters' front matter say which
unit introduces which concept. Every concept a chapter ``requires`` must be
introduced by an earlier unit, earlier meaning before it in outline order or
anywhere in its ``depends_on`` chain. A requirement met only by a later unit
is a forward reference (high); a term from a later unit's ``introduces`` list
that appears in the body is reported at low severity, because a mention is
not always a use.
"""
from __future__ import annotations

import re
from typing import Any

from ...domain.obligations import concept_list, strip_code
from ...domain.outline import Outline, Unit
from ...domain.utils import parse_front_matter
from ..registry import ToolContext
from . import check_result, defect, record_evidence, resolve_text_file, trace_check, verdict
from .readability import strip_markup

BODY_MENTIONS_SHOWN = 10


def unit_concepts(ctx: ToolContext, unit: Unit, own_rel: str | None, own_front: dict[str, Any] | None) -> list[str]:
    """``introduces`` for a unit: its meta first, then its chapter file's front matter."""
    from_meta = concept_list(unit.meta.get("introduces"))
    if from_meta:
        return from_meta
    if unit.path and own_rel is not None and unit.path == own_rel and own_front is not None:
        return concept_list(own_front.get("introduces"))
    if unit.path:
        try:
            p = ctx.workspace.resolve(unit.path)
            if p.is_file():
                front, _ = parse_front_matter(p.read_text(encoding="utf-8", errors="replace"))
                return concept_list(front.get("introduces"))
        except Exception:  # noqa: BLE001 - another unit's unreadable file is that unit's problem
            return []
    return []


def find_unit(outline: Outline, unit_id: Any, front: dict[str, Any], rel: str) -> Unit | None:
    for cand in (unit_id, front.get("unit_id")):
        if isinstance(cand, str) and outline.has(cand):
            return outline.get(cand)
    return next((u for u in outline.units if u.path == rel), None)


def dependency_closure(outline: Outline, unit: Unit) -> set[str]:
    seen: set[str] = set()
    stack = list(unit.depends_on)
    while stack:
        d = stack.pop()
        if d in seen or not outline.has(d):
            continue
        seen.add(d)
        stack.extend(outline.get(d).depends_on)
    return seen


def _mentions(prose: str, concept: str) -> int | None:
    rx = re.compile(r"(?<![a-z0-9])" + re.escape(concept) + r"(?:s|es)?(?![a-z0-9])", re.I)
    for n, line in enumerate(prose.splitlines(), 1):
        if rx.search(line):
            return n
    return None


def check_prereqs(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    _abs, rel, text = resolve_text_file(ctx, args.get("path"))
    front, body = parse_front_matter(text)
    outline = Outline(ctx.workspace.path("outline"))
    unit = find_unit(outline, args.get("unit_id"), front, rel)
    requires = concept_list(front.get("requires")) or (concept_list(unit.meta.get("requires")) if unit else [])
    own = concept_list(front.get("introduces")) or (concept_list(unit.meta.get("introduces")) if unit else [])
    defects: list[dict[str, Any]] = []
    limitations: list[str] = []
    detail: dict[str, Any] = {"unit_id": unit.id if unit else None, "requires": requires, "introduces": own, "untrusted": True}  # concept names come from files
    if unit is None:
        if requires:
            defects.append(defect(f"{rel}:L1", "medium", f"the file is not an outline unit (give unit_id or a unit_id front matter key); requires {requires} cannot be verified"))
        else:
            limitations.append("not an outline unit; nothing to check against")
    else:
        units = outline.units
        idx = next(i for i, u in enumerate(units) if u.id == unit.id)
        earlier_ids = {u.id for u in units[:idx]} | dependency_closure(outline, unit)
        introduced_by: dict[str, list[str]] = {}
        for u in units:
            if u.id == unit.id:
                continue
            for c in unit_concepts(ctx, u, rel, front):
                introduced_by.setdefault(c, []).append(u.id)
        detail["earlier_units"] = sorted(earlier_ids)
        for c in requires:
            if c in own:
                continue
            by = introduced_by.get(c, [])
            if any(b in earlier_ids for b in by):
                continue
            if by:
                defects.append(defect(f"{rel}:L1", "high", f"forward reference: {c!r} is required here but introduced later by {', '.join(by)}"))
            else:
                defects.append(defect(f"{rel}:L1", "high", f"{c!r} is required but no unit introduces it"))
        prose = strip_markup(strip_code(body))
        later_terms = [(c, by) for c, by in introduced_by.items() if c not in own and c not in requires and not any(b in earlier_ids for b in by)]
        mentions: list[dict[str, Any]] = []
        for c, by in later_terms:
            line = _mentions(prose, c)
            if line is not None:
                mentions.append({"concept": c, "line": line, "introduced_by": by})
        for m in mentions[:BODY_MENTIONS_SHOWN]:
            defects.append(defect(f"{rel}:L{m['line']}", "low", f"uses {m['concept']!r}, which {', '.join(m['introduced_by'])} introduces later"))
        if len(mentions) > BODY_MENTIONS_SHOWN:
            defects.append(defect(rel, "low", f"{len(mentions) - BODY_MENTIONS_SHOWN} more terms from later units appear in the body"))
        detail["later_terms_used"] = len(mentions)
    passed = verdict(defects)
    summary = {"passed": passed, "unit_id": detail["unit_id"], "requires": requires, "forward_references": sum(1 for d in defects if d["severity"] == "high")}
    eid = record_evidence(ctx, "check_prereqs", rel, text, summary, confidence="derived", authority="controlled_replay", limitations=limitations, parameters={"unit_id": detail["unit_id"]})
    result = check_result(passed, "controlled_replay", "derived", defects, eid, detail, limitations)
    trace_check(ctx, "prereqs", rel, result)
    return result
