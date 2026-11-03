"""The checks registry and the engine that closes obligations with them.

``build_checks_registry`` registers the six checks and ``run_checks``.
``close_with_checks`` is the done-gate's engine: it groups a unit's open
obligations by closer, runs each closer once per target, maps per-block and
per-citation results onto their obligations, and closes only what a verifier
of sufficient authority passed. Closers owned by other modules (``widget_run``,
``assess_items_check``, ``film_render``, ``book_build``) are reached through
the registry handed in by the caller when their arguments can be derived.
"""
from __future__ import annotations

import importlib
import json
import threading
from collections.abc import Callable
from typing import Any

from ...domain.errors import ErrorCategory, ToolFailure
from ...domain.obligations import (
    KINDS,
    Obligation,
    blocking_obligations,
    close_obligation,
    derive_obligations,
    ledger_summary,
)
from ...domain.outline import Outline, Unit
from ...domain.utils import parse_front_matter
from ..registry import B, N, S, ToolContext, ToolRegistry, ToolSpec, arr, obj, opt, validate_args
from . import resolve_text_file
from .cite import check_cite
from .execute import TOOL_TIMEOUT_S as EXECUTE_TOOL_TIMEOUT_S
from .execute import check_execute
from .judge import TOOL_TIMEOUT_S as JUDGE_TOOL_TIMEOUT_S
from .judge import check_judge
from .prereqs import check_prereqs, find_unit
from .readability import GRADE_BANDS, LEVELS, MIN_WORDS_FOR_GRADE, check_readability, level_from_text
from .render import RENDER_TIMEOUT_LIMIT, check_render

CHECK_FUNCTIONS: dict[str, Callable[[ToolContext, dict[str, Any]], dict[str, Any]]] = {
    "check_render": check_render,
    "check_execute": check_execute,
    "check_cite": check_cite,
    "check_readability": check_readability,
    "check_prereqs": check_prereqs,
    "check_judge": check_judge,
}
CHECK_TOOL_NAMES: tuple[str, ...] = (*CHECK_FUNCTIONS, "run_checks")
REGISTRY_TIMEOUT_MARGIN_S = 30.0
DEFECTS_IN_REASON = 3
ADVISORY_NOT_RUN = "advisory: not run (pass judge=true, or rely on the critic round)"
_EXTERNAL_ID_KEYS = ("film_id", "widget_id", "figure_id", "diagram_id", "set_id", "unit_id", "id")
_FAILING_EXTERNAL_SEVERITIES = frozenset({"medium", "high", "error"})
# Closers the gate may build itself when the caller hands in no registry: verification tools that are safe to re-run.
# The media closers (film_render, widget_run, diagram_render) create their targets from content, so they are reached
# only through the caller's registry.
_FALLBACK_BUILDERS: tuple[tuple[str, str], ...] = (("edgent.tools.edu.assess", "build_assess_registry"), ("edgent.tools.edu.book", "build_book_registry"))
_fallback_registry: ToolRegistry | None = None
_fallback_lock = threading.Lock()

_PATH = opt(S, "workspace-relative path of the Markdown chapter, e.g. book/chapters/01-vectors.md")


# ----------------------------------------------------------------------------- the engine
def _leave_open(obs: list[Obligation], reason: str) -> None:
    for ob in obs:
        ob.status = "open"
        ob.reason = reason


def _defect_text(d: Any) -> str:
    if not isinstance(d, dict):
        return str(d)
    return str(d.get("message") or d.get("detail") or d.get("kind") or "")


def _first_defects(res: dict[str, Any]) -> str:
    defects = [d for d in res.get("defects") or [] if isinstance(d, dict)]
    msgs = [_defect_text(d) for d in defects if d.get("severity") in _FAILING_EXTERNAL_SEVERITIES]
    if not msgs:
        msgs = [_defect_text(d) for d in defects]
    text = "; ".join(m[:160] for m in msgs[:DEFECTS_IN_REASON])
    if len(msgs) > DEFECTS_IN_REASON:
        text += f" (+{len(msgs) - DEFECTS_IN_REASON} more)"
    return text


def _local_args(closer: str, obs: list[Obligation]) -> dict[str, Any]:
    first = obs[0]
    args: dict[str, Any] = {"path": first.target}
    level = first.args.get("level")
    if closer == "check_readability" and isinstance(level, str) and level_from_text(level):
        args["level"] = level  # a persona phrase without a level keyword is left to the check's own fallbacks and limitation
    if closer in ("check_prereqs", "check_judge") and first.args.get("unit_id"):
        args["unit_id"] = first.args["unit_id"]
    if closer == "check_execute" and first.args.get("timeout_s"):
        args["timeout_s"] = first.args["timeout_s"]
    return args


def _apply_local(closer: str, obs: list[Obligation], res: dict[str, Any]) -> None:
    authority = str(res.get("authority") or "controlled_replay")
    ids = [res["evidence_id"]] if res.get("evidence_id") else []
    limitation = "; ".join(res.get("limitations") or [])
    if closer == "check_execute":
        blocks = {e.get("block"): e for e in (res.get("detail") or {}).get("blocks") or []}
        for ob in obs:
            e = blocks.get(ob.args.get("block"))
            if e is None:
                close_obligation(ob, authority, ids, False, "block was not run")
                continue
            passed = bool(e.get("ok")) and e.get("output_matches", True) is not False
            detail = e.get("error") or ("printed something other than the shown output" if e.get("output_matches") is False else f"ran in {e.get('duration_s', 0)} s")
            close_obligation(ob, authority, ids, passed, detail)
        return
    if closer == "check_cite":
        entries = {e.get("locator"): e for e in (res.get("detail") or {}).get("citations") or []}
        for ob in obs:
            e = entries.get(ob.args.get("locator"))
            if e is None:
                close_obligation(ob, authority, ids, False, "citation was not checked")
                continue
            links = ids + ([e["evidence_id"]] if e.get("evidence_id") else [])
            detail = f"resolved via {e.get('via', 'local')}" if e.get("resolved") else (e.get("reason") or "did not resolve")
            close_obligation(ob, authority, links, bool(e.get("resolved")), detail)
        return
    passed = bool(res.get("passed"))
    detail = "passed" + (f" ({limitation})" if limitation else "") if passed else _first_defects(res) or "reported defects"
    for ob in obs:
        close_obligation(ob, authority, ids, passed, detail)


def _external_args(spec: ToolSpec, ob: Obligation) -> dict[str, Any] | None:
    props = spec.parameters.get("properties", {}) if isinstance(spec.parameters, dict) else {}
    args: dict[str, Any] = {}
    if "path" in props:
        args["path"] = ob.target
    for key in _EXTERNAL_ID_KEYS:
        if key in props:
            args[key] = ob.args.get(key) or ob.args.get("unit_id")
    args = {k: v for k, v in args.items() if v is not None}
    return args if not validate_args(spec.parameters, args) else None


def _full_output(ctx: ToolContext, out: dict[str, Any]) -> dict[str, Any] | None:
    """The whole output of a closer whose result the registry compacted into a preview and a persisted file."""
    pointer = out.get("full_output_path")
    if not isinstance(pointer, str) or not pointer:
        return None
    try:
        data = json.loads(ctx.workspace.resolve(pointer).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - an unreadable file means the verdict is unknown, which leaves the obligation open
        return None
    return data if isinstance(data, dict) else None


def _apply_external(obs: list[Obligation], out: dict[str, Any]) -> None:
    authority = str(out.get("authority") or "controlled_replay")
    ids = [x for x in [out.get("evidence_id"), *(out.get("evidence_ids") or [])] if isinstance(x, str) and x]
    verdict = out.get("passed", out.get("ok", True))  # the media and assessment tools say ok; the checks say passed
    for ob in obs:
        passed = verdict is True
        detail = _first_defects(out) if not passed else "passed"
        if ob.kind == "structure" and isinstance(out.get("unresolved_references"), list):
            unresolved = out["unresolved_references"]
            passed = passed and not unresolved
            if unresolved:
                detail = "unresolved references: " + ", ".join(str(r.get("id") if isinstance(r, dict) else r) for r in unresolved[:5])
        close_obligation(ob, authority, ids, passed, detail)


def close_with_checks(
    ctx: ToolContext,
    obligations: list[Obligation],
    *,
    only: list[str] | None = None,
    registry: ToolRegistry | None = None,
    judge: bool = False,
) -> tuple[list[Obligation], dict[str, Any]]:
    """Run the closers for the open obligations and close what passed; returns (ledger, per-closer results)."""
    results: dict[str, Any] = {}
    groups: dict[tuple[str, str], list[Obligation]] = {}
    run_judge = judge or bool(only and "judge" in only)
    for ob in obligations:
        if ob.status != "open" or (only is not None and ob.kind not in only):
            continue
        if ob.kind == "judge" and not run_judge:
            ob.reason = ADVISORY_NOT_RUN
            continue
        groups.setdefault((ob.closer, ob.target), []).append(ob)
    for (closer, target), obs in groups.items():
        key = f"{closer} {target}"
        if closer in CHECK_FUNCTIONS:
            try:
                res = CHECK_FUNCTIONS[closer](ctx, _local_args(closer, obs))
            except ToolFailure as e:
                _leave_open(obs, f"{closer}: {e.error.code}: {e.error.message}")
                results[key] = {"error": e.error.to_dict()}
                continue
            except Exception as e:  # noqa: BLE001 - a crashed closer leaves its obligations open, never the gate broken
                _leave_open(obs, f"{closer} crashed: {type(e).__name__}: {str(e)[:200]}")
                results[key] = {"error": {"code": "execution_error", "message": f"{type(e).__name__}: {str(e)[:200]}"}}
                continue
            _apply_local(closer, obs, res)
            results[key] = compact_result(res)
            continue
        spec = registry.get(closer) if registry is not None else None
        if spec is None:
            _leave_open(obs, f"closer {closer} is not available in this registry")
            results[key] = {"skipped": "closer unavailable"}
            continue
        tool_args = _external_args(spec, obs[0])
        if tool_args is None:
            _leave_open(obs, f"closer {closer} needs arguments this gate cannot derive; run it yourself, then mark the unit again")
            results[key] = {"skipped": "arguments could not be derived"}
            continue
        r = registry.call(closer, tool_args, ctx)
        if not r.ok:
            err = r.error.to_dict() if r.error else {"code": "error"}
            _leave_open(obs, f"{closer}: {err.get('code')}: {err.get('message', '')}")
            results[key] = {"error": err}
            continue
        out = r.output if isinstance(r.output, dict) else {}
        if r.compacted or out.get("compacted") is True:
            full = _full_output(ctx, out)
            if full is None:
                _leave_open(obs, f"{closer}: its output was compacted and the full result could not be read; run it yourself and mark the unit again")
                results[key] = {"error": {"code": "truncated", "message": "closer output compacted"}}
                continue
            out = full
        _apply_external(obs, out)
        results[key] = compact_result(out)
    return obligations, results


def closer_registry(ctx: ToolContext) -> ToolRegistry | None:
    """The registry the gate reaches external closers through: the caller's, else one built from the re-runnable verifiers."""
    extra = getattr(ctx, "extra", None)
    given = extra.get("registry") if isinstance(extra, dict) else None
    if given is not None:
        return given
    global _fallback_registry
    with _fallback_lock:
        if _fallback_registry is None:
            reg = ToolRegistry()
            for module_name, builder in _FALLBACK_BUILDERS:
                try:
                    getattr(importlib.import_module(module_name), builder)(reg)
                except Exception:  # noqa: BLE001 - a sibling module that is absent or broken only removes its closers
                    continue
            _fallback_registry = reg
    return _fallback_registry


def compact_result(res: dict[str, Any]) -> dict[str, Any]:
    defects = res.get("defects") or []
    out: dict[str, Any] = {"passed": res.get("passed"), "defects": len(defects), "evidence_id": res.get("evidence_id")}
    if defects:
        out["first_defects"] = [{"location": d.get("location"), "severity": d.get("severity"), "message": str(d.get("message", ""))[:200]} for d in defects[:DEFECTS_IN_REASON]]
    if res.get("limitations"):
        out["limitations"] = list(res["limitations"])
    return out


def ledger_rows(obligations: list[Obligation]) -> list[dict[str, Any]]:
    rows = []
    for o in obligations:
        row = {"id": o.id, "kind": o.kind, "status": o.status, "closer": o.closer, "required_authority": o.required_authority, "reason": o.reason, "evidence_ids": list(o.evidence_ids)}
        if o.advisory:
            row["advisory"] = True
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------- run_checks
def run_checks(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    abs_path, rel, text = resolve_text_file(ctx, args.get("path"))
    outline = Outline(ctx.workspace.path("outline"))
    front, _ = parse_front_matter(text)
    unit = find_unit(outline, args.get("unit_id"), front, rel)
    if args.get("unit_id") and unit is None:
        raise ToolFailure("not_found", f"unit {args['unit_id']!r} is not in the outline", ErrorCategory.not_found, remediation="Use book_outline_get to list unit ids, or omit unit_id to check the file as a generic chapter.")
    in_outline = unit is not None
    if unit is None:
        meta = {k: front[k] for k in ("level", "persona", "introduces", "requires") if k in front}
        unit = Unit(id=str(front.get("unit_id") or abs_path.stem), kind="chapter", title=str(front.get("title") or abs_path.stem), path=rel, meta=meta)
    only = list(args.get("only") or []) or None
    obligations = derive_obligations(unit, text, outline)
    obligations, results = close_with_checks(ctx, obligations, only=only, registry=closer_registry(ctx), judge=bool(args.get("judge")))
    return {
        "unit_id": unit.id,
        "path": rel,
        "in_outline": in_outline,
        "passed": not blocking_obligations([o for o in obligations if only is None or o.kind in only]),
        "summary": ledger_summary(obligations),
        "ledger": ledger_rows(obligations),
        "results": results,
    }


# ----------------------------------------------------------------------------- registry
def build_checks_registry(registry: ToolRegistry | None = None) -> ToolRegistry:
    reg = registry if registry is not None else ToolRegistry()
    specs = [
        ToolSpec(
            name="check_render",
            description="Render a Markdown chapter and report structural defects: front matter, headings, unclosed fences, unbalanced $ and $$, broken relative links, placeholder targets. With pandoc installed the chapter is rendered to book/build/<stem>.html with KaTeX; without it the result is structure-only and says so. Use it before marking a unit done and after any edit that touched headings, math or links.",
            parameters=obj({"path": _PATH}, ["path"]),
            fn=check_render, permission="compute", timeout_s=RENDER_TIMEOUT_LIMIT.value + REGISTRY_TIMEOUT_MARGIN_S, group="checks",
            when_not_to_use="the file is not Markdown (films, widgets and figures have their own render tools)",
            examples=[{"path": "book/chapters/01-vectors.md"}],
        ),
        ToolSpec(
            name="check_execute",
            description="Run every python code block of a chapter in the sandbox, in order, each as its own process in one shared scratch directory, and compare stdout with the ```output block that follows a code block when there is one (whitespace normalised). A block starting with '# edgent: continue' is concatenated with the code that ran before it. Reports one defect per failing block with the last line of the traceback.",
            parameters=obj({"path": _PATH, "timeout_s": opt(N, "wall-clock seconds allowed per code block (default 60, maximum 300)", minimum=1, maximum=300)}, ["path"]),
            fn=check_execute, permission="compute", timeout_s=EXECUTE_TOOL_TIMEOUT_S, group="checks", cost_class="moderate",
            when_not_to_use="the chapter has no python blocks, or a block needs the network or more than the sandbox's memory; run it with bash instead and record the result",
            examples=[{"path": "book/chapters/01-vectors.md"}],
        ),
        ToolSpec(
            name="check_cite",
            description="Resolve every citation in a chapter ([source-id:p.12], [source-id:t=00:14:32], [source-id:L12]) against the text views in sources/text/. A citation that does not resolve is a high defect; one that points into a region the text view did not cover is 'absence is unknown' (medium) with the coverage note. The resolved passages are recorded as source_passage evidence and returned as untrusted data.",
            parameters=obj({"path": _PATH}, ["path"]),
            fn=check_cite, permission="read", group="checks",
            when_not_to_use="you want the passage text itself for writing; read sources/text/<id>.md around the marker instead",
            examples=[{"path": "book/chapters/01-vectors.md"}],
        ),
        ToolSpec(
            name="check_readability",
            description=f"Measure a chapter's prose (code, math and markup stripped): Flesch-Kincaid grade, average sentence length, long-word ratio, and sentences over the level's ceiling. The level comes from the argument, the front matter, the outline unit or the persona; bands: {', '.join(f'{k} {lo}-{hi}' for k, (lo, hi) in GRADE_BANDS.items())}. Texts under {MIN_WORDS_FOR_GRADE} words get only the sentence-length judgement.",
            parameters=obj({"path": _PATH, "level": opt(S, f"reading level: one of {', '.join(LEVELS)}, or a persona phrase such as 'high school'; omitted means take it from the chapter, the outline or the persona")}, ["path"]),
            fn=check_readability, permission="read", group="checks",
            when_not_to_use="the file is a reference listing or an exercise set rather than prose",
            examples=[{"path": "book/chapters/01-vectors.md", "level": "undergraduate"}],
        ),
        ToolSpec(
            name="check_prereqs",
            description="Check a chapter's prerequisites against the outline: every concept in its front matter 'requires' must be introduced by an earlier unit (outline order or depends_on chain). A requirement first introduced by a later unit is a forward reference (high); terms from later units' 'introduces' lists that appear in the body are reported at low severity.",
            parameters=obj({"path": _PATH, "unit_id": opt(S, "the outline unit this file belongs to; omitted means the front matter unit_id, then the unit whose path matches")}, ["path"]),
            fn=check_prereqs, permission="read", group="checks",
            when_not_to_use="the outline has no units yet; propose it first with book_outline_propose",
            examples=[{"path": "book/chapters/02-matrices.md", "unit_id": "chapter-matrices"}],
        ),
        ToolSpec(
            name="check_judge",
            description="Ask a separate-context critic (the model, with the rubric and the chapter only) for a score 0-10, defects with locations and severities, and a summary. Advisory: its authority is model_inference, so it closes only the judge obligation and never stands in for a render, an execution or a citation check. Costs one model call.",
            parameters=obj({"path": _PATH, "rubric": opt(S, "rubric text; omitted means the chapter rubric (motivation before definition, example density, notation discipline, sections end with what the reader can now do, no unexplained jargon, supported claims)"), "role": opt(S, "the reviewer role marker, default 'critic'"), "unit_id": opt(S, "outline unit id, for the trace")}, ["path"]),
            fn=check_judge, permission="compute", timeout_s=JUDGE_TOOL_TIMEOUT_S, group="checks", cost_class="expensive", idempotent=False,
            when_not_to_use="a mechanical check exists for the question (code runs, citation resolves, grade level): run that instead",
            examples=[{"path": "book/chapters/01-vectors.md"}],
        ),
        ToolSpec(
            name="run_checks",
            description="Derive the obligation ledger for a chapter (from its outline unit, or as a generic chapter when it has none) and run every closer available here: render, execute, cite, readability, prereqs, and the judge only when asked. Returns the ledger with each obligation open or closed and why, plus the per-check results. Use it to see what book_mark_done will demand before you call it.",
            parameters=obj({"path": _PATH, "unit_id": opt(S, "outline unit id; omitted means the front matter unit_id or the unit whose path matches"), "only": opt(arr({"type": "string", "enum": list(KINDS)}), "restrict to these obligation kinds"), "judge": opt(B, "also run the advisory judge (one model call); default false")}, ["path"]),
            fn=run_checks, permission="compute", timeout_s=EXECUTE_TOOL_TIMEOUT_S, group="checks", cost_class="moderate",
            when_not_to_use="you only need one check; call it directly",
            examples=[{"path": "book/chapters/01-vectors.md", "only": ["execute", "cite"]}],
        ),
    ]
    for spec in specs:
        if reg.get(spec.name) is None:
            reg.register(spec)
    return reg
