"""Checks: the verifiers that close a unit's obligations.

Every check is a plain function ``check_xxx(ctx, args) -> dict`` with the keys
``passed``, ``authority``, ``confidence``, ``defects`` (each ``{"location",
"severity", "message"}``), ``evidence_id`` and ``detail``. A check passes when
it found no defect of medium or high severity. Each check writes one
``Evidence`` record (``operation`` ``check_<kind>``) into ``ctx.evidence`` when
the context has a store, so the done-gate and the eval gate can see what was
verified, by what authority, and against which version of the file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ...domain.errors import ErrorCategory, Limit, ToolFailure
from ...domain.evidence import Evidence
from ...domain.utils import sha256_text
from ...domain.workspace import WorkspaceError
from ..registry import ToolContext

SEVERITIES: tuple[str, ...] = ("low", "medium", "high")
FAILING_SEVERITIES: frozenset[str] = frozenset({"medium", "high"})
CHECK_FILE_SIZE_LIMIT = Limit("check file size", 2_000_000, "bytes", does_not_raise=("splitting the chapter into sections, which is the fix",))
DEFECTS_INLINE_LIMIT = Limit("defects returned inline", 50, "defects", does_not_raise=("the detail, which keeps the counts",))
TEXT_SUFFIXES: tuple[str, ...] = (".md", ".markdown", ".txt")


def defect(location: str, severity: str, message: str) -> dict[str, str]:
    if severity not in SEVERITIES:
        severity = "medium"
    return {"location": location, "severity": severity, "message": message}


def verdict(defects: list[dict[str, Any]]) -> bool:
    return not any(d.get("severity") in FAILING_SEVERITIES for d in defects)


def resolve_text_file(ctx: ToolContext, path: Any, suffixes: tuple[str, ...] = TEXT_SUFFIXES) -> tuple[Path, str, str]:
    """(absolute path, workspace-relative path, text) for a chapter-like file, with the usual typed errors."""
    if not isinstance(path, str) or not path.strip():
        raise ToolFailure("invalid_input", "path must be a non-empty workspace-relative path", ErrorCategory.invalid_input, remediation="Pass the chapter path, e.g. book/chapters/01-vectors.md.")
    try:
        p = ctx.workspace.resolve(path)
    except WorkspaceError as e:
        raise ToolFailure("invalid_input", f"{e}; the workspace root is {ctx.workspace.root}.", ErrorCategory.invalid_input, remediation="Give a path relative to the workspace root; files outside it cannot be checked.", path=path) from e
    if not p.is_file():
        raise ToolFailure("not_found", f"no file at {path}", ErrorCategory.not_found, remediation="Check the path with list_files or glob; the unit's path is in the outline.", path=path)
    if suffixes and p.suffix.lower() not in suffixes:
        raise ToolFailure("unsupported", f"{path} is not a Markdown file ({p.suffix or 'no suffix'}); this check reads chapters.", ErrorCategory.unsupported, remediation="Point the check at the unit's Markdown file.", path=path)
    size = p.stat().st_size
    if size > CHECK_FILE_SIZE_LIMIT.value:
        err = CHECK_FILE_SIZE_LIMIT.exceeded(size, "Split the chapter into sections and check each.")
        raise ToolFailure(err.code, err.message, err.category, remediation=err.remediation, details=err.details)
    rel = ctx.workspace.relpath(p)
    return p, rel, p.read_text(encoding="utf-8", errors="replace")


def record_evidence(
    ctx: ToolContext,
    operation: str,
    rel: str,
    text: str,
    result: dict[str, Any],
    *,
    confidence: str,
    authority: str,
    limitations: list[str] | None = None,
    links: list[str] | None = None,
    parameters: dict[str, Any] | None = None,
    provider: str = "edgent",
) -> str | None:
    """Write the check's Evidence record; the subject carries the file digest so a changed file gets a new record."""
    store = getattr(ctx, "evidence", None)
    if store is None:
        return None
    ev = Evidence(
        operation=operation,
        parameters={"path": rel, **(parameters or {})},
        result=result,
        confidence=confidence,  # type: ignore[arg-type]
        authority=authority,  # type: ignore[arg-type]
        limitations=list(limitations or []),
        locations=[{"kind": "path", "ref": rel}],
        links=list(links or []),
        subject={"path": rel, "sha256": sha256_text(text)},
        provider=provider,
    )
    try:
        store.put(ev)
    except Exception as e:  # noqa: BLE001 - a conflicting or unwritable record must not fail the check itself
        _trace(ctx, "error", {"operation": operation, "error": f"{type(e).__name__}: {e}"[:300]})
        return None
    return ev.evidence_id


def check_result(passed: bool, authority: str, confidence: str, defects: list[dict[str, Any]], evidence_id: str | None, detail: dict[str, Any], limitations: list[str] | None = None) -> dict[str, Any]:
    shown = defects[: int(DEFECTS_INLINE_LIMIT.value)]
    out: dict[str, Any] = {
        "passed": bool(passed),
        "authority": authority,
        "confidence": confidence,
        "defects": shown,
        "evidence_id": evidence_id,
        "detail": detail,
    }
    if len(defects) > len(shown):
        out["defects_omitted"] = len(defects) - len(shown)
    if limitations:
        out["limitations"] = list(limitations)
    return out


def _trace(ctx: ToolContext, kind: str, data: dict[str, Any]) -> None:
    tracer = getattr(ctx, "tracer", None)
    if tracer is None:
        return
    try:
        tracer.record(kind, data, node_id=ctx.node_id or None)
    except Exception:  # noqa: BLE001 - tracing never breaks a check
        pass


def trace_check(ctx: ToolContext, kind: str, rel: str, result: dict[str, Any]) -> None:
    _trace(ctx, "check", {"check": kind, "path": rel, "passed": result.get("passed"), "defects": len(result.get("defects") or []), "evidence_id": result.get("evidence_id")})


def last_line(text: str, limit: int = 300) -> str:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return (lines[-1] if lines else "")[:limit]


__all__ = [
    "FAILING_SEVERITIES", "SEVERITIES", "check_result", "defect", "last_line", "record_evidence", "resolve_text_file", "trace_check", "verdict",
]
