"""Ingestion: every input becomes a greppable, citable text view under ``sources/text``.

``ingest_path`` sniffs a file, runs the matching ingester, copies the raw file, writes the text view and
the ``SourceRecord``, and regenerates ``INDEX.md``. ``ingest_url`` does the same for a URL. ``list_sources``
and ``get_source`` read the records back; ``resolve_locator`` turns a citation such as ``lecture-03:t=14:32``
into the text around that marker, saying plainly when the region is uncovered. Everything here is offline
and standard-library; optional toolchains only widen what a view can contain.
"""
from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ...domain.errors import ErrorCategory, Limit, ToolFailure
from ...domain.utils import now_iso, sha256_file
from ...domain.workspace import Workspace
from .base import (
    COVERAGE_STATUSES,
    LOCATOR_FORMS,
    PAGE_MARKER_RE,
    SOURCE_KINDS,
    SUPPORTED_INPUTS,
    TIME_MARKER_RE,
    WORKSPACE_WRITE_LOCK,
    IngestResult,
    Locator,
    LocatorError,
    Sniff,
    SourceRecord,
    all_records,
    allocate_id,
    clean_title,
    coverage,
    coverage_status,
    find_by_origin,
    find_by_sha256,
    frames_rel,
    hms,
    id_taken,
    in_flight_ids,
    parse_hms,
    parse_locator,
    place_raw,
    raw_extension,
    read_record,
    regenerate_index,
    release_id,
    reserve_id,
    sniff,
    sniff_kind,
    title_from_stem,
    validate_id,
    wait_for_in_flight_sha,
    write_meta,
    write_text_view,
)
from .image import ingest_image
from .media import Segment, Transcript, ingest_media
from .office import ingest_office
from .pdf import ingest_pdf
from .text import ingest_html, ingest_text
from .web import Fetcher, ingest_url, make_default_fetcher

LOCATOR_WINDOW_LIMIT = Limit("locator window", 4000, "chars", does_not_raise=("read_file on the text view, which has no such ceiling",))
LOCATOR_WINDOW_DEFAULT_CHARS = 600
NEAREST_MARKERS_SHOWN = 5
MAX_INPUT_BYTES_LIMIT = Limit("source file size", 4 * 1024 ** 3, "bytes")
UNCOVERED_PAGE_NOTE = "page {page} had no extractable text: absence is unknown, not evidence"
Ingester = Callable[..., IngestResult]

_INGESTERS: dict[str, Ingester] = {
    "text": ingest_text,
    "markdown": ingest_text,
    "code": ingest_text,
    "web": ingest_html,
    "pdf": ingest_pdf,
    "image": ingest_image,
    "video": ingest_media,
    "audio": ingest_media,
    "office": ingest_office,
}


# ----------------------------------------------------------------------------- adding sources
def _toolchain(toolchain: Any) -> Any:
    if toolchain is not None:
        return toolchain
    from ..toolchain import Toolchain  # noqa: PLC0415 - detection runs subprocess lookups; only when nothing was passed

    return Toolchain.detect()


def _input_path(workspace: Workspace, path: str | os.PathLike[str]) -> Path:
    """A readable file: absolute paths anywhere are accepted (this is how outside files enter the workspace)."""
    raw = os.path.expanduser(str(path).strip())
    if not raw:
        raise ToolFailure("invalid_input", "path is empty.", ErrorCategory.invalid_input, remediation="Give the file to ingest, absolute or relative to the workspace root.")
    p = Path(raw) if os.path.isabs(raw) else workspace.root / raw
    p = p.resolve()
    if p.is_dir():
        raise ToolFailure("invalid_input", f"{path} is a directory; sources_add takes one file at a time.", ErrorCategory.invalid_input, remediation="Add each file separately (glob or list_files shows what is inside).", path=str(path))
    if not p.is_file():
        raise ToolFailure("not_found", f"no file at {path}.", ErrorCategory.not_found, remediation="Check the path (absolute paths outside the workspace are accepted); for a URL use sources_add_url.", path=str(path))
    size = p.stat().st_size
    if size > MAX_INPUT_BYTES_LIMIT.value:
        err = MAX_INPUT_BYTES_LIMIT.exceeded(size, "Split or re-encode the file before adding it.")
        raise ToolFailure(err.code, err.message, err.category, remediation=err.remediation, details=err.details)
    return p


def ingest_path(
    path: str | os.PathLike[str],
    workspace: Workspace,
    toolchain: Any = None,
    title: str | None = None,
    source_id: str | None = None,
    *,
    replace: bool = False,
    origin: str | None = None,
    transcriber: Any = None,
    extra: dict[str, Any] | None = None,
) -> SourceRecord:
    """Ingest one file. Returns the record; a file already ingested (same sha256) returns its existing record
    with ``extra['already_present']`` set on the returned object (not persisted). ``replace`` re-ingests in place:
    the record with the same content, else the one ingested from the same path or URL, keeps its id.
    Degraded results (missing toolchains) are still recorded: ``extra['usable']`` is False when no text or
    viewable content came out, and ``extra['missing_toolchains']`` names what would fix it.

    Safe to call from parallel tool threads: the id is reserved under ``WORKSPACE_WRITE_LOCK`` before the slow
    work runs, and the files, the record and the index are written under it afterwards."""
    tc = _toolchain(toolchain)
    if source_id is not None:
        source_id = validate_id(source_id)
    src = _input_path(workspace, path)
    info = sniff(src)
    if info.kind == "binary":
        raise _unsupported_file(src, info)
    size = src.stat().st_size
    sha = sha256_file(str(src))
    origin = origin or str(src)
    with WORKSPACE_WRITE_LOCK:
        wait_for_in_flight_sha(workspace, sha)  # a parallel add of the same content lands first, so the dedupe sees it
        existing = None if replace else find_by_sha256(workspace, sha)
        if existing is not None and (source_id is None or existing.id == source_id):
            existing.extra["already_present"] = True
            return existing
        sid = _choose_id(workspace, source_id, title, src, sha, replace, origin)
        reserve_id(workspace, sid, sha)
    try:
        result = _INGESTERS[info.kind](src, sid, workspace, tc, info, title=title, transcriber=transcriber)
        with WORKSPACE_WRITE_LOCK:
            return _commit(workspace, sid, src, info, result, size=size, sha=sha, title=title, origin=origin, extra=extra)
    finally:
        release_id(workspace, sid)


def _commit(workspace: Workspace, sid: str, src: Path, info: Sniff, result: IngestResult, *, size: int, sha: str, title: str | None, origin: str, extra: dict[str, Any] | None) -> SourceRecord:
    """Raw copy, text view, meta and index for a finished ingest. Runs under ``WORKSPACE_WRITE_LOCK``."""
    if result.kind != "video":  # a replaced video's frames are managed by the media ingester; anything else drops stale frames
        shutil.rmtree(workspace.resolve(frames_rel(sid)), ignore_errors=True)
    raw_rel = place_raw(workspace, sid, raw_extension(info, src), src=src)
    record = SourceRecord(
        id=sid,
        kind=result.kind,
        title=clean_title(title or result.title or title_from_stem(src.stem)),
        raw_path=raw_rel,
        text_path="",
        sha256=sha,
        size_bytes=size,
        coverage=result.coverage,
        limitations=list(result.limitations),
        created_at=now_iso(),
        origin=origin,
        extra={**result.extra, **(extra or {}), "usable": bool(result.usable), "missing_toolchains": list(result.missing)},
    )
    write_text_view(workspace, record, result.body)
    write_meta(workspace, record)
    regenerate_index(workspace)
    return record


def _choose_id(workspace: Workspace, source_id: str | None, title: str | None, src: Path, sha: str, replace: bool, origin: str) -> str:
    """Runs under ``WORKSPACE_WRITE_LOCK``: ids reserved by ingests still running count as taken."""
    busy = in_flight_ids(workspace)
    if source_id is None and replace:
        same = find_by_sha256(workspace, sha) or find_by_origin(workspace, origin)
        source_id = same.id if same is not None else None
    if source_id is None:
        return allocate_id(workspace, title or src.stem, reserved=busy)
    if not replace and (id_taken(workspace, source_id) or source_id in busy):
        raise ToolFailure("invalid_input", f"source id {source_id!r} is already taken.", ErrorCategory.invalid_input, remediation="Omit source_id to get a fresh one, choose another, or pass replace=true to overwrite that source.", source_id=source_id)
    return source_id  # on replace the view, meta and raw copy are overwritten in place once the ingester has succeeded


def _unsupported_file(src: Path, info: Sniff) -> ToolFailure:
    hints = {
        "xlsx": "Export the sheet to CSV (bash: libreoffice --headless --convert-to csv) and add the CSV.",
        "zip": "Unpack the archive (bash: unzip) and add the files inside one by one.",
    }
    return ToolFailure(
        "unsupported",
        f"{src.name} is {info.format} ({info.mime}), which sources_add cannot turn into text.",
        ErrorCategory.unsupported,
        remediation=hints.get(info.format, f"Convert it to one of the supported inputs first ({SUPPORTED_INPUTS})."),
        path=str(src),
        format=info.format,
    )


# ----------------------------------------------------------------------------- reading sources
def list_sources(workspace: Workspace) -> list[SourceRecord]:
    return all_records(workspace)


def get_source(workspace: Workspace, source_id: str) -> SourceRecord | None:
    return read_record(workspace, str(source_id).strip().lower())


# ----------------------------------------------------------------------------- locators
def _clamp_window(window_chars: Any) -> int:
    try:
        n = int(window_chars)
    except (TypeError, ValueError):
        n = LOCATOR_WINDOW_DEFAULT_CHARS
    return max(40, min(n, int(LOCATOR_WINDOW_LIMIT.value)))


def _cut(text: str, window: int) -> tuple[str, bool]:
    if len(text) <= window:
        return text, False
    cut = text[:window]
    nl = cut.rfind("\n")
    if nl > window // 2:
        cut = cut[:nl]
    return cut.rstrip() + " …", True


def _marker_lines(lines: list[str]) -> list[tuple[str, int, int]]:
    """(kind, value, line index) for every page and time marker in view order."""
    out: list[tuple[str, int, int]] = []
    for i, line in enumerate(lines):
        m = PAGE_MARKER_RE.match(line)
        if m:
            out.append(("page", int(m.group(1)), i))
            continue
        m = TIME_MARKER_RE.match(line)
        if m:
            out.append(("time", int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)), i))
    return out


def _nearest(markers: list[tuple[str, int, int]], kind: str, value: int) -> list[str]:
    same = [(abs(v - value), v) for k, v, _ in markers if k == kind]
    same.sort()
    fmt = (lambda v: f"p.{v}") if kind == "page" else (lambda v: f"t={hms(v)}")
    return [fmt(v) for _, v in same[:NEAREST_MARKERS_SHOWN]]


def _coverage_note(rec: SourceRecord) -> str | None:
    status = rec.coverage_status
    if status == "complete":
        return None
    detail = "; ".join(rec.limitations[:3]) if rec.limitations else "see sources_coverage"
    return f"coverage of {rec.id} is {status}: {detail}"


def resolve_locator(workspace: Workspace, locator: str, window_chars: int = LOCATOR_WINDOW_DEFAULT_CHARS) -> dict[str, Any]:
    """Resolve ``id:p.N`` / ``id:t=HH:MM:SS`` / ``id:L<n>`` to the text around the marker.

    Never raises for a bad locator or unknown source: ``found`` is False and ``error`` or ``coverage_note``
    says why, with the nearest markers when there are any. ``coverage_note`` is also set on a hit inside a
    region the ingester could not cover (a scanned page, an untranscribed recording)."""
    window = _clamp_window(window_chars)
    out: dict[str, Any] = {"locator": str(locator), "source_id": None, "kind": None, "marker": None, "text": "", "found": False, "coverage_note": None}
    try:
        loc = parse_locator(locator)
    except LocatorError as e:
        out["error"] = str(e)
        out["hint"] = f"Use {LOCATOR_FORMS}."
        return out
    out["source_id"] = loc.source_id
    rec = read_record(workspace, loc.source_id)
    if rec is None:
        known = [r.id for r in all_records(workspace)]
        out["coverage_note"] = f"no source with id {loc.source_id!r}" + (f"; known ids: {', '.join(known[:20])}" if known else "; the workspace has no sources yet")
        out["known_ids"] = known[:20]
        return out
    out["kind"] = rec.kind
    out["text_path"] = rec.text_path
    try:
        view = workspace.read_text(rec.text_path)
    except (FileNotFoundError, OSError):
        out["coverage_note"] = f"the text view {rec.text_path} is missing; add the source again (sources_add with replace=true)"
        return out
    lines = view.split("\n")
    markers = _marker_lines(lines)
    if loc.kind == "page":
        _resolve_page(out, rec, loc, lines, markers, window)
    elif loc.kind == "time":
        _resolve_time(out, rec, loc, lines, markers, window)
    else:
        _resolve_line(out, rec, loc, lines, window)
    return out


def _segment(lines: list[str], start: int, markers: list[tuple[str, int, int]], window: int, stop_kinds: tuple[str, ...]) -> tuple[str, bool]:
    """Text after the marker at ``start`` up to the next marker of a stopping kind, cut to the window."""
    end = next((idx for kind, _, idx in markers if idx > start and kind in stop_kinds), len(lines))
    return _cut("\n".join(lines[start + 1 : end]).strip("\n"), window)


def _tail(lines: list[str], budget: int) -> str:
    """The last whole lines that fit in ``budget`` characters."""
    kept: list[str] = []
    used = 0
    for line in reversed(lines):
        if used + len(line) + 1 > budget:
            break
        kept.append(line)
        used += len(line) + 1
    return "\n".join(reversed(kept))


def _resolve_page(out: dict[str, Any], rec: SourceRecord, loc: Locator, lines: list[str], markers: list[tuple[str, int, int]], window: int) -> None:
    detail = rec.coverage.get("detail") or {}
    pages_total = detail.get("pages_total")
    failed = set(detail.get("pages_failed") or [])
    hit = next((idx for k, v, idx in markers if k == "page" and v == loc.value), None)
    out["marker"] = loc.marker
    if hit is None:
        page_markers = [m for m in markers if m[0] == "page"]
        if isinstance(pages_total, int) and loc.value > pages_total:
            out["coverage_note"] = f"page {loc.value} is beyond the last page ({pages_total}) of {rec.id}"
        elif not page_markers:
            kinds = sorted({k for k, _, _ in markers})
            out["coverage_note"] = f"{rec.id} ({rec.kind}) has no page markers; cite " + ("t=HH:MM:SS" if "time" in kinds else "L<n> lines of the text view") + " instead"
        else:
            out["coverage_note"] = f"page {loc.value} has no marker in the text view of {rec.id}"
        out["nearest"] = _nearest(markers, "page", loc.value)
        return
    text, truncated = _segment(lines, hit, markers, window, ("page", "time"))
    out["line"] = hit + 1
    out["text"] = text
    out["truncated"] = truncated
    if loc.value in failed:
        out["coverage_note"] = UNCOVERED_PAGE_NOTE.format(page=loc.value)
        out["marker_present"] = True
        out["nearest"] = [f"p.{v}" for v in sorted(v for k, v, _ in markers if k == "page" and v not in failed and abs(v - loc.value) <= 3)]
        return
    out["found"] = True
    out["coverage_note"] = _coverage_note(rec)


def _resolve_time(out: dict[str, Any], rec: SourceRecord, loc: Locator, lines: list[str], markers: list[tuple[str, int, int]], window: int) -> None:
    detail = rec.coverage.get("detail") or {}
    duration = rec.extra.get("duration_s")
    time_markers = [(v, idx) for k, v, idx in markers if k == "time"]
    untranscribed = rec.kind in ("video", "audio") and not detail.get("transcriber")
    if not time_markers:
        if rec.kind in ("video", "audio"):
            out["coverage_note"] = f"{rec.id} has no time markers: " + ("; ".join(rec.limitations[:2]) if rec.limitations else "nothing was transcribed") + "; what is said at that time is unknown, not absent"
        else:
            out["coverage_note"] = f"{rec.id} ({rec.kind}) has no time markers; cite p.N or L<n> instead"
        out["nearest"] = []
        return
    if isinstance(duration, (int, float)) and loc.value > duration + 1:
        out["coverage_note"] = f"t={hms(loc.value)} is beyond the duration of {rec.id} ({hms(duration)})"
        out["nearest"] = _nearest(markers, "time", loc.value)
        return
    at_or_before = [m for m in time_markers if m[0] <= loc.value]
    if not at_or_before:  # the first marker starts after the cited time: nothing in the view covers it
        out["coverage_note"] = f"t={hms(loc.value)} is before the first marker [t={hms(time_markers[0][0])}] of {rec.id}: nothing is recorded there"
        out["nearest"] = _nearest(markers, "time", loc.value)
        return
    secs, idx = at_or_before[-1]
    out["marker"] = f"[t={hms(secs)}]"
    out["line"] = idx + 1
    out["offset_s"] = loc.value - secs
    text, truncated = _segment(lines, idx, markers, window, ("page",))
    out["text"] = text
    out["truncated"] = truncated
    out["found"] = True
    if untranscribed:
        out["coverage_note"] = f"no transcript for {rec.id}: only keyframes and slide text are listed at t={hms(secs)}; what is said there is unknown, not absent"
    else:
        out["coverage_note"] = _coverage_note(rec)
    if out["offset_s"]:
        out["nearest"] = _nearest(markers, "time", loc.value)


def _resolve_line(out: dict[str, Any], rec: SourceRecord, loc: Locator, lines: list[str], window: int) -> None:
    total = len(lines) - (1 if lines and lines[-1] == "" else 0)
    out["marker"] = f"L{loc.value}"
    if loc.value < 1 or loc.value > total:
        out["coverage_note"] = f"line {loc.value} is beyond the end of the text view of {rec.id} ({total} lines)"
        out["nearest"] = [f"L{total}"] if total else []
        return
    i = loc.value - 1
    half = window // 2
    before = _tail(lines[max(0, i - 40) : i], half)
    after = _cut("\n".join(lines[i + 1 : i + 41]).rstrip("\n"), half)[0]
    out["line"] = loc.value
    out["line_text"] = lines[i]
    out["text"] = "\n".join(part for part in (before, lines[i], after) if part)
    out["found"] = True
    out["coverage_note"] = _coverage_note(rec)


__all__ = [
    "COVERAGE_STATUSES", "Fetcher", "IngestResult", "LOCATOR_WINDOW_DEFAULT_CHARS", "LOCATOR_WINDOW_LIMIT", "Locator", "LocatorError", "SOURCE_KINDS",
    "SUPPORTED_INPUTS", "Segment", "Sniff", "SourceRecord", "Transcript", "allocate_id", "coverage", "coverage_status", "get_source", "hms", "ingest_path",
    "ingest_url", "list_sources", "make_default_fetcher", "parse_hms", "parse_locator", "regenerate_index", "resolve_locator", "sniff", "sniff_kind",
]
