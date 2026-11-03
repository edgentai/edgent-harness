"""Media tools: films (Manim), diagrams (SVG), interactive widgets (HTML/JS), narration (TTS) and assembly.

Every artifact gets render, measure and look, in that order, and the first two are code. The model never
writes a scene file: it sends a storyboard and ops to ``film_compile``; ``film_render`` compiles, renders in the
sandbox and measures the report the scene wrote about itself; ``film_look`` attaches frames for eyes. Diagrams
and widgets go through headless Chromium with the page lab. A missing toolchain is a typed ``unavailable``
error that says what to install and what to do meanwhile; a film that cannot be rendered stays 'unrendered'
and never blocks a book. Tests inject ``ctx.extra["media_runner"]`` (a subprocess runner with
``run_subprocess``'s signature), ``ctx.extra["chromium"]`` (a browser path, or None for none) and
``ctx.extra["which"]`` (PATH lookup for speech backends).
"""
from __future__ import annotations

import html as html_lib
import re
import shutil
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ...adapters.browser.chromium import find_browser, report_defects, run_page
from ...adapters.browser.chromium import unavailable as chromium_unavailable
from ...adapters.renderers.av import assemble, extract_frames, probe
from ...adapters.renderers.manim_compiler import (
    WORDS_PER_SECOND,
    CompileError,
    apply_ops,
    compile_storyboard,
    narration_words,
    normalise,
    outline,
)
from ...adapters.renderers.manim_render import (
    FILM_RENDER_TIMEOUT_LIMIT,
    FRAMES_DIRNAME,
    ONLY_SCENE_FILENAME,
    QUALITY_FLAGS,
    REPORT_FILENAME,
    SCENE_FILENAME,
    VIDEO_FILENAME,
    RenderResult,
    beat_midpoints,
    beat_timings,
    last_render,
    measure,
    render,
)
from ...adapters.renderers.svg import (
    DIAGRAM_SEVERITIES,
    RENDER_WIDTH_PX_DEFAULT,
    render_svg_png,
    static_defects,
    validate_svg,
)
from ...adapters.renderers.tts import synthesize
from ...adapters.toolchain import INSTALL_HINTS, Toolchain
from ...domain.errors import ErrorCategory, Limit, ToolFailure
from ...domain.utils import ensure_dir, now_iso, read_json, write_json
from ...domain.workspace import WorkspaceError
from ..registry import I, N, S, ToolContext, ToolRegistry, ToolSpec, arr, obj, opt

MEDIA_TOOL_NAMES: tuple[str, ...] = ("film_compile", "film_render", "film_look", "film_outline", "diagram_render", "widget_run", "tts_narrate", "video_assemble")
STORYBOARD_FILENAME = "storyboard.json"
DEFECTS_FILENAME = "defects.json"
FINAL_FILENAME = "final.mp4"
REGISTRY_TIMEOUT_MARGIN_S = 30.0  # the registry's own timeout sits above the sandbox's so the sandbox reports first
FILM_LOOK_FRAMES_LIMIT = Limit("film_look frames", 12, "frames")
FILM_LOOK_FRAMES_DEFAULT = 6
FILM_DEFECTS_INLINE_LIMIT = Limit("film_render inline defects", 60, "defects", does_not_raise=("the full list is always in media/films/<id>/defects.json",))
FILM_LOCK_WAIT_S = 5.0  # how long a film tool waits for another tool working on the same film before it says busy
FRAME_WIDTH_PX = 640
DIAGRAM_WIDTH_PX = RENDER_WIDTH_PX_DEFAULT
DIAGRAM_TIMEOUT_S = 60.0
WIDGET_VIEWPORT_PX = (1280, 800)
WIDGET_TIMEOUT_S = 60.0
WIDGET_TOOL_TIMEOUT_S = WIDGET_TIMEOUT_S * 2 + REGISTRY_TIMEOUT_MARGIN_S  # two Chromium runs at most
TTS_TOOL_TIMEOUT_S = 330.0
ASSEMBLE_TOOL_TIMEOUT_S = 630.0
MEASUREMENT_KEYS: tuple[str, ...] = ("overflow", "clipped", "overlaps", "small_text", "low_contrast")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

HOUSE_THEME_CSS = """
:root{--bg:#ffffff;--surface:#f4f6f8;--ink:#1b1f24;--muted:#5b6470;--accent:#0b67c2;--accent-ink:#ffffff;--ok:#1d7a3e;--warn:#b35c00;--border:#d5dae0;--radius:8px;--font:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
@media (prefers-color-scheme:dark){:root{--bg:#0f1318;--surface:#171c23;--ink:#eef1f5;--muted:#aab4c0;--accent:#5fb0ff;--accent-ink:#0f1318;--border:#2a323c}}
*,*::before,*::after{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--font);font-size:16px;line-height:1.5}
body{padding:16px}
h1,h2,h3{line-height:1.25;margin:0 0 .5em}
h1{font-size:1.5rem}h2{font-size:1.25rem}h3{font-size:1.1rem}
p,li{max-width:70ch}
small,.caption{font-size:.875rem;color:var(--muted)}
button,input,select,textarea{font:inherit;color:inherit}
button{background:var(--accent);color:var(--accent-ink);border:0;border-radius:var(--radius);padding:.5em 1em;min-height:40px;cursor:pointer}
button:focus-visible,input:focus-visible,select:focus-visible,textarea:focus-visible{outline:3px solid var(--accent);outline-offset:2px}
input[type=range]{width:100%}
label{display:inline-block;margin:.25em 0}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:16px;margin:0 0 16px}
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center}
.row{display:flex;gap:16px;flex-wrap:wrap}
canvas,svg,img{max-width:100%;height:auto}
code,pre{font-family:var(--mono);font-size:.9375rem}
.ok{color:var(--ok)}.warn{color:var(--warn)}
""".strip()


# ----------------------------------------------------------------------------- helpers
def _fail(code: str, message: str, category: ErrorCategory, remediation: str, **details: Any) -> ToolFailure:
    return ToolFailure(code, message, category, remediation=remediation, **details)


def _ident(value: Any, what: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise _fail("invalid_input", f"{what} must be 1-64 letters, digits, '-' or '_' (got {value!r}).", ErrorCategory.invalid_input, f"Choose a short slug for {what}, such as 'ch03-bayes'.", **{what: value})
    return value


def _rel(ctx: ToolContext, p: str | Path | None) -> str | None:
    if p is None:
        return None
    try:
        return ctx.workspace.relpath(p)
    except ValueError:
        return str(p)


def _resolve(ctx: ToolContext, path: str) -> Path:
    try:
        return ctx.workspace.resolve(path)
    except WorkspaceError as e:
        raise _fail("invalid_input", f"{e}; the workspace root is {ctx.workspace.root}.", ErrorCategory.invalid_input, "Give a path inside the workspace.", path=path) from e


def _existing(ctx: ToolContext, path: str, what: str) -> Path:
    p = _resolve(ctx, path)
    if not p.is_file():
        raise _fail("not_found", f"{what} not found at {path}.", ErrorCategory.not_found, "Check the path with list_files; produce the file first (tts_narrate for narration).", path=path)
    return p


def _media_dir(ctx: ToolContext, key: str, ident: str) -> Path:
    return _resolve(ctx, str(ctx.workspace.path(key) / ident))


_FILM_LOCKS: dict[str, threading.Lock] = {}
_FILM_LOCKS_GUARD = threading.Lock()


@contextmanager
def _film_lock(film_dir: Path, film_id: str, what: str) -> Iterator[None]:
    """One film tool at a time per film: tools run in parallel threads and all of them rewrite the film directory."""
    with _FILM_LOCKS_GUARD:
        lock = _FILM_LOCKS.setdefault(str(film_dir), threading.Lock())
    if not lock.acquire(timeout=FILM_LOCK_WAIT_S):
        raise ToolFailure(
            "busy",
            f"film {film_id!r} is being worked on by another call (a render in progress, most likely); {what} waited {FILM_LOCK_WAIT_S:g} s.",
            ErrorCategory.resource_constraint,
            retryable=True,
            remediation="Wait for the running film_render or film_compile on this film to finish, then call again once.",
            film_id=film_id,
        )
    try:
        yield
    finally:
        lock.release()


def _toolchain(ctx: ToolContext) -> Toolchain:
    return ctx.toolchain if ctx.toolchain is not None else Toolchain.detect()


def _runner(ctx: ToolContext) -> Any:
    return ctx.extra.get("media_runner")


def _chromium(ctx: ToolContext) -> str | None:
    if "chromium" in ctx.extra:
        return ctx.extra["chromium"] or None
    path = _toolchain(ctx).path("chromium")
    return find_browser(path) if path else None


def _pagelab_dir(ctx: ToolContext, name: str) -> Path:
    return Path(ensure_dir(ctx.workspace.path("cache") / "pagelab" / name))


def _clamp(value: Any, limit: Limit, default: int, floor: int = 1) -> int:
    try:
        n = int(value) if value is not None else default
    except (TypeError, ValueError):
        n = default
    return max(floor, min(n, int(limit.value)))


def _load_storyboard(ctx: ToolContext, film_dir: Path, film_id: str) -> dict[str, Any]:
    path = film_dir / STORYBOARD_FILENAME
    data = read_json(path) if path.is_file() else None
    if not isinstance(data, dict):
        raise _fail("not_found", f"film {film_id!r} has no storyboard.", ErrorCategory.not_found, "Create it with film_compile(film_id, storyboard={...}) first.", film_id=film_id)
    return data


def _compile_failure(e: CompileError, film_id: str) -> ToolFailure:
    where = f"beat {e.beat}" if e.beat is not None else "the storyboard prelude"
    at = f" at line {e.line}" if e.line is not None else ""
    return ToolFailure(
        "invalid_input",
        f"film {film_id!r} does not compile: {where}{at}: {e.message}" + (f" ({e.text.strip()[:80]})" if e.text else ""),
        ErrorCategory.invalid_input,
        remediation="Fix that code and resend it: patch_beat with find/replace for a small change, set_beat with the full code otherwise. Do not repeat the unchanged call.",
        recovery={"kind": "inspect_input_contract", "operation": "film_compile"},
        film_id=film_id,
        beat=e.beat,
        line=e.line,
        text=(e.text or "").strip()[:200] or None,
    )


def _defect_counts(defects: list[dict[str, Any]]) -> dict[str, int]:
    return {"error": sum(1 for d in defects if d.get("severity") == "error"), "warn": sum(1 for d in defects if d.get("severity") == "warn")}


# ----------------------------------------------------------------------------- films
def film_compile(ctx: ToolContext, a: dict[str, Any]) -> dict[str, Any]:
    film_id = _ident(a["film_id"], "film_id")
    film_dir = _media_dir(ctx, "films", film_id)
    with _film_lock(film_dir, film_id, "film_compile"):
        return _film_compile(ctx, a, film_id, film_dir)


def _film_compile(ctx: ToolContext, a: dict[str, Any], film_id: str, film_dir: Path) -> dict[str, Any]:
    sb_path = film_dir / STORYBOARD_FILENAME
    given, ops = a.get("storyboard"), a.get("ops") or []
    existing = read_json(sb_path) if sb_path.is_file() else None
    if given is None and existing is None and not ops:
        raise _fail("invalid_input", f"film {film_id!r} has no storyboard yet.", ErrorCategory.invalid_input, "Pass storyboard={title, persona, beats:[...]} to create it, or ops=[{op:'add_beat', ...}].", film_id=film_id)
    base: dict[str, Any] = dict(given) if isinstance(given, dict) else dict(existing or {})
    if isinstance(given, dict) and isinstance(existing, dict):  # a replacement keeps the retired ids retired
        base.setdefault("next_beat_id", existing.get("next_beat_id"))
        base.setdefault("removed_beats", existing.get("removed_beats"))
    applied: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    try:
        sb = normalise(base)
        if ops:
            sb, applied, rejected = apply_ops(sb, ops)
        source, warnings = compile_storyboard(sb)
    except CompileError as e:
        raise _compile_failure(e, film_id) from e
    ensure_dir(film_dir)  # only once there is something to keep; a rejected call leaves no empty film behind
    write_json(sb_path, sb)
    scene_path = film_dir / SCENE_FILENAME
    scene_path.write_text(source, encoding="utf-8")
    rows = outline(sb)
    return {
        "film_id": film_id,
        "title": sb["title"],
        "beats": rows,
        "beat_count": len(rows),
        "estimated_seconds": round(sum(r["estimated_seconds"] or 0.0 for r in rows), 1),
        "warnings": warnings,
        "applied": applied,
        "rejected": rejected,
        "scene_path": _rel(ctx, scene_path),
        "storyboard_path": _rel(ctx, sb_path),
    }


def _unrendered(e: ToolFailure, ctx: ToolContext, film_id: str, film_dir: Path) -> ToolFailure:
    err = e.error
    return ToolFailure(
        "unavailable",
        f"{err.message} Film {film_id!r} stays unrendered; its storyboard and scene.py are saved under {_rel(ctx, film_dir)}.",
        ErrorCategory.unavailable,
        remediation=err.remediation + " A film that cannot be rendered never blocks the book: mark its unit partial, keep the storyboard, and record the render as an unknown.",
        film_id=film_id,
        status="unrendered",
        scene_path=_rel(ctx, film_dir / SCENE_FILENAME),
        toolchain="manim",
    )


def _render_failure(result: RenderResult, ctx: ToolContext, film_id: str) -> ToolFailure:
    completed = result.beats_completed()
    partial = {"log_tail": result.log, "beats_completed": completed, "report_path": _rel(ctx, result.report_path), "failed_beat": result.failed_beat()}
    if result.timed_out:
        return ToolFailure(
            "timeout",
            f"manim did not finish film {film_id!r} within {FILM_RENDER_TIMEOUT_LIMIT.value:g} s; beats completed: {completed}. {FILM_RENDER_TIMEOUT_LIMIT.describe()}",
            ErrorCategory.timeout,
            remediation="Render at quality 'draft', or a few beats at a time with only=[...]; shorten waits. Do not repeat the unchanged call.",
            partial=partial,
            film_id=film_id,
        )
    failed = result.failed_beat()
    where = f" Beat {failed['beat']} ({failed['title']!r}) raised {failed['error']}." if failed else ""
    return ToolFailure(
        "render_failed",
        f"manim exited {result.returncode} rendering film {film_id!r}.{where} Log tail: {result.log[-1200:] or 'empty'}",
        ErrorCategory.execution_failure,
        remediation="Read the traceback, fix the beat it names with film_compile (patch_beat or set_beat), then render again. Do not repeat the unchanged call.",
        partial=partial,
        film_id=film_id,
        beat=failed["beat"] if failed else None,
    )


def film_render(ctx: ToolContext, a: dict[str, Any]) -> dict[str, Any]:
    film_id = _ident(a["film_id"], "film_id")
    film_dir = _media_dir(ctx, "films", film_id)
    with _film_lock(film_dir, film_id, "film_render"):
        return _film_render(ctx, a, film_id, film_dir)


def _film_render(ctx: ToolContext, a: dict[str, Any], film_id: str, film_dir: Path) -> dict[str, Any]:
    quality = a.get("quality") or "draft"
    sb = _load_storyboard(ctx, film_dir, film_id)
    only = sorted({int(i) for i in a.get("only") or []}) or None
    try:
        sb = normalise(sb)
        if only:
            known = {b["id"] for b in sb["beats"]}
            unknown = [i for i in only if i not in known]
            if unknown:
                raise _fail("invalid_input", f"beats {unknown} are not in film {film_id!r}; its beats are {sorted(known)}.", ErrorCategory.invalid_input, "Use ids from film_outline.", film_id=film_id)
        source, warnings = compile_storyboard(sb, only=only)
    except CompileError as e:
        raise _compile_failure(e, film_id) from e
    scene_filename = ONLY_SCENE_FILENAME if only else SCENE_FILENAME  # a partial render never overwrites the whole film's scene.py
    try:
        result = render(film_dir, source, quality=quality, only=only, toolchain=_toolchain(ctx), timeout_s=FILM_RENDER_TIMEOUT_LIMIT.value, runner=_runner(ctx), scene_filename=scene_filename)
    except ToolFailure as e:
        if e.error.code == "unavailable":
            raise _unrendered(e, ctx, film_id, film_dir) from e
        raise
    if not result.ok:
        raise _render_failure(result, ctx, film_id)
    defects = measure(result.report, sb, only=only)
    counts = _defect_counts(defects)
    defects_path = film_dir / DEFECTS_FILENAME
    write_json(defects_path, {"quality": quality, "only": only, "defects": defects, "counts": counts, "measured_at": now_iso()})
    report = result.report or {}
    inline = int(FILM_DEFECTS_INLINE_LIMIT.value)
    return {
        "film_id": film_id,
        "quality": quality,
        "only": only,
        "video_path": _rel(ctx, result.video_path),
        "report_path": _rel(ctx, result.report_path),
        "defects": defects[:inline],
        "defects_omitted": max(0, len(defects) - inline),
        "defects_path": _rel(ctx, defects_path),
        "defect_counts": counts,
        "beat_timings": beat_timings(result.report),
        "duration_s": report.get("total_s"),
        "render_seconds": round(result.duration_s, 1),
        "warnings": warnings,
        "ok": counts["error"] == 0,
    }


def film_look(ctx: ToolContext, a: dict[str, Any]) -> dict[str, Any]:
    film_id = _ident(a["film_id"], "film_id")
    film_dir = _media_dir(ctx, "films", film_id)
    beats = a.get("beats")
    max_frames = _clamp(a.get("max_frames"), FILM_LOOK_FRAMES_LIMIT, FILM_LOOK_FRAMES_DEFAULT)
    info = last_render(film_dir)
    video = film_dir / VIDEO_FILENAME
    if info is None:
        raise _fail("unavailable", f"film {film_id!r} has no render to look at.", ErrorCategory.unavailable, "Run film_render first (quality 'draft' is enough to look); without manim the film stays unrendered.", film_id=film_id)
    if not info.get("ok") or not video.is_file():
        raise _fail("unavailable", f"the last render of film {film_id!r} produced no video, or a render is in progress (beats completed: {info.get('beats_completed')}).", ErrorCategory.unavailable, "Fix what the film_render error named with film_compile, then film_render again before looking.", film_id=film_id)
    report = read_json(film_dir / REPORT_FILENAME) if (film_dir / REPORT_FILENAME).is_file() else None
    points = beat_midpoints(report, beats)
    if not points:
        if beats:
            rendered = [b.get("id") for b in (report or {}).get("beats") or []]
            raise _fail("invalid_input", f"beats {list(beats)} are not in the last render of {film_id!r}; rendered beats: {rendered}.", ErrorCategory.invalid_input, "Pick ids from the last render, or render those beats first.", film_id=film_id)
        duration = probe(video, _toolchain(ctx), _runner(ctx))["duration_s"] or 0.0
        points = [{"beat": None, "title": "", "time_s": round(duration * (i + 0.5) / max_frames, 3)} for i in range(max_frames)]
    skipped = max(0, len(points) - max_frames)
    points = points[:max_frames]
    frames = extract_frames(video, [p["time_s"] for p in points], film_dir / FRAMES_DIRNAME, width=FRAME_WIDTH_PX, toolchain=_toolchain(ctx), runner=_runner(ctx))
    rows: list[dict[str, Any]] = []
    attach: list[dict[str, Any]] = []
    for point, frame in zip(points, frames, strict=True):
        rows.append({"beat": point["beat"], "title": point["title"], "time_s": frame["time_s"], "path": _rel(ctx, frame["path"])})
        if frame["path"]:
            attach.append({"type": "image", "path": frame["path"], "text": f"[frame: beat {point['beat']} {point['title']!r} at {frame['time_s']:.1f} s]"})
    out: dict[str, Any] = {"film_id": film_id, "quality": info.get("quality"), "frames": rows, "skipped": skipped, "_attach": attach}
    if skipped:
        out["note"] = f"{skipped} more beats were not shown; call again with beats=[...] (at most {FILM_LOOK_FRAMES_LIMIT.value:g} frames per call)."
    return out


def film_outline(ctx: ToolContext, a: dict[str, Any]) -> dict[str, Any]:
    film_id = _ident(a["film_id"], "film_id")
    film_dir = _media_dir(ctx, "films", film_id)
    sb = normalise(_load_storyboard(ctx, film_dir, film_id))
    rows = outline(sb)
    info = last_render(film_dir)
    defects = read_json(film_dir / DEFECTS_FILENAME) if (film_dir / DEFECTS_FILENAME).is_file() else None
    return {
        "film_id": film_id,
        "title": sb["title"],
        "persona": sb["persona"],
        "show_titles": sb["show_titles"],
        "prelude_lines": len(sb["prelude"].splitlines()),
        "beats": rows,
        "beat_count": len(rows),
        "estimated_seconds": round(sum(r["estimated_seconds"] or 0.0 for r in rows), 1),
        "removed_beats": sb["removed_beats"],
        "rendered": bool(info and info.get("ok")),
        "last_render": {k: info.get(k) for k in ("quality", "ok", "only", "rendered_at", "beats_completed")} if info else None,
        "last_defects": defects.get("counts") if isinstance(defects, dict) else None,
        "final_path": _rel(ctx, film_dir / FINAL_FILENAME) if (film_dir / FINAL_FILENAME).is_file() else None,
    }


# ----------------------------------------------------------------------------- diagrams and widgets
def diagram_render(ctx: ToolContext, a: dict[str, Any]) -> dict[str, Any]:
    diagram_id = _ident(a["diagram_id"], "diagram_id")
    svg_text = str(a["svg"])
    v = validate_svg(svg_text)
    if not v.ok:
        raise _fail(
            "invalid_input",
            f"the SVG for {diagram_id!r} was rejected: " + "; ".join(v.errors),
            ErrorCategory.invalid_input,
            "Fix the listed problems and resend the whole SVG. Do not repeat the unchanged call.",
            errors=v.errors,
            warnings=v.warnings,
        )
    svg_path = _media_dir(ctx, "diagrams", f"{diagram_id}.svg")
    ensure_dir(svg_path.parent)
    svg_path.write_text(svg_text, encoding="utf-8")
    chromium = _chromium(ctx)
    defects = static_defects(v, include_small_text=chromium is None)
    limitations: list[str] = []
    png_path: str | None = None
    size: dict[str, int] | None = None
    if chromium:
        try:
            rendered = render_svg_png(svg_text, svg_path.with_suffix(".png"), width=DIAGRAM_WIDTH_PX, chromium=chromium, runner=_runner(ctx), out_dir=_pagelab_dir(ctx, f"diagram-{diagram_id}"), timeout_s=DIAGRAM_TIMEOUT_S, validation=v)
        except ToolFailure as e:
            e.error.partial = {"svg_path": _rel(ctx, svg_path), "validation_defects": defects}  # the file is saved; only the picture is missing
            raise
        png_path, size = rendered["png_path"], {"width": rendered["width"], "height": rendered["height"]}
        if rendered["report"] is None:
            defects.append({"kind": "no_report", "severity": "error", "detail": "; ".join(rendered["errors"]) or "the page lab produced no report"})
        else:
            defects += report_defects(rendered["report"], DIAGRAM_SEVERITIES)
    else:
        limitations.append("chromium is not installed: the SVG was validated but not rendered or measured. " + INSTALL_HINTS["chromium"])
    counts = _defect_counts(defects)
    out: dict[str, Any] = {
        "diagram_id": diagram_id,
        "svg_path": _rel(ctx, svg_path),
        "png_path": _rel(ctx, png_path),
        "size": size,
        "validation": {"text_count": v.text_count, "font_sizes": sorted(set(v.font_sizes)), "view_box": v.view_box, "width": v.width, "height": v.height, "element_count": v.element_count},
        "defects": defects,
        "defect_counts": counts,
        "limitations": limitations,
        "ok": counts["error"] == 0,
        "untrusted": True,
    }
    if png_path:
        out["_attach"] = [{"type": "image", "path": png_path, "text": f"[diagram {diagram_id} rendered at {size['width']}x{size['height']}]" if size else f"[diagram {diagram_id}]"}]
    return out


def build_widget_document(html: str, css: str = "", js: str = "", title: str = "") -> str:
    """A self-contained page: the house theme first (so the widget's CSS overrides it), then the content and script."""
    theme = f'<style id="edgent-theme">{HOUSE_THEME_CSS}</style>'
    extra_css = f"<style>{css}</style>" if css.strip() else ""
    extra_js = f"<script>{js}</script>" if js.strip() else ""
    if re.search(r"<html[\s>]", html, re.IGNORECASE):
        doc = html
        m = re.search(r"<head[^>]*>", doc, re.IGNORECASE)
        if m:
            doc = doc[: m.end()] + theme + extra_css + doc[m.end() :]
        else:
            m = re.search(r"<html[^>]*>", doc, re.IGNORECASE)
            assert m is not None
            doc = doc[: m.end()] + "<head>" + theme + extra_css + "</head>" + doc[m.end() :]
        if extra_js:
            m = re.search(r"</body>", doc, re.IGNORECASE)
            doc = doc[: m.start()] + extra_js + doc[m.start() :] if m else doc + extra_js
        return doc
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html_lib.escape(title)}</title>\n{theme}\n{extra_css}\n</head>\n<body>\n{html}\n{extra_js}\n</body>\n</html>\n"
    )


def widget_run(ctx: ToolContext, a: dict[str, Any]) -> dict[str, Any]:
    widget_id = _ident(a["widget_id"], "widget_id")
    title = str(a.get("title") or widget_id)
    doc = build_widget_document(str(a["html"]), str(a.get("css") or ""), str(a.get("js") or ""), title)
    html_path = _media_dir(ctx, "widgets", f"{widget_id}.html")
    ensure_dir(html_path.parent)
    html_path.write_text(doc, encoding="utf-8")
    chromium = _chromium(ctx)
    if not chromium:
        e = chromium_unavailable().error
        raise ToolFailure("unavailable", f"{e.message} The widget was saved at {_rel(ctx, html_path)} but not run.", ErrorCategory.unavailable, remediation=e.remediation, partial={"html_path": _rel(ctx, html_path)}, toolchain="chromium")
    width, height = WIDGET_VIEWPORT_PX
    page = run_page(doc, _pagelab_dir(ctx, f"widget-{widget_id}"), width=width, height=height, timeout_s=WIDGET_TIMEOUT_S, screenshot=True, chromium=chromium, runner=_runner(ctx), name="widget", title=title)
    screenshot: str | None = None
    if page.screenshot_path:
        target = html_path.with_suffix(".png")
        shutil.copyfile(page.screenshot_path, target)
        screenshot = str(target)
    report = page.report or {}
    measurements = report.get("measurements") if isinstance(report.get("measurements"), dict) else {}
    defects = report_defects(page.report)
    if page.report is None:
        defects.append({"kind": "no_report", "severity": "error", "detail": "; ".join(page.errors) or "the page lab produced no report"})
    elif not report.get("tests_ran"):
        defects.append({"kind": "no_self_test", "severity": "warn", "detail": "the widget defines no window.__edgentTest; add one that exercises the controls and returns a list of failure strings"})
    counts = _defect_counts(defects)
    out: dict[str, Any] = {
        "widget_id": widget_id,
        "html_path": _rel(ctx, html_path),
        "screenshot_path": _rel(ctx, screenshot),
        "console_errors": list(report.get("console_errors") or []),
        "test_failures": list(report.get("test_failures") or []),
        "tests_ran": bool(report.get("tests_ran")),
        "measurements": {k: list(measurements.get(k) or []) for k in MEASUREMENT_KEYS},
        "counts": report.get("counts"),
        "document": report.get("document"),
        "viewport": {"width": width, "height": height},
        "defects": defects,
        "defect_counts": counts,
        "run_errors": list(page.errors),
        "ok": counts["error"] == 0,
        "untrusted": True,
    }
    if screenshot:
        out["_attach"] = [{"type": "image", "path": screenshot, "text": f"[widget {widget_id} screenshot at {width}x{height}]"}]
    return out


# ----------------------------------------------------------------------------- narration and assembly
def tts_narrate(ctx: ToolContext, a: dict[str, Any]) -> dict[str, Any]:
    text = str(a["text"])
    out_name = _ident(a["out_name"], "out_name")
    voice = a.get("voice") or None
    out = _media_dir(ctx, "audio", f"{out_name}.wav")
    result = synthesize(text, out, voice=voice, toolchain=_toolchain(ctx), runner=_runner(ctx), which=ctx.extra.get("which"))
    words = narration_words(text)
    return {
        "path": _rel(ctx, result["path"]),
        "backend": result["backend"],
        "voice": result.get("voice"),
        "duration_s": result.get("duration_s"),
        "words": words,
        "estimated_seconds": round(words / WORDS_PER_SECOND, 1),
    }


def storyboard_captions(sb: dict[str, Any], report: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Caption segments: each beat's narration over the span the last render measured for it."""
    if not report or not report.get("beats"):
        raise _fail("invalid_input", "captions='storyboard' needs a render report to time the narration by, and there is none.", ErrorCategory.invalid_input, "Run film_render first, or pass captions as segments [{start_s, end_s, text}].")
    narration = {b["id"]: b["narration"] for b in normalise(sb)["beats"]}
    segments = []
    for entry in report["beats"]:
        if not isinstance(entry, dict):
            continue
        text = narration.get(entry.get("id"), "")
        if text.strip():
            segments.append({"start_s": entry.get("start_s") or 0.0, "end_s": entry.get("end_s") or 0.0, "text": text})
    return segments


def video_assemble(ctx: ToolContext, a: dict[str, Any]) -> dict[str, Any]:
    film_id = _ident(a["film_id"], "film_id")
    film_dir = _media_dir(ctx, "films", film_id)
    video = film_dir / VIDEO_FILENAME
    if not video.is_file():
        raise _fail("unavailable", f"film {film_id!r} has no render to assemble.", ErrorCategory.unavailable, "Run film_render (quality 'final' for the deliverable) first; without manim the film stays unrendered.", film_id=film_id)
    narration = a.get("narration")
    audio = _existing(ctx, str(narration), "narration audio") if narration else None
    captions = a.get("captions")
    captions_arg: Any = None
    source: str | None = None
    if captions == "storyboard":
        report = read_json(film_dir / REPORT_FILENAME) if (film_dir / REPORT_FILENAME).is_file() else None
        captions_arg, source = storyboard_captions(_load_storyboard(ctx, film_dir, film_id), report), "storyboard"
    elif isinstance(captions, str):
        captions_arg, source = _existing(ctx, captions, "captions file"), "file"
    elif isinstance(captions, list):
        captions_arg, source = captions, "segments"
    info = assemble(video, audio, captions_arg, film_dir / FINAL_FILENAME, toolchain=_toolchain(ctx), runner=_runner(ctx))
    return {
        "film_id": film_id,
        "path": _rel(ctx, info["path"]),
        "duration_s": info["duration_s"],
        "has_audio": info["has_audio"],
        "has_captions": info["has_captions"],
        "captions_path": _rel(ctx, info["captions_path"]),
        "captions_source": source,
    }


# ----------------------------------------------------------------------------- registry
def _spec(
    name: str,
    fn: Any,
    description: str,
    parameters: dict[str, Any],
    *,
    permission: str,
    when_not_to_use: str,
    side_effects: bool = False,
    idempotent: bool = True,
    timeout_s: float = 60.0,
    cost_class: str = "cheap",
    attaches_media: bool = False,
    examples: list[dict[str, Any]] | None = None,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        parameters=parameters,
        fn=fn,
        permission=permission,
        timeout_s=timeout_s,
        idempotent=idempotent,
        side_effects=side_effects,
        cost_class=cost_class,
        when_not_to_use=when_not_to_use,
        examples=list(examples or []),
        group="media",
        attaches_media=attaches_media,
    )


_FILM_ID = opt(S, "Short film id (letters, digits, '-' or '_'): the folder media/films/<film_id>/.")
_STORYBOARD = opt(
    {"type": "object"},
    "The whole storyboard: {title, persona, prelude?, show_titles?, beats:[{id?, title, narration, code, seconds?}]}. It replaces the saved one. "
    "code is the body of a Manim method with self as the Scene; the prelude gives stage(group), focus(group, keep), unfocus(group), "
    "clamp_to_safe(m), the sizes TITLE/BODY/LABEL/CAPTION and the colours INK/MUTED/DIM/ACCENT/ACCENT_2 on a dark background.",
)
_OPS = opt(
    arr({"type": "object"}),
    "Edits applied in order to the saved (or given) storyboard: add_beat {after?, title, narration, code, seconds?}, "
    "set_beat {beat, title?, narration?, code?, seconds?}, patch_beat {beat, find, replace} (find must occur once), "
    "move_beat {beat, after} (after 0 = front), remove_beat {beat, reason}, prelude {py, why}, set_meta {name, value}. "
    "A rejected op is reported with its reason and changes nothing.",
)


def media_specs() -> list[ToolSpec]:
    """Every media tool's contract; ``build_media_registry`` registers them."""
    return [
        _spec(
            "film_compile", film_compile,
            "Create or edit a film's storyboard and compile it to a Manim scene (media/films/<id>/scene.py). Send the whole "
            "storyboard to start, then ops for every change; each beat's code is syntax-checked and linted and the outline "
            "comes back with target seconds. Nothing is rendered here.",
            obj({"film_id": _FILM_ID, "storyboard": _STORYBOARD, "ops": _OPS}, ["film_id"]),
            permission="write",
            when_not_to_use="you want to see the film (film_render, then film_look), or you only need the outline (film_outline).",
            side_effects=True,
            idempotent=False,
            examples=[{"film_id": "bayes-intro", "ops": [{"op": "patch_beat", "beat": 2, "find": "Square(side_length=2)", "replace": "Square(side_length=1.5)"}]}],
        ),
        _spec(
            "film_render", film_render,
            "Render a compiled film with Manim in the sandbox and measure it by code: per-beat timings, text overlaps, anything "
            "outside the safe frame, fonts below the floor, still-card beats and durations that do not match the narration. "
            "Use quality 'draft' while iterating. Without manim this returns unavailable and the film stays unrendered: carry on.",
            obj({
                "film_id": _FILM_ID,
                "quality": opt({"type": "string", "enum": list(QUALITY_FLAGS)}, "draft (480p, fastest) while fixing defects; medium (720p); final (1080p) for the deliverable."),
                "only": opt(arr(I), "Render only these beat ids, a quick check of one idea; mobjects from skipped beats are absent. The partial scene goes to scene_only.py; scene.py stays the whole film."),
            }, ["film_id"]),
            permission="compute",
            when_not_to_use="the storyboard has not compiled cleanly yet (film_compile reports the errors first).",
            idempotent=False,
            timeout_s=FILM_RENDER_TIMEOUT_LIMIT.value + REGISTRY_TIMEOUT_MARGIN_S,
            cost_class="expensive",
            examples=[{"film_id": "bayes-intro", "quality": "draft"}],
        ),
        _spec(
            "film_look", film_look,
            "Look at the last render: one frame from the middle of each beat, attached as images, so you can judge what "
            "the measurements cannot (composition, emphasis, whether the picture explains the idea).",
            obj({
                "film_id": _FILM_ID,
                "beats": opt(arr(I), "Beat ids to look at (default: all, in order)."),
                "max_frames": opt(I, f"Frames to attach (default {FILM_LOOK_FRAMES_DEFAULT}, capped at {FILM_LOOK_FRAMES_LIMIT.value:g}).", minimum=1),
            }, ["film_id"]),
            permission="read",
            when_not_to_use="the film has not been rendered (film_render first) or the defects list already tells you what to fix.",
            timeout_s=120.0,
            cost_class="moderate",
            attaches_media=True,
            examples=[{"film_id": "bayes-intro", "beats": [2, 3]}],
        ),
        _spec(
            "film_outline", film_outline,
            "The film's outline: beats with ids, titles, target seconds and narration length, plus whether it was rendered "
            "and how many defects the last measurement found. Read it before editing with ops.",
            obj({"film_id": _FILM_ID}, ["film_id"]),
            permission="read",
            when_not_to_use="you need the code of a beat (read_file on media/films/<id>/storyboard.json).",
            examples=[{"film_id": "bayes-intro"}],
        ),
        _spec(
            "diagram_render", diagram_render,
            "Save an SVG diagram under media/diagrams/<id>.svg, validate it (well-formed, svg namespace, viewBox, no scripts, "
            "labels at 12 px or larger), rasterise it in Chromium on a white page the size of its viewBox, measure label "
            "overlaps, clipping, small text and contrast, and attach the picture. Without Chromium it validates only.",
            obj({
                "diagram_id": opt(S, "Short diagram id (letters, digits, '-' or '_')."),
                "svg": opt(S, 'The complete SVG with xmlns="http://www.w3.org/2000/svg" and a viewBox; text labels 12 px or larger; no <script>.'),
            }, ["diagram_id", "svg"]),
            permission="write",
            when_not_to_use="the figure is interactive (widget_run) or moves (a film).",
            side_effects=True,
            idempotent=False,
            timeout_s=DIAGRAM_TIMEOUT_S * 2 + REGISTRY_TIMEOUT_MARGIN_S,
            cost_class="moderate",
            attaches_media=True,
            examples=[{"diagram_id": "bayes-grid", "svg": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 200"><text x="10" y="30" font-size="16">prior</text></svg>'}],
        ),
        _spec(
            "widget_run", widget_run,
            "Save a self-contained interactive widget (media/widgets/<id>.html, house theme included) and run it in headless "
            "Chromium with no network: its window.__edgentTest self-test, console errors, elements overflowing the viewport, "
            "text overlaps, text under 12 px and contrast under 4.5:1, plus a screenshot attached as an image.",
            obj({
                "widget_id": opt(S, "Short widget id (letters, digits, '-' or '_')."),
                "html": opt(S, "Body markup (a full document is accepted too)."),
                "css": opt(S, "Extra stylesheet; the house theme is applied first and provides --bg, --ink, --accent, .panel and .controls."),
                "js": opt(S, "Script run after the markup. Define window.__edgentTest returning a list of failure strings (empty when all pass); it may return a promise."),
                "title": opt(S, "Page title (default: the widget id)."),
            }, ["widget_id", "html"]),
            permission="write",
            when_not_to_use="the content is static (diagram_render) or needs a server or external scripts (not allowed; widgets are offline).",
            side_effects=True,
            idempotent=False,
            timeout_s=WIDGET_TOOL_TIMEOUT_S,
            cost_class="moderate",
            attaches_media=True,
            examples=[{"widget_id": "coin-flip", "html": "<div class=\"panel\"><button id=\"flip\">Flip</button> <output id=\"out\">?</output></div>", "js": "const o=document.getElementById('out');document.getElementById('flip').onclick=()=>{o.textContent=Math.random()<0.5?'H':'T'};window.__edgentTest=()=>{document.getElementById('flip').click();return ['H','T'].includes(o.textContent)?[]:['flip did not update the output']}"}],
        ),
        _spec(
            "tts_narrate", tts_narrate,
            "Speak narration text into media/audio/<out_name>.wav with the installed on-device synthesiser (piper, say, "
            "espeak-ng or espeak, in that order). Returns the backend and the duration so you can compare it with the beat.",
            obj({
                "text": opt(S, "The narration to speak (plain text, no markup)."),
                "out_name": opt(S, "Audio file name without extension (letters, digits, '-' or '_')."),
                "voice": opt(S, "Optional voice: a piper .onnx model path, a `say` voice name, or an espeak voice such as en-us."),
            }, ["text", "out_name"]),
            permission="compute",
            when_not_to_use="no synthesiser is installed (the error lists the options; keep the narration as captions meanwhile).",
            idempotent=False,
            timeout_s=TTS_TOOL_TIMEOUT_S,
            cost_class="moderate",
            examples=[{"text": "A right triangle has one right angle.", "out_name": "pythagoras-beat-1"}],
        ),
        _spec(
            "video_assemble", video_assemble,
            "Assemble the film's last render with optional narration audio and captions into media/films/<id>/final.mp4 "
            "(ffmpeg; video copied, audio AAC padded to the picture, captions as a soft subtitle track).",
            obj({
                "film_id": _FILM_ID,
                "narration": opt(S, "Workspace path of the narration audio (the tts_narrate output)."),
                "captions": opt(
                    {"anyOf": [S, arr(obj({"start_s": N, "end_s": N, "text": S}, ["start_s", "end_s", "text"]))]},
                    "An .srt path, the word 'storyboard' (each beat's narration over the span the last render measured), or segments [{start_s, end_s, text}].",
                ),
            }, ["film_id"]),
            permission="compute",
            when_not_to_use="the film has not been rendered at the quality you want to ship (film_render with quality 'final' first).",
            idempotent=False,
            timeout_s=ASSEMBLE_TOOL_TIMEOUT_S,
            cost_class="moderate",
            examples=[{"film_id": "bayes-intro", "narration": "media/audio/bayes-intro.wav", "captions": "storyboard"}],
        ),
    ]


def build_media_registry(registry: ToolRegistry | None = None, include: set[str] | None = None) -> ToolRegistry:
    """Register the media tools (group 'media') on ``registry`` (a new one when None); ``include`` selects a subset by name."""
    reg = registry if registry is not None else ToolRegistry()
    if include is not None:
        unknown = sorted(set(include) - set(MEDIA_TOOL_NAMES))
        if unknown:
            raise ValueError(f"unknown media tools {unknown}; choose from {list(MEDIA_TOOL_NAMES)}")
    for spec in media_specs():
        if include is not None and spec.name not in include:
            continue
        if reg.get(spec.name) is None:
            reg.register(spec)
    return reg


__all__ = ["HOUSE_THEME_CSS", "MEDIA_TOOL_NAMES", "build_media_registry", "build_widget_document", "media_specs", "storyboard_captions"]
