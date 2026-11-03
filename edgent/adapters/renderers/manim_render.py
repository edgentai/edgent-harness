"""Render a compiled scene with Manim in the sandbox, read its self-report, measure it, pull frames.

``render`` writes ``scene.py`` into the film directory and runs ``manim`` with the network off, a generous
address-space ceiling and the named timeout; the prelude inside the scene writes ``report.json`` (per-beat
timings, animation counts and the bounding boxes, font sizes and text of what was on screen). ``measure``
turns that report into defects by code: overlapping text, anything outside the safe frame, fonts below the
floor, still-card beats and beats whose duration does not match their narration. ``frames`` extracts stills
for eyes. Manim is never imported here; when it is missing, ``render`` raises the typed ``unavailable`` error.
"""
from __future__ import annotations

import math
import shutil
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...domain.errors import ErrorCategory, Limit, ToolFailure
from ...domain.utils import ensure_dir, now_iso, read_json, write_json
from ..sandbox import SandboxResult, run_subprocess
from ..toolchain import Toolchain
from . import av
from .manim_compiler import SCENE_CLASS, WORDS_PER_SECOND, narration_words, normalise
from .manim_prelude import FONT_FLOOR, REPORT_ENV, SAFE_MARGIN_UNITS

QUALITY_FLAGS: dict[str, str] = {"draft": "l", "medium": "m", "final": "h"}
FILM_RENDER_TIMEOUT_LIMIT = Limit("film render timeout", 1800, "s", floor=30, does_not_raise=("the registry timeout, which follows it",))
MANIM_MEMORY_BYTES = 8 << 30  # address-space ceiling for the manim process (numpy, cairo, pango and its ffmpeg child)
RENDER_LOG_MAX_CHARS = 2_000_000  # captured manim output; the sandbox keeps the head, and a traceback sits at the end
RENDER_LOG_TAIL_CHARS = 4000
OUTPUT_BASENAME = "film"
SCENE_FILENAME = "scene.py"
ONLY_SCENE_FILENAME = "scene_only.py"  # a partial (only=[...]) render, so scene.py always holds the whole film
REPORT_FILENAME = "report.json"
RENDER_INFO_FILENAME = "render.json"
VIDEO_FILENAME = "out.mp4"
FRAMES_DIRNAME = "frames"
DEFAULT_FRAME_WIDTH_UNITS = 14.222222222222221
DEFAULT_FRAME_HEIGHT_UNITS = 8.0
REFERENCE_PIXEL_HEIGHT = 1080
# Manim's Text renders font_size/4.8 pt through Pango and scales the SVG by 0.05: one unit of font size is
# 0.0104 scene units of em, 1.406 px at 1080p. The shortest single line at the floor (x-height only) is half an em.
MANIM_EM_PX_PER_FONT_SIZE = 1.40625
TEXT_HEIGHT_FLOOR_PX = round(FONT_FLOOR * MANIM_EM_PX_PER_FONT_SIZE * 0.5, 1)
TEXT_OVERLAP_TOLERANCE_UNITS = 0.02
SAFE_FRAME_TOLERANCE_UNITS = 0.05
VISIBLE_OPACITY_MIN = 0.1
STILL_CARD_MIN_S = 0.5
SECONDS_TOLERANCE_FRACTION = 0.25
SECONDS_TOLERANCE_MIN_S = 1.5
NARRATION_TOLERANCE_FRACTION = 0.35
NARRATION_TOLERANCE_MIN_S = 2.0
MAX_DEFECTS_PER_BEAT = 40


@dataclass
class RenderResult:
    ok: bool
    video_path: str | None
    report_path: str | None
    report: dict[str, Any] | None
    log: str
    duration_s: float
    scene_path: str
    returncode: int
    timed_out: bool = False
    quality: str = "draft"
    only: list[int] | None = None
    argv: list[str] = field(default_factory=list)

    def beats_completed(self) -> list[int]:
        return [int(b["id"]) for b in (self.report or {}).get("beats") or [] if isinstance(b, dict) and isinstance(b.get("id"), int)]

    def failed_beat(self) -> dict[str, Any] | None:
        for b in (self.report or {}).get("beats") or []:
            if isinstance(b, dict) and b.get("error"):
                return {"beat": b.get("id"), "title": b.get("title"), "error": b.get("error")}
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "video_path": self.video_path,
            "report_path": self.report_path,
            "log": self.log,
            "duration_s": round(self.duration_s, 3),
            "scene_path": self.scene_path,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "quality": self.quality,
            "only": self.only,
            "beats_completed": self.beats_completed(),
        }


# ----------------------------------------------------------------------------- render
def manim_command(toolchain: Toolchain) -> list[str]:
    """The manim CLI, or ``python -m manim`` when only the module is installed; typed unavailable otherwise."""
    exe = toolchain.path("manim")
    if exe:
        return [exe]
    if toolchain.modules.get("manim_py"):
        return [sys.executable, "-m", "manim"]
    toolchain.require("manim", "rendering films")
    raise AssertionError("require() returns only when manim is present")  # pragma: no cover


def locate_video(media_dir: str | Path, since: float | None = None) -> Path | None:
    """The final mp4 under manim's media dir (partial movie files are never the answer; with ``since``, only one written after that time)."""
    candidates = [p for p in Path(media_dir).glob("videos/**/*.mp4") if "partial_movie_files" not in p.parts and p.is_file()]
    if since is not None:
        candidates = [p for p in candidates if p.stat().st_mtime >= since - 1.0]  # a second of slack for coarse file clocks
    named = [p for p in candidates if p.stem == OUTPUT_BASENAME]
    pool = named or candidates
    return max(pool, key=lambda p: p.stat().st_mtime) if pool else None


def _tail(text: str, chars: int = RENDER_LOG_TAIL_CHARS) -> str:
    text = (text or "").strip()
    return text if len(text) <= chars else "…" + text[-chars:]


def _timeout(timeout_s: float | None) -> float:
    if timeout_s is None:
        return float(FILM_RENDER_TIMEOUT_LIMIT.value)
    return float(min(max(timeout_s, FILM_RENDER_TIMEOUT_LIMIT.floor or 0), FILM_RENDER_TIMEOUT_LIMIT.value))


def render(
    film_dir: str | Path,
    scene_source: str,
    quality: str = "draft",
    only: Iterable[int] | None = None,
    toolchain: Toolchain | None = None,
    timeout_s: float | None = None,
    runner: Any = None,
    scene_filename: str = SCENE_FILENAME,
) -> RenderResult:
    """Write the scene file and render ``GeneratedFilm``; the result carries the report even when the render failed.

    A render starts by removing the previous report and video, so a failure never leaves an older film behind
    labelled as this one. ``scene_filename`` lets a partial (``only``) render keep ``scene.py`` as the whole film.
    """
    if quality not in QUALITY_FLAGS:
        raise ToolFailure("invalid_input", f"quality must be one of {list(QUALITY_FLAGS)}, got {quality!r}", ErrorCategory.invalid_input, remediation="Use 'draft' while iterating and 'final' once the measurements pass.")
    tc = toolchain if toolchain is not None else Toolchain.detect()
    command = manim_command(tc)
    film = Path(ensure_dir(film_dir)).resolve()  # the sandbox pins HOME and TMPDIR to the cwd, which must be absolute
    scene_path = film / scene_filename
    scene_path.write_text(scene_source, encoding="utf-8")
    for stale in (film / REPORT_FILENAME, film / VIDEO_FILENAME):
        if stale.exists():
            stale.unlink()
    report_path = film / REPORT_FILENAME
    media_dir = Path(ensure_dir(film / "media"))
    timeout = _timeout(timeout_s)
    argv = [*command, "-q", QUALITY_FLAGS[quality], "--progress_bar", "none", "--media_dir", str(media_dir), "-o", OUTPUT_BASENAME, scene_filename, SCENE_CLASS]
    run = runner or run_subprocess
    started_at = time.time()
    r: SandboxResult = run(
        argv,
        str(film),
        timeout_s=timeout,
        cpu_seconds=int(math.ceil(timeout)),
        memory_bytes=MANIM_MEMORY_BYTES,
        allow_network=False,
        env_extra={REPORT_ENV: str(report_path)},
        max_output=RENDER_LOG_MAX_CHARS,
    )
    report = read_json(report_path) if report_path.is_file() else None
    if report is not None and not isinstance(report, dict):
        report = None
    produced = locate_video(media_dir, since=started_at)
    video_path: str | None = None
    if produced is not None and r.ok:
        target = film / VIDEO_FILENAME
        shutil.copyfile(produced, target)
        video_path = str(target)
    only_list = sorted({int(i) for i in only}) if only is not None else None
    result = RenderResult(
        ok=r.ok and video_path is not None,
        video_path=video_path,
        report_path=str(report_path) if report is not None else None,
        report=report,
        log=_tail((r.stderr or "") + ("\n" + r.stdout if r.stdout else "")),
        duration_s=r.duration_s,
        scene_path=str(scene_path),
        returncode=r.returncode,
        timed_out=r.timed_out,
        quality=quality,
        only=only_list,
        argv=list(argv),
    )
    write_json(film / RENDER_INFO_FILENAME, {**result.to_dict(), "rendered_at": now_iso()})
    return result


def last_render(film_dir: str | Path) -> dict[str, Any] | None:
    """The sidecar written by the last ``render`` of this film, or None when it was never rendered."""
    info = read_json(Path(film_dir) / RENDER_INFO_FILENAME)
    return info if isinstance(info, dict) else None


# ----------------------------------------------------------------------------- measure
def _safe_frame(config: dict[str, Any]) -> dict[str, float]:
    safe = config.get("safe") if isinstance(config.get("safe"), dict) else {}
    fw = float(config.get("frame_width") or DEFAULT_FRAME_WIDTH_UNITS)
    fh = float(config.get("frame_height") or DEFAULT_FRAME_HEIGHT_UNITS)
    margin = float(safe.get("margin", SAFE_MARGIN_UNITS))
    return {
        "left": float(safe.get("left", -fw / 2 + margin)),
        "right": float(safe.get("right", fw / 2 - margin)),
        "top": float(safe.get("top", fh / 2 - margin)),
        "bottom": float(safe.get("bottom", -fh / 2 + margin)),
        "margin": margin,
        "frame_height": fh,
    }


def _bbox(m: dict[str, Any]) -> tuple[float, float, float, float] | None:
    box = m.get("bbox")
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    return x0, y0, x1, y1


def _visible(m: dict[str, Any]) -> bool:
    try:
        return float(m.get("opacity", 1.0)) >= VISIBLE_OPACITY_MIN and _bbox(m) is not None
    except (TypeError, ValueError):
        return False


def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float]:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w, h


def _label(m: dict[str, Any]) -> str:
    text = " ".join(str(m.get("text") or "").split())
    if text:
        return f"{m.get('type', 'text')} {text[:40]!r}"
    return str(m.get("type") or m.get("kind") or "mobject")


def _outside(box: tuple[float, float, float, float], safe: dict[str, float]) -> list[str]:
    tol = SAFE_FRAME_TOLERANCE_UNITS
    sides = []
    if box[0] < safe["left"] - tol:
        sides.append("left")
    if box[2] > safe["right"] + tol:
        sides.append("right")
    if box[3] > safe["top"] + tol:
        sides.append("top")
    if box[1] < safe["bottom"] - tol:
        sides.append("bottom")
    return sides


def _measure_beat(entry: dict[str, Any], beat: dict[str, Any], safe: dict[str, float], floor: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    bid = entry.get("id")

    def add(kind: str, severity: str, detail: str, **extra: Any) -> None:
        out.append({"beat": bid, "kind": kind, "severity": severity, "detail": detail, **extra})

    if entry.get("error"):
        add("beat_error", "error", f"the beat raised {entry['error']}")
    try:
        duration = float(entry.get("duration_s") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    plays = int(entry.get("plays") or 0)
    if plays == 0:
        severity = "error" if duration >= STILL_CARD_MIN_S else "warn"
        add("still_card", severity, f"no self.play in {duration:.1f} s: a still card. Animate the idea (Create, Transform, Indicate) or merge the beat into its neighbour", duration_s=duration)
    target = beat.get("seconds")
    if target is not None:
        target = float(target)
        tol = max(SECONDS_TOLERANCE_MIN_S, SECONDS_TOLERANCE_FRACTION * target)
        if abs(duration - target) > tol:
            add("duration_mismatch", "warn", f"the beat ran {duration:.1f} s; the storyboard targets {target:g} s (tolerance {tol:.1f} s)", observed_s=duration, target_s=target, source="seconds")
    else:
        words = narration_words(str(beat.get("narration") or ""))
        if words:
            estimate = words / WORDS_PER_SECOND
            tol = max(NARRATION_TOLERANCE_MIN_S, NARRATION_TOLERANCE_FRACTION * estimate)
            if abs(duration - estimate) > tol:
                add("duration_mismatch", "warn", f"the beat ran {duration:.1f} s but its narration ({words} words) takes about {estimate:.1f} s at {WORDS_PER_SECOND} words/s (tolerance {tol:.1f} s)", observed_s=duration, target_s=round(estimate, 1), source="narration")
    mobs = [m for m in entry.get("mobjects") or [] if isinstance(m, dict) and _visible(m)]
    texts = [m for m in mobs if m.get("kind") == "text"]
    for i, a in enumerate(texts):
        box_a = _bbox(a)
        for b in texts[i + 1 :]:
            box_b = _bbox(b)
            if box_a is None or box_b is None:
                continue
            w, h = _overlap(box_a, box_b)
            if w > TEXT_OVERLAP_TOLERANCE_UNITS and h > TEXT_OVERLAP_TOLERANCE_UNITS:
                add("text_overlap", "error", f"{_label(a)} overlaps {_label(b)} by {w:.2f} x {h:.2f} units", a=a.get("text"), b=b.get("text"))
    for m in mobs:
        box = _bbox(m)
        if box is None:
            continue
        sides = _outside(box, safe)
        if sides:
            severity = "error" if m.get("kind") == "text" else "warn"
            add("outside_safe_frame", severity, f"{_label(m)} crosses the safe frame on the {', '.join(sides)} (margin {safe['margin']:g} units)", sides=sides)
    px_per_unit = REFERENCE_PIXEL_HEIGHT / safe["frame_height"]
    for m in texts:
        box = _bbox(m)
        size = m.get("font_size")
        if isinstance(size, (int, float)) and not isinstance(size, bool) and size < floor:
            add("font_below_floor", "error", f"{_label(m)} uses font size {size:g}; the floor is {floor:g} (CAPTION)", font_size=size)
            continue
        if box is not None:
            height_px = (box[3] - box[1]) * px_per_unit
            if 0 < height_px < TEXT_HEIGHT_FLOOR_PX:
                add("text_too_small", "warn", f"{_label(m)} renders about {height_px:.0f} px tall at 1080p; it was scaled below the floor ({TEXT_HEIGHT_FLOOR_PX:g} px)", height_px=round(height_px, 1))
    return out


def measure(report: dict[str, Any] | None, storyboard: dict[str, Any], only: Iterable[int] | None = None) -> list[dict[str, Any]]:
    """Defects found in a render report by code. Each: ``{"beat", "kind", "severity", "detail", ...}``.

    Kinds: ``beat_error``, ``still_card``, ``duration_mismatch``, ``text_overlap``, ``outside_safe_frame``,
    ``font_below_floor``, ``text_too_small``, ``missing_beat``, ``render_incomplete``, ``no_report``.
    """
    sb = normalise(storyboard)
    wanted = None if only is None else {int(i) for i in only}
    beats_by_id = {b["id"]: b for b in sb["beats"]}
    expected = [b["id"] for b in sb["beats"] if wanted is None or b["id"] in wanted]
    if not report:
        return [{"beat": None, "kind": "no_report", "severity": "error", "detail": "the render produced no report; the scene did not reach its first beat (see the log)"}]
    config = report.get("config") if isinstance(report.get("config"), dict) else {}
    safe = _safe_frame(config)
    floor = float(config.get("font_floor") or FONT_FLOOR)
    defects: list[dict[str, Any]] = []
    seen: set[int] = set()
    for entry in report.get("beats") or []:
        if not isinstance(entry, dict):
            continue
        bid = entry.get("id")
        if isinstance(bid, int):
            seen.add(bid)
        defects.extend(_measure_beat(entry, beats_by_id.get(bid, {}), safe, floor)[:MAX_DEFECTS_PER_BEAT])
    for bid in expected:
        if bid not in seen:
            defects.append({"beat": bid, "kind": "missing_beat", "severity": "error", "detail": "the beat never ran (the render stopped before it)"})
    if not report.get("complete", False):
        defects.append({"beat": None, "kind": "render_incomplete", "severity": "error", "detail": "the scene did not finish; the report covers the beats that completed"})
    return defects


def beat_timings(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows = []
    for entry in (report or {}).get("beats") or []:
        if isinstance(entry, dict):
            rows.append({k: entry.get(k) for k in ("id", "title", "start_s", "end_s", "duration_s", "plays", "waits", "adds")})
    return rows


def beat_midpoints(report: dict[str, Any] | None, beats: Iterable[int] | None = None) -> list[dict[str, Any]]:
    """Where to look: the middle of each beat (``beats`` restricts the ids)."""
    wanted = None if beats is None else {int(b) for b in beats}
    out = []
    for entry in (report or {}).get("beats") or []:
        if not isinstance(entry, dict) or (wanted is not None and entry.get("id") not in wanted):
            continue
        try:
            start, end = float(entry.get("start_s") or 0.0), float(entry.get("end_s") or 0.0)
        except (TypeError, ValueError):
            continue
        mid = start + max(0.0, end - start) / 2
        out.append({"beat": entry.get("id"), "title": entry.get("title") or "", "time_s": round(mid, 3)})
    return out


def frames(video_path: str | Path, timestamps: Iterable[float], out_dir: str | Path, width: int = av.FRAME_WIDTH_PX_DEFAULT, toolchain: Toolchain | None = None, runner: Any = None) -> list[dict[str, Any]]:
    """Stills at the given times (seconds) via ffmpeg; see ``av.extract_frames``."""
    return av.extract_frames(video_path, timestamps, out_dir, width=width, toolchain=toolchain, runner=runner)


__all__ = [
    "FILM_RENDER_TIMEOUT_LIMIT",
    "MANIM_MEMORY_BYTES",
    "ONLY_SCENE_FILENAME",
    "QUALITY_FLAGS",
    "RENDER_LOG_MAX_CHARS",
    "REPORT_FILENAME",
    "RENDER_INFO_FILENAME",
    "TEXT_HEIGHT_FLOOR_PX",
    "VIDEO_FILENAME",
    "RenderResult",
    "beat_midpoints",
    "beat_timings",
    "frames",
    "last_render",
    "locate_video",
    "manim_command",
    "measure",
    "render",
]
