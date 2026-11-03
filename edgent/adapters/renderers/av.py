"""Audio and video assembly through ffmpeg: probe, frames, concat, mux, captions, assemble.

Everything runs through the sandbox runner with the network off. ``ffmpeg`` and ``ffprobe`` are optional
toolchains: when absent the caller gets the typed ``unavailable`` error with the install hint, never an
``OSError``. Tests inject a fake ``runner`` with the signature of ``run_subprocess``.
"""
from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from ...domain.errors import ErrorCategory, Limit, ToolFailure
from ...domain.utils import ensure_dir
from ..sandbox import SandboxResult, run_subprocess
from ..toolchain import Toolchain

FFMPEG_TIMEOUT_LIMIT = Limit("ffmpeg timeout", 600, "s", floor=5)
FFPROBE_TIMEOUT_S = 60.0
FFMPEG_MEMORY_BYTES = 2 << 30  # address-space ceiling for one ffmpeg process
FRAME_WIDTH_PX_DEFAULT = 640
FRAME_WIDTH_PX_MAX = 1920
FRAMES_PER_CALL_LIMIT = Limit("frames per call", 24, "frames")
CONCAT_INPUTS_LIMIT = Limit("concat inputs", 200, "files")
LOG_TAIL_CHARS = 1500
AUDIO_BITRATE = "160k"

Runner = Callable[..., SandboxResult]


# ----------------------------------------------------------------------------- plumbing
def _toolchain(toolchain: Toolchain | None) -> Toolchain:
    return toolchain if toolchain is not None else Toolchain.detect()


def _run(runner: Runner | None, argv: list[str], cwd: str | os.PathLike[str], timeout_s: float, stdin: bytes | None = None) -> SandboxResult:
    run = runner or run_subprocess
    cwd = str(Path(cwd).resolve())  # the sandbox pins TMPDIR and HOME to the cwd, so it must be absolute
    return run(argv, cwd, timeout_s=timeout_s, cpu_seconds=int(math.ceil(timeout_s)), memory_bytes=FFMPEG_MEMORY_BYTES, allow_network=False, stdin=stdin)


def _tail(text: str, chars: int = LOG_TAIL_CHARS) -> str:
    text = (text or "").strip()
    return text if len(text) <= chars else "…" + text[-chars:]


def _check(result: SandboxResult, what: str, timeout_s: float) -> None:
    if result.timed_out:
        raise ToolFailure(
            "timeout",
            f"ffmpeg did not finish {what} within {timeout_s:g} s. {FFMPEG_TIMEOUT_LIMIT.describe()}",
            ErrorCategory.timeout,
            remediation="Work on a shorter clip or fewer frames; do not repeat the unchanged call.",
            what=what,
        )
    if result.returncode != 0:
        raise ToolFailure(
            "ffmpeg_failed",
            f"ffmpeg failed {what} (exit {result.returncode}): {_tail(result.stderr) or 'no output'}",
            ErrorCategory.execution_failure,
            remediation="Read the ffmpeg message; check the input exists and is a media file (probe it), then change the input or the request.",
            what=what,
            returncode=result.returncode,
        )


def _existing(path: str | os.PathLike[str], what: str) -> Path:
    p = Path(path)
    if not p.is_file():
        raise ToolFailure("not_found", f"{what} not found: {p}", ErrorCategory.not_found, remediation="Check the path; render or synthesise the input first.", path=str(p))
    return p


def _timeout(timeout_s: float | None) -> float:
    if timeout_s is None:
        return float(FFMPEG_TIMEOUT_LIMIT.value)
    return float(min(max(timeout_s, FFMPEG_TIMEOUT_LIMIT.floor or 0), FFMPEG_TIMEOUT_LIMIT.value))


def _ratio(text: Any) -> float | None:
    if not isinstance(text, str) or "/" not in text:
        return None
    num, _, den = text.partition("/")
    try:
        n, d = float(num), float(den)
    except ValueError:
        return None
    return round(n / d, 3) if d else None


# ----------------------------------------------------------------------------- probe
def probe(path: str | os.PathLike[str], toolchain: Toolchain | None = None, runner: Runner | None = None) -> dict[str, Any]:
    """Duration, dimensions, frame rate and streams of a media file (ffprobe JSON, reduced to what matters)."""
    p = _existing(path, "media file")
    ffprobe = _toolchain(toolchain).require("ffprobe", "probing media")
    argv = [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(p)]
    r = _run(runner, argv, p.parent, FFPROBE_TIMEOUT_S)
    _check(r, f"probing {p.name}", FFPROBE_TIMEOUT_S)
    try:
        data = json.loads(r.stdout or "{}")
    except json.JSONDecodeError as e:
        raise ToolFailure("ffprobe_output", f"ffprobe returned no JSON for {p.name}: {e}", ErrorCategory.execution_failure, remediation="The file is probably not a media file; check it with `file` or re-create it.") from e
    fmt = data.get("format") or {}
    streams = [s for s in data.get("streams") or [] if isinstance(s, dict)]
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    duration = fmt.get("duration") or (video or {}).get("duration") or (audio or {}).get("duration") or 0
    try:
        duration_s = round(float(duration), 3)
    except (TypeError, ValueError):
        duration_s = 0.0
    return {
        "path": str(p),
        "duration_s": duration_s,
        "width": int(video["width"]) if video and video.get("width") else None,
        "height": int(video["height"]) if video and video.get("height") else None,
        "fps": _ratio((video or {}).get("avg_frame_rate")) or _ratio((video or {}).get("r_frame_rate")),
        "has_video": video is not None,
        "has_audio": audio is not None,
        "video_codec": (video or {}).get("codec_name"),
        "audio_codec": (audio or {}).get("codec_name"),
        "format": fmt.get("format_name"),
        "size_bytes": int(fmt.get("size") or 0),
        "streams": [{"type": s.get("codec_type"), "codec": s.get("codec_name")} for s in streams],
    }


# ----------------------------------------------------------------------------- frames
def extract_frames(
    video: str | os.PathLike[str],
    times: Iterable[float],
    out_dir: str | os.PathLike[str],
    width: int = FRAME_WIDTH_PX_DEFAULT,
    toolchain: Toolchain | None = None,
    runner: Runner | None = None,
    timeout_s: float | None = None,
) -> list[dict[str, Any]]:
    """One PNG per timestamp (seconds), scaled to ``width`` px; a time past the end yields ``path: None``."""
    v = _existing(video, "video")
    stamps = [float(t) for t in times]
    if any(t < 0 or math.isnan(t) for t in stamps):
        raise ToolFailure("invalid_input", "frame times must be non-negative seconds", ErrorCategory.invalid_input, remediation="Give times within the video's duration (probe it).")
    if len(stamps) > FRAMES_PER_CALL_LIMIT.value:
        err = FRAMES_PER_CALL_LIMIT.exceeded(len(stamps), "Ask for fewer frames per call.")
        raise ToolFailure(err.code, err.message, err.category, remediation=err.remediation)
    width = int(min(max(width, 64), FRAME_WIDTH_PX_MAX))
    ffmpeg = _toolchain(toolchain).require("ffmpeg", "extracting frames")
    out = Path(ensure_dir(out_dir))
    timeout = _timeout(timeout_s)
    frames: list[dict[str, Any]] = []
    for i, t in enumerate(stamps):
        target = out / f"frame_{i:02d}_{int(round(t * 1000)):07d}ms.png"
        if target.exists():
            target.unlink()
        argv = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t:.3f}", "-i", str(v), "-frames:v", "1", "-vf", f"scale={width}:-2", str(target)]
        r = _run(runner, argv, out, timeout)
        _check(r, f"extracting the frame at {t:.2f} s", timeout)
        if target.is_file() and target.stat().st_size > 0:
            frames.append({"time_s": round(t, 3), "path": str(target)})
        else:
            frames.append({"time_s": round(t, 3), "path": None, "note": "no frame at this time (past the end of the video?)"})
    return frames


# ----------------------------------------------------------------------------- concat and mux
def _concat_list(videos: list[Path], list_path: Path) -> None:
    lines = ["file '" + str(v).replace("'", "'\\''") + "'" for v in videos]
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def concat(videos: Iterable[str | os.PathLike[str]], out: str | os.PathLike[str], toolchain: Toolchain | None = None, runner: Runner | None = None, timeout_s: float | None = None) -> str:
    """Join clips in order into ``out``. Tries a stream copy first and re-encodes when the clips differ."""
    paths = [_existing(v, "video") for v in videos]
    if not paths:
        raise ToolFailure("invalid_input", "concat needs at least one video", ErrorCategory.invalid_input, remediation="Pass the clips in playback order.")
    if len(paths) > CONCAT_INPUTS_LIMIT.value:
        err = CONCAT_INPUTS_LIMIT.exceeded(len(paths), "Concatenate in batches.")
        raise ToolFailure(err.code, err.message, err.category, remediation=err.remediation)
    ffmpeg = _toolchain(toolchain).require("ffmpeg", "concatenating video")
    target = Path(out)
    ensure_dir(target.parent)
    list_path = target.with_name(target.stem + ".concat.txt")
    _concat_list(paths, list_path)
    timeout = _timeout(timeout_s)
    base = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path)]
    r = _run(runner, [*base, "-c", "copy", "-movflags", "+faststart", str(target)], target.parent, timeout)
    if r.returncode != 0 and not r.timed_out:
        r = _run(runner, [*base, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-movflags", "+faststart", str(target)], target.parent, timeout)
    _check(r, "concatenating clips", timeout)
    try:
        list_path.unlink()
    except OSError:
        pass
    return str(target)


def mux_audio(video: str | os.PathLike[str], audio: str | os.PathLike[str], out: str | os.PathLike[str], toolchain: Toolchain | None = None, runner: Runner | None = None, timeout_s: float | None = None) -> str:
    """Put ``audio`` under ``video`` (video copied, audio AAC, padded with silence to the video's length)."""
    v, a = _existing(video, "video"), _existing(audio, "audio")
    ffmpeg = _toolchain(toolchain).require("ffmpeg", "muxing audio")
    target = Path(out)
    ensure_dir(target.parent)
    timeout = _timeout(timeout_s)
    argv = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(v), "-i", str(a),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-af", "apad", "-shortest",
        "-movflags", "+faststart", str(target),
    ]
    r = _run(runner, argv, target.parent, timeout)
    _check(r, "muxing audio", timeout)
    return str(target)


# ----------------------------------------------------------------------------- captions
def format_srt_time(seconds: float) -> str:
    ms = int(round(max(0.0, float(seconds)) * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def srt_text(segments: Iterable[Mapping[str, Any]]) -> str:
    """SRT for ``[{"start_s", "end_s", "text"}]``; empty texts are skipped, ends are clamped after starts."""
    blocks: list[str] = []
    for seg in segments:
        if not isinstance(seg, Mapping):
            raise ToolFailure("invalid_input", f"caption segments must be objects with start_s, end_s and text; got {type(seg).__name__}", ErrorCategory.invalid_input, remediation="Send captions as [{start_s, end_s, text}] or as an .srt path.")
        try:
            start, end = float(seg.get("start_s", 0)), float(seg.get("end_s", 0))
        except (TypeError, ValueError) as e:
            raise ToolFailure("invalid_input", f"caption segment times must be numbers: {seg}", ErrorCategory.invalid_input, remediation="Give start_s and end_s in seconds for every segment.") from e
        text = " ".join(str(seg.get("text") or "").split())
        if not text:
            continue
        if end <= start:
            end = start + 0.5
        blocks.append(f"{len(blocks) + 1}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{text}\n")
    return "\n".join(blocks) + ("\n" if blocks else "")


def write_srt(segments: Iterable[Mapping[str, Any]], path: str | os.PathLike[str]) -> str:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(srt_text(segments), encoding="utf-8")
    return str(p)


# ----------------------------------------------------------------------------- assemble
def assemble(
    film_video: str | os.PathLike[str],
    narration_audio: str | os.PathLike[str] | None,
    captions: str | os.PathLike[str] | Iterable[Mapping[str, Any]] | None,
    out: str | os.PathLike[str],
    toolchain: Toolchain | None = None,
    runner: Runner | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Film plus optional narration (AAC, padded) plus optional captions (an .srt path or segments, embedded as a soft mov_text track)."""
    v = _existing(film_video, "film video")
    target = Path(out)
    ensure_dir(target.parent)
    audio = _existing(narration_audio, "narration audio") if narration_audio else None
    srt: Path | None = None
    if captions is not None:
        if isinstance(captions, (str, os.PathLike)):
            srt = _existing(captions, "captions file")
        else:
            srt = Path(write_srt(captions, target.with_suffix(".srt")))
    tc = _toolchain(toolchain)
    ffmpeg = tc.require("ffmpeg", "assembling the film")
    argv = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(v)]
    inputs = 1
    audio_index = captions_index = None
    if audio is not None:
        argv += ["-i", str(audio)]
        audio_index, inputs = inputs, inputs + 1
    if srt is not None:
        argv += ["-i", str(srt)]
        captions_index, inputs = inputs, inputs + 1
    argv += ["-map", "0:v:0", "-c:v", "copy"]
    if audio_index is not None:
        argv += ["-map", f"{audio_index}:a:0", "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-af", "apad", "-shortest"]
    else:
        argv += ["-map", "0:a?", "-c:a", "copy"]
    if captions_index is not None:
        argv += ["-map", f"{captions_index}:s:0", "-c:s", "mov_text", "-metadata:s:s:0", "language=eng"]
    argv += ["-movflags", "+faststart", str(target)]
    timeout = _timeout(timeout_s)
    r = _run(runner, argv, target.parent, timeout)
    _check(r, "assembling the film", timeout)
    info: dict[str, Any] = {"path": str(target), "has_audio": audio is not None, "has_captions": srt is not None, "captions_path": str(srt) if srt else None, "duration_s": None}
    if tc.has("ffprobe"):
        try:
            info["duration_s"] = probe(target, tc, runner)["duration_s"]
        except ToolFailure:
            pass
    return info


__all__ = [
    "CONCAT_INPUTS_LIMIT",
    "FFMPEG_TIMEOUT_LIMIT",
    "FRAMES_PER_CALL_LIMIT",
    "FRAME_WIDTH_PX_DEFAULT",
    "assemble",
    "concat",
    "extract_frames",
    "format_srt_time",
    "mux_audio",
    "probe",
    "srt_text",
    "write_srt",
]
