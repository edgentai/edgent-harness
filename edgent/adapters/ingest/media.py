"""Audio and video sources: a timeline of ``[t=HH:MM:SS]`` markers with keyframes, slide text and transcript.

ffprobe is required (duration and streams). For video, ffmpeg writes a keyframe at every scene change
(``select='gt(scene,0.3)'``, capped at ``KEYFRAMES_LIMIT``) plus one frame per ``KEYFRAME_FLOOR_INTERVAL_S``
as a floor, into ``sources/text/<id>.frames/<HH-MM-SS>.jpg``; tesseract, when present, reads slide text
off them. Audio is extracted to 16 kHz mono WAV in a temporary directory and transcribed with faster-whisper
when it is importable (an injected ``transcriber`` replaces it in tests). Without a transcriber coverage is
``unknown`` and the view says so: silence in the view is not silence in the lecture.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...domain.errors import ErrorCategory, Limit, ToolFailure
from ...domain.utils import ensure_dir
from ...domain.workspace import Workspace
from .base import (
    IngestResult,
    Sniff,
    coverage,
    frames_rel,
    hms,
    missing_tool,
    run_command,
    time_marker,
    title_from_stem,
)

SCENE_CHANGE_THRESHOLD = 0.3
KEYFRAME_FLOOR_INTERVAL_S = 60
KEYFRAMES_LIMIT = Limit("video keyframes", 400, "frames", does_not_raise=("the scene threshold",))
KEYFRAME_MAX_WIDTH_PX = 1280
KEYFRAME_JPEG_QUALITY = 4  # ffmpeg -q:v: 2 is best, 31 worst
MEDIA_DURATION_LIMIT = Limit("media duration", 6 * 3600, "s", does_not_raise=("the sources_add timeout",))
FFPROBE_TIMEOUT_S = 120.0
FFMPEG_TIMEOUT_S = 3600.0
FRAME_SEEK_TIMEOUT_S = 120.0
OCR_FRAME_TIMEOUT_S = 120.0
AUDIO_SAMPLE_RATE_HZ = 16000
WHISPER_MODEL_SIZE = os.environ.get("EDGENT_WHISPER_MODEL", "base")
STT_INSTALL_HINT = "pip install edgent[stt]"
_PTS_RE = re.compile(r"Parsed_showinfo\S*\s[^\n]*?pts_time:\s*([0-9.]+)")


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    segments: list[Segment] = field(default_factory=list)
    language: str | None = None
    engine: str = "unknown"


Transcriber = Callable[[str], Transcript]


@dataclass
class Keyframe:
    seconds: int
    rel_path: str
    slide_text: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------- probing
def probe(ffprobe: str, src: Path) -> dict[str, Any]:
    r = run_command([ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(src)], FFPROBE_TIMEOUT_S)
    data: dict[str, Any] = {}
    if r.ok:
        try:
            data = json.loads(r.stdout.decode("utf-8", "replace") or "{}")
        except ValueError:
            data = {}
    if not r.ok or not data.get("streams"):
        reason = "timed out" if r.timed_out else (r.stderr.strip().splitlines() or ["no streams found"])[-1][:300]
        raise ToolFailure(
            "unsupported",
            f"ffprobe could not read {src.name} as audio or video: {reason}",
            ErrorCategory.unsupported,
            remediation="Check that the file is a complete media file; re-encode it with ffmpeg (ffmpeg -i in -c copy out.mp4) and add it again.",
            path=str(src),
        )
    streams = data["streams"]
    video = [s for s in streams if s.get("codec_type") == "video" and not int((s.get("disposition") or {}).get("attached_pic", 0) or 0)]
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    duration = _float((data.get("format") or {}).get("duration"))
    if duration is None:
        duration = max((_float(s.get("duration")) or 0.0 for s in streams), default=0.0)
    first_video = video[0] if video else None
    language = next((str(s["tags"]["language"]) for s in audio if isinstance(s.get("tags"), dict) and s["tags"].get("language") not in (None, "", "und")), None)
    return {
        "duration_s": round(duration, 3),
        "video_streams": len(video),
        "audio_streams": len(audio),
        "dimensions": [int(first_video.get("width", 0)), int(first_video.get("height", 0))] if first_video else None,
        "video_codec": first_video.get("codec_name") if first_video else None,
        "audio_codec": audio[0].get("codec_name") if audio else None,
        "language": language,
    }


def _float(v: Any) -> float | None:
    try:
        return float(v) if v not in (None, "", "N/A") else None
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------------- keyframes
def _scale_filter() -> str:
    return f"scale='min({KEYFRAME_MAX_WIDTH_PX},iw)':-2"


def _scene_frames(ffmpeg: str, src: Path, tmp: Path, notes: list[str]) -> list[tuple[float, Path]]:
    pattern = tmp / "scene_%05d.jpg"
    vf = f"select='gt(scene,{SCENE_CHANGE_THRESHOLD})',showinfo,{_scale_filter()}"
    argv = [ffmpeg, "-hide_banner", "-nostdin", "-nostats", "-loglevel", "info", "-y", "-i", str(src), "-vf", vf, "-vsync", "vfr", "-q:v", str(KEYFRAME_JPEG_QUALITY), str(pattern)]
    r = run_command(argv, FFMPEG_TIMEOUT_S)
    times = [float(t) for t in _PTS_RE.findall(r.stderr)]
    files = sorted(tmp.glob("scene_*.jpg"))
    if r.timed_out:
        notes.append(f"scene detection stopped after {FFMPEG_TIMEOUT_S:g} s; keyframes cover only the decoded part")
    elif not r.ok and not files:
        notes.append(f"ffmpeg scene detection failed: {(r.stderr.strip().splitlines() or ['unknown error'])[-1][:200]}")
    if len(times) != len(files):
        notes.append(f"keyframe timestamps could not all be matched ({len(times)} timestamps for {len(files)} frames); the extra frames were dropped")
    return list(zip(times, files, strict=False))


def _seek_frame(ffmpeg: str, src: Path, seconds: int, out: Path) -> bool:
    argv = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-ss", str(seconds), "-i", str(src), "-frames:v", "1", "-vf", _scale_filter(), "-q:v", str(KEYFRAME_JPEG_QUALITY), str(out)]
    r = run_command(argv, FRAME_SEEK_TIMEOUT_S)
    return r.ok and out.exists() and out.stat().st_size > 0


def _sample(items: list[Any], limit: int) -> list[Any]:
    if len(items) <= limit:
        return items
    step = len(items) / limit
    return [items[min(len(items) - 1, int(i * step))] for i in range(limit)]


def extract_keyframes(ffmpeg: str, src: Path, duration_s: float, frames_dir: Path, rel_dir: str, notes: list[str]) -> list[Keyframe]:
    """Scene-change frames (capped) plus a floor of one frame per interval, named ``HH-MM-SS.jpg``."""
    ensure_dir(frames_dir)
    chosen: dict[int, Path] = {}
    with tempfile.TemporaryDirectory(prefix="edgent-frames-") as tmp_s:
        tmp = Path(tmp_s)
        scene = _scene_frames(ffmpeg, src, tmp, notes)
        if len(scene) > KEYFRAMES_LIMIT.value:
            notes.append(f"{len(scene)} scene changes found; kept {int(KEYFRAMES_LIMIT.value)} spread evenly ({KEYFRAMES_LIMIT.describe()})")
            scene = _sample(scene, int(KEYFRAMES_LIMIT.value))
        for t, path in scene:
            sec = int(t)
            if sec not in chosen:  # one frame per second; the first wins
                chosen[sec] = path
        floors = [0] + list(range(KEYFRAME_FLOOR_INTERVAL_S, int(math.ceil(duration_s)), KEYFRAME_FLOOR_INTERVAL_S))
        for t in floors:
            window = 1 if t == 0 else KEYFRAME_FLOOR_INTERVAL_S  # the first frame is always wanted; later floors only fill gaps
            if any(t <= sec < t + window for sec in chosen):
                continue
            out = tmp / f"floor_{t}.jpg"
            if _seek_frame(ffmpeg, src, t, out):
                chosen[t] = out
        frames: list[Keyframe] = []
        for sec in sorted(chosen):
            name = hms(sec).replace(":", "-") + ".jpg"
            shutil.move(str(chosen[sec]), str(frames_dir / name))
            frames.append(Keyframe(sec, f"{rel_dir}/{name}"))
    return frames


def ocr_keyframes(tesseract: str, frames: list[Keyframe], frames_dir: Path, notes: list[str]) -> int:
    read = 0
    failures = 0
    for kf in frames:
        path = frames_dir / Path(kf.rel_path).name
        r = run_command([tesseract, str(path), "-"], OCR_FRAME_TIMEOUT_S)
        if not r.ok:
            failures += 1
            continue
        lines = [" ".join(ln.split()) for ln in r.stdout.decode("utf-8", "replace").splitlines()]
        kf.slide_text = [ln for ln in lines if len(ln) > 1]
        read += 1
    if failures:
        notes.append(f"tesseract failed on {failures} of {len(frames)} keyframes")
    return read


# ----------------------------------------------------------------------------- audio and transcription
def extract_audio(ffmpeg: str, src: Path, out: Path) -> str | None:
    """16 kHz mono PCM WAV for a speech model; returns an error message or None."""
    argv = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE_HZ), "-c:a", "pcm_s16le", str(out)]
    r = run_command(argv, FFMPEG_TIMEOUT_S)
    if r.ok and out.exists():
        return None
    return "audio extraction timed out" if r.timed_out else f"audio extraction failed: {(r.stderr.strip().splitlines() or ['unknown error'])[-1][:200]}"


def faster_whisper_transcriber(wav_path: str) -> Transcript:
    """On-device transcription with faster-whisper; only called when the module is importable."""
    from faster_whisper import WhisperModel  # noqa: PLC0415 - optional dependency

    model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    segments, info = model.transcribe(wav_path, vad_filter=True)
    out = [Segment(float(s.start), float(s.end), " ".join(str(s.text).split())) for s in segments]
    return Transcript([s for s in out if s.text], getattr(info, "language", None), f"faster-whisper:{WHISPER_MODEL_SIZE}")


def default_transcriber(toolchain: Any) -> Transcriber | None:
    return faster_whisper_transcriber if toolchain is not None and toolchain.has("faster_whisper") else None


# ----------------------------------------------------------------------------- the view
def _timeline(frames: list[Keyframe], transcript: Transcript | None) -> str:
    events: dict[int, dict[str, list[str]]] = {}

    def at(sec: int) -> dict[str, list[str]]:
        return events.setdefault(sec, {"frames": [], "slides": [], "text": []})

    for kf in frames:
        at(kf.seconds)["frames"].append(f"(keyframe: {kf.rel_path})")
        at(kf.seconds)["slides"].extend(f"slide text: {ln}" for ln in kf.slide_text)
    for seg in transcript.segments if transcript else []:
        text = " ".join(seg.text.split())
        if text:
            at(int(seg.start))["text"].append(text)
    lines: list[str] = []
    for sec in sorted(events):
        ev = events[sec]
        lines.append(time_marker(sec))
        lines.extend(ev["frames"] + ev["slides"] + ev["text"])
        lines.append("")
    return "\n".join(lines)


def ingest_media(src: Path, source_id: str, workspace: Workspace, toolchain: Any, info: Sniff, title: str | None = None, transcriber: Transcriber | None = None, **_: Any) -> IngestResult:
    limitations: list[str] = []
    missing: list[dict[str, str]] = []
    ffprobe = toolchain.path("ffprobe") if toolchain is not None else None
    if not ffprobe:
        missing.append(missing_tool("ffprobe", "audio and video probing"))
        limitations.append("ffprobe is not installed: the file was kept but neither duration, keyframes nor a transcript could be derived (install ffmpeg, then sources_add again with replace=true)")
        body = "(ffprobe missing: no timeline could be built for this media file)\n"
        return IngestResult(info.kind, body, title or title_from_stem(src.stem), coverage("unknown", minutes_transcribed=0), limitations, {}, missing, usable=False)
    meta = probe(ffprobe, src)
    duration = float(meta["duration_s"] or 0.0)
    if duration > MEDIA_DURATION_LIMIT.value:
        err = MEDIA_DURATION_LIMIT.exceeded(duration, "Split the recording with ffmpeg (-ss/-t) and add the parts as separate sources.")
        raise ToolFailure(err.code, err.message, err.category, remediation=err.remediation, details=err.details)
    kind = "video" if meta["video_streams"] else "audio"
    minutes_total = round(duration / 60.0, 2)
    ffmpeg = toolchain.path("ffmpeg")
    frames: list[Keyframe] = []
    frames_dir = workspace.resolve(frames_rel(source_id))
    if kind == "video":
        if frames_dir.is_dir():
            shutil.rmtree(frames_dir, ignore_errors=True)
        if ffmpeg:
            frames = extract_keyframes(ffmpeg, src, duration, frames_dir, frames_rel(source_id), limitations)
            if not frames:
                limitations.append("no keyframes could be extracted from the video stream")
                shutil.rmtree(frames_dir, ignore_errors=True)  # an empty frames directory would look like a lost extraction
        else:
            missing.append(missing_tool("ffmpeg", "keyframe extraction"))
            limitations.append("ffmpeg is not installed: no keyframes were extracted")
    ocr_ran = False
    tesseract = toolchain.path("tesseract")
    if frames:
        if tesseract:
            ocr_ran = ocr_keyframes(tesseract, frames, frames_dir, limitations) > 0
        else:
            missing.append(missing_tool("tesseract", "slide text OCR of keyframes"))
            limitations.append(f"no OCR engine: slide text was not read from {len(frames)} keyframes (look at the frames directly, or install tesseract-ocr)")
    transcript: Transcript | None = None
    transcriber = transcriber or default_transcriber(toolchain)
    if not meta["audio_streams"]:
        limitations.append("the file has no audio stream, so there is nothing to transcribe")
    elif transcriber is None:
        missing.append(missing_tool("faster_whisper", "transcription"))
        limitations.append(f"no transcriber installed ({STT_INSTALL_HINT}): spoken content is unknown, not absent; add an SRT/VTT caption file with sources_add if you have one")
    elif not ffmpeg:
        missing.append(missing_tool("ffmpeg", "audio extraction for transcription"))
        limitations.append("ffmpeg is not installed: audio could not be extracted for transcription")
    else:
        with tempfile.TemporaryDirectory(prefix="edgent-audio-") as tmp:
            wav = Path(tmp) / "audio.wav"
            error = extract_audio(ffmpeg, src, wav)
            if error:
                limitations.append(error)
            else:
                try:
                    transcript = transcriber(str(wav))
                except Exception as e:  # noqa: BLE001 - a transcriber crash degrades coverage, it does not lose the source
                    limitations.append(f"transcription failed ({type(e).__name__}: {str(e)[:200]}); coverage is unknown")
    if transcript is not None:
        status = "complete"
        minutes_transcribed = minutes_total
        engine = transcript.engine
        if not transcript.segments:
            limitations.append(f"the transcriber ({engine}) returned no speech segments: the audio may be silent or music")
    else:
        status = "unknown"
        minutes_transcribed = 0.0
        engine = None
    header = [f"duration: {hms(duration)} | kind: {kind}" + (f" | keyframes: {len(frames)}" if kind == "video" else "") + (f" | transcript: {engine} ({transcript.language or 'language unknown'})" if transcript else " | transcript: none"), ""]
    timeline = _timeline(frames, transcript)
    if not timeline:
        timeline = "(no transcript and no keyframes: nothing in this file is greppable yet; see limitations)\n"
    cov = coverage(status, minutes_total=minutes_total, minutes_transcribed=minutes_transcribed, ocr=ocr_ran, transcriber=engine)
    extra = {
        "duration_s": meta["duration_s"],
        "dimensions": meta["dimensions"],
        "language": transcript.language if transcript else meta["language"],
        "streams": {"video": meta["video_streams"], "audio": meta["audio_streams"]},
        "codecs": {"video": meta["video_codec"], "audio": meta["audio_codec"]},
        "keyframes": len(frames),
        "frames_dir": frames_rel(source_id) if frames else None,
        "transcript_segments": len(transcript.segments) if transcript else 0,
    }
    usable = bool(frames) or transcript is not None
    return IngestResult(kind, "\n".join(header) + "\n" + timeline, title or title_from_stem(src.stem), cov, limitations, extra, missing, usable=usable)
