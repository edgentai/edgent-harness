"""Video adapter on top of ffprobe/ffmpeg (sandboxed). Hierarchical sampling (§5.3)."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from ..sandbox.runner import run_subprocess, which


def available() -> bool:
    return which("ffprobe") is not None and which("ffmpeg") is not None


def probe(path: str) -> Optional[dict[str, Any]]:
    exe = which("ffprobe")
    if not exe:
        return None
    r = run_subprocess([exe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
                       os.path.dirname(os.path.abspath(path)) or ".", timeout_s=60, cpu_seconds=60, memory_bytes=2 << 30)
    if not r.ok:
        return None
    try:
        data = json.loads(r.stdout)
    except Exception:
        return None
    fmt = data.get("format", {})
    out: dict[str, Any] = {"duration_s": float(fmt.get("duration", 0) or 0), "format": fmt.get("format_name"), "has_audio": False, "has_video": False}
    for s in data.get("streams", []):
        if s.get("codec_type") == "video" and not out["has_video"]:
            out["has_video"] = True
            out["width"], out["height"] = s.get("width"), s.get("height")
            fr = s.get("avg_frame_rate") or "0/1"
            try:
                num, den = fr.split("/")
                out["fps"] = float(num) / float(den) if float(den) else 0.0
            except Exception:
                out["fps"] = 0.0
            out["video_codec"] = s.get("codec_name")
        elif s.get("codec_type") == "audio":
            out["has_audio"] = True
            out["audio_codec"] = s.get("codec_name")
            out["sample_rate"] = s.get("sample_rate")
    return out


def _ffmpeg(args: list[str], cwd: str, timeout_s: float = 300.0):
    exe = which("ffmpeg")
    if not exe:
        return None
    return run_subprocess([exe, "-hide_banner", "-nostats", "-y", *args], cwd, timeout_s=timeout_s, cpu_seconds=int(timeout_s), memory_bytes=3 << 30)


def scene_changes(path: str, threshold: float = 0.4, timeout_s: float = 600.0) -> Optional[list[float]]:
    """Timestamps where ffmpeg's scene detector fires. Always includes 0.0."""
    r = _ffmpeg(["-i", path, "-vf", f"select='gt(scene,{threshold})',showinfo", "-an", "-f", "null", "-"],
                os.path.dirname(os.path.abspath(path)) or ".", timeout_s)
    if r is None:
        return None
    times = [0.0]
    for m in re.finditer(r"pts_time:([0-9.]+)", r.stderr):
        try:
            times.append(float(m.group(1)))
        except ValueError:
            pass
    return sorted(set(round(t, 3) for t in times))


def frame_at(path: str, t: float, dst: str, max_side: int = 768) -> bool:
    r = _ffmpeg(["-ss", f"{t:.3f}", "-i", path, "-frames:v", "1", "-vf",
                 f"scale='min({max_side},iw)':'min({max_side},ih)':force_original_aspect_ratio=decrease", dst],
                os.path.dirname(dst) or ".", timeout_s=60)
    return bool(r and r.ok and os.path.exists(dst))


def frames_between(path: str, t1: float, t2: float, fps: float, out_dir: str, max_side: int = 512, max_frames: int = 32) -> list[dict[str, Any]]:
    os.makedirs(out_dir, exist_ok=True)
    span = max(0.0, t2 - t1)
    n = int(span * fps) + 1
    if n > max_frames:
        fps = max_frames / span if span else fps
    pattern = os.path.join(out_dir, "f_%05d.jpg")
    r = _ffmpeg(["-ss", f"{t1:.3f}", "-to", f"{t2:.3f}", "-i", path, "-vf",
                 f"fps={fps},scale='min({max_side},iw)':'min({max_side},ih)':force_original_aspect_ratio=decrease", pattern], out_dir, timeout_s=300)
    if not r or not r.ok:
        return []
    files = sorted(f for f in os.listdir(out_dir) if f.startswith("f_") and f.endswith(".jpg"))
    return [{"path": os.path.join(out_dir, f), "t": round(t1 + i / fps, 3)} for i, f in enumerate(files)]


def extract_audio(path: str, dst: str, sample_rate: int = 16000) -> bool:
    r = _ffmpeg(["-i", path, "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", dst], os.path.dirname(dst) or ".", timeout_s=600)
    return bool(r and r.ok and os.path.exists(dst))
