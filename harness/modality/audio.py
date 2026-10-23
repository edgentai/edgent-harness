"""Audio adapter: WAV via the ``wave`` module, everything else through ffmpeg.

Silence-based splitting is implemented in pure Python on 16-bit PCM so long
recordings are cut between words, not mid-word (§5.4).
"""
from __future__ import annotations

import array
import math
import os
import wave
from typing import Any, Optional

from ..sandbox.runner import run_subprocess, which


def probe_wav(path: str) -> Optional[dict[str, Any]]:
    try:
        with wave.open(path, "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            return {"channels": w.getnchannels(), "sample_rate": rate, "sample_width": w.getsampwidth(),
                    "frames": frames, "duration_s": frames / float(rate) if rate else 0.0}
    except Exception:
        return None


def normalise(src: str, dst: str, sample_rate: int = 16000, channels: int = 1, timeout_s: float = 300.0) -> bool:
    """Transcode to 16 kHz mono PCM WAV (what speech models expect)."""
    exe = which("ffmpeg")
    if not exe:
        return False
    r = run_subprocess([exe, "-hide_banner", "-loglevel", "error", "-y", "-i", src, "-ac", str(channels), "-ar", str(sample_rate),
                        "-c:a", "pcm_s16le", dst], os.path.dirname(dst) or ".", timeout_s=timeout_s, cpu_seconds=int(timeout_s), memory_bytes=2 << 30)
    return r.ok and os.path.exists(dst)


def rms_envelope(path: str, window_s: float = 0.05) -> tuple[list[float], float]:
    """Per-window RMS (0..1) for 16-bit PCM WAV. Returns (values, window_s)."""
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        ch = w.getnchannels()
        width = w.getsampwidth()
        if width != 2:
            raise ValueError("rms_envelope needs 16-bit PCM; normalise first")
        frames_per_window = max(1, int(rate * window_s))
        out: list[float] = []
        while True:
            raw = w.readframes(frames_per_window)
            if not raw:
                break
            samples = array.array("h")
            samples.frombytes(raw[: len(raw) - (len(raw) % 2)])
            if ch > 1:
                samples = array.array("h", samples[::ch])
            if not samples:
                break
            acc = 0
            for s in samples:
                acc += s * s
            out.append(math.sqrt(acc / len(samples)) / 32768.0)
    return out, window_s


def split_on_silence(path: str, max_chunk_s: float = 30.0, min_silence_s: float = 0.4, threshold: float = 0.02) -> list[dict[str, float]]:
    """Chunk boundaries (start, end) chosen at the quietest point inside each silence run."""
    env, win = rms_envelope(path)
    total = len(env) * win
    if total <= max_chunk_s:
        return [{"start": 0.0, "end": total}]
    min_run = max(1, int(min_silence_s / win))
    silences = []
    run_start = None
    for i, v in enumerate(env + [1.0]):
        if v < threshold:
            if run_start is None:
                run_start = i
        elif run_start is not None:
            if i - run_start >= min_run:
                seg = env[run_start:i]
                qi = run_start + min(range(len(seg)), key=seg.__getitem__)
                silences.append(qi * win)
            run_start = None
    chunks = []
    start = 0.0
    for cut in silences:
        if cut - start >= max_chunk_s * 0.5:
            chunks.append({"start": start, "end": cut})
            start = cut
        if cut - start > max_chunk_s:  # no silence found in time: hard cut
            chunks.append({"start": start, "end": start + max_chunk_s})
            start += max_chunk_s
    while total - start > max_chunk_s:
        chunks.append({"start": start, "end": start + max_chunk_s})
        start += max_chunk_s
    if total - start > 0.05:
        chunks.append({"start": start, "end": total})
    return chunks


def extract_range(src: str, dst: str, start: float, end: float) -> bool:
    exe = which("ffmpeg")
    if not exe:
        return False
    r = run_subprocess([exe, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", src, "-c:a", "pcm_s16le", dst],
                       os.path.dirname(dst) or ".", timeout_s=120, cpu_seconds=120, memory_bytes=2 << 30)
    return r.ok and os.path.exists(dst)


def transcript_text(segments: list[dict[str, Any]], start: Optional[float] = None, end: Optional[float] = None) -> str:
    lines = []
    for s in segments:
        if start is not None and s.get("end", 0) < start:
            continue
        if end is not None and s.get("start", 0) > end:
            continue
        spk = f"{s['speaker']}: " if s.get("speaker") else ""
        flag = " (low confidence)" if s.get("conf", 1.0) < 0.6 else ""
        lines.append(f"[{s.get('start', 0):07.2f}-{s.get('end', 0):07.2f}] {spk}{s.get('text', '')}{flag}")
    return "\n".join(lines)
