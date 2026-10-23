"""Shared fixtures: temp state dirs, synthetic media, deterministic IDs/clock."""
from __future__ import annotations

import math
import os
import shutil
import struct
import subprocess
import tempfile
import wave
import zlib

from harness.core.config import HarnessConfig
from harness.core.utils import FakeClock, SequentialIdGenerator


def temp_state() -> tuple[str, HarnessConfig]:
    d = tempfile.mkdtemp(prefix="harness_test_")
    cfg = HarnessConfig(state_dir=os.path.join(d, "state"))
    return d, cfg


def deterministic():
    return SequentialIdGenerator(), FakeClock()


def write_text(d: str, name: str, text: str) -> str:
    p = os.path.join(d, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p


def png_bytes(w: int = 4, h: int = 3, rgb: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))

    def chunk(t: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + t + data + struct.pack(">I", zlib.crc32(t + data) & 0xFFFFFFFF)

    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def write_png(d: str, name: str = "img.png", **kw) -> str:
    p = os.path.join(d, name)
    with open(p, "wb") as f:
        f.write(png_bytes(**kw))
    return p


def write_wav(d: str, name: str = "tone.wav", seconds: float = 3.0, silence_from: float = 1.0, silence_to: float = 2.0, rate: int = 16000) -> str:
    p = os.path.join(d, name)
    n = int(seconds * rate)
    with wave.open(p, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = bytearray()
        for i in range(n):
            t = i / rate
            v = 0 if silence_from <= t < silence_to else int(8000 * math.sin(2 * math.pi * 440 * t))
            frames += struct.pack("<h", v)
        w.writeframes(bytes(frames))
    return p


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def write_mp4(d: str, name: str = "clip.mp4", seconds: int = 3) -> str | None:
    if not have_ffmpeg():
        return None
    p = os.path.join(d, name)
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=160x120:rate=10",
                        "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-shortest", "-pix_fmt", "yuv420p", p], capture_output=True)
    return p if r.returncode == 0 else None
