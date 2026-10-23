"""Image adapter.

Pure-Python header parsing gives dimensions and format for PNG, JPEG, GIF,
BMP and WebP. Resizing, cropping and zooming use ffmpeg when present (run
inside the sandbox). OCR and captioning are sidecars.
"""
from __future__ import annotations

import base64
import os
import struct
from typing import Any, Optional

from ..sandbox.runner import run_subprocess, which


def dimensions(data: bytes) -> Optional[tuple[int, int]]:
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        w, h = struct.unpack(">II", data[16:24])
        return w, h
    if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
        w, h = struct.unpack("<HH", data[6:10])
        return w, h
    if data[:2] == b"BM" and len(data) >= 26:
        w, h = struct.unpack("<ii", data[18:26])
        return abs(w), abs(h)
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP" and len(data) >= 30:
        chunk = data[12:16]
        if chunk == b"VP8 ":
            w, h = struct.unpack("<HH", data[26:30])
            return w & 0x3FFF, h & 0x3FFF
        if chunk == b"VP8L":
            b = data[21:25]
            bits = b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)
            return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        if chunk == b"VP8X":
            w = 1 + (data[24] | (data[25] << 8) | (data[26] << 16))
            h = 1 + (data[27] | (data[28] << 8) | (data[29] << 16))
            return w, h
    if data[:2] == b"\xff\xd8":
        i = 2
        n = len(data)
        while i + 9 <= n:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            seg_len = struct.unpack(">H", data[i + 2 : i + 4])[0]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                h, w = struct.unpack(">HH", data[i + 5 : i + 9])
                return w, h
            i += 2 + seg_len
    return None


def data_url(path: str, mime: str) -> str:
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode("ascii")


def _ffmpeg(args: list[str], cwd: str, timeout_s: float = 60.0):
    exe = which("ffmpeg")
    if not exe:
        return None
    return run_subprocess([exe, "-hide_banner", "-loglevel", "error", "-y", *args], cwd, timeout_s=timeout_s, cpu_seconds=int(timeout_s), memory_bytes=2 << 30)


def make_preview(src: str, dst: str, max_side: int = 768) -> bool:
    r = _ffmpeg(["-i", src, "-vf", f"scale='min({max_side},iw)':'min({max_side},ih)':force_original_aspect_ratio=decrease", dst], os.path.dirname(dst) or ".")
    return bool(r and r.ok and os.path.exists(dst))


def crop(src: str, dst: str, x: int, y: int, w: int, h: int, zoom: float = 1.0) -> bool:
    vf = f"crop={w}:{h}:{x}:{y}"
    if zoom and zoom != 1.0:
        vf += f",scale=iw*{zoom}:ih*{zoom}:flags=lanczos"
    r = _ffmpeg(["-i", src, "-vf", vf, dst], os.path.dirname(dst) or ".")
    return bool(r and r.ok and os.path.exists(dst))


def enhance(src: str, dst: str, contrast: float = 1.5, brightness: float = 0.0, rotate_deg: int = 0) -> bool:
    filters = [f"eq=contrast={contrast}:brightness={brightness}"]
    if rotate_deg % 360:
        turns = (rotate_deg // 90) % 4
        filters += ["transpose=1"] * turns
    r = _ffmpeg(["-i", src, "-vf", ",".join(filters), dst], os.path.dirname(dst) or ".")
    return bool(r and r.ok and os.path.exists(dst))


def describe(data: bytes, mime: str) -> dict[str, Any]:
    dims = dimensions(data)
    out: dict[str, Any] = {"mime": mime, "bytes": len(data)}
    if dims:
        out["width"], out["height"] = dims
        out["pixels"] = dims[0] * dims[1]
    return out
