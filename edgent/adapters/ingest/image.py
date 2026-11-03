"""Images: dimensions from the file header (no PIL) and OCR through tesseract when it is installed.

The text view is one page, ``[p.1]``, holding the OCR text or a note that no OCR engine exists. Coverage is
``complete`` only when OCR ran: without it, a grep miss says nothing about what the image shows.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

from ...domain.workspace import Workspace
from .base import IngestResult, Sniff, coverage_complete, coverage_unknown, missing_tool, run_command, title_from_stem

IMAGE_HEADER_BYTES = 65536
OCR_TIMEOUT_S = 300.0
_JPEG_SOF_MARKERS = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def _jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    i = 2
    n = len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xFF, 0x01) or 0xD0 <= marker <= 0xD8:
            i += 1 if marker == 0xFF else 2
            continue
        if marker == 0xD9 or marker == 0xDA:
            return None
        if i + 4 > n:
            return None
        length = struct.unpack(">H", data[i + 2 : i + 4])[0]
        if marker in _JPEG_SOF_MARKERS:
            if i + 9 > n:
                return None
            height, width = struct.unpack(">HH", data[i + 5 : i + 9])
            return (width, height) if width and height else None
        i += 2 + length
    return None


def _webp_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < 30:
        return None
    chunk = data[12:16]
    if chunk == b"VP8 " and data[23:26] == b"\x9d\x01\x2a":
        w, h = struct.unpack("<HH", data[26:30])
        return (w & 0x3FFF, h & 0x3FFF)
    if chunk == b"VP8L" and data[20] == 0x2F:
        bits = struct.unpack("<I", data[21:25])[0]
        return ((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1)
    if chunk == b"VP8X":
        w = int.from_bytes(data[24:27], "little") + 1
        h = int.from_bytes(data[27:30], "little") + 1
        return (w, h)
    return None


def image_dimensions(path: str | Path) -> tuple[int, int] | None:
    """(width, height) from PNG, JPEG, GIF, WEBP or BMP headers; None for other formats or damaged files."""
    with open(path, "rb") as f:
        data = f.read(IMAGE_HEADER_BYTES)
    try:
        if data.startswith(b"\x89PNG\r\n\x1a\n") and data[12:16] == b"IHDR":
            w, h = struct.unpack(">II", data[16:24])
            return (w, h)
        if data.startswith(b"\xff\xd8\xff"):
            return _jpeg_dimensions(data)
        if data[:6] in (b"GIF87a", b"GIF89a"):
            w, h = struct.unpack("<HH", data[6:10])
            return (w, h)
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return _webp_dimensions(data)
        if data.startswith(b"BM") and len(data) >= 26:
            w, h = struct.unpack("<ii", data[18:26])
            return (abs(w), abs(h))
    except struct.error:
        return None
    return None


def ocr_image(path: Path, tesseract: str, timeout_s: float = OCR_TIMEOUT_S) -> tuple[str | None, str | None]:
    """(text, error). Text is '' when tesseract ran and found nothing."""
    r = run_command([tesseract, str(path), "-"], timeout_s)
    if not r.ok:
        reason = "timed out" if r.timed_out else (r.stderr.strip().splitlines() or ["unknown error"])[-1][:200]
        return None, f"tesseract failed on {path.name}: {reason}"
    lines = [ln.rstrip() for ln in r.stdout.decode("utf-8", "replace").splitlines()]
    return "\n".join(ln for ln in lines if ln.strip()), None


def ingest_image(src: Path, source_id: str, workspace: Workspace, toolchain: Any, info: Sniff, title: str | None = None, **_: Any) -> IngestResult:
    dims = image_dimensions(src)
    extra: dict[str, Any] = {"format": info.format, "dimensions": list(dims) if dims else None}
    size_note = f"{dims[0]}x{dims[1]}" if dims else "dimensions unknown"
    lines = ["[p.1]", f"(image: {info.format}, {size_note}, raw copy at sources/raw/{source_id})"]
    limitations: list[str] = []
    missing: list[dict[str, str]] = []
    tesseract = toolchain.path("tesseract") if toolchain is not None else None
    if tesseract:
        text, error = ocr_image(src, tesseract)
        if error:
            limitations.append(error)
            lines.append("(OCR failed; see limitations)")
            cov = coverage_unknown(ocr=False)
        elif text:
            lines.extend(text.splitlines())
            cov = coverage_complete(pages_total=1, ocr=True)
        else:
            lines.append("(OCR found no text in this image)")
            cov = coverage_complete(pages_total=1, ocr=True)
    else:
        missing.append(missing_tool("tesseract", "OCR of image text"))
        lines.append("(no OCR engine)")
        limitations.append("no OCR engine: any text in the image is unread (install tesseract-ocr and sources_add again with replace=true)")
        cov = coverage_unknown(pages_total=1, ocr=False)
    return IngestResult("image", "\n".join(lines) + "\n", title or title_from_stem(src.stem), cov, limitations, extra, missing, usable=True)
