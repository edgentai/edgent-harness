"""Replaceable sidecar adapters (§1 principle 6).

Each sidecar is a callable. Defaults try local CLIs (tesseract) or an
OpenAI-compatible transcription endpoint; when nothing is available the
sidecar reports ``unavailable`` so the harness can say so honestly instead
of guessing.
"""
from __future__ import annotations

import json
import os
import shutil
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..sandbox.runner import run_subprocess

OcrFn = Callable[[str], dict[str, Any]]  # path -> {"text": str, "boxes": [...], "available": bool}
SttFn = Callable[[str], dict[str, Any]]  # path -> {"segments": [...], "text": str, "available": bool}
CaptionFn = Callable[[str], dict[str, Any]]  # path -> {"caption": str, "available": bool}
EmbedFn = Callable[[list[str]], list[list[float]]]
TagFn = Callable[[str], dict[str, Any]]  # audio path -> {"events": [...], "available": bool}


def unavailable(what: str) -> dict[str, Any]:
    return {"available": False, "reason": f"{what} sidecar not configured", "text": "", "segments": [], "boxes": [], "events": []}


def tesseract_ocr(path: str, timeout_s: float = 60.0) -> dict[str, Any]:
    exe = shutil.which("tesseract")
    if not exe:
        return unavailable("OCR (tesseract)")
    cwd = os.path.dirname(os.path.abspath(path)) or "."
    r = run_subprocess([exe, path, "stdout", "tsv"], cwd, timeout_s=timeout_s, cpu_seconds=int(timeout_s), memory_bytes=2 << 30)
    if not r.ok:
        return {"available": True, "error": r.stderr[:300], "text": "", "boxes": []}
    boxes = []
    words = []
    for line in r.stdout.splitlines()[1:]:
        cols = line.split("\t")
        if len(cols) < 12 or not cols[11].strip():
            continue
        try:
            conf = float(cols[10])
            x, y, w, h = (int(cols[6]), int(cols[7]), int(cols[8]), int(cols[9]))
        except ValueError:
            continue
        boxes.append({"text": cols[11], "box": [x, y, w, h], "conf": conf, "line": int(cols[4]), "block": int(cols[2])})
        words.append(cols[11])
    return {"available": True, "text": " ".join(words), "boxes": boxes}


def openai_transcription(base_url: str, api_key: str = "", model: str = "whisper-1", timeout_s: float = 300.0) -> SttFn:
    """Speech-to-text through an OpenAI-compatible ``/audio/transcriptions`` endpoint."""

    def stt(path: str) -> dict[str, Any]:
        boundary = uuid.uuid4().hex
        with open(path, "rb") as f:
            audio = f.read()
        parts = [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n".encode(),
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\nverbose_json\r\n".encode(),
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{os.path.basename(path)}\"\r\nContent-Type: application/octet-stream\r\n\r\n".encode() + audio + b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
        body = b"".join(parts)
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(base_url.rstrip("/") + "/audio/transcriptions", data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            return {"available": True, "error": str(e), "text": "", "segments": []}
        segments = []
        for s in data.get("segments", []):
            segments.append({"start": float(s.get("start", 0)), "end": float(s.get("end", 0)), "text": s.get("text", "").strip(),
                             "conf": float(1.0 + s.get("avg_logprob", 0.0)) if "avg_logprob" in s else 0.9, "speaker": s.get("speaker")})
        return {"available": True, "text": data.get("text", ""), "segments": segments, "language": data.get("language")}

    return stt


@dataclass
class Sidecars:
    ocr: OcrFn = field(default=tesseract_ocr)
    stt: Optional[SttFn] = None
    caption: Optional[CaptionFn] = None
    embed: Optional[EmbedFn] = None
    audio_tags: Optional[TagFn] = None

    def run_ocr(self, path: str) -> dict[str, Any]:
        try:
            return self.ocr(path)
        except Exception as e:  # noqa: BLE001
            return {"available": True, "error": str(e), "text": "", "boxes": []}

    def run_stt(self, path: str) -> dict[str, Any]:
        if not self.stt:
            return unavailable("speech-to-text")
        try:
            return self.stt(path)
        except Exception as e:  # noqa: BLE001
            return {"available": True, "error": str(e), "text": "", "segments": []}

    def run_caption(self, path: str) -> dict[str, Any]:
        if not self.caption:
            return unavailable("captioning")
        try:
            return self.caption(path)
        except Exception as e:  # noqa: BLE001
            return {"available": True, "error": str(e), "caption": ""}

    def run_audio_tags(self, path: str) -> dict[str, Any]:
        if not self.audio_tags:
            return unavailable("audio tagging")
        try:
            return self.audio_tags(path)
        except Exception as e:  # noqa: BLE001
            return {"available": True, "error": str(e), "events": []}

    def status(self) -> dict[str, bool]:
        return {"ocr": shutil.which("tesseract") is not None or self.ocr is not tesseract_ocr, "stt": self.stt is not None,
                "caption": self.caption is not None, "embed": self.embed is not None, "audio_tags": self.audio_tags is not None,
                "ffmpeg": shutil.which("ffmpeg") is not None, "ffprobe": shutil.which("ffprobe") is not None}
