"""Ingestion: validate by content, store, and build derived views per modality (§5.1, §5.6)."""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Optional

from ..core.config import ModalityLimits
from ..core.types import Modality
from ..core.utils import estimate_tokens
from . import audio as audio_mod
from . import image as image_mod
from . import text as text_mod
from . import video as video_mod
from .artifacts import Artifact, ArtifactStore
from .sidecars import Sidecars


class IngestError(Exception):
    pass


_MAGIC: list[tuple[bytes, str, Modality]] = [
    (b"\x89PNG\r\n\x1a\n", "image/png", Modality.IMAGE),
    (b"\xff\xd8\xff", "image/jpeg", Modality.IMAGE),
    (b"GIF87a", "image/gif", Modality.IMAGE),
    (b"GIF89a", "image/gif", Modality.IMAGE),
    (b"BM", "image/bmp", Modality.IMAGE),
    (b"%PDF", "application/pdf", Modality.DOCUMENT),
    (b"ID3", "audio/mpeg", Modality.AUDIO),
    (b"\xff\xfb", "audio/mpeg", Modality.AUDIO),
    (b"\xff\xf3", "audio/mpeg", Modality.AUDIO),
    (b"fLaC", "audio/flac", Modality.AUDIO),
    (b"OggS", "audio/ogg", Modality.AUDIO),
    (b"\x1aE\xdf\xa3", "video/webm", Modality.VIDEO),
    (b"FLV", "video/x-flv", Modality.VIDEO),
    (b"PK\x03\x04", "application/zip", Modality.DOCUMENT),
]


def sniff_mime(head: bytes) -> tuple[str, Modality]:
    """Detect type from content, never from the extension."""
    if head[:4] == b"RIFF" and len(head) >= 12:
        tag = head[8:12]
        if tag == b"WAVE":
            return "audio/wav", Modality.AUDIO
        if tag == b"AVI ":
            return "video/x-msvideo", Modality.VIDEO
        if tag == b"WEBP":
            return "image/webp", Modality.IMAGE
    if len(head) >= 12 and head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in (b"M4A ", b"M4B "):
            return "audio/mp4", Modality.AUDIO
        if brand in (b"qt  ",):
            return "video/quicktime", Modality.VIDEO
        return "video/mp4", Modality.VIDEO
    for magic, mime, mod in _MAGIC:
        if head.startswith(magic):
            return mime, mod
    try:
        sample = head.decode("utf-8")
    except UnicodeDecodeError:
        return "application/octet-stream", Modality.UNKNOWN
    if "\x00" in sample:
        return "application/octet-stream", Modality.UNKNOWN
    stripped = sample.lstrip()
    if stripped.startswith(("{", "[")):
        return "application/json", Modality.TEXT
    if stripped.lower().startswith(("<!doctype html", "<html")):
        return "text/html", Modality.DOCUMENT
    if stripped.startswith("#") or "\n#" in sample:
        return "text/markdown", Modality.TEXT
    return "text/plain", Modality.TEXT


class Ingestor:
    def __init__(self, store: ArtifactStore, limits: Optional[ModalityLimits] = None, sidecars: Optional[Sidecars] = None,
                 work_dir: Optional[str] = None) -> None:
        self.store = store
        self.limits = limits or ModalityLimits()
        self.sidecars = sidecars or Sidecars()
        self.work_dir = work_dir or tempfile.mkdtemp(prefix="ingest_")
        os.makedirs(self.work_dir, exist_ok=True)

    # -- entry points -----------------------------------------------------
    def ingest_path(self, path: str, source: Optional[str] = None) -> Artifact:
        if not os.path.isfile(path):
            raise IngestError(f"not a file: {path}")
        size = os.path.getsize(path)
        if size > self.limits.max_file_bytes:
            raise IngestError(f"file too large: {size} bytes > {self.limits.max_file_bytes}")
        with open(path, "rb") as f:
            head = f.read(4096)
        mime, modality = sniff_mime(head)
        art = self.store.put_file(path, mime, modality, source=source or path)
        self._derive(art)
        return art

    def ingest_text(self, text: str, source: str = "inline") -> Artifact:
        data = text.encode("utf-8")
        mime, _ = sniff_mime(data[:4096])
        if not mime.startswith("text/") and mime != "application/json":
            mime = "text/plain"
        art = self.store.put_bytes(data, mime, Modality.TEXT, source=source, ext=".txt")
        self._derive(art)
        return art

    def ingest_bytes(self, data: bytes, source: str = "upload") -> Artifact:
        if len(data) > self.limits.max_file_bytes:
            raise IngestError("payload too large")
        mime, modality = sniff_mime(data[:4096])
        ext = {"image/png": ".png", "image/jpeg": ".jpg", "audio/wav": ".wav", "video/mp4": ".mp4", "text/plain": ".txt"}.get(mime, "")
        art = self.store.put_bytes(data, mime, modality, source=source, ext=ext)
        self._derive(art)
        return art

    # -- derived views ----------------------------------------------------
    def _derive(self, art: Artifact) -> None:
        if self.store.get_view(art.id, "summary"):
            return  # cached by content hash
        if art.modality == Modality.TEXT:
            self._derive_text(art)
        elif art.modality == Modality.DOCUMENT:
            self._derive_document(art)
        elif art.modality == Modality.IMAGE:
            self._derive_image(art)
        elif art.modality == Modality.AUDIO:
            self._derive_audio(art)
        elif art.modality == Modality.VIDEO:
            self._derive_video(art)
        else:
            self.store.put_json_view(art.id, "summary", {"note": "unknown binary content; no derived views"})

    def _derive_text(self, art: Artifact) -> None:
        raw = self.store.read_bytes(art.id).decode("utf-8", "replace")
        text = text_mod.clean_text(raw)
        self.store.put_text_view(art.id, "text", text, {"chars": len(text), "tokens_est": estimate_tokens(text)})
        self.store.put_json_view(art.id, "structure", text_mod.structure(text))
        self.store.put_json_view(art.id, "pages", text_mod.page_map(text))
        self.store.put_json_view(art.id, "chunks", text_mod.chunk(text))
        self.store.put_json_view(art.id, "summary", {"modality": "text", "chars": len(text), "tokens_est": estimate_tokens(text), "preview": text[:200]})

    def _derive_document(self, art: Artifact) -> None:
        # PDFs and office files need a converter sidecar; without one, be honest.
        if art.mime == "text/html":
            import html
            import re

            raw = self.store.read_bytes(art.id).decode("utf-8", "replace")
            raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
            text = text_mod.clean_text(html.unescape(re.sub(r"<[^>]+>", " ", raw)))
            self.store.put_text_view(art.id, "text", text)
            self.store.put_json_view(art.id, "chunks", text_mod.chunk(text))
            self.store.put_json_view(art.id, "summary", {"modality": "document", "format": "html", "chars": len(text), "preview": text[:200]})
            return
        self.store.put_json_view(art.id, "summary", {"modality": "document", "format": art.mime, "note": "no text extractor configured for this format; content unverified"})

    def _derive_image(self, art: Artifact) -> None:
        data = self.store.read_bytes(art.id)
        info = image_mod.describe(data, art.mime)
        if info.get("pixels", 0) > self.limits.max_image_pixels:
            raise IngestError("image exceeds pixel limit")
        self.store.put_json_view(art.id, "metadata", info)
        preview_path = os.path.join(self.work_dir, f"{art.id[:16]}_preview.png")
        if image_mod.make_preview(art.path, preview_path, self.limits.preview_max_side):
            prev = self.store.put_file(preview_path, "image/png", Modality.IMAGE, parent_id=art.id, derived_kind="preview",
                                       derived_params={"max_side": self.limits.preview_max_side}, move=True)
            info["preview_artifact"] = prev.id
        ocr = self.sidecars.run_ocr(art.path)
        if ocr.get("available"):
            self.store.put_text_view(art.id, "ocr", ocr.get("text", ""), {"boxes": ocr.get("boxes", [])[:500], "error": ocr.get("error")})
        cap = self.sidecars.run_caption(art.path)
        if cap.get("available") and cap.get("caption"):
            self.store.put_text_view(art.id, "caption", cap["caption"])
        text_parts = []
        if ocr.get("text"):
            text_parts.append("OCR: " + ocr["text"])
        if cap.get("caption"):
            text_parts.append("Caption: " + cap["caption"])
        self.store.put_text_view(art.id, "text", "\n".join(text_parts) if text_parts else "(no text view: OCR/caption sidecars unavailable; inspect the image with image tools)")
        self.store.put_json_view(art.id, "summary", {"modality": "image", **{k: info[k] for k in ("width", "height") if k in info},
                                                     "ocr_available": bool(ocr.get("available")), "caption_available": bool(cap.get("available")),
                                                     "preview": (ocr.get("text") or cap.get("caption") or "")[:200]})

    def _derive_audio(self, art: Artifact) -> None:
        wav_path = art.path
        info = audio_mod.probe_wav(art.path) if art.mime == "audio/wav" else None
        if info is None or info.get("sample_width") != 2 or info.get("channels") != 1:
            norm = os.path.join(self.work_dir, f"{art.id[:16]}_16k.wav")
            if audio_mod.normalise(art.path, norm):
                child = self.store.put_file(norm, "audio/wav", Modality.AUDIO, parent_id=art.id, derived_kind="normalised",
                                            derived_params={"sr": 16000, "ch": 1}, move=True)
                wav_path = child.path
                info = audio_mod.probe_wav(wav_path)
        info = info or {}
        duration = float(info.get("duration_s") or 0.0)
        if duration > self.limits.max_audio_seconds:
            raise IngestError("audio exceeds duration limit")
        self.store.put_json_view(art.id, "metadata", {**info, "normalised_path": wav_path})
        chunks: list[dict[str, float]] = []
        if info.get("sample_width") == 2:
            try:
                chunks = audio_mod.split_on_silence(wav_path)
            except Exception:
                chunks = []
        self.store.put_json_view(art.id, "chunks", chunks)
        stt = self.sidecars.run_stt(wav_path)
        segments = stt.get("segments", [])
        self.store.put_json_view(art.id, "transcript_segments", segments, {"available": stt.get("available"), "error": stt.get("error"), "reason": stt.get("reason")})
        transcript = audio_mod.transcript_text(segments) if segments else ""
        tags = self.sidecars.run_audio_tags(wav_path)
        if tags.get("available"):
            self.store.put_json_view(art.id, "events", tags.get("events", []))
        if transcript:
            self.store.put_text_view(art.id, "transcript", transcript, {"segments": len(segments)})
            self.store.put_text_view(art.id, "text", transcript)
        else:
            why = stt.get("error") or stt.get("reason") or "no speech detected"
            self.store.put_text_view(art.id, "text", f"(no transcript: {why})")
        self.store.put_json_view(art.id, "summary", {"modality": "audio", "duration_s": round(duration, 2), "chunks": len(chunks),
                                                     "transcript_available": bool(transcript), "low_confidence_spans": sum(1 for s in segments if s.get("conf", 1) < 0.6),
                                                     "preview": transcript[:200]})

    def _derive_video(self, art: Artifact) -> None:
        info = video_mod.probe(art.path)
        if info is None:
            self.store.put_text_view(art.id, "text", "(video could not be probed: ffprobe unavailable or file unreadable)")
            self.store.put_json_view(art.id, "summary", {"modality": "video", "error": "probe failed"})
            return
        duration = info.get("duration_s", 0.0)
        if duration > self.limits.max_video_seconds:
            raise IngestError("video exceeds duration limit")
        self.store.put_json_view(art.id, "metadata", info)
        scenes = video_mod.scene_changes(art.path, self.limits.scene_threshold) or [0.0]
        # Pass 1: coarse keyframes at scene starts plus a fixed low-rate grid.
        grid = [round(t, 3) for t in _frange(0.0, duration, self.limits.keyframe_interval_s)]
        times = sorted(set(scenes) | set(grid))
        keyframes = []
        for t in times[:120]:
            cached = self.store.find_derived(art.id, "frame", {"t": t})
            if cached is None:
                dst = os.path.join(self.work_dir, f"{art.id[:16]}_{int(t * 1000):09d}.jpg")
                if not video_mod.frame_at(art.path, t, dst, self.limits.preview_max_side):
                    continue
                cached = self.store.put_file(dst, "image/jpeg", Modality.IMAGE, parent_id=art.id, derived_kind="frame", derived_params={"t": t}, move=True)
                self.store.put_json_view(cached.id, "summary", {"modality": "image", "frame_of": art.id, "t": t})
            keyframes.append({"t": t, "artifact_id": cached.id, "scene_start": t in scenes})
        self.store.put_json_view(art.id, "scenes", [{"start": s, "end": (scenes[i + 1] if i + 1 < len(scenes) else duration)} for i, s in enumerate(scenes)])
        self.store.put_json_view(art.id, "keyframes", keyframes)
        transcript = ""
        if info.get("has_audio"):
            wav = os.path.join(self.work_dir, f"{art.id[:16]}_audio.wav")
            cached_audio = self.store.find_derived(art.id, "audio_track", {"sr": 16000})
            if cached_audio is None and video_mod.extract_audio(art.path, wav):
                cached_audio = self.store.put_file(wav, "audio/wav", Modality.AUDIO, parent_id=art.id, derived_kind="audio_track", derived_params={"sr": 16000}, move=True)
            if cached_audio is not None:
                self._derive_audio(cached_audio)
                tv = self.store.get_view(cached_audio.id, "transcript")
                transcript = tv.text if tv and tv.text else ""
                self.store.put_json_view(art.id, "audio_track", {"artifact_id": cached_audio.id})
        timeline = []
        for kf in keyframes:
            timeline.append(f"[{kf['t']:07.2f}] frame artifact_id={kf['artifact_id']}" + (" (scene start)" if kf["scene_start"] else ""))
        text = "Visual timeline:\n" + "\n".join(timeline)
        if transcript:
            self.store.put_text_view(art.id, "transcript", transcript)
            text += "\n\nTranscript:\n" + transcript
        else:
            text += "\n\nTranscript: (unavailable)"
        self.store.put_text_view(art.id, "text", text)
        self.store.put_json_view(art.id, "summary", {"modality": "video", "duration_s": round(duration, 2), "scenes": len(scenes), "keyframes": len(keyframes),
                                                     "has_audio": info.get("has_audio"), "transcript_available": bool(transcript),
                                                     "width": info.get("width"), "height": info.get("height")})


def _frange(start: float, stop: float, step: float) -> list[float]:
    out = []
    t = start
    while t < stop and step > 0:
        out.append(t)
        t += step
    return out
