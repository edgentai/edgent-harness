"""Warm-up and health checks after every start (§4.2.3): text, image, tool call, long context."""
from __future__ import annotations

import base64
import struct
import time
import zlib
from typing import Any

from ..core.client import ModelClient, ModelRequest
from ..core.types import ContentPart, Message


def _tiny_png() -> str:
    raw = b"\x00\xff\x00\x00"
    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


class HealthChecker:
    def __init__(self, client: ModelClient, native_image: bool = True, long_context_tokens: int = 8000) -> None:
        self.client = client
        self.native_image = native_image
        self.long_context_tokens = long_context_tokens

    def _timed(self, name: str, fn: Any) -> dict[str, Any]:
        t0 = time.monotonic()
        try:
            detail = fn()
            return {"check": name, "ok": True, "latency_s": round(time.monotonic() - t0, 3), "detail": detail}
        except Exception as e:  # noqa: BLE001
            return {"check": name, "ok": False, "latency_s": round(time.monotonic() - t0, 3), "error": str(e)[:300]}

    def text(self) -> Any:
        r = self.client.complete(ModelRequest([Message.text("user", "Reply with the single word OK.")], temperature=0.0, max_tokens=5))
        return {"text": r.text.strip()[:20], "usage": r.usage.to_dict()}

    def image(self) -> Any:
        if not self.native_image:
            return "skipped: model has no native image input"
        msg = Message(role="user", content=[ContentPart.of_text("What colour is this 1x1 image? One word."), ContentPart(type="image", data_url=_tiny_png(), mime="image/png")])
        r = self.client.complete(ModelRequest([msg], temperature=0.0, max_tokens=5))
        return {"text": r.text.strip()[:20]}

    def tool_call(self) -> Any:
        tool = {"type": "function", "function": {"name": "add", "description": "Add two integers", "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}
        r = self.client.complete(ModelRequest([Message.text("user", "Use the add tool to add 2 and 3.")], tools=[tool], tool_choice="auto", temperature=0.0, max_tokens=50))
        return {"tool_calls": [tc.to_dict() for tc in r.tool_calls], "text": r.text[:50]}

    def long_context(self) -> Any:
        filler = ("The quick brown fox jumps over the lazy dog. " * 10 + "\n") * max(1, self.long_context_tokens // 100)
        r = self.client.complete(ModelRequest([Message.text("user", filler + "\nThe secret word is PELICAN.\nWhat is the secret word? One word.")], temperature=0.0, max_tokens=5))
        return {"text": r.text.strip()[:20], "prompt_tokens": r.usage.prompt_tokens}

    def run_all(self) -> dict[str, Any]:
        results = [self._timed("text", self.text), self._timed("image", self.image), self._timed("tool_call", self.tool_call), self._timed("long_context", self.long_context)]
        return {"ok": all(r["ok"] for r in results), "checks": results, "backend": self.client.health()}
