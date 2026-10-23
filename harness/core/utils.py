"""Clock, ID and small helpers. Tests inject deterministic versions."""
from __future__ import annotations

import datetime as _dt
import hashlib
import itertools
import json
import os
import re
import time
import uuid
from typing import Any, Iterator, Optional


class Clock:
    """Wall clock. Replace with ``FakeClock`` in tests."""

    def now(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()

    def iso(self) -> str:
        return _dt.datetime.fromtimestamp(self.now(), tz=_dt.timezone.utc).isoformat()


class FakeClock(Clock):
    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self._t = start

    def now(self) -> float:
        return self._t

    def monotonic(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds


class IdGenerator:
    """Random IDs by default; ``SequentialIdGenerator`` for tests."""

    def __init__(self, prefix: str = "") -> None:
        self.prefix = prefix

    def new(self, kind: str = "id") -> str:
        return f"{self.prefix}{kind}_{uuid.uuid4().hex[:12]}"


class SequentialIdGenerator(IdGenerator):
    def __init__(self, prefix: str = "") -> None:
        super().__init__(prefix)
        self._counters: dict[str, Iterator[int]] = {}

    def new(self, kind: str = "id") -> str:
        it = self._counters.setdefault(kind, itertools.count(1))
        return f"{self.prefix}{kind}_{next(it):04d}"


def now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Optional[Any]:
    """Pull the first JSON object or array out of free text.

    Accepts bare JSON, fenced JSON, or JSON embedded in prose. Returns
    ``None`` when nothing parses. Models are unreliable at emitting clean
    JSON, so every consumer goes through this.
    """
    if text is None:
        return None
    text = text.strip()
    candidates = [text]
    candidates += [m.strip() for m in _FENCE_RE.findall(text)]
    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            pass
    # Scan for the first balanced {...} or [...] block.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        while start != -1:
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(text)):
                ch = text[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except Exception:
                            break
            start = text.find(opener, start + 1)
    return None


def truncate(text: str, limit: int, marker: str = " …[truncated]") -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(marker))] + marker


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=_default)


def _default(o: Any) -> Any:
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (about 4 characters per token) for budgeting."""
    return max(1, len(text) // 4)


def redact(obj: Any, keys: tuple[str, ...] = ("api_key", "authorization", "token", "secret", "password")) -> Any:
    """Recursively redact sensitive fields for logs and traces."""
    if isinstance(obj, dict):
        return {k: ("[REDACTED]" if k.lower() in keys else redact(v, keys)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v, keys) for v in obj]
    return obj
