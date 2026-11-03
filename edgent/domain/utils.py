"""Clock, IDs, hashing, JSON helpers. Tests inject deterministic versions."""
from __future__ import annotations

import datetime as _dt
import hashlib
import itertools
import json
import os
import re
import time
import uuid
from collections.abc import Iterator
from typing import Any


class Clock:
    """Wall clock. Replace with ``FakeClock`` in tests."""

    def now(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()

    def iso(self) -> str:
        return _dt.datetime.fromtimestamp(self.now(), tz=_dt.UTC).isoformat()


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
    return _dt.datetime.now(tz=_dt.UTC).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, UTF-8 preserved."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_default)


def json_dumps(obj: Any, indent: int | None = None) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=indent, default=_default)


def _default(o: Any) -> Any:
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, bytes):
        return o.decode("utf-8", "replace")
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str | None) -> Any | None:
    """Pull the first JSON object or array out of free text (bare, fenced, or embedded)."""
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


def ensure_dir(path: str | os.PathLike[str]) -> str:
    os.makedirs(path, exist_ok=True)
    return str(path)


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (about 4 characters per token) for budgeting."""
    return max(1, len(text) // 4) if text else 0


def redact(obj: Any, keys: tuple[str, ...] = ("api_key", "authorization", "token", "secret", "password", "x-goog-api-key")) -> Any:
    """Recursively redact sensitive fields for logs and traces."""
    if isinstance(obj, dict):
        return {k: ("[REDACTED]" if str(k).lower() in keys else redact(v, keys)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v, keys) for v in obj]
    return obj


def atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    """Write via a temp file and rename, so a reader never sees half a file."""
    path = str(path)
    ensure_dir(os.path.dirname(path) or ".")
    tmp = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex[:6]}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def read_json(path: str | os.PathLike[str], default: Any = None) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def write_json(path: str | os.PathLike[str], obj: Any) -> None:
    atomic_write_text(path, json_dumps(obj, indent=2) + "\n")


def slugify(text: str, max_len: int = 60) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:max_len].rstrip("-") or "untitled"


def parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Parse a ``---`` YAML-ish front matter block (flat keys, lists, booleans, ints) and return (meta, body).

    Deliberately small: skills and agent definitions use flat keys and simple lists; a full YAML
    parser is not worth a dependency.
    """
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    block = text[3:end].strip("\n")
    body = text[end + 4 :].lstrip("\n")
    meta: dict[str, Any] = {}
    current_list_key: str | None = None
    for raw in block.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.strip().startswith("#"):
            continue
        if line.startswith((" ", "\t")) and current_list_key and line.strip().startswith("- "):
            meta.setdefault(current_list_key, []).append(_scalar(line.strip()[2:]))
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if val == "":
            meta[key] = []
            current_list_key = key
            continue
        current_list_key = None
        if val.startswith("[") and val.endswith("]"):
            meta[key] = [_scalar(v.strip()) for v in val[1:-1].split(",") if v.strip()]
        else:
            meta[key] = _scalar(val)
    return meta, body


def _scalar(v: str) -> Any:
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    low = v.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "none", "~"):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v
