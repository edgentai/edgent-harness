"""Execution trace as a first-class, replayable artifact: one JSONL file per session."""
from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from ..domain.utils import Clock, ensure_dir, redact


@dataclass
class TraceEvent:
    seq: int
    ts: float
    session_id: str
    kind: str  # model_call | tool_call | permission | compaction | unit | check | budget | subagent | note | error
    data: dict[str, Any] = field(default_factory=dict)
    node_id: str | None = None
    parent_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"seq": self.seq, "ts": self.ts, "session_id": self.session_id, "kind": self.kind, "node_id": self.node_id, "parent_id": self.parent_id, "data": self.data}


class Tracer:
    def __init__(self, traces_dir: str | os.PathLike[str], session_id: str, clock: Clock | None = None, echo: Any | None = None) -> None:
        self.dir = ensure_dir(traces_dir)
        self.session_id = session_id
        self.clock = clock or Clock()
        self.path = os.path.join(self.dir, f"{session_id}.jsonl")
        self._lock = threading.Lock()
        self._seq = self._last_seq()
        self._echo = echo

    def _last_seq(self) -> int:
        if not os.path.exists(self.path):
            return 0
        last = 0
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                try:
                    last = json.loads(line)["seq"]
                except Exception:
                    continue
        return last

    def record(self, kind: str, data: dict[str, Any] | None = None, node_id: str | None = None, parent_id: str | None = None) -> TraceEvent:
        with self._lock:
            self._seq += 1
            ev = TraceEvent(self._seq, self.clock.now(), self.session_id, kind, redact(data or {}), node_id, parent_id)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev.to_dict(), ensure_ascii=False, default=str) + "\n")
        if self._echo:
            try:
                self._echo(ev)
            except Exception:
                pass
        return ev

    def events(self) -> list[TraceEvent]:
        return list(TraceReader(self.path))


class TraceReader:
    def __init__(self, path: str) -> None:
        self.path = path

    def __iter__(self) -> Iterator[TraceEvent]:
        if not os.path.exists(self.path):
            return iter(())
        return self._iter()

    def _iter(self) -> Iterator[TraceEvent]:
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                yield TraceEvent(d["seq"], d["ts"], d.get("session_id", ""), d["kind"], d.get("data", {}), d.get("node_id"), d.get("parent_id"))

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        prompt = completion = cached = 0
        tool_errors = 0
        tool_names: dict[str, int] = {}
        repeated = 0
        seen_calls: set[str] = set()
        for ev in self:
            counts[ev.kind] = counts.get(ev.kind, 0) + 1
            if ev.kind == "model_call":
                u = ev.data.get("usage") or {}
                prompt += int(u.get("prompt_tokens", 0))
                completion += int(u.get("completion_tokens", 0))
                cached += int(u.get("cached_tokens", 0))
            elif ev.kind == "tool_call":
                name = ev.data.get("name", "?")
                tool_names[name] = tool_names.get(name, 0) + 1
                if not ev.data.get("ok", True):
                    tool_errors += 1
                key = name + json.dumps(ev.data.get("arguments", {}), sort_keys=True, default=str)
                if key in seen_calls and not ev.data.get("replayed"):
                    repeated += 1
                seen_calls.add(key)
        return {
            "events": counts,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "cached_tokens": cached,
            "cache_hit_rate": (cached / prompt) if prompt else 0.0,
            "tool_calls": tool_names,
            "tool_errors": tool_errors,
            "repeated_identical_calls": repeated,
        }

    def replay(self, handler: Any) -> None:
        for ev in self:
            handler(ev)
