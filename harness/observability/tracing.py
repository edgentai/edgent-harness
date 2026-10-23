"""Execution trace as a first-class, replayable artifact.

One JSONL file per run. Every model call, tool call, verifier verdict,
state transition and budget check is an event with a monotonically
increasing sequence number so the run can be replayed in order.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from ..core.utils import Clock, ensure_dir, redact


@dataclass
class TraceEvent:
    seq: int
    ts: float
    run_id: str
    kind: str  # model_call | tool_call | verdict | node | budget | plan | gate | error | note
    data: dict[str, Any] = field(default_factory=dict)
    node_id: Optional[str] = None
    parent_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {"seq": self.seq, "ts": self.ts, "run_id": self.run_id, "kind": self.kind,
                "node_id": self.node_id, "parent_id": self.parent_id, "data": self.data}


class Tracer:
    def __init__(self, traces_dir: str, run_id: str, clock: Optional[Clock] = None, echo: Optional[Any] = None) -> None:
        self.dir = ensure_dir(traces_dir)
        self.run_id = run_id
        self.clock = clock or Clock()
        self.path = os.path.join(self.dir, f"{run_id}.jsonl")
        self._lock = threading.Lock()
        self._seq = self._last_seq()
        self._echo = echo  # optional callable for live display

    def _last_seq(self) -> int:
        if not os.path.exists(self.path):
            return 0
        last = 0
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    last = json.loads(line)["seq"]
                except Exception:
                    continue
        return last

    def record(self, kind: str, data: Optional[dict[str, Any]] = None, node_id: Optional[str] = None,
               parent_id: Optional[str] = None) -> TraceEvent:
        with self._lock:
            self._seq += 1
            ev = TraceEvent(self._seq, self.clock.now(), self.run_id, kind, redact(data or {}), node_id, parent_id)
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
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                yield TraceEvent(d["seq"], d["ts"], d["run_id"], d["kind"], d.get("data", {}), d.get("node_id"), d.get("parent_id"))

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        tokens = 0
        tool_errors = 0
        verdicts: dict[str, int] = {}
        for ev in self:
            counts[ev.kind] = counts.get(ev.kind, 0) + 1
            if ev.kind == "model_call":
                u = ev.data.get("usage") or {}
                tokens += int(u.get("prompt_tokens", 0)) + int(u.get("completion_tokens", 0))
            elif ev.kind == "tool_call" and not ev.data.get("ok", True):
                tool_errors += 1
            elif ev.kind == "verdict":
                s = str(ev.data.get("status") or ("passed" if ev.data.get("passed") else "failed"))
                verdicts[s] = verdicts.get(s, 0) + 1
            elif ev.kind == "gate":
                s = "gate:" + str(ev.data.get("status"))
                verdicts[s] = verdicts.get(s, 0) + 1
        return {"events": counts, "tokens": tokens, "tool_errors": tool_errors, "verdicts": verdicts}

    def replay(self, handler: Any) -> None:
        """Feed events in order to ``handler(event)`` (debugging / audit)."""
        for ev in self:
            handler(ev)
