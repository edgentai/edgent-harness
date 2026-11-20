"""Persistent session transcripts, tiered compaction notes, grep recall & HITL state.

Zero embeddings: every exchange is appended verbatim to ``transcript.jsonl``.
Older turns roll into ``notes.md`` when context pressure rises, and ``recall()``
greps ``transcript.jsonl`` to pull matching past exchanges back verbatim.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..core.types import Message
from ..core.utils import ensure_dir, now_iso, truncate


@dataclass
class SessionState:
    session_id: str
    status: str = "active"  # active | awaiting_review | completed | stopped
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    turn_count: int = 0
    compacted_through_seq: int = 0
    loaded_skills: list[str] = field(default_factory=list)
    pending_review: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SessionState":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


class SessionStore:
    """File-backed session store inside ``<root>/<session_id>/``."""

    def __init__(self, root: str) -> None:
        self.root = ensure_dir(root)

    def _dir(self, session_id: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.\-]", "_", session_id)
        return ensure_dir(os.path.join(self.root, safe))

    def state_path(self, session_id: str) -> str:
        return os.path.join(self._dir(session_id), "state.json")

    def transcript_path(self, session_id: str) -> str:
        return os.path.join(self._dir(session_id), "transcript.jsonl")

    def notes_path(self, session_id: str) -> str:
        return os.path.join(self._dir(session_id), "notes.md")

    def load_state(self, session_id: str) -> SessionState:
        p = self.state_path(session_id)
        if not os.path.exists(p):
            st = SessionState(session_id=session_id)
            self.save_state(st)
            return st
        with open(p, "r", encoding="utf-8") as f:
            return SessionState.from_dict(json.load(f))

    def save_state(self, state: SessionState) -> None:
        state.updated_at = now_iso()
        p = self.state_path(state.session_id)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp, p)

    def list_sessions(self) -> list[dict[str, Any]]:
        out = []
        for name in sorted(os.listdir(self.root)):
            p = os.path.join(self.root, name, "state.json")
            if os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        out.append(json.load(f))
                except Exception:
                    continue
        return out

    def append_message(self, session_id: str, msg: Message) -> int:
        path = self.transcript_path(session_id)
        seq = self._next_seq(path)
        rec = {"seq": seq, "ts": now_iso(), "message": msg.to_dict()}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        return seq

    def _next_seq(self, path: str) -> int:
        if not os.path.exists(path):
            return 1
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f) + 1

    def read_transcript(self, session_id: str, *, after_seq: int = 0) -> list[tuple[int, Message]]:
        path = self.transcript_path(session_id)
        if not os.path.exists(path):
            return []
        out: list[tuple[int, Message]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                seq = int(rec.get("seq", 0))
                if seq > after_seq:
                    out.append((seq, Message.from_dict(rec["message"])))
        return out

    def read_notes(self, session_id: str) -> str:
        p = self.notes_path(session_id)
        if not os.path.exists(p):
            return ""
        with open(p, "r", encoding="utf-8") as f:
            return f.read()

    def write_notes(self, session_id: str, text: str) -> None:
        p = self.notes_path(session_id)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text.strip() + "\n")
        os.replace(tmp, p)

    def compact(
        self,
        session_id: str,
        *,
        keep_recent: int = 8,
        summariser: Optional[Callable[[list[Message]], str]] = None,
        max_notes_chars: int = 8_000,
    ) -> Optional[str]:
        """Roll older transcript messages into ``notes.md`` and advance ``compacted_through_seq``."""
        st = self.load_state(session_id)
        entries = self.read_transcript(session_id, after_seq=st.compacted_through_seq)
        if len(entries) <= keep_recent:
            return None
        to_compact = entries[:-keep_recent]
        cut_seq = to_compact[-1][0]
        msgs = [m for _, m in to_compact]
        block = ""
        if summariser is not None:
            try:
                block = (summariser(msgs) or "").strip()
            except Exception:
                block = ""
        if not block:
            lines = []
            for seq, m in to_compact:
                if m.role == "tool":
                    lines.append(f"- [#{seq}] tool `{m.name}`: {truncate(m.plain_text(), 180)}")
                elif m.tool_calls:
                    lines.append(f"- [#{seq}] assistant called {', '.join(tc.name for tc in m.tool_calls)}")
                else:
                    lines.append(f"- [#{seq}] {m.role}: {truncate(m.plain_text(), 240)}")
            block = "\n".join(lines[-50:])
        existing = self.read_notes(session_id).strip()
        combined = ((existing + "\n\n" if existing else "") + f"### Turns through #{cut_seq}\n{block}").strip()
        if len(combined) > max_notes_chars:
            # Keep the newest notes inside the hard cap.
            combined = "…(earlier notes rolled up)…\n\n" + combined[-max_notes_chars:]
        self.write_notes(session_id, combined)
        st.compacted_through_seq = cut_seq
        self.save_state(st)
        return block

    def recall(self, session_id: str, query: str, *, limit: int = 8) -> list[dict[str, Any]]:
        """Grep-based verbatim recall over ``transcript.jsonl`` (regex or keyword tokens)."""
        entries = self.read_transcript(session_id, after_seq=0)
        q = (query or "").strip()
        if not q:
            return []
        tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", q) if len(t) >= 2]
        try:
            rx = re.compile(q, re.IGNORECASE)
        except re.error:
            rx = None
        scored: list[tuple[int, int, dict[str, Any]]] = []
        for seq, m in entries:
            txt = m.plain_text()
            if not txt:
                continue
            low = txt.lower()
            score = 0
            if rx is not None and rx.search(txt):
                score += 5
            score += sum(2 for tok in tokens if tok in low)
            if score > 0:
                scored.append((score, seq, {"seq": seq, "role": m.role, "name": m.name, "text": truncate(txt, 1200)}))
        scored.sort(key=lambda item: (-item[0], -item[1]))
        return [x[2] for x in scored[:limit]]

    def submit_review(self, session_id: str, *, approved: bool, feedback: str = "") -> SessionState:
        """Record human review decision and reactivate a paused session."""
        st = self.load_state(session_id)
        review = st.pending_review or {}
        st.pending_review = None
        st.status = "active"
        verdict = "APPROVED" if approved else "CHANGES REQUESTED"
        note = f"[HUMAN REVIEW — {verdict}]"
        if review.get("title"):
            note += f" ({review['title']})"
        if feedback.strip():
            note += f"\nFeedback: {feedback.strip()}"
        self.append_message(session_id, Message.text("user", note))
        self.save_state(st)
        return st
