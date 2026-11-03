"""Sessions: the live conversation, its append-only transcript, tiered notes and stance.

A session is a directory under ``sessions/`` in the workspace. Every message is
appended to ``transcript.jsonl`` before anything else happens, so a crash loses
nothing; ``state.json`` is the resumable view (post-compaction messages, notes,
stance, todo, usage) and ``notes.md`` is the same for humans. Recall is a
keyword or regex grep over the transcript and returns exchanges verbatim: no
embeddings anywhere.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..domain.types import Message, ToolCall, Usage
from ..domain.utils import atomic_write_text, ensure_dir, estimate_tokens, now_iso, read_json, truncate, write_json
from ..domain.workspace import Workspace

MODES = ("chat", "build", "subagent")
STANCE_MAX_ENTRIES = 40
RECALL_TEXT_MAX_CHARS = 1500
RECALL_CONTEXT_MAX_CHARS = 400
RECALL_MAX_HITS = 50
SESSION_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]{0,119}")

TRANSCRIPT_FILE = "transcript.jsonl"
STATE_FILE = "state.json"
NOTES_FILE = "notes.md"
CLEARED_DIR = "cleared"


def validate_session_id(session_id: str) -> str:
    """Session ids name directories; anything that could escape ``sessions/`` is refused."""
    if not isinstance(session_id, str) or not SESSION_ID_RE.fullmatch(session_id) or ".." in session_id:
        raise ValueError(f"invalid session id {session_id!r}: use letters, digits, '-', '_' and '.' (max 120 chars)")
    return session_id


def new_session_id(mode: str = "chat") -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return f"{mode}-{stamp}-{uuid.uuid4().hex[:6]}"


@dataclass
class Session:
    session_id: str
    workspace: Workspace
    mode: str = "chat"
    title: str = ""
    parent_id: str | None = None
    created_at: str = field(default_factory=now_iso)
    messages: list[Message] = field(default_factory=list)  # live, post-compaction; the system prompt is never stored
    notes: list[dict[str, Any]] = field(default_factory=list)  # {'level', 'covers': [first_seq, last_seq], 'text', 'at'}
    stance: list[str] = field(default_factory=list)
    todo: list[dict[str, Any]] = field(default_factory=list)  # {'id', 'content', 'status'}
    meta: dict[str, Any] = field(default_factory=dict)
    usage_total: Usage = field(default_factory=Usage)
    seq: int = 0
    dir: Path = field(init=False)

    def __post_init__(self) -> None:
        validate_session_id(self.session_id)
        if not self.mode:
            raise ValueError("session mode must be a non-empty string")
        self.dir = self.workspace.path("sessions") / self.session_id
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ paths
    @property
    def transcript_path(self) -> Path:
        return self.dir / TRANSCRIPT_FILE

    @property
    def state_path(self) -> Path:
        return self.dir / STATE_FILE

    @property
    def notes_path(self) -> Path:
        return self.dir / NOTES_FILE

    @property
    def cleared_dir(self) -> Path:
        return self.dir / CLEARED_DIR

    # ------------------------------------------------------------------ lifecycle
    @classmethod
    def create(
        cls,
        workspace: Workspace,
        mode: str = "chat",
        title: str | None = None,
        session_id: str | None = None,
        parent_id: str | None = None,
    ) -> Session:
        session = cls(session_id=session_id or new_session_id(mode), workspace=workspace, mode=mode, title=title or "", parent_id=parent_id)
        if session.state_path.exists():
            raise FileExistsError(f"session {session.session_id!r} already exists at {session.dir}")
        ensure_dir(session.dir)
        ensure_dir(session.cleared_dir)
        session.save()
        return session

    @classmethod
    def load(cls, workspace: Workspace, session_id: str) -> Session:
        validate_session_id(session_id)
        state_path = workspace.path("sessions") / session_id / STATE_FILE
        try:
            data = read_json(state_path, None)
        except ValueError as e:
            raise ValueError(f"session {session_id!r} cannot be loaded: {state_path} is not valid JSON ({e}); restore it from git or start a new session") from e
        if data is None:
            raise FileNotFoundError(f"session {session_id!r} not found (no {state_path})")
        if not isinstance(data, dict):
            raise ValueError(f"session {session_id!r} cannot be loaded: {state_path} does not hold a JSON object")
        session = cls(
            session_id=session_id,
            workspace=workspace,
            mode=data.get("mode") or "chat",
            title=data.get("title") or "",
            parent_id=data.get("parent_id"),
            created_at=data.get("created_at") or now_iso(),
            messages=[Message.from_dict(m) for m in data.get("messages") or []],
            notes=list(data.get("notes") or []),
            stance=list(data.get("stance") or []),
            todo=list(data.get("todo") or []),
            meta=dict(data.get("meta") or {}),
            usage_total=Usage.from_dict(data.get("usage_total") or {}),
            seq=int(data.get("seq") or 0),
        )
        session._replay_tail()
        return session

    @staticmethod
    def exists(workspace: Workspace, session_id: str) -> bool:
        try:
            validate_session_id(session_id)
        except ValueError:
            return False
        return (workspace.path("sessions") / session_id / STATE_FILE).is_file()

    def _replay_tail(self) -> None:
        """Messages appended after the last save (a crash mid-turn) live only in the transcript; take them back."""
        for entry in self._transcript_entries():
            if entry["seq"] > self.seq:
                msg = Message.from_dict(entry)
                msg.meta["seq"] = entry["seq"]
                self.messages.append(msg)
                self.seq = entry["seq"]

    # ------------------------------------------------------------------ writing
    def append(self, message: Message) -> int:
        """Persist the message to the transcript, then add it to the live conversation. Returns its seq."""
        with self._lock:
            self.seq += 1
            message.meta["seq"] = self.seq
            line = {"seq": self.seq, "ts": now_iso(), **message.to_dict()}
            ensure_dir(self.dir)
            with open(self.transcript_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
            self.messages.append(message)
            return self.seq

    def save(self) -> None:
        with self._lock:
            ensure_dir(self.dir)
            write_json(self.state_path, self.to_dict())
            atomic_write_text(self.notes_path, self.render_notes())

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "mode": self.mode,
            "title": self.title,
            "parent_id": self.parent_id,
            "created_at": self.created_at,
            "seq": self.seq,
            "messages": [m.to_dict() for m in self.messages],
            "notes": list(self.notes),
            "stance": list(self.stance),
            "todo": list(self.todo),
            "meta": dict(self.meta),
            "usage_total": self.usage_total.to_dict(),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "mode": self.mode,
            "title": self.title,
            "created_at": self.created_at,
            "messages": len(self.messages),
            "parent_id": self.parent_id,
        }

    # ------------------------------------------------------------------ small helpers
    def tokens(self) -> int:
        return sum(m.tokens() for m in self.messages)

    def notes_tokens(self) -> int:
        return sum(estimate_tokens(str(n.get("text", ""))) for n in self.notes) + sum(estimate_tokens(s) for s in self.stance)

    def recent(self, n: int) -> list[Message]:
        return self.messages[-n:] if n > 0 else []

    def last_text(self, role: str = "assistant") -> str:
        for m in reversed(self.messages):
            if m.role == role and m.text():
                return m.text()
        return ""

    def add_stance(self, text: str) -> bool:
        """Keep a short conclusion or constraint to carry forward; duplicates are ignored, the oldest drop past the cap."""
        clean = " ".join(str(text).split())
        if not clean or clean in self.stance:
            return False
        self.stance.append(clean)
        del self.stance[:-STANCE_MAX_ENTRIES]
        return True

    def add_usage(self, usage: Usage) -> Usage:
        self.usage_total = self.usage_total + usage
        return self.usage_total

    def record_child(self, child_id: str) -> None:
        """Remember a subagent session spawned from this one; safe from the tool threads that spawn them."""
        with self._lock:
            children = self.meta.setdefault("children", [])
            if child_id not in children:
                children.append(child_id)

    def dangling_tool_calls(self) -> list[ToolCall]:
        """Calls of the last assistant message that have no tool result after it (a crash or a budget stop mid-call)."""
        for i in range(len(self.messages) - 1, -1, -1):
            m = self.messages[i]
            if m.role == "assistant" and m.tool_calls:
                answered = {t.tool_call_id for t in self.messages[i + 1 :] if t.role == "tool"}
                return [tc for tc in m.tool_calls if tc.id not in answered]
        return []

    # ------------------------------------------------------------------ recall
    def _transcript_entries(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        if not self.transcript_path.is_file():
            return entries
        with open(self.transcript_path, encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    d = json.loads(raw)
                except ValueError:
                    continue  # a torn last line after a crash
                if isinstance(d, dict) and "seq" in d and "role" in d:
                    entries.append(d)
        return entries

    @staticmethod
    def _entry_text(entry: dict[str, Any]) -> str:
        content = entry.get("content", [])
        if isinstance(content, str):
            parts = [content]
        else:
            parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("text")]
        for tc in entry.get("tool_calls") or []:
            if isinstance(tc, dict):
                parts.append(f"[call {tc.get('name')} {json.dumps(tc.get('arguments', {}), ensure_ascii=False, sort_keys=True)}]")
        return "\n".join(parts)

    def recall(self, query: str, k: int = 5, window: int = 1) -> list[dict[str, Any]]:
        """Keyword (or ``re:`` regex) search over the whole transcript; returns verbatim hits with neighbours."""
        entries = self._transcript_entries()
        regex: re.Pattern[str] | None = None
        terms: list[str] = []
        if query.startswith("re:"):
            try:
                regex = re.compile(query[3:], re.IGNORECASE)
            except re.error as e:
                raise ValueError(f"invalid regex in recall query: {e}") from e
        else:
            terms = sorted({t for t in query.lower().split() if t})
        if regex is None and not terms:
            return []
        scored: list[tuple[int, int, int]] = []
        texts = [self._entry_text(e) for e in entries]
        for idx, text in enumerate(texts):
            if regex is not None:
                score = len(regex.findall(text))
            else:
                low = text.lower()
                score = sum(1 for t in terms if t in low)
            if score > 0:
                scored.append((score, int(entries[idx]["seq"]), idx))
        scored.sort(key=lambda s: (-s[0], -s[1]))
        out: list[dict[str, Any]] = []
        for score, seq, idx in scored[: max(0, min(k, RECALL_MAX_HITS))]:
            lo, hi = max(0, idx - window), min(len(entries), idx + window + 1)
            context = [
                {"seq": int(entries[j]["seq"]), "role": entries[j]["role"], "text": truncate(texts[j], RECALL_CONTEXT_MAX_CHARS)}
                for j in range(lo, hi)
                if j != idx
            ]
            out.append({"seq": seq, "role": entries[idx]["role"], "text": truncate(texts[idx], RECALL_TEXT_MAX_CHARS), "score": score, "context": context})
        return out

    # ------------------------------------------------------------------ rendering
    def render_notes(self) -> str:
        head = f"# Session {self.session_id} ({self.mode})"
        if self.title:
            head += f": {self.title}"
        lines = [head, ""]
        if self.stance:
            lines += ["## Stance", *(f"- {s}" for s in self.stance), ""]
        if self.todo:
            lines += ["## Todo", *(f"- [{t.get('status', 'pending')}] {t.get('content', '')}" for t in self.todo), ""]
        lines.append("## Notes")
        if not self.notes:
            lines.append("_(no compaction notes yet)_")
        for n in self.notes:
            covers = n.get("covers") or [0, 0]
            lines += [f"### Level {n.get('level', 0)}, messages {covers[0]}-{covers[1]} (at {n.get('at', '')})", str(n.get("text", "")), ""]
        return "\n".join(lines).rstrip() + "\n"


class SessionStore:
    """Lists, loads and creates sessions under the workspace's ``sessions/`` directory."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.dir = workspace.path("sessions")

    def list(self, mode: str | None = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not self.dir.is_dir():
            return out
        for d in self.dir.iterdir():
            try:
                data = read_json(d / STATE_FILE, None) if d.is_dir() else None
            except ValueError:
                continue  # one corrupt state.json must not hide every other session
            if not isinstance(data, dict):
                continue
            if mode is not None and data.get("mode") != mode:
                continue
            out.append(
                {
                    "session_id": data.get("session_id") or d.name,
                    "mode": data.get("mode") or "chat",
                    "title": data.get("title") or "",
                    "created_at": data.get("created_at") or "",
                    "messages": len(data.get("messages") or []),
                    "parent_id": data.get("parent_id"),
                }
            )
        out.sort(key=lambda s: (s["created_at"], s["session_id"]), reverse=True)
        return out

    def latest(self, mode: str | None = None) -> Session | None:
        rows = self.list(mode)
        return self.get(rows[0]["session_id"]) if rows else None

    def get(self, session_id: str) -> Session:
        return Session.load(self.workspace, session_id)

    def create(self, mode: str = "chat", title: str | None = None, session_id: str | None = None, parent_id: str | None = None) -> Session:
        return Session.create(self.workspace, mode=mode, title=title, session_id=session_id, parent_id=parent_id)

    def exists(self, session_id: str) -> bool:
        return Session.exists(self.workspace, session_id)
