"""The four tiers and the facade that gates writes and supports inspection/deletion."""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.types import Message
from ..core.utils import IdGenerator, ensure_dir, estimate_tokens, now_iso, truncate
from .retrieval import BM25Index, hybrid_search


# ----------------------------------------------------------------- working
class WorkingMemory:
    """Messages of one sub-agent plus compaction into the episodic tier."""

    def __init__(self, max_tokens: int = 12_000, keep_recent: int = 6) -> None:
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent
        self.messages: list[Message] = []
        self.stance: list[str] = []  # compact record of conclusions reached (§10.1)

    def add(self, msg: Message) -> None:
        self.messages.append(msg)

    def tokens(self) -> int:
        return sum(estimate_tokens(m.plain_text()) + 20 * len(m.tool_calls) for m in self.messages)

    def needs_compaction(self) -> bool:
        return self.tokens() > self.max_tokens and len(self.messages) > self.keep_recent + 2

    def compact(self, summariser: Optional[Any] = None) -> Optional[str]:
        """Summarise old turns (keeping the system and the newest messages). Returns the summary text."""
        if not self.needs_compaction():
            return None
        head = [m for m in self.messages[:1] if m.role == "system"]
        body = self.messages[len(head) : -self.keep_recent]
        tail = self.messages[-self.keep_recent :]
        if summariser is not None:
            try:
                summary = summariser(body)
            except Exception:
                summary = None
        else:
            summary = None
        if not summary:
            lines = []
            for m in body:
                if m.role == "tool":
                    lines.append(f"- tool {m.name}: {truncate(m.plain_text(), 160)}")
                elif m.tool_calls:
                    lines.append(f"- called {', '.join(tc.name for tc in m.tool_calls)}")
                elif m.plain_text():
                    lines.append(f"- {m.role}: {truncate(m.plain_text(), 200)}")
            summary = "Earlier turns (compacted):\n" + "\n".join(lines[-40:])
        if self.stance:
            summary += "\nConclusions so far:\n" + "\n".join(f"- {s}" for s in self.stance[-10:])
        self.messages = head + [Message.text("user", summary)] + tail
        return summary


# ----------------------------------------------------------------- episodic
class EpisodicMemory:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        with closing(self._conn()) as c:
            c.execute("CREATE TABLE IF NOT EXISTS episodes(id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, node_id TEXT, kind TEXT, content TEXT, ts TEXT)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_ep_run ON episodes(run_id)")

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def record(self, run_id: str, node_id: str, kind: str, content: Any) -> None:
        with self._lock, closing(self._conn()) as c:
            c.execute("INSERT INTO episodes(run_id,node_id,kind,content,ts) VALUES (?,?,?,?,?)", (run_id, node_id, kind, json.dumps(content, default=str), now_iso()))

    def for_run(self, run_id: str, kind: Optional[str] = None) -> list[dict[str, Any]]:
        q = "SELECT node_id, kind, content, ts FROM episodes WHERE run_id=?"
        args: list[Any] = [run_id]
        if kind:
            q += " AND kind=?"
            args.append(kind)
        with closing(self._conn()) as c:
            rows = c.execute(q + " ORDER BY id", args).fetchall()
        return [{"node_id": r[0], "kind": r[1], "content": json.loads(r[2]), "ts": r[3]} for r in rows]

    def lessons(self, limit: int = 10) -> list[str]:
        with closing(self._conn()) as c:
            rows = c.execute("SELECT content FROM episodes WHERE kind='lesson' ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [json.loads(r[0]) for r in rows]


# ----------------------------------------------------------------- semantic
@dataclass
class Fact:
    id: str
    text: str
    subject: str = ""
    valid_from: Optional[str] = None  # when true in the world
    valid_to: Optional[str] = None
    learned_at: str = ""  # when the harness learned it
    retracted_at: Optional[str] = None
    source: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    verified: bool = False
    user_id: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class SemanticMemory:
    """Bi-temporal fact store + indexed documents (artifact views) with hybrid retrieval."""

    def __init__(self, db_path: str, embed: Optional[Any] = None, ids: Optional[IdGenerator] = None) -> None:
        self.db_path = db_path
        self.embed = embed
        self.ids = ids or IdGenerator()
        self._lock = threading.Lock()
        self.index = BM25Index()
        self.texts: dict[str, str] = {}
        self.vectors: dict[str, list[float]] = {}
        with closing(self._conn()) as c:
            c.execute("CREATE TABLE IF NOT EXISTS facts(id TEXT PRIMARY KEY, text TEXT, subject TEXT, valid_from TEXT, valid_to TEXT, learned_at TEXT, retracted_at TEXT, source TEXT, provenance TEXT, verified INTEGER, user_id TEXT)")
            c.execute("CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY, text TEXT, artifact_id TEXT, locator TEXT, user_id TEXT, added_at TEXT)")
        self._load()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def _load(self) -> None:
        with closing(self._conn()) as c:
            for fid, text in c.execute("SELECT id, text FROM facts WHERE retracted_at IS NULL"):
                self.index.add(fid, text)
                self.texts[fid] = text
            for did, text in c.execute("SELECT id, text FROM documents"):
                self.index.add(did, text)
                self.texts[did] = text

    # facts
    def add_fact(self, text: str, source: str, provenance: dict[str, Any], verified: bool, subject: str = "", valid_from: Optional[str] = None,
                 valid_to: Optional[str] = None, user_id: str = "default") -> Fact:
        if not verified:
            raise PermissionError("writes to semantic memory are gated: only verified facts with provenance are stored")
        if not provenance:
            raise PermissionError("a fact needs provenance (artifact IDs, run ID, locator)")
        fact = Fact(self.ids.new("fact"), text, subject, valid_from, valid_to, now_iso(), None, source, provenance, True, user_id)
        with self._lock, closing(self._conn()) as c:
            c.execute("INSERT INTO facts VALUES (?,?,?,?,?,?,?,?,?,?,?)", (fact.id, text, subject, valid_from, valid_to, fact.learned_at, None, source, json.dumps(provenance, default=str), 1, user_id))
        self._index(fact.id, text)
        return fact

    def retract(self, fact_id: str) -> bool:
        with self._lock, closing(self._conn()) as c:
            n = c.execute("UPDATE facts SET retracted_at=? WHERE id=? AND retracted_at IS NULL", (now_iso(), fact_id)).rowcount
        if n:
            self.index.remove(fact_id)
            self.texts.pop(fact_id, None)
            self.vectors.pop(fact_id, None)
        return bool(n)

    def facts(self, subject: Optional[str] = None, as_of_learned: Optional[str] = None, user_id: Optional[str] = None, include_retracted: bool = False) -> list[Fact]:
        """``as_of_learned`` answers "what did we believe at time T?" (bi-temporal query)."""
        q = "SELECT * FROM facts WHERE 1=1"
        args: list[Any] = []
        if subject:
            q += " AND subject=?"
            args.append(subject)
        if user_id:
            q += " AND user_id=?"
            args.append(user_id)
        if as_of_learned:
            q += " AND learned_at<=? AND (retracted_at IS NULL OR retracted_at>?)"
            args += [as_of_learned, as_of_learned]
        elif not include_retracted:
            q += " AND retracted_at IS NULL"
        with closing(self._conn()) as c:
            rows = c.execute(q + " ORDER BY learned_at", args).fetchall()
        return [Fact(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], json.loads(r[8] or "{}"), bool(r[9]), r[10]) for r in rows]

    def delete_user(self, user_id: str) -> int:
        with self._lock, closing(self._conn()) as c:
            ids = [r[0] for r in c.execute("SELECT id FROM facts WHERE user_id=?", (user_id,))] + [r[0] for r in c.execute("SELECT id FROM documents WHERE user_id=?", (user_id,))]
            c.execute("DELETE FROM facts WHERE user_id=?", (user_id,))
            c.execute("DELETE FROM documents WHERE user_id=?", (user_id,))
        for i in ids:
            self.index.remove(i)
            self.texts.pop(i, None)
            self.vectors.pop(i, None)
        return len(ids)

    # documents
    def add_document(self, text: str, artifact_id: str, locator: dict[str, Any], user_id: str = "default") -> str:
        did = "doc_" + artifact_id[:12] + "_" + str(locator.get("index", locator.get("offset", 0)))
        with self._lock, closing(self._conn()) as c:
            c.execute("INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?)", (did, text, artifact_id, json.dumps(locator), user_id, now_iso()))
        self._index(did, text)
        return did

    def document_meta(self, doc_id: str) -> Optional[dict[str, Any]]:
        with closing(self._conn()) as c:
            row = c.execute("SELECT artifact_id, locator FROM documents WHERE id=?", (doc_id,)).fetchone()
        return {"artifact_id": row[0], "locator": json.loads(row[1])} if row else None

    def _index(self, doc_id: str, text: str) -> None:
        self.index.add(doc_id, text)
        self.texts[doc_id] = text
        if self.embed is not None:
            try:
                self.vectors[doc_id] = self.embed([text])[0]
            except Exception:
                pass

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        hits = hybrid_search(query, self.index, self.texts, k, self.embed, self.vectors or None)
        out = []
        for doc_id, score in hits:
            item: dict[str, Any] = {"id": doc_id, "score": round(score, 3), "text": truncate(self.texts.get(doc_id, ""), 500)}
            if doc_id.startswith("doc_"):
                item.update(self.document_meta(doc_id) or {})
                item["tier"] = "semantic:document"
            else:
                item["tier"] = "semantic:fact"
            out.append(item)
        return out


# ----------------------------------------------------------------- procedural
class ProceduralMemory:
    """Versioned files: reusable plans, tool recipes, improved tool descriptions, prompts."""

    def __init__(self, root: str) -> None:
        self.root = ensure_dir(root)

    def _dir(self, kind: str, name: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
        return ensure_dir(os.path.join(self.root, kind, safe))

    def save(self, kind: str, name: str, content: Any, note: str = "") -> int:
        d = self._dir(kind, name)
        versions = self.versions(kind, name)
        v = (max(versions) + 1) if versions else 1
        with open(os.path.join(d, f"v{v}.json"), "w", encoding="utf-8") as f:
            json.dump({"version": v, "saved_at": now_iso(), "note": note, "content": content}, f, indent=2, default=str)
        return v

    def versions(self, kind: str, name: str) -> list[int]:
        d = self._dir(kind, name)
        out = []
        for fn in os.listdir(d):
            if fn.startswith("v") and fn.endswith(".json"):
                try:
                    out.append(int(fn[1:-5]))
                except ValueError:
                    pass
        return sorted(out)

    def load(self, kind: str, name: str, version: Optional[int] = None) -> Optional[dict[str, Any]]:
        versions = self.versions(kind, name)
        if not versions:
            return None
        v = version or versions[-1]
        with open(os.path.join(self._dir(kind, name), f"v{v}.json"), "r", encoding="utf-8") as f:
            return json.load(f)

    def list(self, kind: str) -> list[str]:
        d = os.path.join(self.root, kind)
        return sorted(os.listdir(d)) if os.path.isdir(d) else []


# ----------------------------------------------------------------- facade
class MemorySystem:
    def __init__(self, memory_dir: str, embed: Optional[Any] = None, ids: Optional[IdGenerator] = None) -> None:
        ensure_dir(memory_dir)
        self.episodic = EpisodicMemory(os.path.join(memory_dir, "episodic.sqlite3"))
        self.semantic = SemanticMemory(os.path.join(memory_dir, "semantic.sqlite3"), embed, ids)
        self.procedural = ProceduralMemory(os.path.join(memory_dir, "procedural"))

    def new_working(self, max_tokens: int = 12_000) -> WorkingMemory:
        return WorkingMemory(max_tokens=max_tokens)

    def index_artifact(self, store: Any, artifact_id: str, user_id: str = "default") -> int:
        """Multimodal retrieval goes through derived text views with pointers back to the artifact."""
        n = 0
        chunks = store.get_view(artifact_id, "chunks")
        if chunks and chunks.data():
            for ch in chunks.data():
                self.semantic.add_document(ch["text"], artifact_id, {"index": ch["index"], "offset": ch["offset"], "length": ch["length"], "view": "text"}, user_id)
                n += 1
        for name in ("transcript", "ocr", "caption"):
            v = store.get_view(artifact_id, name)
            if v and v.text and name != "text":
                self.semantic.add_document(v.text[:4000], artifact_id, {"view": name, "offset": 0}, user_id)
                n += 1
        return n

    def search(self, query: str, k: int = 5, tiers: Optional[list[str]] = None) -> list[dict[str, Any]]:
        tiers = tiers or ["semantic"]
        out: list[dict[str, Any]] = []
        if "semantic" in tiers:
            out += self.semantic.search(query, k)
        if "procedural" in tiers:
            for kind in ("lessons", "plans", "recipes"):
                for name in self.procedural.list(kind):
                    if any(t in name.lower() for t in query.lower().split()):
                        out.append({"id": f"{kind}/{name}", "tier": "procedural", "text": truncate(json.dumps(self.procedural.load(kind, name), default=str), 500), "score": 0.5})
        return out[:k]

    def commit_verified(self, run_id: str, claims: list[Any], user_id: str = "default") -> list[Fact]:
        """Gated write: only claims marked verified, with their evidence as provenance."""
        stored = []
        for c in claims:
            if getattr(c, "verified", False) and c.evidence:
                prov = {"run_id": run_id, "evidence": [e.to_dict() for e in c.evidence]}
                stored.append(self.semantic.add_fact(c.text, f"run:{run_id}", prov, True, user_id=user_id))
        return stored

    def forget_user(self, user_id: str) -> int:
        return self.semantic.delete_user(user_id)
