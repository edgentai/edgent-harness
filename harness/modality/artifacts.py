"""Content-addressed artifact store with derived views and provenance.

Raw bytes live under ``artifacts_dir/<aa>/<hash>``; metadata, views and the
provenance graph live in SQLite. Derived products (frames, transcripts,
crops) are artifacts too, keyed by ``(parent, operation, params)`` so that
repeated inputs are free (§14.4).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.types import Modality
from ..core.utils import ensure_dir, now_iso, sha256_bytes, sha256_file


@dataclass
class View:
    artifact_id: str
    name: str  # "text" | "ocr" | "caption" | "transcript" | "scenes" | "structure" | "chunks" | ...
    kind: str  # text | json | file
    text: Optional[str] = None
    path: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)
    untrusted: bool = True  # derived from media/files: data, never instructions

    def to_dict(self) -> dict[str, Any]:
        return {"artifact_id": self.artifact_id, "name": self.name, "kind": self.kind, "text": self.text,
                "path": self.path, "meta": self.meta, "untrusted": self.untrusted}

    def data(self) -> Any:
        if self.kind == "json" and self.text is not None:
            return json.loads(self.text)
        return self.text


@dataclass
class Artifact:
    id: str  # sha256 of content
    mime: str
    modality: Modality
    size: int
    path: str
    source: str = ""
    ingested_at: str = ""
    duration_s: Optional[float] = None
    meta: dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    derived_kind: Optional[str] = None  # e.g. "frame", "audio_track", "crop"
    derived_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["modality"] = self.modality.value
        return d

    @property
    def short(self) -> str:
        return self.id[:12]


def _derived_key(parent_id: str, kind: str, params: dict[str, Any]) -> str:
    return sha256_bytes((parent_id + "|" + kind + "|" + json.dumps(params, sort_keys=True)).encode())


class ArtifactStore:
    def __init__(self, root: str, db_path: Optional[str] = None) -> None:
        self.root = ensure_dir(root)
        self.db_path = db_path or os.path.join(root, "artifacts.sqlite3")
        ensure_dir(os.path.dirname(self.db_path))
        self._local = threading.local()
        self._all_conns: list[sqlite3.Connection] = []
        self._conn_lock = threading.Lock()
        self._init()

    @property
    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.db_path, timeout=30, isolation_level=None, check_same_thread=False)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA busy_timeout=30000")
            c.row_factory = sqlite3.Row
            self._local.conn = c
            with self._conn_lock:
                self._all_conns.append(c)
        return c

    def _init(self) -> None:
        c = self._conn
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS artifacts(
                id TEXT PRIMARY KEY, mime TEXT, modality TEXT, size INTEGER, path TEXT, source TEXT,
                ingested_at TEXT, duration_s REAL, meta TEXT, parent_id TEXT, derived_kind TEXT, derived_params TEXT
            );
            CREATE TABLE IF NOT EXISTS views(
                artifact_id TEXT, name TEXT, kind TEXT, text TEXT, path TEXT, meta TEXT, untrusted INTEGER,
                PRIMARY KEY(artifact_id, name)
            );
            CREATE TABLE IF NOT EXISTS derived(key TEXT PRIMARY KEY, artifact_id TEXT);
            CREATE INDEX IF NOT EXISTS idx_parent ON artifacts(parent_id);
            """
        )

    # -- raw storage ------------------------------------------------------
    def _blob_path(self, digest: str, ext: str = "") -> str:
        d = ensure_dir(os.path.join(self.root, digest[:2]))
        return os.path.join(d, digest + ext)

    def put_bytes(self, data: bytes, mime: str, modality: Modality, source: str = "", ext: str = "",
                  duration_s: Optional[float] = None, meta: Optional[dict[str, Any]] = None,
                  parent_id: Optional[str] = None, derived_kind: Optional[str] = None,
                  derived_params: Optional[dict[str, Any]] = None) -> Artifact:
        digest = sha256_bytes(data)
        path = self._blob_path(digest, ext)
        if not os.path.exists(path):
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
        return self._register(digest, mime, modality, len(data), path, source, duration_s, meta or {}, parent_id, derived_kind, derived_params or {})

    def put_file(self, src: str, mime: str, modality: Modality, source: str = "", duration_s: Optional[float] = None,
                 meta: Optional[dict[str, Any]] = None, parent_id: Optional[str] = None, derived_kind: Optional[str] = None,
                 derived_params: Optional[dict[str, Any]] = None, move: bool = False) -> Artifact:
        digest = sha256_file(src)
        ext = os.path.splitext(src)[1].lower()
        path = self._blob_path(digest, ext)
        if not os.path.exists(path):
            if move:
                os.replace(src, path)
            else:
                import shutil

                shutil.copy2(src, path)
        return self._register(digest, mime, modality, os.path.getsize(path), path, source or src, duration_s, meta or {}, parent_id, derived_kind, derived_params or {})

    def _register(self, digest: str, mime: str, modality: Modality, size: int, path: str, source: str,
                  duration_s: Optional[float], meta: dict[str, Any], parent_id: Optional[str],
                  derived_kind: Optional[str], derived_params: dict[str, Any]) -> Artifact:
        existing = self.get(digest)
        if existing:
            # Immutable: keep the first record, but merge any new metadata keys.
            if meta and any(k not in existing.meta for k in meta):
                existing.meta.update({k: v for k, v in meta.items() if k not in existing.meta})
                self._conn.execute("UPDATE artifacts SET meta=? WHERE id=?", (json.dumps(existing.meta), digest))
            return existing
        art = Artifact(digest, mime, modality, size, path, source, now_iso(), duration_s, meta, parent_id, derived_kind, derived_params)
        self._conn.execute(
            "INSERT INTO artifacts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (art.id, mime, modality.value, size, path, source, art.ingested_at, duration_s, json.dumps(meta), parent_id,
             derived_kind, json.dumps(derived_params, sort_keys=True)),
        )
        if parent_id and derived_kind:
            self._conn.execute("INSERT OR REPLACE INTO derived VALUES (?,?)", (_derived_key(parent_id, derived_kind, derived_params), art.id))
        return art

    def get(self, artifact_id: str) -> Optional[Artifact]:
        row = self._conn.execute("SELECT * FROM artifacts WHERE id=?", (artifact_id,)).fetchone()
        if not row and len(artifact_id) >= 8:
            rows = self._conn.execute("SELECT * FROM artifacts WHERE id LIKE ?", (artifact_id + "%",)).fetchall()
            row = rows[0] if len(rows) == 1 else None
        return self._row(row) if row else None

    def _row(self, row: sqlite3.Row) -> Artifact:
        return Artifact(row["id"], row["mime"], Modality(row["modality"]), row["size"], row["path"], row["source"],
                        row["ingested_at"], row["duration_s"], json.loads(row["meta"] or "{}"), row["parent_id"],
                        row["derived_kind"], json.loads(row["derived_params"] or "{}"))

    def read_bytes(self, artifact_id: str) -> bytes:
        art = self.get(artifact_id)
        if not art:
            raise KeyError(artifact_id)
        with open(art.path, "rb") as f:
            return f.read()

    def list(self, modality: Optional[Modality] = None, roots_only: bool = False) -> list[Artifact]:
        q = "SELECT * FROM artifacts WHERE 1=1"
        args: list[Any] = []
        if modality:
            q += " AND modality=?"
            args.append(modality.value)
        if roots_only:
            q += " AND parent_id IS NULL"
        q += " ORDER BY ingested_at"
        return [self._row(r) for r in self._conn.execute(q, args).fetchall()]

    def children(self, parent_id: str, kind: Optional[str] = None) -> list[Artifact]:
        q = "SELECT * FROM artifacts WHERE parent_id=?"
        args: list[Any] = [parent_id]
        if kind:
            q += " AND derived_kind=?"
            args.append(kind)
        return [self._row(r) for r in self._conn.execute(q, args).fetchall()]

    def find_derived(self, parent_id: str, kind: str, params: dict[str, Any]) -> Optional[Artifact]:
        row = self._conn.execute("SELECT artifact_id FROM derived WHERE key=?", (_derived_key(parent_id, kind, params),)).fetchone()
        return self.get(row["artifact_id"]) if row else None

    def provenance(self, artifact_id: str) -> list[Artifact]:
        """Chain from this artifact back to its root."""
        chain = []
        cur = self.get(artifact_id)
        while cur:
            chain.append(cur)
            cur = self.get(cur.parent_id) if cur.parent_id else None
        return chain

    # -- derived views ----------------------------------------------------
    def put_view(self, view: View) -> View:
        self._conn.execute(
            "INSERT OR REPLACE INTO views VALUES (?,?,?,?,?,?,?)",
            (view.artifact_id, view.name, view.kind, view.text, view.path, json.dumps(view.meta, default=str), int(view.untrusted)),
        )
        return view

    def put_text_view(self, artifact_id: str, name: str, text: str, meta: Optional[dict[str, Any]] = None) -> View:
        return self.put_view(View(artifact_id, name, "text", text=text, meta=meta or {}))

    def put_json_view(self, artifact_id: str, name: str, data: Any, meta: Optional[dict[str, Any]] = None) -> View:
        return self.put_view(View(artifact_id, name, "json", text=json.dumps(data, ensure_ascii=False, default=str), meta=meta or {}))

    def get_view(self, artifact_id: str, name: str) -> Optional[View]:
        art = self.get(artifact_id)
        if not art:
            return None
        row = self._conn.execute("SELECT * FROM views WHERE artifact_id=? AND name=?", (art.id, name)).fetchone()
        return self._view_row(row) if row else None

    def views(self, artifact_id: str) -> list[View]:
        art = self.get(artifact_id)
        if not art:
            return []
        return [self._view_row(r) for r in self._conn.execute("SELECT * FROM views WHERE artifact_id=? ORDER BY name", (art.id,)).fetchall()]

    @staticmethod
    def _view_row(row: sqlite3.Row) -> View:
        return View(row["artifact_id"], row["name"], row["kind"], row["text"], row["path"], json.loads(row["meta"] or "{}"), bool(row["untrusted"]))

    def close(self) -> None:
        with self._conn_lock:
            for c in self._all_conns:
                try:
                    c.close()
                except Exception:
                    pass
            self._all_conns.clear()
        self._local = threading.local()
