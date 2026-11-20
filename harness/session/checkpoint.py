"""Per-turn workspace checkpointing and undo (works in git and non-git workspaces).

Before mutating tools (``write_file``, ``edit_file``, ``delete_file``, ``todo_manage``)
modify a file inside the workspace, ``CheckpointManager`` records the file's exact
pre-turn bytes in content-addressed storage under ``<state_dir>/checkpoints/``.
Calling ``undo()`` restores overwritten/deleted files and removes newly-created
files from the last turn (or last N turns).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.utils import ensure_dir, now_iso
from ..sandbox.workspace import Workspace


@dataclass
class FileSnapshot:
    path: str
    existed: bool
    blob_sha: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "existed": self.existed, "blob_sha": self.blob_sha}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FileSnapshot":
        return cls(path=str(d["path"]), existed=bool(d["existed"]), blob_sha=d.get("blob_sha"))


@dataclass
class TurnCheckpoint:
    session_id: str
    summary: str = ""
    snapshots: dict[str, FileSnapshot] = field(default_factory=dict)


class CheckpointManager:
    """Content-addressed file-snapshot checkpoint & undo manager."""

    def __init__(self, workspace: Workspace, root: str) -> None:
        self.workspace = workspace
        self.root = ensure_dir(root)
        self.blobs_dir = ensure_dir(os.path.join(self.root, "blobs"))

    def _session_dir(self, session_id: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.\-]", "_", session_id)
        return ensure_dir(os.path.join(self.root, "sessions", safe))

    def _index_path(self, session_id: str) -> str:
        return os.path.join(self._session_dir(session_id), "index.json")

    def _load_index(self, session_id: str) -> list[dict[str, Any]]:
        p = self._index_path(session_id)
        if not os.path.exists(p):
            return []
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return list(data) if isinstance(data, list) else []
        except Exception:
            return []

    def _save_index(self, session_id: str, entries: list[dict[str, Any]]) -> None:
        p = self._index_path(session_id)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
        os.replace(tmp, p)

    def _store_blob(self, data: bytes) -> str:
        sha = hashlib.sha256(data).hexdigest()
        dst = os.path.join(self.blobs_dir, sha)
        if not os.path.exists(dst):
            tmp = dst + ".tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dst)
        return sha

    def _read_blob(self, sha: str) -> bytes:
        with open(os.path.join(self.blobs_dir, sha), "rb") as f:
            return f.read()

    def begin_turn(self, session_id: str, summary: str = "") -> TurnCheckpoint:
        return TurnCheckpoint(session_id=session_id, summary=summary[:200])

    def capture_before_write(self, turn_cp: TurnCheckpoint, rel_path: str) -> None:
        """Record pre-mutation state of ``rel_path`` (once per path per turn)."""
        try:
            full = self.workspace.resolve(rel_path)
        except Exception:
            return
        norm = os.path.relpath(full, self.workspace.path)
        if norm.startswith(".edgent"):
            return
        if os.path.isdir(full):
            for dirpath, _, filenames in os.walk(full):
                for fn in filenames:
                    f_abs = os.path.join(dirpath, fn)
                    f_rel = os.path.relpath(f_abs, self.workspace.path)
                    if f_rel not in turn_cp.snapshots:
                        try:
                            with open(f_abs, "rb") as f:
                                sha = self._store_blob(f.read())
                            turn_cp.snapshots[f_rel] = FileSnapshot(f_rel, True, sha)
                        except OSError:
                            pass
            return
        if norm in turn_cp.snapshots:
            return
        if os.path.isfile(full):
            try:
                with open(full, "rb") as f:
                    sha = self._store_blob(f.read())
                turn_cp.snapshots[norm] = FileSnapshot(norm, True, sha)
            except OSError:
                pass
        else:
            turn_cp.snapshots[norm] = FileSnapshot(norm, False, None)

    def commit(self, turn_cp: TurnCheckpoint) -> Optional[dict[str, Any]]:
        """Persist a turn checkpoint if any files were captured."""
        if not turn_cp.snapshots:
            return None
        entries = self._load_index(turn_cp.session_id)
        seq = (entries[-1]["seq"] + 1) if entries else 1
        rec = {
            "id": f"cp_{seq:04d}",
            "seq": seq,
            "created_at": now_iso(),
            "summary": turn_cp.summary,
            "files": [snap.to_dict() for snap in turn_cp.snapshots.values()],
        }
        entries.append(rec)
        self._save_index(turn_cp.session_id, entries)
        return rec

    def list_checkpoints(self, session_id: str) -> list[dict[str, Any]]:
        return self._load_index(session_id)

    def undo(self, session_id: str, steps: int = 1) -> dict[str, Any]:
        """Rewind the workspace by the last ``steps`` checkpoints."""
        entries = self._load_index(session_id)
        if not entries:
            return {"ok": False, "reason": "no checkpoints available for this session", "undone": [], "restored_files": [], "removed_files": []}
        count = max(1, min(steps, len(entries)))
        undone: list[dict[str, Any]] = []
        restored: list[str] = []
        removed: list[str] = []

        for _ in range(count):
            cp = entries.pop()
            undone.append({"id": cp["id"], "summary": cp.get("summary", ""), "created_at": cp.get("created_at", "")})
            for fdict in cp.get("files") or []:
                snap = FileSnapshot.from_dict(fdict)
                full = self.workspace.resolve(snap.path)
                if snap.existed and snap.blob_sha:
                    data = self._read_blob(snap.blob_sha)
                    os.makedirs(os.path.dirname(full) or self.workspace.path, exist_ok=True)
                    tmp = full + ".edg_undo_tmp"
                    with open(tmp, "wb") as f:
                        f.write(data)
                    os.replace(tmp, full)
                    if snap.path not in restored:
                        restored.append(snap.path)
                else:
                    if os.path.exists(full):
                        self.workspace.delete(snap.path)
                        if snap.path not in removed:
                            removed.append(snap.path)

        self._save_index(session_id, entries)
        return {
            "ok": True,
            "undone": undone,
            "restored_files": restored,
            "removed_files": removed,
            "remaining_checkpoints": len(entries),
        }

    def git_status(self) -> Optional[dict[str, Any]]:
        """Return lightweight read-only git branch & dirty summary if workspace is a git repo."""
        try:
            r = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.workspace.path,
                capture_output=True,
                text=True,
                timeout=3,
            )
            if r.returncode != 0:
                return None
            branch = r.stdout.strip()
            s = subprocess.run(
                ["git", "status", "--short"],
                cwd=self.workspace.path,
                capture_output=True,
                text=True,
                timeout=3,
            )
            dirty = [ln for ln in s.stdout.splitlines() if ln.strip()]
            return {"branch": branch, "dirty_count": len(dirty), "dirty_preview": dirty[:15]}
        except Exception:
            return None
