"""Per-run workspace directory: the agent's working memory for big objects."""
from __future__ import annotations

import json
import os
import shutil
from typing import Any, Optional

from ..core.utils import ensure_dir, now_iso


class WorkspaceEscape(Exception):
    pass


class Workspace:
    SUBDIRS = ("inputs", "outputs", "scratch", "snapshots")

    def __init__(self, root: str, run_id: str) -> None:
        self.run_id = run_id
        self.path = ensure_dir(os.path.join(root, run_id))
        for s in self.SUBDIRS:
            ensure_dir(os.path.join(self.path, s))
        self.manifest_path = os.path.join(self.path, "manifest.json")
        if not os.path.exists(self.manifest_path):
            self._write_manifest({"run_id": run_id, "created_at": now_iso(), "files": {}})

    def _write_manifest(self, m: dict[str, Any]) -> None:
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2, sort_keys=True)

    def manifest(self) -> dict[str, Any]:
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def resolve(self, rel: str) -> str:
        """Resolve a workspace-relative path and refuse anything that escapes."""
        if os.path.isabs(rel):
            cand = os.path.realpath(rel)
        else:
            cand = os.path.realpath(os.path.join(self.path, rel))
        root = os.path.realpath(self.path)
        if cand != root and not cand.startswith(root + os.sep):
            raise WorkspaceEscape(f"path escapes workspace: {rel}")
        return cand

    def write_text(self, rel: str, text: str) -> str:
        p = self.resolve(rel)
        ensure_dir(os.path.dirname(p))
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        self._track(rel)
        return p

    def write_bytes(self, rel: str, data: bytes) -> str:
        p = self.resolve(rel)
        ensure_dir(os.path.dirname(p))
        with open(p, "wb") as f:
            f.write(data)
        self._track(rel)
        return p

    def read_text(self, rel: str, limit: Optional[int] = None) -> str:
        with open(self.resolve(rel), "r", encoding="utf-8", errors="replace") as f:
            return f.read(limit) if limit else f.read()

    def read_bytes(self, rel: str) -> bytes:
        with open(self.resolve(rel), "rb") as f:
            return f.read()

    def exists(self, rel: str) -> bool:
        try:
            return os.path.exists(self.resolve(rel))
        except WorkspaceEscape:
            return False

    def list(self, rel: str = ".") -> list[dict[str, Any]]:
        base = self.resolve(rel)
        out = []
        for dirpath, _dirs, files in os.walk(base):
            for fn in sorted(files):
                full = os.path.join(dirpath, fn)
                out.append({"path": os.path.relpath(full, self.path), "bytes": os.path.getsize(full)})
        return out

    def search(self, pattern: str, rel: str = ".", max_hits: int = 50) -> list[dict[str, Any]]:
        import re

        rx = re.compile(pattern)
        hits = []
        for entry in self.list(rel):
            try:
                text = self.read_text(entry["path"])
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    hits.append({"path": entry["path"], "line": i, "text": line[:300]})
                    if len(hits) >= max_hits:
                        return hits
        return hits

    def _track(self, rel: str) -> None:
        m = self.manifest()
        m["files"][rel] = now_iso()
        self._write_manifest(m)

    def copy_in(self, src: str, name: Optional[str] = None) -> str:
        name = name or os.path.basename(src)
        dst = self.resolve(os.path.join("inputs", name))
        shutil.copy2(src, dst)
        self._track(os.path.join("inputs", name))
        return dst

    def destroy(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)
