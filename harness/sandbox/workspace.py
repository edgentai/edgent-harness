"""Per-run workspace directory: the agent's working memory for big objects."""
from __future__ import annotations

import fnmatch
import json
import os
import shutil
import threading
from typing import Any, Optional

from ..core.utils import ensure_dir, now_iso


class WorkspaceEscape(Exception):
    pass


class Workspace:
    SUBDIRS = ("inputs", "outputs", "scratch", "snapshots")

    def __init__(self, root: str, run_id: str = "", *, create_subdirs: bool = True) -> None:
        self.run_id = run_id or os.path.basename(os.path.abspath(root)) or "workspace"
        raw_path = os.path.join(root, run_id) if run_id else root
        # Canonicalise via realpath so symlinked temp roots (e.g. /var -> /private/var on macOS)
        # agree with resolve() and os.path.relpath() never emits escaping ../../ prefixes.
        self.path = os.path.realpath(ensure_dir(raw_path))
        self._lock = threading.Lock()
        if create_subdirs:
            for s in self.SUBDIRS:
                ensure_dir(os.path.join(self.path, s))
        self.manifest_path = os.path.join(self.path, "manifest.json")
        if create_subdirs and not os.path.exists(self.manifest_path):
            self._write_manifest({"run_id": self.run_id, "created_at": now_iso(), "files": {}})

    @classmethod
    def open(cls, directory: str, *, create_subdirs: bool = False) -> "Workspace":
        """Open a persistent workspace directory directly (e.g., a course or project root)."""
        return cls(directory, run_id="", create_subdirs=create_subdirs)

    def _write_manifest(self, m: dict[str, Any]) -> None:
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2, sort_keys=True)
        os.replace(tmp, self.manifest_path)

    def manifest(self) -> dict[str, Any]:
        if not os.path.exists(self.manifest_path):
            return {"run_id": self.run_id, "created_at": now_iso(), "files": {}}
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def resolve(self, rel: str) -> str:
        """Resolve a workspace-relative path and refuse anything that escapes."""
        if os.path.isabs(rel):
            cand = os.path.realpath(rel)
        else:
            cand = os.path.realpath(os.path.join(self.path, rel))
        root = self.path
        if cand != root and not cand.startswith(root + os.sep):
            raise WorkspaceEscape(f"path escapes workspace: {rel}")
        return cand

    def write_text(self, rel: str, text: str) -> str:
        p = self.resolve(rel)
        ensure_dir(os.path.dirname(p))
        tmp = f"{p}.{threading.get_ident()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, p)
        self._track(rel)
        return p

    def write_bytes(self, rel: str, data: bytes) -> str:
        p = self.resolve(rel)
        ensure_dir(os.path.dirname(p))
        tmp = f"{p}.{threading.get_ident()}.tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, p)
        self._track(rel)
        return p

    def delete(self, rel: str) -> bool:
        p = self.resolve(rel)
        if p == self.path:
            raise WorkspaceEscape("cannot delete workspace root")
        if not os.path.exists(p):
            return False
        if os.path.isdir(p):
            shutil.rmtree(p)
        else:
            os.remove(p)
        return True

    def glob(self, pattern: str, rel: str = ".", limit: int = 500) -> list[dict[str, Any]]:
        out = []
        for entry in self.list(rel):
            p = entry["path"]
            if fnmatch.fnmatch(p, pattern) or fnmatch.fnmatch(os.path.basename(p), pattern):
                out.append(entry)
                if len(out) >= limit:
                    break
        return out

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
        for dirpath, dirs, files in os.walk(base):
            dirs[:] = sorted(d for d in dirs if d not in (".git", "__pycache__"))
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
        if not os.path.exists(self.manifest_path):
            return
        with self._lock:
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
