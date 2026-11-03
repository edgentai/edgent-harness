"""The study workspace: a directory that is the agent's memory.

Layout, instruction-file chain, path-escape protection, search, settings
precedence and git checkpoints. The model reads and writes it with the same
tools it uses for everything else; a human can read it too.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import atomic_write_text, ensure_dir, now_iso, read_json, write_json

LAYOUT: dict[str, str] = {
    "sources": "sources",
    "sources_raw": "sources/raw",
    "sources_text": "sources/text",
    "book": "book",
    "chapters": "book/chapters",
    "build": "book/build",
    "media": "media",
    "films": "media/films",
    "diagrams": "media/diagrams",
    "widgets": "media/widgets",
    "audio": "media/audio",
    "assessments": "assessments",
    "sessions": "sessions",
    "learner": "learner",
    "evidence": "evidence",
    "knowledge": "knowledge",
    "state": ".edgent",
    "traces": ".edgent/traces",
    "tool_results": ".edgent/tool-results",
    "cache": ".edgent/cache",
}

FILES: dict[str, str] = {
    "study": "STUDY.md",
    "index": "sources/INDEX.md",
    "outline": "book/outline.json",
    "progress": "progress.md",
    "unknowns": "unknowns.json",
    "house": "knowledge/house.md",
    "mastery": "learner/mastery.json",
    "learner_progress": "learner/progress.json",
    "misconceptions": "learner/misconceptions.md",
    "settings": ".edgent/settings.json",
}

STUDY_TEMPLATE = """# {name}

<!-- This file is read at the start of every call. Keep it short and specific. -->

## Learner
- Who: (e.g. second-year undergraduate, comfortable with calculus, new to probability)
- Goal: (e.g. pass the linear algebra final; understand attention from scratch)
- Language: English
- Level: (KG / school / undergraduate / graduate / professional)

## How I like things explained
- Example before definition. Intuition before formula.
- Short sections. One idea per section.

## Conventions
- Citations are `source-id:p.12` or `source-id:t=00:14:32` and must resolve.
- Never mark a unit done without running its checks.

## Never
- Do not invent sources. Say "unverified" when something could not be checked.
"""

HOUSE_TEMPLATE = """# House memory

Learned preferences about how this user likes things made. Rewritten whole by the
distiller, never appended. Size-capped.

## Pinned (authoritative, never auto-edited)

_(nothing pinned yet)_

## Learned

_(nothing learned yet)_
"""

PROGRESS_TEMPLATE = """# Progress

Append-only log. Each session writes what it did and what it left open.

"""

GITIGNORE = """.edgent/
sources/raw/*.mp4
sources/raw/*.mkv
sources/raw/*.mov
media/films/*/frames/
media/films/*/partial/
*.tmp.*
__pycache__/
"""

_SKIP_DIRS = {".git", ".edgent", "__pycache__", "node_modules", ".venv"}
_BINARY_SNIFF = 2048


class WorkspaceError(Exception):
    pass


@dataclass
class SearchHit:
    path: str
    line: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "line": self.line, "text": self.text}


class Workspace:
    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).resolve()

    # ------------------------------------------------------------------ creation
    @classmethod
    def init(cls, root: str | os.PathLike[str], name: str | None = None, git: bool = True) -> Workspace:
        ws = cls(root)
        ensure_dir(ws.root)
        for rel in LAYOUT.values():
            ensure_dir(ws.root / rel)
        name = name or ws.root.name
        if not (ws.root / FILES["study"]).exists():
            atomic_write_text(ws.root / FILES["study"], STUDY_TEMPLATE.format(name=name))
        if not (ws.root / FILES["house"]).exists():
            atomic_write_text(ws.root / FILES["house"], HOUSE_TEMPLATE)
        if not (ws.root / FILES["progress"]).exists():
            atomic_write_text(ws.root / FILES["progress"], PROGRESS_TEMPLATE)
        if not (ws.root / FILES["index"]).exists():
            atomic_write_text(ws.root / FILES["index"], "# Sources\n\n<!-- one line per source: id | kind | title | coverage | text view -->\n")
        if not (ws.root / FILES["unknowns"]).exists():
            write_json(ws.root / FILES["unknowns"], {"version": 1, "unknowns": []})
        if not (ws.root / FILES["outline"]).exists():
            write_json(ws.root / FILES["outline"], {"version": 1, "title": name, "persona": "", "units": []})
        if not (ws.root / FILES["mastery"]).exists():
            write_json(ws.root / FILES["mastery"], {"version": 1, "concepts": {}})
        if not (ws.root / FILES["learner_progress"]).exists():
            write_json(ws.root / FILES["learner_progress"], {"version": 1, "events": [], "settings": {"streaks": True, "badges": True, "points": True}})
        if not (ws.root / FILES["misconceptions"]).exists():
            atomic_write_text(ws.root / FILES["misconceptions"], "# Misconceptions\n\n")
        if not (ws.root / ".gitignore").exists():
            atomic_write_text(ws.root / ".gitignore", GITIGNORE)
        if git and ws.git_available() and not (ws.root / ".git").exists():
            ws.git("init", "-q")
            ws.git("add", "-A")
            ws.git("-c", "user.name=edgent", "-c", "user.email=edgent@localhost", "commit", "-q", "-m", "edgent init")
        return ws

    @classmethod
    def find(cls, start: str | os.PathLike[str] = ".") -> Workspace | None:
        p = Path(start).resolve()
        chain = [p, *p.parents]
        for cand in chain:  # the state directory marks the root; nested STUDY.md files are subdirectory instructions
            if (cand / LAYOUT["state"]).is_dir():
                return cls(cand)
        for cand in chain:
            if (cand / FILES["study"]).exists():
                return cls(cand)
        return None

    @property
    def name(self) -> str:
        return self.root.name

    def exists(self) -> bool:
        return (self.root / FILES["study"]).exists()

    # ------------------------------------------------------------------ paths
    def path(self, key: str) -> Path:
        if key in LAYOUT:
            return self.root / LAYOUT[key]
        if key in FILES:
            return self.root / FILES[key]
        raise KeyError(key)

    def resolve(self, rel: str | os.PathLike[str]) -> Path:
        """Resolve a workspace-relative path; refuse anything that escapes the root."""
        s = str(rel)
        if s.startswith("~"):
            raise WorkspaceError(f"path escapes the workspace: {rel}")
        p = (self.root / s).resolve() if not os.path.isabs(s) else Path(s).resolve()
        try:
            p.relative_to(self.root)
        except ValueError:
            raise WorkspaceError(f"path escapes the workspace: {rel}")
        return p

    def relpath(self, p: str | os.PathLike[str]) -> str:
        return str(Path(p).resolve().relative_to(self.root))

    def is_inside(self, p: str | os.PathLike[str]) -> bool:
        try:
            Path(p).resolve().relative_to(self.root)
            return True
        except ValueError:
            return False

    # ------------------------------------------------------------------ files
    def read_text(self, rel: str, offset: int = 0, length: int | None = None) -> str:
        p = self.resolve(rel)
        if not p.is_file():
            raise FileNotFoundError(rel)
        text = p.read_text(encoding="utf-8", errors="replace")
        return text[offset : offset + length] if length is not None else text[offset:]

    def write_text(self, rel: str, text: str) -> Path:
        p = self.resolve(rel)
        ensure_dir(p.parent)
        atomic_write_text(p, text)
        return p

    def append_text(self, rel: str, text: str) -> Path:
        p = self.resolve(rel)
        ensure_dir(p.parent)
        with open(p, "a", encoding="utf-8") as f:
            f.write(text)
        return p

    def read_json(self, rel: str, default: Any = None) -> Any:
        return read_json(self.resolve(rel), default)

    def write_json(self, rel: str, obj: Any) -> None:
        write_json(self.resolve(rel), obj)

    def file_exists(self, rel: str) -> bool:
        try:
            return self.resolve(rel).exists()
        except WorkspaceError:
            return False

    def list(self, rel: str = ".", max_entries: int = 500, recursive: bool = False) -> list[dict[str, Any]]:
        base = self.resolve(rel)
        if not base.exists():
            raise FileNotFoundError(rel)
        out: list[dict[str, Any]] = []
        it: Iterable[Path] = sorted(base.rglob("*")) if recursive else sorted(base.iterdir())
        for p in it:
            if any(part in _SKIP_DIRS for part in p.relative_to(self.root).parts):
                continue
            out.append({"path": self.relpath(p), "type": "dir" if p.is_dir() else "file", "size": p.stat().st_size if p.is_file() else None})
            if len(out) >= max_entries:
                break
        return out

    def glob(self, pattern: str, max_entries: int = 500) -> list[str]:
        out = []
        for p in sorted(self.root.glob(pattern)):
            if any(part in _SKIP_DIRS for part in p.relative_to(self.root).parts):
                continue
            out.append(self.relpath(p))
            if len(out) >= max_entries:
                break
        return out

    def search(self, pattern: str, rel: str = ".", max_hits: int = 50, ignore_case: bool = True, glob: str | None = None, context: int = 0) -> list[SearchHit]:
        """Regex search over text files, skipping state and binaries. The agent's grep."""
        rx = re.compile(pattern, re.I if ignore_case else 0)
        base = self.resolve(rel)
        hits: list[SearchHit] = []
        files = [base] if base.is_file() else sorted(base.rglob(glob or "*"))
        for p in files:
            if not p.is_file() or any(part in _SKIP_DIRS for part in p.relative_to(self.root).parts):
                continue
            try:
                with open(p, "rb") as fb:
                    head = fb.read(_BINARY_SNIFF)
                if b"\x00" in head:
                    continue
                lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for i, line in enumerate(lines, 1):
                if rx.search(line):
                    lo, hi = max(0, i - 1 - context), min(len(lines), i + context)
                    text = "\n".join(lines[lo:hi]) if context else line
                    hits.append(SearchHit(self.relpath(p), i, text[:500]))
                    if len(hits) >= max_hits:
                        return hits
        return hits

    # ------------------------------------------------------------------ instructions and settings
    def instruction_chain(self, subdir: str | None = None, home: str | os.PathLike[str] | None = None) -> list[tuple[str, str]]:
        """STUDY.md files that apply: user-wide, workspace root, then each directory down to ``subdir``."""
        out: list[tuple[str, str]] = []
        home_dir = Path(home) if home is not None else Path(os.environ.get("EDGENT_HOME", Path.home() / ".edgent"))
        user_file = home_dir / "STUDY.md"
        if user_file.is_file():
            out.append((str(user_file), user_file.read_text(encoding="utf-8", errors="replace")))
        root_file = self.root / FILES["study"]
        if root_file.is_file():
            out.append((FILES["study"], root_file.read_text(encoding="utf-8", errors="replace")))
        if subdir:
            try:
                target = self.resolve(subdir)
            except WorkspaceError:
                return out
            rel_parts = target.relative_to(self.root).parts
            cur = self.root
            for part in rel_parts:
                cur = cur / part
                f = cur / "STUDY.md"
                if f.is_file() and cur != self.root:
                    out.append((self.relpath(f), f.read_text(encoding="utf-8", errors="replace")))
        return out

    def settings(self, home: str | os.PathLike[str] | None = None) -> dict[str, Any]:
        """Merged settings: user-wide then workspace; workspace keys win; lists are concatenated."""
        home_dir = Path(home) if home is not None else Path(os.environ.get("EDGENT_HOME", Path.home() / ".edgent"))
        merged: dict[str, Any] = {}
        for p in (home_dir / "settings.json", self.root / FILES["settings"]):
            data = read_json(p, {}) if p.is_file() else {}
            merged = _merge_settings(merged, data)
        return merged

    # ------------------------------------------------------------------ progress
    def append_progress(self, text: str, session_id: str | None = None) -> None:
        stamp = now_iso()
        head = f"\n## {stamp}" + (f" ({session_id})" if session_id else "") + "\n"
        self.append_text(FILES["progress"], head + text.rstrip() + "\n")

    def progress_tail(self, lines: int = 40) -> str:
        try:
            all_lines = self.read_text(FILES["progress"]).splitlines()
        except FileNotFoundError:
            return ""
        return "\n".join(all_lines[-lines:])

    # ------------------------------------------------------------------ git
    @staticmethod
    def git_available() -> bool:
        return shutil.which("git") is not None

    def git(self, *args: str, timeout: float = 60) -> tuple[int, str]:
        if not self.git_available():
            return 127, "git not available"
        try:
            r = subprocess.run(["git", *args], cwd=self.root, capture_output=True, text=True, timeout=timeout)
        except (subprocess.TimeoutExpired, OSError) as e:
            return 1, str(e)
        return r.returncode, (r.stdout + r.stderr).strip()

    def is_git_repo(self) -> bool:
        return (self.root / ".git").exists()

    def commit(self, message: str, paths: list[str] | None = None) -> str | None:
        """Stage and commit; returns the short sha, or None when nothing changed or git is absent."""
        if not self.is_git_repo():
            return None
        self.git("add", "-A", *(paths or []))
        rc, _ = self.git("diff", "--cached", "--quiet")
        if rc == 0:
            return None
        rc, out = self.git("-c", "user.name=edgent", "-c", "user.email=edgent@localhost", "commit", "-q", "-m", message)
        if rc != 0:
            return None
        _, sha = self.git("rev-parse", "--short", "HEAD")
        return sha or None

    def git_log(self, n: int = 15) -> str:
        if not self.is_git_repo():
            return ""
        _, out = self.git("log", "--oneline", f"-{n}")
        return out

    def git_status(self) -> str:
        if not self.is_git_repo():
            return ""
        _, out = self.git("status", "--short")
        return out


def _merge_settings(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_settings(out[k], v)
        elif isinstance(v, list) and isinstance(out.get(k), list):
            out[k] = out[k] + [x for x in v if x not in out[k]]
        else:
            out[k] = v
    return out
