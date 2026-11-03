"""The real ``Service`` behind the local web UI: a thin facade over ``App``."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from ...adapters.markdown import markdown_to_html
from ...domain.utils import read_json


class AppService:
    def __init__(self, app: Any) -> None:
        self.app = app
        self._build_thread: threading.Thread | None = None
        self._build_state: dict[str, Any] = {"running": False}
        self._heartbeat_thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ workspace
    def workspace_info(self) -> dict[str, Any]:
        ws = self.app.workspace
        return {"name": ws.name, "root": str(ws.root), "provider": getattr(self.app.provider, "name", "?"), "model": getattr(self.app.provider, "model", "")}

    # ------------------------------------------------------------------ sessions and chat
    def list_sessions(self) -> list[dict[str, Any]]:
        return self.app.sessions().list()

    def create_session(self, mode: str, title: str) -> dict[str, Any]:
        s = self.app.new_session(mode or "chat", title=title or None)
        return {"session_id": s.session_id, "mode": s.mode, "title": s.title}

    def session_messages(self, session_id: str, limit: int = 100) -> list[dict[str, Any]]:
        s = self.app.load_session(session_id)
        out = []
        for m in s.messages[-limit:]:
            out.append({"role": m.role, "text": m.text(), "name": m.name, "tool_calls": [tc.to_dict() for tc in m.tool_calls]})
        return out

    def send(self, session_id: str, text: str, screen: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:  # one turn at a time per process keeps the transcript ordered
            extra: dict[str, Any] = {"learner_summary": self.app.learner_summary()}
            if screen:
                extra["screen"] = screen
                if screen.get("chapter"):
                    extra["subdir"] = str(Path(screen["chapter"]).parent)
            _, result = self.app.chat(text, session_id=session_id, extra_context=extra)
            d = result.to_dict() if hasattr(result, "to_dict") else {"text": str(result)}
            d.setdefault("text", getattr(result, "text", ""))
            d.setdefault("stop_reason", getattr(result, "stop_reason", "done"))
            return d

    # ------------------------------------------------------------------ book
    def outline(self) -> dict[str, Any]:
        return self.app.outline().to_dict()

    def chapters(self) -> list[dict[str, Any]]:
        ws = self.app.workspace
        out = []
        for rel in ws.glob("book/chapters/*.md"):
            head = ws.read_text(rel, 0, 2000)
            title = rel.rsplit("/", 1)[-1]
            for line in head.splitlines():
                if line.startswith("title:"):
                    title = line.split(":", 1)[1].strip().strip("\"'")
                    break
                if line.startswith("# "):
                    title = line[2:].strip()
                    break
            out.append({"path": rel, "title": title})
        return out

    def chapter_html(self, path: str) -> dict[str, Any]:
        ws = self.app.workspace
        if not path.startswith("book/") or not path.endswith(".md"):
            raise FileNotFoundError(path)
        text = ws.read_text(path)
        html, meta = markdown_to_html(text)
        return {"path": path, "html": html, "meta": meta}

    # ------------------------------------------------------------------ sources
    def sources(self) -> list[dict[str, Any]]:
        try:
            from ...adapters import ingest

            return [r.to_dict() if hasattr(r, "to_dict") else r for r in ingest.list_sources(self.app.workspace)]
        except Exception:  # noqa: BLE001
            return []

    def locator(self, locator: str) -> dict[str, Any]:
        from ...adapters import ingest

        return ingest.resolve_locator(self.app.workspace, locator)

    # ------------------------------------------------------------------ learner, unknowns, build, doctor
    def learner(self) -> dict[str, Any]:
        ws = self.app.workspace
        try:
            from ...domain.learner import LearnerStore

            store = LearnerStore(ws)
            due = store.due()
            progress = store.progress()
        except Exception:  # noqa: BLE001
            due, progress = [], {}
        mastery = (read_json(ws.path("mastery"), {}) or {}).get("concepts", {})
        return {"progress": progress, "due": [d if isinstance(d, dict) else {"concept": d} for d in due], "mastery": mastery}

    def unknowns(self) -> dict[str, Any]:
        ledger = self.app.unknowns()
        return {"markdown": ledger.render_markdown(), "open": [u.to_dict() for u in ledger.open()]}

    def build_status(self) -> dict[str, Any]:
        d = {"outline": self.app.outline().summary(), "unknowns_open": len(self.app.unknowns().open()), "progress_tail": self.app.workspace.progress_tail(15)}
        d.update(self._build_state)
        return d

    def start_build(self, goal: str) -> dict[str, Any]:
        if self._build_thread and self._build_thread.is_alive():
            return {"started": False, "reason": "a build is already running", **self._build_state}

        def run() -> None:
            self._build_state = {"running": True, "goal": goal, "error": None}
            try:
                if not self.app.outline().units:
                    self.app.plan_build(goal)
                job = self.app.build_job()
                report = job.run()
                self._build_state = {"running": False, "goal": goal, "report": report.to_dict() if hasattr(report, "to_dict") else str(report), "error": None}
            except Exception as e:  # noqa: BLE001
                self._build_state = {"running": False, "goal": goal, "error": f"{type(e).__name__}: {e}"}

        self._build_thread = threading.Thread(target=run, name="edgent-build", daemon=True)
        self._build_thread.start()
        return {"started": True, "goal": goal}

    def doctor(self) -> dict[str, Any]:
        return self.app.doctor()

    def events(self, since: int = 0) -> list[dict[str, Any]]:
        return self.app.events(since)

    def start_heartbeat(self) -> None:
        from ...application.heartbeat import Heartbeat

        settings = self.app.settings.get("heartbeat") or {}

        def run(prompt: str) -> Any:
            with self._lock:
                _, result = self.app.chat(prompt)
            return result.text

        hb = Heartbeat(self.app.workspace, run, times=settings.get("times"), interval_minutes=settings.get("interval_minutes"), prompt=settings.get("prompt"), tz=settings.get("timezone", "UTC"))
        stop = threading.Event()
        self._heartbeat_thread = threading.Thread(target=hb.run_forever, args=(stop,), name="edgent-heartbeat", daemon=True)
        self._heartbeat_thread.start()
