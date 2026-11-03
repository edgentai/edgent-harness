"""The real ``Service`` behind the local web UI: a thin facade over ``App``."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from ...adapters.markdown import markdown_to_html

CHAPTER_HEAD_CHARS = 2000  # how much of a chapter is read to find its title (chars)
DEFAULT_UI_MODE = "accept-edits"  # the reader has no permission prompt: workspace writes are allowed, commands and the network are not


class AppService:
    def __init__(self, app: Any) -> None:
        self.app = app
        self._build_thread: threading.Thread | None = None
        self._build_state: dict[str, Any] = {"running": False}
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_stop = threading.Event()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ workspace
    def workspace_info(self) -> dict[str, Any]:
        ws = self.app.workspace
        meta = self.app.provider_meta
        return {
            "name": ws.name,
            "root": str(ws.root),
            "provider": getattr(self.app.provider, "name", "?"),
            "model": getattr(self.app.provider, "model", ""),
            "provider_source": meta.get("source", ""),
            "warnings": [w for w in [meta.get("warning"), *meta.get("warnings", [])] if w],
        }

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
                    extra["subdir"] = str(Path(str(screen["chapter"])).parent)
            mode = None if (self.app.config.mode or "mode" in self.app.settings) else DEFAULT_UI_MODE
            _, result = self.app.chat(text, session_id=session_id, extra_context=extra, permission_mode=mode)
            d = result.to_dict()
            d["session_id"] = session_id
            return d

    # ------------------------------------------------------------------ book
    def outline(self) -> dict[str, Any]:
        return self.app.outline().to_dict()

    def chapters(self) -> list[dict[str, Any]]:
        ws = self.app.workspace
        out = []
        for rel in sorted(ws.glob("book/chapters/*.md")):
            head = ws.read_text(rel, 0, CHAPTER_HEAD_CHARS)
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
        from ...adapters import ingest

        try:
            return [r.to_dict() for r in ingest.list_sources(self.app.workspace)]
        except Exception:  # noqa: BLE001 - a broken sources folder shows as empty, the doctor says why
            return []

    def locator(self, locator: str) -> dict[str, Any]:
        from ...adapters import ingest

        return ingest.resolve_locator(self.app.workspace, locator)

    # ------------------------------------------------------------------ learner, unknowns, build, doctor
    def learner(self) -> dict[str, Any]:
        from ...domain.learner import LearnerStore

        try:
            store = LearnerStore(self.app.workspace)
            return {"progress": store.progress(), "due": store.due(), "mastery": store.mastery().get("concepts", {}), "misconceptions": store.open_misconceptions()}
        except Exception as e:  # noqa: BLE001 - an unreadable learner file is reported, not fatal
            return {"progress": {}, "due": [], "mastery": {}, "misconceptions": [], "error": f"{type(e).__name__}: {e}"}

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
                report = self.app.build_job().run()
                self._build_state = {"running": False, "goal": goal, "report": report.to_dict(), "error": None}
            except Exception as e:  # noqa: BLE001 - the failure is shown in the build tab
                self._build_state = {"running": False, "goal": goal, "error": f"{type(e).__name__}: {e}"}

        self._build_thread = threading.Thread(target=run, name="edgent-build", daemon=True)
        self._build_thread.start()
        return {"started": True, "goal": goal}

    def doctor(self) -> dict[str, Any]:
        return self.app.doctor()

    def events(self, since: int = 0) -> list[dict[str, Any]]:
        return self.app.events(since)

    # ------------------------------------------------------------------ heartbeat
    def start_heartbeat(self) -> bool:
        def run(prompt: str) -> Any:
            with self._lock:
                _, result = self.app.chat(prompt)
            return result.text

        hb = self.app.heartbeat(run)
        if not hb.configured():
            return False
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(target=hb.run_forever, args=(self._heartbeat_stop,), name="edgent-heartbeat", daemon=True)
        self._heartbeat_thread.start()
        return True

    def stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
