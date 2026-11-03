"""The composition root: one place that wires a workspace, a provider, the registries,
skills, agents, permissions, hooks, budgets and the loop into something a CLI, a web UI,
an MCP host or a build job can drive.

Nothing in here is clever. It reads settings, builds objects, and hands them over.
"""
from __future__ import annotations

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..adapters.providers.base import Provider
from ..adapters.toolchain import Toolchain
from ..domain.evidence import EvidenceStore
from ..domain.outline import Outline
from ..domain.types import ContentPart
from ..domain.unknowns import UnknownsLedger
from ..domain.workspace import Workspace
from ..observability import Metrics, Tracer
from ..tools.registry import IdempotencyStore, ToolContext, ToolRegistry


def _home() -> Path:
    return Path(os.environ.get("EDGENT_HOME", Path.home() / ".edgent"))


@dataclass
class AppConfig:
    provider: str | None = None  # gemini | openai | agy | claude | mock
    model: str | None = None
    effort: str = "medium"
    mode: str | None = None  # permission mode override
    max_tokens: int = 4096
    verbose: bool = False
    settings_overrides: dict[str, Any] = field(default_factory=dict)


class App:
    """Everything a surface needs, built once per workspace."""

    def __init__(self, workspace: Workspace, config: AppConfig | None = None, provider: Provider | None = None) -> None:
        from ..adapters.providers.factory import make_provider

        self.workspace = workspace
        self.config = config or AppConfig()
        self.home = _home()
        self.settings: dict[str, Any] = workspace.settings(self.home)
        if self.config.settings_overrides:
            self.settings.update(self.config.settings_overrides)
        self.toolchain = Toolchain.detect()
        self.metrics = Metrics()
        self.provider: Provider = provider or make_provider(self.settings, which=self.config.provider)
        if self.config.model and hasattr(self.provider, "model"):
            self.provider.model = self.config.model
        self.registry = self.build_registry()
        self.evidence = EvidenceStore(workspace.path("evidence"))
        self.skills = self._load_skills()
        self.agents = self._load_agents()
        self._events: list[dict[str, Any]] = []
        self._events_lock = threading.Lock()
        self._seq = 0

    # ------------------------------------------------------------------ construction
    @classmethod
    def open(cls, path: str | os.PathLike[str] = ".", config: AppConfig | None = None, provider: Provider | None = None, create: bool = False) -> App:
        ws = Workspace.find(path)
        if ws is None:
            if not create:
                raise FileNotFoundError(f"no Edgent workspace at or above {Path(path).resolve()}; run `edgent init` first")
            ws = Workspace.init(path)
        return cls(ws, config, provider)

    def build_registry(self) -> ToolRegistry:
        from ..tools.checks.registry import build_checks_registry
        from ..tools.core import build_core_registry
        from ..tools.edu.assess import build_assess_registry
        from ..tools.edu.book import build_book_registry
        from ..tools.edu.learner import build_learner_registry
        from ..tools.edu.media import build_media_registry
        from ..tools.edu.sources import build_sources_registry

        reg = ToolRegistry(idempotency=IdempotencyStore(self.workspace.path("state") / "state.db"), metrics=self.metrics)
        for builder in (build_core_registry, build_sources_registry, build_book_registry, build_checks_registry, build_media_registry, build_assess_registry, build_learner_registry):
            builder(reg)
        return reg

    def _load_skills(self) -> Any:
        from .skills import SkillIndex

        idx = SkillIndex()
        idx.load(SkillIndex.default_dirs(self.workspace, self.home), ["workspace", "user", "builtin"])
        return idx

    def _load_agents(self) -> Any:
        from .subagents import AgentIndex

        idx = AgentIndex()
        idx.load(AgentIndex.default_dirs(self.workspace, self.home), ["workspace", "user", "builtin"])
        return idx

    # ------------------------------------------------------------------ pieces
    def permissions(self, mode: str | None = None) -> Any:
        from .permissions import Permissions

        settings = dict(self.settings)
        if mode or self.config.mode:
            settings["mode"] = mode or self.config.mode
        return Permissions.from_settings(settings, self.workspace)

    def hooks(self, tracer: Any = None) -> Any:
        from .hooks import HookRunner

        return HookRunner(self.settings, self.workspace, tracer=tracer)

    def budget(self) -> Any:
        from .budgets import Budget, BudgetTracker

        b = Budget.from_env()
        if "budget" in self.settings:
            b = Budget.from_settings(self.settings)
        return BudgetTracker(b)

    def system_prompt(self, role: str = "main", mode: str = "chat", extra: str = "") -> str:
        from .. import prompts

        return prompts.build_system_prompt(role, mode=mode, caps=self.provider.caps, extra=extra)

    def tracer(self, session_id: str) -> Tracer:
        return Tracer(self.workspace.path("traces"), session_id, echo=self._on_trace)

    def _on_trace(self, ev: Any) -> None:
        with self._events_lock:
            self._seq += 1
            d = {"seq": self._seq, "kind": ev.kind, "session_id": ev.session_id, "name": ev.data.get("name"), "detail": (ev.data.get("output_preview") or ev.data.get("finish_reason") or "")[:200]}
            self._events.append(d)
            del self._events[:-500]

    def events(self, since: int = 0) -> list[dict[str, Any]]:
        with self._events_lock:
            return [e for e in self._events if e["seq"] > since]

    # ------------------------------------------------------------------ loops
    def loop(
        self,
        session: Any,
        *,
        role: str = "main",
        mode: str = "chat",
        system_extra: str = "",
        ask_user: Callable[[str], str] | None = None,
        ask_permission: Callable[[Any, dict[str, Any]], bool] | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        permission_mode: str | None = None,
        allowlist: set[str] | None = None,
        effort: str | None = None,
    ) -> Any:
        from .compaction import Compactor
        from .context import ContextAssembler
        from .loop import AgentLoop

        tracer = self.tracer(session.session_id)
        return AgentLoop(
            self.provider,
            self.registry,
            self.workspace,
            session,
            system_prompt=self.system_prompt(role, mode, system_extra),
            permissions=self.permissions(permission_mode),
            hooks=self.hooks(tracer),
            budget=self.budget(),
            tracer=tracer,
            metrics=self.metrics,
            skills=self.skills,
            agents=self.agents,
            allowlist=allowlist,
            effort=effort or self.config.effort,
            max_tokens=self.config.max_tokens,
            compactor=Compactor(self.provider),
            context=ContextAssembler(),
            ask_user=ask_user,
            ask_permission=ask_permission,
            role=role,
            on_event=on_event,
            evidence_store=self.evidence,
            toolchain=self.toolchain,
            settings=self.settings,
        )

    def new_session(self, mode: str = "chat", title: str | None = None) -> Any:
        from .sessions import Session

        return Session.create(self.workspace, mode, title=title)

    def load_session(self, session_id: str) -> Any:
        from .sessions import Session

        return Session.load(self.workspace, session_id)

    def sessions(self) -> Any:
        from .sessions import SessionStore

        return SessionStore(self.workspace)

    def chat(self, text: str, session_id: str | None = None, parts: list[ContentPart] | None = None, extra_context: dict[str, Any] | None = None, **loop_kw: Any) -> tuple[Any, Any]:
        """One turn in a chat session; returns (session, TurnResult)."""
        session = self.load_session(session_id) if session_id else self.new_session("chat")
        loop = self.loop(session, **loop_kw)
        extra = dict(extra_context or {})
        extra.setdefault("learner_summary", self.learner_summary())
        result = loop.run_turn(text, parts=parts, extra_context=extra)
        return session, result

    # ------------------------------------------------------------------ tool context (MCP host mode, scripts)
    def tool_context(self, session: Any = None, role: str = "host", permissions: Any = None) -> ToolContext:
        return ToolContext(
            workspace=self.workspace,
            session_id=session.session_id if session is not None else "host",
            role=role,
            provider=self.provider,
            evidence=self.evidence,
            settings=self.settings,
            toolchain=self.toolchain,
            permissions=permissions or self.permissions(),
            tracer=None,
            metrics=self.metrics,
            session=session,
            skills=self.skills,
        )

    # ------------------------------------------------------------------ build
    def build_job(self, on_event: Callable[[dict[str, Any]], None] | None = None, max_parallel: int = 1) -> Any:
        from .jobs.build import BuildJob, UnitOutcome

        def run_unit(unit: Any, brief: str) -> UnitOutcome:
            session = self.new_session("build", title=f"unit {unit.id}")
            session.meta["unit_id"] = unit.id
            session.save()
            loop = self.loop(session, role="worker", mode="build", on_event=on_event, permission_mode="auto")
            result = loop.run_turn(brief)
            status = "done" if result.stop_reason == "done" else "error"
            return UnitOutcome(unit_id=unit.id, status=status, ledger=[], text=result.text, usage=result.usage, error=None if status == "done" else result.stop_reason)

        return BuildJob(self.workspace, run_unit=run_unit, on_event=on_event, max_parallel=max_parallel)

    def plan_build(self, goal: str, on_event: Callable[[dict[str, Any]], None] | None = None) -> Any:
        """Initializer session: propose the outline for a goal through the model."""
        from ..domain.utils import extract_json
        from .jobs.briefs import initializer_brief

        job = self.build_job(on_event=on_event)

        def propose(brief: str) -> list[dict[str, Any]]:
            session = self.new_session("build", title="initializer")
            loop = self.loop(session, role="initializer", mode="build", on_event=on_event, permission_mode="auto")
            result = loop.run_turn(brief)
            data = extract_json(result.text)
            if isinstance(data, dict):
                data = data.get("units", [])
            if not isinstance(data, list):
                outline = Outline(self.workspace.path("outline"))
                if outline.units:  # the model may have used book_outline_propose directly
                    return []
                raise RuntimeError("the initializer did not return a unit list; see the session transcript")
            return data

        index = self.workspace.read_text("sources/INDEX.md") if self.workspace.file_exists("sources/INDEX.md") else ""
        return job.plan(goal, lambda _brief: propose(initializer_brief(goal, self.workspace, index)))

    # ------------------------------------------------------------------ reads for UIs
    def outline(self) -> Outline:
        return Outline(self.workspace.path("outline"))

    def unknowns(self) -> UnknownsLedger:
        return UnknownsLedger(self.workspace.path("unknowns"))

    def learner_summary(self) -> str:
        try:
            from ..domain.learner import LearnerStore

            return LearnerStore(self.workspace).summary()
        except Exception:  # noqa: BLE001 - the summary is a convenience, never a blocker
            return ""

    def doctor(self) -> dict[str, Any]:
        report = self.toolchain.report()
        try:
            health = self.provider.health()
        except Exception as e:  # noqa: BLE001
            health = {"ok": False, "error": str(e)}
        report["provider"] = {"name": getattr(self.provider, "name", "?"), "model": getattr(self.provider, "model", ""), "health": health, "caps": self.provider.caps.__dict__}
        report["workspace"] = {"root": str(self.workspace.root), "git": self.workspace.is_git_repo(), "settings": {k: v for k, v in self.settings.items() if k != "provider"}}
        report["skills"] = self.skills.names() if hasattr(self.skills, "names") else []
        report["agents"] = self.agents.names() if hasattr(self.agents, "names") else []
        return report
