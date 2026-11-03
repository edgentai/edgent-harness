"""The composition root: one place that wires a workspace, a provider, the registries,
skills, agents, permissions, hooks, budgets and the loop into something a CLI, a web UI,
an MCP host or a build job can drive.

Nothing in here is clever. It reads settings, builds objects, and hands them over.
"""
from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from ..adapters.providers.base import Provider
from ..adapters.toolchain import Toolchain
from ..domain.evidence import EvidenceStore
from ..domain.outline import Outline
from ..domain.types import ContentPart
from ..domain.unknowns import UnknownsLedger
from ..domain.workspace import Workspace, _merge_settings
from ..observability import Metrics, Tracer
from ..tools.registry import AllowAll, IdempotencyStore, ToolContext, ToolRegistry

EVENT_BUFFER_MAX = 500  # progress events kept for UI polling (count)
EVENT_DETAIL_CHARS = 300  # longest event detail kept (chars)
WORKER_STATUS = {"done": "done", "max_iterations": "partial", "budget": "partial", "cancelled": "error", "error": "error"}
BUILD_NETWORK_TOOLS = ("web_fetch", "web_search", "sources_add_url")  # research units need them and a build has nobody to ask
JUDGEABLE_SUFFIXES = (".md", ".markdown", ".txt")


def default_home(home: str | os.PathLike[str] | None = None) -> Path:
    """The user directory (skills, agents, settings): an explicit path, else EDGENT_HOME, else ~/.edgent."""
    if home is not None:
        return Path(home)
    return Path(os.environ.get("EDGENT_HOME", Path.home() / ".edgent"))


@dataclass
class AppConfig:
    provider: str | None = None  # gemini | openai | agy | claude | mock; None lets settings and the environment decide
    model: str | None = None  # overrides settings.provider.model
    effort: str = "medium"
    mode: str | None = None  # permission mode override: ask | accept-edits | auto | plan | bypass
    max_tokens: int = 4096
    verbose: bool = False
    critic: bool = True  # review newly done build units with a separate-context judge
    home: str | None = None  # user directory; None means EDGENT_HOME or ~/.edgent
    settings_overrides: dict[str, Any] = field(default_factory=dict)


class App:
    """Everything a surface needs, built once per workspace."""

    def __init__(self, workspace: Workspace, config: AppConfig | None = None, provider: Provider | None = None) -> None:
        from ..adapters.providers.factory import make_provider
        from .skills import SkillIndex
        from .subagents import AgentIndex

        self.workspace = workspace
        self.config = config or AppConfig()
        self.home = default_home(self.config.home)
        self.settings: dict[str, Any] = workspace.settings(self.home)
        if self.config.settings_overrides:
            self.settings = _merge_settings(self.settings, self.config.settings_overrides)
        if self.config.model:
            conf = self.settings.get("provider")
            self.settings["provider"] = {**(conf if isinstance(conf, dict) else {}), "model": self.config.model}
        self.toolchain = Toolchain.detect()
        self.metrics = Metrics()
        self.provider: Provider = provider or make_provider(self.settings, which=self.config.provider)
        if not isinstance(getattr(self.provider, "meta", None), dict):
            self.provider.meta = {"kind": getattr(self.provider, "name", "?"), "source": "argument", "warnings": []}  # type: ignore[attr-defined]
        self.evidence = EvidenceStore(workspace.path("evidence"))
        self.registry = self.build_registry()
        self.skills = SkillIndex.discover(workspace, self.home)
        self.agents = AgentIndex.discover(workspace, self.home)
        self._events: list[dict[str, Any]] = []
        self._events_lock = threading.Lock()
        self._seq = 0
        self._tracers: dict[str, Tracer] = {}
        self._tracers_lock = threading.Lock()

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

    @property
    def provider_meta(self) -> dict[str, Any]:
        meta = getattr(self.provider, "meta", None)
        return dict(meta) if isinstance(meta, dict) else {}

    # ------------------------------------------------------------------ pieces
    def permissions(self, mode: str | None = None) -> Any:
        from .permissions import Permissions

        settings = dict(self.settings)
        chosen = mode or self.config.mode
        if chosen:
            settings["mode"] = chosen
        return Permissions.from_settings(settings, self.workspace)

    def hooks(self, tracer: Any = None) -> Any:
        from .hooks import HookRunner

        return HookRunner(self.settings, self.workspace, tracer=tracer)

    def budget(self) -> Any:
        from .budgets import Budget, BudgetTracker

        return BudgetTracker(Budget.from_settings(self.settings, base=Budget.from_env()))

    def system_prompt(self, role: str = "main", mode: str = "chat", extra: str = "") -> str:
        from .. import prompts

        return prompts.build_system_prompt(role, mode=mode, caps=self.provider.caps, extra=extra)

    def tracer(self, session_id: str) -> Tracer:
        """One tracer per session id for the life of the app, so seq numbers in a trace file stay monotonic."""
        with self._tracers_lock:
            tracer = self._tracers.get(session_id)
            if tracer is None:
                tracer = self._tracers[session_id] = Tracer(self.workspace.path("traces"), session_id)
            return tracer

    def build_allow(self) -> list[str]:
        """Patterns a build session may use without anyone to ask: the network tools, unless settings.build.network is false."""
        build = self.settings.get("build") if isinstance(self.settings.get("build"), dict) else {}
        return list(BUILD_NETWORK_TOOLS) if build.get("network", True) else []

    # ------------------------------------------------------------------ progress events (for UIs polling a process)
    def emit(self, event: dict[str, Any]) -> None:
        kind = event.get("type") or event.get("event") or event.get("kind") or "event"
        detail = event.get("preview") or event.get("text") or event.get("reason") or event.get("status") or event.get("error") or ""
        row = {
            "seq": 0,
            "ts": time.time(),
            "kind": str(kind),
            "session_id": event.get("session_id"),
            "name": event.get("name") or event.get("unit_id") or event.get("agent") or "",
            "ok": event.get("ok"),
            "detail": str(detail)[:EVENT_DETAIL_CHARS],
        }
        with self._events_lock:
            self._seq += 1
            row["seq"] = self._seq
            self._events.append(row)
            del self._events[:-EVENT_BUFFER_MAX]

    def events(self, since: int = 0) -> list[dict[str, Any]]:
        with self._events_lock:
            return [dict(e) for e in self._events if e["seq"] > since]

    def _chain(self, on_event: Callable[[dict[str, Any]], None] | None) -> Callable[[dict[str, Any]], None]:
        def fn(event: dict[str, Any]) -> None:
            self.emit(event)
            if on_event is not None:
                on_event(event)

        return fn

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
        allow: list[str] | None = None,
    ) -> Any:
        from .compaction import Compactor
        from .context import ContextAssembler
        from .loop import AgentLoop
        from .permissions import Permissions

        tracer = self.tracer(session.session_id)
        permissions = self.permissions(permission_mode)
        if allow:
            permissions = Permissions(permissions.mode, [*map(str, permissions.allow), *allow], [*map(str, permissions.deny)], self.workspace, permissions.always_ask)
        return AgentLoop(
            self.provider,
            self.registry,
            self.workspace,
            session,
            system_prompt=self.system_prompt(role, mode, system_extra),
            permissions=permissions,
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
            on_event=self._chain(on_event),
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

    # ------------------------------------------------------------------ tool context (MCP host mode, CLI commands, the build gate)
    def tool_context(
        self,
        session: Any = None,
        role: str = "host",
        permissions: Any = None,
        ask_permission: Callable[[Any, dict[str, Any]], bool] | None = None,
        ask_user: Callable[[str], str] | None = None,
        node_id: str = "",
    ) -> ToolContext:
        session_id = session.session_id if session is not None else role
        return ToolContext(
            workspace=self.workspace,
            session_id=session_id,
            node_id=node_id,
            role=role,
            provider=self.provider,
            evidence=self.evidence,
            settings=self.settings,
            toolchain=self.toolchain,
            permissions=permissions if permissions is not None else self.permissions(),
            ask_user=ask_user,
            ask_permission=ask_permission,
            tracer=self.tracer(session_id),
            metrics=self.metrics,
            session=session,
            skills=self.skills,
            extra={"registry": self.registry},
        )

    # ------------------------------------------------------------------ build
    def build_job(self, on_event: Callable[[dict[str, Any]], None] | None = None, max_parallel: int = 1, max_rounds: int = 2, critic: bool | None = None) -> Any:
        from ..tools.edu.book import mark_unit_done
        from .jobs.build import BuildJob, UnitOutcome

        hooks = self.hooks(self.tracer("build"))
        chained = self._chain(on_event)

        def events(event: dict[str, Any]) -> None:
            chained(event)
            kind = event.get("event")
            if kind == "unit_finished" and hooks.has("UnitDone"):
                hooks.run("UnitDone", dict(event))
            elif kind in ("run_finished", "unit_blocked", "critic_exhausted") and hooks.has("Notification"):
                hooks.run("Notification", {"event": kind, **{k: v for k, v in event.items() if k != "event"}})

        def run_unit(unit: Any, brief: str) -> UnitOutcome:
            session = self.new_session("build", title=f"unit {unit.id}")
            session.meta["unit_id"] = unit.id
            session.save()
            loop = self.loop(session, role="worker", mode="build", on_event=on_event, permission_mode="auto", allow=self.build_allow())
            subdir = str(PurePosixPath(unit.path).parent) if unit.path else None  # the nearest STUDY.md wins for the unit's folder
            result = loop.run_turn(brief, extra_context={"subdir": subdir} if subdir and subdir != "." else None)
            status = WORKER_STATUS.get(result.stop_reason, "error")
            error = None
            if status != "done":
                error = f"worker stopped: {result.stop_reason}" + ("; " + "; ".join(result.errors[-3:]) if result.errors else "")
            return UnitOutcome(unit_id=unit.id, status=status, text=result.text, usage=result.usage, error=error)

        def mark_done(unit: Any) -> dict[str, Any]:
            return mark_unit_done(self.tool_context(role="build", permissions=AllowAll(), node_id=unit.id), unit.id)

        def judge(unit: Any) -> list[dict[str, Any]]:
            if not unit.path or not str(unit.path).endswith(JUDGEABLE_SUFFIXES):
                return []
            ctx = self.tool_context(role="critic", permissions=AllowAll(), node_id=unit.id)
            res = self.registry.call("check_judge", {"path": unit.path, "unit_id": unit.id}, ctx)
            if not res.ok:
                raise RuntimeError(res.error.message if res.error else "check_judge failed")
            output = res.output if isinstance(res.output, dict) else {}
            return [d for d in output.get("defects") or [] if isinstance(d, dict)]

        use_critic = self.config.critic if critic is None else critic
        return BuildJob(
            self.workspace,
            run_unit=run_unit,
            mark_done=mark_done,
            critic=judge if use_critic else None,
            max_rounds=max_rounds,
            max_parallel=max_parallel,
            on_event=events,
            tracer=self.tracer("build"),
        )

    def plan_build(self, goal: str, on_event: Callable[[dict[str, Any]], None] | None = None) -> Any:
        """Initializer session: the model proposes the outline for a goal; the job validates and saves it."""
        from ..domain.utils import extract_json

        job = self.build_job(on_event=on_event)

        def propose(brief: str) -> list[dict[str, Any]] | dict[str, Any]:
            session = self.new_session("build", title="initializer")
            loop = self.loop(session, role="initializer", mode="build", on_event=on_event, permission_mode="auto", allow=self.build_allow())
            result = loop.run_turn(brief)
            data = extract_json(result.text)
            if isinstance(data, dict) and isinstance(data.get("units"), list):
                return data
            if isinstance(data, list):
                return data
            return {"units": []}  # the initializer may have used book_outline_propose itself; the job checks the outline

        return job.plan(goal, propose)

    # ------------------------------------------------------------------ heartbeat
    def heartbeat(self, run: Callable[[str], Any] | None = None, on_event: Callable[[dict[str, Any]], None] | None = None) -> Any:
        from .heartbeat import Heartbeat

        def default_run(prompt: str) -> Any:
            _, result = self.chat(prompt, on_event=on_event)
            return result.text

        return Heartbeat.from_settings(self.workspace, run or default_run, self.settings, tracer=self.tracer("heartbeat"))

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
        report["provider"] = {
            "name": getattr(self.provider, "name", "?"),
            "model": getattr(self.provider, "model", ""),
            "health": health,
            "caps": dict(self.provider.caps.__dict__),
            **{k: v for k, v in self.provider_meta.items() if k in ("kind", "source", "warnings", "warning", "hint")},
        }
        report["workspace"] = {"root": str(self.workspace.root), "git": self.workspace.is_git_repo(), "settings": {k: v for k, v in self.settings.items() if k != "provider"}}
        report["skills"] = self.skills.names()
        report["agents"] = self.agents.names()
        report["tools"] = len(self.registry.names())
        return report
