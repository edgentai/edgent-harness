"""Subagents: Markdown definitions with front matter, and the runner that gives each a fresh context.

A subagent is a narrow role with a tool allowlist and a budget. It sees its
own system prompt, the workspace instructions and the delegation message,
nothing of the parent's conversation, and only its final message returns.
Depth and concurrency are capped in code.
"""
from __future__ import annotations

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..domain.errors import ConfigurationError, ErrorCategory, ToolFailure
from ..domain.types import Usage
from ..domain.utils import extract_json, parse_front_matter, truncate
from .permissions import MODES as PERMISSION_MODES
from .permissions import expand_tool_names
from .sessions import Session
from .skills import user_home

if TYPE_CHECKING:  # pragma: no cover
    from .loop import AgentLoop

SOURCES = ("workspace", "user", "builtin")
INDEX_HEADER = "## Subagents (delegate with the agent tool)"
INDEX_LINE_MAX_CHARS = 200
SUBAGENT_QUEUE_TIMEOUT_S = 600.0
SUMMARY_MAX_CHARS = 12_000
TASK_PREVIEW_CHARS = 200
EFFORTS = ("low", "medium", "high")
STATUS_BY_STOP_REASON = {"done": "done", "max_iterations": "max_turns", "budget": "budget"}


def builtin_agents_dir() -> Path:
    import edgent

    return Path(edgent.__file__).resolve().parent / "agents"


def _as_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = [v.strip() for v in value.split(",")]
    elif isinstance(value, list):
        items = [str(v).strip() for v in value]
    else:
        items = [str(value).strip()]
    return [v for v in items if v]


@dataclass
class AgentDefinition:
    name: str
    description: str = ""
    tools: list[str] | None = None  # None = inherit every tool the parent has; "mcp__x__*" and "mcp__x" expand at spawn
    disallowed_tools: list[str] = field(default_factory=list)
    model: str | None = None
    effort: str | None = None
    skills: list[str] = field(default_factory=list)
    max_turns: int | None = None
    background: bool = False
    memory: str | None = None
    permission_mode: str | None = None
    system_prompt: str = ""
    source: str = "builtin"
    path: Path | None = None

    @classmethod
    def from_markdown(cls, text: str, source: str = "builtin", path: Path | None = None, default_name: str = "") -> AgentDefinition:
        meta, body = parse_front_matter(text)
        where = str(path) if path else "<agent definition>"
        name = str(meta.get("name") or default_name).strip()
        if not name:
            raise ConfigurationError(f"{where}: an agent definition needs a 'name' in its front matter")
        tools = _as_list(meta.get("tools"))
        if tools is not None and any(t in ("*", "all") for t in tools):
            tools = None
        max_turns = meta.get("maxTurns", meta.get("max_turns"))
        if max_turns is not None and (isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1):
            raise ConfigurationError(f"{where}: maxTurns must be a positive integer; got {max_turns!r}")
        effort = meta.get("effort")
        if effort is not None and effort not in EFFORTS:
            raise ConfigurationError(f"{where}: effort must be one of {', '.join(EFFORTS)}; got {effort!r}")
        permission_mode = meta.get("permissionMode", meta.get("permission_mode"))
        if permission_mode is not None and permission_mode not in PERMISSION_MODES:
            raise ConfigurationError(f"{where}: permissionMode must be one of {', '.join(PERMISSION_MODES)}; got {permission_mode!r}")
        return cls(
            name=name,
            description=" ".join(str(meta.get("description") or "").split()),
            tools=tools,
            disallowed_tools=_as_list(meta.get("disallowedTools", meta.get("disallowed_tools"))) or [],
            model=str(meta["model"]) if meta.get("model") is not None else None,
            effort=effort,
            skills=_as_list(meta.get("skills")) or [],
            max_turns=max_turns,
            background=bool(meta.get("background", False)),
            memory=str(meta["memory"]) if meta.get("memory") is not None else None,
            permission_mode=permission_mode,
            system_prompt=body.strip(),
            source=source,
            path=path,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "tools": list(self.tools) if self.tools is not None else None,
            "disallowed_tools": list(self.disallowed_tools),
            "model": self.model,
            "effort": self.effort,
            "skills": list(self.skills),
            "max_turns": self.max_turns,
            "background": self.background,
            "memory": self.memory,
            "permission_mode": self.permission_mode,
            "source": self.source,
            "path": str(self.path) if self.path else None,
        }


class AgentIndex:
    def __init__(self) -> None:
        self._agents: dict[str, AgentDefinition] = {}

    @staticmethod
    def default_dirs(workspace: Any, home: str | os.PathLike[str] | None = None) -> list[Path]:
        return [Path(workspace.root) / ".edgent" / "agents", user_home(home) / "agents", builtin_agents_dir()]

    @classmethod
    def discover(cls, workspace: Any, home: str | os.PathLike[str] | None = None) -> AgentIndex:
        return cls().load(cls.default_dirs(workspace, home), list(SOURCES))

    def load(self, dirs: list[Path], sources: list[str]) -> AgentIndex:
        """Scan each directory for ``<name>.md``; the first directory that defines a name wins."""
        for directory, source in zip(dirs, sources, strict=True):
            directory = Path(directory)
            if not directory.is_dir():
                continue
            for f in sorted(directory.glob("*.md")):
                if not f.is_file() or f.name.startswith("."):
                    continue
                definition = AgentDefinition.from_markdown(f.read_text(encoding="utf-8", errors="replace"), source, f, default_name=f.stem)
                if definition.name not in self._agents:
                    self._agents[definition.name] = definition
        return self

    def add(self, definition: AgentDefinition) -> None:
        self._agents.setdefault(definition.name, definition)

    def get(self, name: str) -> AgentDefinition | None:
        return self._agents.get(name)

    def names(self) -> list[str]:
        return sorted(self._agents)

    def __len__(self) -> int:
        return len(self._agents)

    def index_text(self, max_chars: int = 3000) -> str:
        if not self._agents:
            return ""
        lines = [truncate(f"- {name}: {self._agents[name].description or '(no description)'}", INDEX_LINE_MAX_CHARS) for name in self.names()]
        kept: list[str] = []
        size = len(INDEX_HEADER)
        for line in lines:
            if size + len(line) + 1 > max_chars:
                break
            kept.append(line)
            size += len(line) + 1
        if len(kept) < len(lines):
            kept.append(f"- … {len(lines) - len(kept)} more subagents")
        return "\n".join([INDEX_HEADER, *kept])


@dataclass
class SubagentResult:
    name: str
    session_id: str
    status: str  # done | max_turns | budget | error
    summary: str = ""
    structured: dict[str, Any] | None = None
    usage: Usage = field(default_factory=Usage)
    turns: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "session_id": self.session_id,
            "status": self.status,
            "summary": self.summary,
            "structured": self.structured,
            "usage": self.usage.to_dict(),
            "turns": self.turns,
            "errors": list(self.errors),
        }


LoopFactory = Callable[..., "AgentLoop"]


class SubagentRunner:
    """Runs one subagent per spawn in a child session, with depth and concurrency caps enforced here."""

    def __init__(self, loop_factory: LoopFactory, max_depth: int = 2, max_concurrent: int = 4) -> None:
        self.loop_factory = loop_factory
        self.max_depth = max_depth
        self.max_concurrent = max(1, max_concurrent)
        self._lock = threading.Lock()
        self._slots: dict[int, threading.BoundedSemaphore] = {}  # one cap per depth: a parent holding a slot never starves its children
        self.active = 0
        self.peak_active = 0

    def _slots_at(self, depth: int) -> threading.BoundedSemaphore:
        with self._lock:
            return self._slots.setdefault(depth, threading.BoundedSemaphore(self.max_concurrent))

    def spawn(self, definition: AgentDefinition, task: str, parent: AgentLoop, depth: int) -> SubagentResult:
        if depth >= self.max_depth:
            raise ToolFailure(
                "permission_denied",
                f"max subagent depth {self.max_depth} reached (this agent runs at depth {depth}); do the work here instead",
                ErrorCategory.permission_denied,
                remediation="Do the task in this context, or return what you have so the parent can decide.",
                depth=depth,
                max_depth=self.max_depth,
            )
        slots = self._slots_at(depth)
        if not slots.acquire(timeout=SUBAGENT_QUEUE_TIMEOUT_S):
            raise ToolFailure(
                "resource_constraint",
                f"all {self.max_concurrent} subagent slots at depth {depth} stayed busy for {SUBAGENT_QUEUE_TIMEOUT_S:g} s",
                ErrorCategory.resource_constraint,
                retryable=True,
                remediation="Wait for running subagents to finish, or do the task in this context.",
            )
        with self._lock:
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
        try:
            return self._run(definition, task, parent, depth)
        finally:
            with self._lock:
                self.active -= 1
            slots.release()

    # ------------------------------------------------------------------ pieces
    def child_allowlist(self, definition: AgentDefinition, parent: AgentLoop, depth: int) -> set[str]:
        """The child's tools. Wildcards ("mcp__playwright__*") and bare connector names ("mcp__composio") expand against
        the parent registry at spawn time, so a connector that is not configured contributes nothing."""
        names = parent.registry.names()
        if definition.tools is not None:
            allowed = expand_tool_names(definition.tools, names)
        elif parent.allowlist is not None:
            allowed = set(parent.allowlist)
        else:
            allowed = set(names)
        allowed -= expand_tool_names(definition.disallowed_tools, names)
        allowed.discard("ask_user")
        if depth + 1 >= self.max_depth:
            allowed.discard("agent")
        return allowed

    @staticmethod
    def child_system_prompt(definition: AgentDefinition, skills: Any, errors: list[str]) -> str:
        prompt = definition.system_prompt
        for name in definition.skills:
            if skills is None:
                errors.append(f"skill {name!r} not loaded: no skill index available")
                continue
            try:
                prompt += f"\n\n# Skill: {name}\n{skills.load_body(name)}"
            except KeyError as e:
                errors.append(str(e))
        return prompt

    def _run(self, definition: AgentDefinition, task: str, parent: AgentLoop, depth: int) -> SubagentResult:
        child_session = Session.create(parent.workspace, mode="subagent", title=definition.name, parent_id=parent.session.session_id)
        child_session.meta.update({"agent": definition.name, "depth": depth + 1, "model": definition.model, "task": truncate(task, TASK_PREVIEW_CHARS)})
        child_session.save()
        parent.session.record_child(child_session.session_id)
        parent.session.save()
        errors: list[str] = []
        allowlist = self.child_allowlist(definition, parent, depth)
        system_prompt = self.child_system_prompt(definition, parent.skills, errors)
        max_iterations = definition.max_turns or parent.budget.budget.max_iterations
        payload = {"agent": definition.name, "task": truncate(task, TASK_PREVIEW_CHARS), "session_id": parent.session.session_id, "child_session_id": child_session.session_id, "depth": depth + 1}
        if parent.hooks is not None and parent.hooks.has("SubagentStart"):
            outcome = parent.hooks.run("SubagentStart", payload)
            if not outcome.allow:
                raise ToolFailure("permission_denied", f"spawn of {definition.name!r} blocked by a SubagentStart hook: {outcome.message}", ErrorCategory.permission_denied,
                                  remediation="Follow the hook's message or do the task in this context.")
        self._trace(parent, "subagent", {"phase": "start", **payload, "allowlist": sorted(allowlist), "max_iterations": max_iterations})
        loop = self.loop_factory(
            session=child_session,
            system_prompt=system_prompt,
            allowlist=allowlist,
            role=definition.name,
            effort=definition.effort or parent.effort,
            max_iterations=max_iterations,
            depth=depth + 1,
            permission_mode=definition.permission_mode,
            model=definition.model,
        )
        try:
            turn = loop.run_turn(task)
        except Exception as e:  # noqa: BLE001 - the parent gets a typed result, never a stack trace
            result = SubagentResult(definition.name, child_session.session_id, "error", "", None, Usage(), 0, errors + [f"{type(e).__name__}: {e}"[:300]])
        else:
            text = turn.text or ""
            structured = extract_json(text) if text.strip().startswith("{") or "```" in text else None
            result = SubagentResult(
                name=definition.name,
                session_id=child_session.session_id,
                status=STATUS_BY_STOP_REASON.get(turn.stop_reason, "error"),
                summary=truncate(text, SUMMARY_MAX_CHARS),
                structured=structured if isinstance(structured, dict) else None,
                usage=turn.usage,
                turns=turn.iterations,
                errors=errors + list(turn.errors),
            )
        parent.budget.charge_usage(result.usage)
        self._trace(parent, "subagent", {"phase": "stop", **payload, "status": result.status, "turns": result.turns, "usage": result.usage.to_dict(), "errors": result.errors[:10]})
        if parent.hooks is not None and parent.hooks.has("SubagentStop"):
            parent.hooks.run("SubagentStop", {**payload, "status": result.status, "summary": truncate(result.summary, 2000)})
        return result

    @staticmethod
    def _trace(parent: AgentLoop, kind: str, data: dict[str, Any]) -> None:
        if parent.tracer is None:
            return
        try:
            parent.tracer.record(kind, data, node_id=parent.node_id or None)
        except Exception:  # noqa: BLE001
            pass
