"""Lifecycle hooks: shell commands from settings and Python callables, run at fixed points.

Settings shape::

    "hooks": {"PreToolUse": [{"matcher": "bash|write_file", "command": "python3 .edgent/hooks/check.py", "timeout_s": 30}]}

A command hook receives the JSON payload on stdin and runs in the sandbox with
the workspace root as cwd and no network. Exit 0 continues (stdout may be JSON
with ``updated_arguments`` and ``message``); exit 2 blocks with stderr as the
message; any other exit is logged and the call continues; a timeout continues
with a note. Hooks are how a user adds a rule the model cannot argue with.
"""
from __future__ import annotations

import json
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..adapters.sandbox.runner import run_subprocess
from ..domain.errors import ConfigurationError
from ..domain.utils import extract_json, truncate

EVENTS = ("SessionStart", "PreToolUse", "PostToolUse", "Stop", "SubagentStart", "SubagentStop", "UnitDone", "Compaction", "Notification")
DEFAULT_HOOK_TIMEOUT_S = 30.0
MAX_HOOK_TIMEOUT_S = 600.0
HOOK_MESSAGE_MAX_CHARS = 4000
BLOCK_EXIT_CODE = 2

HookCallable = Callable[[dict[str, Any]], Any]


@dataclass
class HookOutcome:
    allow: bool = True
    message: str = ""
    updated_arguments: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"allow": self.allow, "message": self.message, "updated_arguments": self.updated_arguments}


@dataclass
class CommandHook:
    event: str
    command: str
    matcher: str = "*"
    timeout_s: float = DEFAULT_HOOK_TIMEOUT_S

    def __post_init__(self) -> None:
        self.argv = shlex.split(self.command)
        if not self.argv:
            raise ConfigurationError(f"hooks.{self.event}: 'command' must not be empty")
        if not isinstance(self.timeout_s, (int, float)) or isinstance(self.timeout_s, bool) or not 0 < self.timeout_s <= MAX_HOOK_TIMEOUT_S:
            raise ConfigurationError(f"hooks.{self.event}: timeout_s must be a number of seconds in (0, {MAX_HOOK_TIMEOUT_S:g}]; got {self.timeout_s!r}")
        self.regex: re.Pattern[str] | None = None
        if self.matcher not in ("", "*"):
            try:
                self.regex = re.compile(self.matcher)
            except re.error as e:
                raise ConfigurationError(f"hooks.{self.event}: matcher {self.matcher!r} is not a valid regex: {e}") from e

    def applies(self, name: str) -> bool:
        if self.regex is None or not name:
            return True
        return self.regex.fullmatch(name) is not None


def merge_outcomes(outcomes: list[HookOutcome]) -> HookOutcome:
    """Any block wins; the last updated_arguments wins; messages are joined."""
    merged = HookOutcome()
    for o in outcomes:
        if not o.allow:
            merged.allow = False
        if o.updated_arguments is not None:
            merged.updated_arguments = dict(o.updated_arguments)
    merged.message = "\n".join(o.message for o in outcomes if o.message)
    return merged


class HookRunner:
    def __init__(self, settings: dict[str, Any] | None, workspace: Any, tracer: Any = None) -> None:
        self.workspace = workspace
        self.tracer = tracer
        self._commands: dict[str, list[CommandHook]] = {}
        self._callables: dict[str, list[HookCallable]] = {}
        self._load((settings or {}).get("hooks") or {})

    # ------------------------------------------------------------------ configuration
    def _load(self, hooks: Any) -> None:
        if not isinstance(hooks, dict):
            raise ConfigurationError("settings.hooks must be an object mapping event names to lists of hooks")
        for event, entries in hooks.items():
            if event not in EVENTS:
                raise ConfigurationError(f"unknown hook event {event!r}; known events: {', '.join(EVENTS)}")
            if not isinstance(entries, list):
                raise ConfigurationError(f"hooks.{event} must be a list of hook objects")
            for entry in entries:
                if not isinstance(entry, dict) or not isinstance(entry.get("command"), str):
                    raise ConfigurationError(f"hooks.{event}: each hook needs a 'command' string; got {entry!r}")
                matcher = entry.get("matcher", "*")
                if not isinstance(matcher, str):
                    raise ConfigurationError(f"hooks.{event}: 'matcher' must be a regex string; got {matcher!r}")
                self._commands.setdefault(event, []).append(CommandHook(event, entry["command"], matcher, entry.get("timeout_s", DEFAULT_HOOK_TIMEOUT_S)))

    def register(self, event: str, fn: HookCallable) -> None:
        if event not in EVENTS:
            raise ConfigurationError(f"unknown hook event {event!r}; known events: {', '.join(EVENTS)}")
        self._callables.setdefault(event, []).append(fn)

    def has(self, event: str) -> bool:
        return bool(self._commands.get(event) or self._callables.get(event))

    def describe(self) -> dict[str, Any]:
        return {
            event: [{"command": h.command, "matcher": h.matcher, "timeout_s": h.timeout_s} for h in self._commands.get(event, [])]
            + [{"callable": getattr(fn, "__name__", repr(fn))} for fn in self._callables.get(event, [])]
            for event in EVENTS
            if self.has(event)
        }

    # ------------------------------------------------------------------ running
    def run(self, event: str, payload: dict[str, Any] | None = None) -> HookOutcome:
        data = dict(payload or {})
        data["event"] = event
        data.setdefault("workspace", str(getattr(self.workspace, "root", "")))
        name = str(data.get("tool_name") or data.get("agent") or "")
        outcomes = [self._run_command(h, data) for h in self._commands.get(event, []) if h.applies(name)]
        outcomes += [self._run_callable(fn, event, data) for fn in self._callables.get(event, [])]
        return merge_outcomes(outcomes)

    def _run_command(self, hook: CommandHook, payload: dict[str, Any]) -> HookOutcome:
        stdin = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        result = run_subprocess(hook.argv, cwd=str(self.workspace.root), timeout_s=hook.timeout_s, allow_network=False, stdin=stdin)
        outcome = HookOutcome()
        if result.timed_out:
            outcome.message = f"[hook {hook.command!r} timed out after {hook.timeout_s:g} s; continued]"
        elif result.returncode == BLOCK_EXIT_CODE:
            outcome.allow = False
            outcome.message = truncate(result.stderr.strip() or result.stdout.strip() or f"blocked by hook {hook.command!r}", HOOK_MESSAGE_MAX_CHARS)
        elif result.returncode == 0:
            parsed = extract_json(result.stdout) if result.stdout.strip() else None
            if isinstance(parsed, dict):
                upd = parsed.get("updated_arguments")
                outcome.updated_arguments = dict(upd) if isinstance(upd, dict) else None
                outcome.message = truncate(str(parsed.get("message") or ""), HOOK_MESSAGE_MAX_CHARS)
        self._trace(
            {
                "event": payload["event"],
                "command": hook.command,
                "tool_name": payload.get("tool_name"),
                "returncode": result.returncode,
                "timed_out": result.timed_out,
                "duration_s": round(result.duration_s, 3),
                "allow": outcome.allow,
                "stderr": truncate(result.stderr, 500),
            }
        )
        return outcome

    def _run_callable(self, fn: HookCallable, event: str, payload: dict[str, Any]) -> HookOutcome:
        try:
            result = fn(payload)
        except Exception as e:  # noqa: BLE001 - a broken hook is logged, never fatal
            self._trace({"event": event, "callable": getattr(fn, "__name__", repr(fn)), "error": f"{type(e).__name__}: {e}"[:500], "allow": True})
            return HookOutcome()
        if isinstance(result, HookOutcome):
            return result
        if isinstance(result, dict):
            upd = result.get("updated_arguments")
            return HookOutcome(bool(result.get("allow", True)), str(result.get("message") or ""), dict(upd) if isinstance(upd, dict) else None)
        return HookOutcome()

    def _trace(self, data: dict[str, Any]) -> None:
        if self.tracer is None:
            return
        try:
            self.tracer.record("hook", data)
        except Exception:  # noqa: BLE001
            pass
