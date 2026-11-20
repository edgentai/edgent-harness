"""Permission modes, pattern allow/deny rules, and lifecycle hooks.

Enforces safety boundaries in code rather than prompts:
- Modes: ``ask`` (default), ``accept_edits``, ``auto``, ``plan`` (read-only).
- Rules: ``tool_name`` or ``tool_name(glob_pattern)`` matched against the tool's
  primary target (path or command argv).
- Hooks: ``PreToolUse``, ``PostToolUse``, ``UnitDone``, ``Stop``.
"""
from __future__ import annotations

import enum
import fnmatch
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..tools.registry import ToolContext, ToolResult, ToolSpec


class PermissionMode(str, enum.Enum):
    ASK = "ask"
    ACCEPT_EDITS = "accept_edits"
    AUTO = "auto"
    PLAN = "plan"


def _target_for(name: str, args: dict[str, Any]) -> str:
    if name == "bash" and args.get("command") is not None:
        return str(args["command"]).strip()
    if name == "run_command" and isinstance(args.get("argv"), list):
        return " ".join(str(x) for x in args["argv"])
    for key in ("path", "target", "url", "query", "name"):
        if args.get(key) is not None:
            return str(args[key])
    return ""


def _rule_matches(rule: str, tool_name: str, args: dict[str, Any]) -> bool:
    rule = rule.strip()
    if "(" in rule and rule.endswith(")"):
        head, pat = rule.split("(", 1)
        pat = pat[:-1]
        if head.strip() != tool_name:
            return False
        return fnmatch.fnmatch(_target_for(tool_name, args), pat)
    return fnmatch.fnmatch(tool_name, rule)


@dataclass
class PermissionPolicy:
    mode: PermissionMode = PermissionMode.ASK
    allow: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)

    def check(
        self,
        spec: ToolSpec,
        args: dict[str, Any],
        ask_approve: Optional[Callable[[str, dict[str, Any]], bool]] = None,
    ) -> tuple[bool, str]:
        # Deny rules always win.
        for rule in self.deny:
            if _rule_matches(rule, spec.name, args):
                return False, f"tool '{spec.name}' denied by rule '{rule}'"

        # Plan mode is strictly read-only + interactive.
        if self.mode == PermissionMode.PLAN and spec.permission not in ("read", "interactive"):
            return False, f"plan mode is read-only; '{spec.name}' ({spec.permission}) is blocked"

        # Auto mode approves everything not explicitly denied above.
        if self.mode == PermissionMode.AUTO:
            return True, ""

        # Explicit allow rules bypass mode prompts (except plan mode above).
        for rule in self.allow:
            if _rule_matches(rule, spec.name, args):
                return True, ""

        if spec.permission in ("read", "interactive"):
            return True, ""
        if spec.permission == "compute":
            if self.mode in (PermissionMode.AUTO, PermissionMode.ACCEPT_EDITS, PermissionMode.ASK):
                return True, ""
        if spec.permission == "spawn":
            return True, ""
        if spec.permission == "write":
            if self.mode in (PermissionMode.AUTO, PermissionMode.ACCEPT_EDITS):
                return True, ""
        if spec.permission in ("write", "network", "delete"):
            if ask_approve is not None and ask_approve(spec.name, args):
                return True, ""
            return False, f"tool '{spec.name}' ({spec.permission}) requires approval in '{self.mode.value}' mode"
        return True, ""


@dataclass
class HookDecision:
    allow: bool = True
    reason: str = ""
    updated_arguments: Optional[dict[str, Any]] = None


PreHook = Callable[[str, dict[str, Any], ToolContext], Optional[HookDecision]]
PostHook = Callable[[str, dict[str, Any], ToolResult, ToolContext], None]
UnitHook = Callable[[dict[str, Any], ToolContext], tuple[bool, list[str]]]
StopHook = Callable[[dict[str, Any]], None]


class HookManager:
    """Extensible lifecycle hooks (PreToolUse, PostToolUse, UnitDone, Stop)."""

    def __init__(self) -> None:
        self._pre: list[PreHook] = []
        self._post: list[PostHook] = []
        self._unit_done: list[UnitHook] = []
        self._stop: list[StopHook] = []

    def on_pre_tool_use(self, fn: PreHook) -> None:
        self._pre.append(fn)

    def on_post_tool_use(self, fn: PostHook) -> None:
        self._post.append(fn)

    def on_unit_done(self, fn: UnitHook) -> None:
        self._unit_done.append(fn)

    def on_stop(self, fn: StopHook) -> None:
        self._stop.append(fn)

    def run_pre_tool_use(self, name: str, args: dict[str, Any], ctx: ToolContext) -> HookDecision:
        cur_args = dict(args)
        modified = False
        for fn in self._pre:
            dec = fn(name, cur_args, ctx)
            if dec is None:
                continue
            if not dec.allow:
                return dec
            if dec.updated_arguments is not None:
                cur_args = dict(dec.updated_arguments)
                modified = True
        return HookDecision(allow=True, updated_arguments=cur_args if modified else None)

    def run_post_tool_use(self, name: str, args: dict[str, Any], res: ToolResult, ctx: ToolContext) -> None:
        for fn in self._post:
            fn(name, args, res, ctx)

    def run_unit_done(self, unit: dict[str, Any], ctx: ToolContext) -> tuple[bool, list[str]]:
        findings: list[str] = []
        for fn in self._unit_done:
            ok, issues = fn(unit, ctx)
            if not ok:
                findings.extend(issues or ["unit_done hook rejected completion"])
        return not findings, findings

    def run_stop(self, summary: dict[str, Any]) -> None:
        for fn in self._stop:
            fn(summary)
