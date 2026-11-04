"""Permission policy: modes and allow/deny patterns, enforced in code before any tool runs.

Pattern syntax (the Claude Code shape): ``tool`` (any arguments), ``tool(*)``,
``tool(prefix *)`` (prefix match on the call's subject) and ``tool(exact)``.
The subject of a call is the command for ``bash``, the path or pattern for the
file tools, the url or query for web tools, the agent name for ``agent``, and
otherwise the canonical JSON of the arguments. Tool names match
case-insensitively, subjects case-sensitively.

The tool part may end in ``*`` to match every tool whose name starts with what
precedes it (``mcp__playwright__browser_*``, ``mcp__*``), and a bare server name
``mcp__<server>`` matches every tool of that connector (``mcp__<server>__...``),
as in Claude Code. A lone ``*`` is refused: it would be a mode, not a pattern.
"""
from __future__ import annotations

import fnmatch
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ..domain.errors import ConfigurationError
from ..domain.utils import canonical_json
from ..domain.workspace import Workspace, WorkspaceError
from ..tools.registry import Decision, ToolSpec

MODES = ("ask", "accept-edits", "auto", "plan", "bypass")
DEFAULT_ALWAYS_ASK = frozenset({"network", "delete"})
AUTO_ALLOWED_CLASSES = frozenset({"read", "compute", "write", "spawn", "interactive"})
PATH_LIKE_ARGS = ("path", "pattern", "out_path", "paths")
CONTROL_PLANE_DIR = ".edgent"  # settings, hooks, skills and agents: the model never changes its own permissions without a person
HARNESS_OWNED_SUBDIRS = ("traces", "tool-results", "cache")  # under .edgent, written by the harness itself
NETWORK_ESCALATION_ARGS = ("allow_network",)  # a compute tool asking for the network is judged as a network tool
_SHELL_SPLIT_RE = re.compile(r"[;&|\n`]|\$\(")  # command separators; a deny pattern must match every segment it could run as

SUBJECT_KEYS: dict[str, tuple[str, ...]] = {
    "bash": ("command",),
    "read_file": ("path",),
    "write_file": ("path",),
    "edit_file": ("path",),
    "list_files": ("path",),
    "glob": ("pattern", "path"),
    "grep": ("pattern", "path"),
    "agent": ("agent",),
}
WEB_KEYS = ("url", "query")

_PATTERN_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)(\*)?\s*(?:\((.*)\))?\s*$", re.DOTALL)
MCP_TOOL_PREFIX = "mcp__"  # connector tools are named mcp__<server>__<tool>
MCP_NAME_SEPARATOR = "__"


def is_server_prefix(name: str) -> bool:
    """True for a bare connector name such as ``mcp__composio``: the mcp prefix and a server, but no tool part."""
    return name.lower().startswith(MCP_TOOL_PREFIX) and len(name) > len(MCP_TOOL_PREFIX) and MCP_NAME_SEPARATOR not in name[len(MCP_TOOL_PREFIX) :]


def tool_name_matches(pattern_tool: str, tool_name: str, prefix: bool = False) -> bool:
    """Case-insensitive tool-name match: exact, a ``*`` prefix (``prefix=True``), or every tool of a bare server name."""
    pattern, name = pattern_tool.lower(), tool_name.lower()
    if prefix:
        return name.startswith(pattern)
    return name == pattern or (is_server_prefix(pattern) and name.startswith(pattern + MCP_NAME_SEPARATOR))


def expand_tool_names(entries: Iterable[str], names: Iterable[str]) -> set[str]:
    """Allowlist entries as concrete names: an entry with ``*`` is an fnmatch pattern over ``names``, a bare
    ``mcp__<server>`` stands for that server's tools, and anything else is kept as written. A pattern that matches
    nothing contributes nothing: a connector that is not configured simply has no tools."""
    known = list(names)
    out: set[str] = set()
    for entry in entries:
        if "*" in entry:
            out.update(n for n in known if fnmatch.fnmatchcase(n.lower(), entry.lower()))
        elif is_server_prefix(entry):
            out.update(n for n in known if tool_name_matches(entry, n))
        else:
            out.add(entry)
    return out


def shell_segments(command: str) -> list[str]:
    """The simple commands a shell line could run: `x; rm -rf .` must not slip past a `bash(rm *)` deny because it starts with x."""
    return [seg.strip() for seg in _SHELL_SPLIT_RE.split(command or "") if seg.strip()]


def subject_of(tool_name: str, args: dict[str, Any]) -> str:
    """The part of a call that patterns match against."""
    name = tool_name.lower()
    keys = SUBJECT_KEYS.get(name)
    if keys is None and name.startswith("web"):
        keys = WEB_KEYS
    if keys is not None:
        for key in keys:
            value = args.get(key)
            if isinstance(value, str):
                return value
        return ""
    return canonical_json(args)


@dataclass(frozen=True)
class Pattern:
    tool: str  # lower-cased tool name, or the name prefix when tool_prefix is set
    kind: str  # any | prefix | exact (how the subject matches)
    subject: str = ""
    tool_prefix: bool = False  # the tool part ended in "*": match every tool name starting with ``tool``

    @classmethod
    def parse(cls, text: str) -> Pattern:
        m = _PATTERN_RE.match(text or "")
        if not m:
            raise ConfigurationError(
                f"invalid permission pattern {text!r}; use tool, tool(*), tool(prefix *), tool(exact), a tool-name prefix such as mcp__playwright__browser_*, or a connector such as mcp__composio"
            )
        tool, star, inner = m.group(1).lower(), bool(m.group(2)), m.group(3)
        if inner is None or inner.strip() == "*":
            return cls(tool, "any", tool_prefix=star)
        if inner.endswith("*"):
            return cls(tool, "prefix", inner[:-1], tool_prefix=star)
        return cls(tool, "exact", inner, tool_prefix=star)

    def matches_tool(self, tool_name: str) -> bool:
        return tool_name_matches(self.tool, tool_name, self.tool_prefix)

    def matches(self, tool_name: str, subject: str) -> bool:
        if not self.matches_tool(tool_name):
            return False
        if self.kind == "any":
            return True
        if self.kind == "prefix":
            return subject.startswith(self.subject) or subject == self.subject.rstrip()
        return subject == self.subject

    def __str__(self) -> str:
        tool = f"{self.tool}*" if self.tool_prefix else self.tool
        if self.kind == "any":
            return tool
        return f"{tool}({self.subject}{'*' if self.kind == 'prefix' else ''})"


class Permissions:
    """Decides allow / deny / ask for one call. Deny patterns win, then allow patterns, then the mode."""

    def __init__(
        self,
        mode: str = "ask",
        allow: Iterable[str] = (),
        deny: Iterable[str] = (),
        workspace: Workspace | None = None,
        always_ask: Iterable[str] = DEFAULT_ALWAYS_ASK,
    ) -> None:
        if mode not in MODES:
            raise ConfigurationError(f"unknown permission mode {mode!r}; choose one of {', '.join(MODES)}")
        self.mode = mode
        self.allow = [Pattern.parse(p) for p in allow]
        self.deny = [Pattern.parse(p) for p in deny]
        self.workspace = workspace
        self.always_ask = frozenset(always_ask)

    @classmethod
    def from_settings(cls, settings: dict[str, Any], workspace: Workspace | None = None) -> Permissions:
        mode = settings.get("mode", "ask")
        if not isinstance(mode, str):
            raise ConfigurationError(f"settings.mode must be a string; got {mode!r}")
        perms = settings.get("permissions") or {}
        if not isinstance(perms, dict):
            raise ConfigurationError("settings.permissions must be an object with 'allow' and 'deny' lists")
        allow = _string_list(perms.get("allow"), "permissions.allow")
        deny = _string_list(perms.get("deny"), "permissions.deny")
        always_ask = perms.get("always_ask")
        kwargs: dict[str, Any] = {}
        if always_ask is not None:
            kwargs["always_ask"] = frozenset(_string_list(always_ask, "permissions.always_ask"))
        return cls(mode, allow, deny, workspace, **kwargs)

    def with_mode(self, mode: str) -> Permissions:
        return Permissions(mode, [str(p) for p in self.allow], [str(p) for p in self.deny], self.workspace, self.always_ask)

    # ------------------------------------------------------------------ decision
    def decide(self, spec: ToolSpec, args: dict[str, Any]) -> Decision:
        args = args if isinstance(args, dict) else {}
        subject = subject_of(spec.name, args)
        subjects = [subject, *shell_segments(subject)] if spec.name.lower() == "bash" else [subject]
        if any(p.matches(spec.name, s) for p in self.deny for s in subjects):
            return "deny"
        if any(p.matches(spec.name, subject) for p in self.allow):
            return "allow"
        perm = spec.permission
        if perm == "compute" and any(args.get(k) is True for k in NETWORK_ESCALATION_ARGS):
            perm = "network"
        if self.mode == "bypass":
            return "allow"
        if self.mode == "plan":
            return "allow" if perm == "read" else "deny"
        if perm in self.always_ask:
            return "ask"
        if perm in ("write", "delete", "compute") and self.touches_control_plane(args):
            return "ask"
        if self.mode == "auto":
            return "allow" if perm in AUTO_ALLOWED_CLASSES and self.inside_workspace(args) else "ask"
        if perm in ("read", "compute"):
            return "allow"
        if self.mode == "accept-edits" and perm == "write":
            return "allow" if self.inside_workspace(args) else "ask"
        return "ask"

    def touches_control_plane(self, args: dict[str, Any]) -> bool:
        """True when a path-like argument points into the workspace's .edgent directory (settings, hooks, skills, agents)."""
        if self.workspace is None:
            return False
        for key in PATH_LIKE_ARGS:
            v = args.get(key) if isinstance(args, dict) else None
            values = [v] if isinstance(v, str) else [x for x in (v or []) if isinstance(x, str)] if isinstance(v, list) else []
            for value in values:
                try:
                    rel = self.workspace.relpath(self.workspace.resolve(value))
                except (WorkspaceError, ValueError):
                    continue
                parts = rel.replace("\\", "/").split("/")
                if parts and parts[0] == CONTROL_PLANE_DIR and not (len(parts) > 1 and parts[1] in HARNESS_OWNED_SUBDIRS):
                    return True
        return False

    def inside_workspace(self, args: dict[str, Any]) -> bool:
        """True when every path-like argument resolves inside the workspace (no path-like argument counts as inside)."""
        values: list[str] = []
        for key in PATH_LIKE_ARGS:
            v = args.get(key) if isinstance(args, dict) else None
            if isinstance(v, str):
                values.append(v)
            elif isinstance(v, list):
                values.extend(x for x in v if isinstance(x, str))
        if not values:
            return True
        if self.workspace is None:
            return False
        for v in values:
            try:
                self.workspace.resolve(v)
            except WorkspaceError:
                return False
        return True

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "allow": [str(p) for p in self.allow],
            "deny": [str(p) for p in self.deny],
            "always_ask": sorted(self.always_ask),
            "workspace": str(self.workspace.root) if self.workspace is not None else None,
        }


def _string_list(value: Any, where: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ConfigurationError(f"settings.{where} must be a list of pattern strings; got {value!r}")
    return list(value)
