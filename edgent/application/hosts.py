"""Host setup: register Edgent as an MCP server with a host and install its skills, without surprises.

Every host is a small spec: where its MCP configuration lives and in which
format, and where it reads skills and agents. ``plan`` reads what is there
and returns the exact changes as ``Change`` records (a dry run by
construction), merging additively so no other server or key is lost, and a
second plan on an applied state is all noops. ``apply`` writes them with
numbered backups and refuses a plan whose files changed underneath it.
``doctor`` only reads. Paths are parameterised by ``home`` and the
workspace so tests run in temporary directories.

Skill and agent files land in directories the model can write to, so setup
never writes through a symbolic link there (the plan and the apply both
refuse one); only a host's configuration file may be a dotfiles link. Agent
definitions are rewritten for the host that reads them: Claude Code gets
its own tool names in ``tools:`` (``host_agent_text``).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import stat
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..domain.errors import ConfigurationError
from ..domain.utils import atomic_write_text, ensure_dir
from .skills import SKILL_FILE, builtin_skills_dir

HOSTS: tuple[str, ...] = ("claude", "agy", "codex", "cursor", "generic")
SERVER_NAME = "edgent"
BACKUP_SUFFIX = ".bak-"
BACKUPS_PER_FILE_MAX = 1_000  # numbered backups beside one file (count) before setup refuses to add another
SKILL_FILES_MAX = 500  # files copied per skill (count)
_SKILL_SUBDIRS = ("references", "scripts")
_HEADER_RE = re.compile(r"^\s*\[\s*(?:\"?mcp_servers\"?)\s*\.\s*\"?([A-Za-z0-9_.-]+)\"?\s*\]\s*(?:#.*)?$")

# Claude Code reads a subagent's ``tools:`` as an allowlist of its own tool names; Edgent's tools reach it over MCP as
# ``mcp__<server>__<tool>``. Edgent ids with a host-native equivalent get it too (the host's own file tools know the
# project); ids that need Edgent's loop or session (agent, ask_user, todo_*, session_recall, remember) map to the
# host's counterpart or to nothing, since the MCP server cannot serve them. Any other id becomes its MCP name.
MCP_TOOL_PREFIX = f"mcp__{SERVER_NAME}__"
CLAUDE_TOOL_NAMES: dict[str, tuple[str, ...]] = {
    "read_file": ("Read", f"{MCP_TOOL_PREFIX}read_file"),
    "write_file": ("Write", f"{MCP_TOOL_PREFIX}write_file"),
    "edit_file": ("Edit", f"{MCP_TOOL_PREFIX}edit_file"),
    "list_files": ("Glob", f"{MCP_TOOL_PREFIX}list_files"),
    "glob": ("Glob", f"{MCP_TOOL_PREFIX}glob"),
    "grep": ("Grep", f"{MCP_TOOL_PREFIX}grep"),
    "bash": ("Bash", f"{MCP_TOOL_PREFIX}bash"),
    "web_search": ("WebSearch", f"{MCP_TOOL_PREFIX}web_search"),
    "web_fetch": ("WebFetch", f"{MCP_TOOL_PREFIX}web_fetch"),
    "skill": ("Skill", f"{MCP_TOOL_PREFIX}skill"),
    "agent": ("Task",),
    "ask_user": ("AskUserQuestion",),
    "todo_write": ("TodoWrite",),
    "todo_read": ("TodoWrite",),
    "session_recall": (),
    "remember": (),
    "tool_search": ("ToolSearch",),  # Claude Code loads deferred tool schemas with its own ToolSearch
}
_AGENT_TOOL_KEYS = ("tools", "disallowedTools", "disallowed_tools")
# Edgent's connectors are not re-exported over MCP: a host uses its own, so the skill that explains Edgent's stays home
HOST_EXCLUDED_SKILLS = frozenset({"connectors"})
_AGENT_EDGENT_ONLY_KEYS = ("effort",)  # front-matter keys Claude Code does not define; name, description, skills, maxTurns, background and model it does

Action = Literal["create", "update", "noop"]
ConfigRoot = Literal["workspace", "home", "none"]
ConfigFormat = Literal["json", "toml", "none"]


@dataclass(frozen=True)
class HostSpec:
    name: str
    config: str | None  # path of the MCP configuration, relative to ``config_root``
    config_root: ConfigRoot
    format: ConfigFormat
    skills: str | None  # skills directory, relative to the workspace
    agents: str | None  # subagent definitions, relative to the workspace
    description: str


HOST_SPECS: dict[str, HostSpec] = {
    "claude": HostSpec("claude", ".mcp.json", "workspace", "json", ".claude/skills", ".claude/agents", "Claude Code: project servers in <workspace>/.mcp.json; skills and subagents under <workspace>/.claude/"),
    "agy": HostSpec("agy", ".gemini/antigravity-cli/settings.json", "home", "json", ".agents/skills", None, "Antigravity CLI: user settings in ~/.gemini/antigravity-cli/settings.json; skills under <workspace>/.agents/skills/"),
    "codex": HostSpec("codex", ".codex/config.toml", "home", "toml", ".agents/skills", None, "Codex CLI: a [mcp_servers.edgent] table in ~/.codex/config.toml; skills under <workspace>/.agents/skills/"),
    "cursor": HostSpec("cursor", ".cursor/mcp.json", "workspace", "json", ".cursor/skills", None, "Cursor: project servers in <workspace>/.cursor/mcp.json; skills under <workspace>/.cursor/skills/"),
    "generic": HostSpec("generic", None, "none", "none", None, None, "Any MCP client: paste the JSON snippet into its configuration"),
}


@dataclass
class Change:
    """One planned file write. ``before`` is the current text (``None`` when the file does not exist)."""

    path: str
    action: Action
    before: str | None
    after: str
    note: str = ""
    follow_symlinks: bool = True  # a host config may be a dotfiles link (the real file is rewritten); skill and agent files never are

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "action": self.action, "note": self.note, "before": self.before, "after": self.after}


@dataclass(frozen=True)
class HostPaths:
    host: str
    config: Path | None
    format: ConfigFormat
    skills: Path | None
    agents: Path | None


# --------------------------------------------------------------------------- resolution
def _spec(host: str) -> HostSpec:
    spec = HOST_SPECS.get(host)
    if spec is None:
        raise ConfigurationError(f"unknown host {host!r}; choose one of {', '.join(HOSTS)}")
    return spec


def _root(workspace: Any) -> Path:
    """The workspace root from a ``Workspace`` or a path (``Path.root`` is the filesystem root, so paths are not duck-typed)."""
    raw = workspace if isinstance(workspace, (str, os.PathLike)) else workspace.root
    return Path(raw).expanduser().resolve()


def _home(home: str | os.PathLike[str] | None) -> Path:
    return Path(home).expanduser().resolve() if home is not None else Path.home()


def host_paths(host: str, workspace: Any, home: str | os.PathLike[str] | None = None) -> HostPaths:
    spec = _spec(host)
    root = _root(workspace)
    config: Path | None = None
    if spec.config is not None:
        config = (root if spec.config_root == "workspace" else _home(home)) / spec.config
    return HostPaths(host, config, spec.format, root / spec.skills if spec.skills else None, root / spec.agents if spec.agents else None)


def edgent_command() -> list[str]:
    """The ``edgent`` executable when installed, else this interpreter with ``-m edgent``."""
    exe = shutil.which("edgent")
    return [exe] if exe else [sys.executable, "-m", "edgent"]


def server_entry(workspace: Any, command: list[str] | str | None = None) -> dict[str, Any]:
    """The MCP server record every host gets: ``edgent --workspace <abs> mcp`` (the global option precedes the subcommand)."""
    cmd = [command] if isinstance(command, str) else (edgent_command() if command is None else list(command))
    if not cmd or not cmd[0]:
        raise ConfigurationError("command must name an executable")
    return {"command": cmd[0], "args": [*cmd[1:], "--workspace", str(_root(workspace)), "mcp"]}


def snippet(workspace: Any, command: list[str] | str | None = None) -> str:
    return json.dumps({"mcpServers": {SERVER_NAME: server_entry(workspace, command)}}, indent=2, ensure_ascii=False) + "\n"


def _same_entry(existing: Any, entry: dict[str, Any]) -> bool:
    return isinstance(existing, dict) and existing.get("command") == entry["command"] and list(existing.get("args") or []) == entry["args"]


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None
    except IsADirectoryError as e:
        raise ConfigurationError(f"{path} is a directory, not a configuration file; move it aside and run setup again") from e
    except NotADirectoryError as e:
        raise ConfigurationError(f"{path}: a parent of this path is a file, not a directory; move it aside and run setup again") from e
    except PermissionError as e:
        raise ConfigurationError(f"{path} is not readable ({e.strerror or e}); fix its permissions and run setup again") from e


# --------------------------------------------------------------------------- JSON hosts
def _parse_json(text: str, path: Path) -> dict[str, Any]:
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except ValueError as e:
        raise ConfigurationError(f"{path} is not valid JSON ({e}); fix it or move it aside, then run setup again") from e
    if not isinstance(data, dict):
        raise ConfigurationError(f"{path} must hold a JSON object at the top level; move it aside and run setup again")
    return data


def _merge_json(before: str | None, entry: dict[str, Any], path: Path) -> tuple[str, str]:
    data = _parse_json(before, path) if before is not None else {}
    servers = data.get("mcpServers")
    if servers is None:
        servers = {}
    if not isinstance(servers, dict):
        raise ConfigurationError(f"{path}: 'mcpServers' must be a JSON object; fix it by hand, then run setup again")
    existing = servers.get(SERVER_NAME)
    if _same_entry(existing, entry):
        return before if before is not None else "", f"{SERVER_NAME} is already registered in {path.name}"
    others = sorted(k for k in servers if k != SERVER_NAME)
    kept = f"; other servers kept: {', '.join(others)}" if others else ""
    if isinstance(existing, dict):
        servers[SERVER_NAME] = {**existing, **entry}  # the user's extra keys (env, cwd) survive
        note = f"update the {SERVER_NAME} entry in {path.name} (command or args changed){kept}"
    else:
        servers[SERVER_NAME] = entry
        note = f"register {SERVER_NAME} in mcpServers of {path.name}{kept}"
    data["mcpServers"] = servers
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n", note


def _json_entry(text: str | None, path: Path) -> dict[str, Any] | None:
    if text is None:
        return None
    servers = _parse_json(text, path).get("mcpServers")
    entry = servers.get(SERVER_NAME) if isinstance(servers, dict) else None
    return entry if isinstance(entry, dict) else None


# --------------------------------------------------------------------------- TOML host (codex)
def _toml_value(value: Any) -> str | None:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)  # JSON string escapes are valid TOML basic-string escapes
    if isinstance(value, list) and all(isinstance(v, (str, int, float, bool)) for v in value):
        return "[" + ", ".join(_toml_value(v) or "" for v in value) + "]"
    return None  # tables and nested arrays are left to their own headers


def _toml_block(entry: dict[str, Any], extra: dict[str, Any] | None = None) -> str:
    lines = [f"[mcp_servers.{SERVER_NAME}]", f"command = {_toml_value(entry['command'])}", f"args = {_toml_value(entry['args'])}"]
    for key, value in (extra or {}).items():
        rendered = _toml_value(value)
        if key not in ("command", "args") and rendered is not None:
            lines.append(f"{key} = {rendered}")
    return "\n".join(lines) + "\n"


def _parse_toml(text: str, path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ConfigurationError(f"{path} is not valid TOML ({e}); fix it or move it aside, then run setup again") from e


def _toml_entry(text: str | None, path: Path) -> dict[str, Any] | None:
    if text is None:
        return None
    servers = _parse_toml(text, path).get("mcp_servers")
    entry = servers.get(SERVER_NAME) if isinstance(servers, dict) else None
    return entry if isinstance(entry, dict) else None


def _replace_toml_table(text: str, block: str, path: Path) -> str:
    """Replace the ``[mcp_servers.edgent]`` table body (up to the next header) with ``block``; sub-tables stay."""
    lines = text.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if (m := _HEADER_RE.match(line)) and m.group(1) == SERVER_NAME), None)
    if start is None:
        raise ConfigurationError(f"{path}: mcp_servers.{SERVER_NAME} is defined in a form setup cannot rewrite (an inline table); edit it by hand")
    end = next((j for j in range(start + 1, len(lines)) if lines[j].lstrip().startswith("[")), len(lines))
    tail = lines[end:]
    new_block = block if not tail else block + "\n"
    return "".join(lines[:start]) + new_block + "".join(tail)


def _merge_toml(before: str | None, entry: dict[str, Any], path: Path) -> tuple[str, str]:
    text = before or ""
    existing = _toml_entry(text, path) if before is not None else None
    if _same_entry(existing, entry):
        return text, f"{SERVER_NAME} is already registered in {path.name}"
    if existing is None:
        sep = "" if not text else ("\n" if text.endswith("\n\n") else ("\n\n" if not text.endswith("\n") else "\n"))
        return text + sep + _toml_block(entry), f"append [mcp_servers.{SERVER_NAME}] to {path.name}"
    return _replace_toml_table(text, _toml_block(entry, existing), path), f"replace [mcp_servers.{SERVER_NAME}] in {path.name} (command or args changed)"


# --------------------------------------------------------------------------- plan / apply
def plan(host: str, workspace: Any, home: str | os.PathLike[str] | None = None, command: list[str] | str | None = None) -> list[Change]:
    """The registration changes for ``host``: at most one file, merged additively; all noops when already applied."""
    spec = _spec(host)
    root = _root(workspace)
    entry = server_entry(root, command)
    if spec.format == "none":
        text = snippet(root, command)
        return [Change("", "noop", None, text, "generic host: add this to your MCP client's configuration (nothing is written):\n" + text.rstrip())]
    path = host_paths(host, root, home).config
    assert path is not None
    before = _read_text(path)
    after, note = _merge_json(before, entry, path) if spec.format == "json" else _merge_toml(before, entry, path)
    action: Action = "noop" if before == after else ("update" if before is not None else "create")
    return [Change(str(path), action, before, after, note)]


def next_backup_path(path: Path) -> Path:
    """``<file>.bak-<n>`` with n above every existing backup, so a backup is never overwritten."""
    prefix = path.name + BACKUP_SUFFIX
    siblings = [p.name for p in path.parent.iterdir()] if path.parent.is_dir() else []  # not glob: a file name may hold glob characters
    tails = [name[len(prefix) :] for name in siblings if name.startswith(prefix)]
    used = [int(t) for t in tails if t.isascii() and t.isdigit()]
    n = max(used, default=0) + 1
    if n > BACKUPS_PER_FILE_MAX:
        raise ConfigurationError(f"{path} already has {BACKUPS_PER_FILE_MAX} backups ({prefix}<n>); delete some before running setup again")
    return path.with_name(f"{prefix}{n}")


def apply(changes: list[Change], backup: bool = True) -> list[str]:
    """Write the planned creates and updates; an update first copies the file to a numbered backup. Returns the written paths."""
    written: list[str] = []
    pending = [c for c in changes if c.action in ("create", "update") and c.path]
    for change in pending:  # before any write: a link planted after the plan refuses the whole apply, not half of it
        if not change.follow_symlinks:
            _refuse_symlink(Path(change.path))
    for change in pending:
        path = Path(change.path)
        current = _read_text(path)
        if current != change.before:
            raise ConfigurationError(f"{path} changed since the plan was made; run the plan again before applying")
        target = path.resolve()  # a symlinked dotfile stays a link: the real file is rewritten and backed up beside itself
        try:
            mode = stat.S_IMODE(target.stat().st_mode) if current is not None else None
            if change.action == "update" and backup and current is not None:
                shutil.copy2(target, next_backup_path(target))
            ensure_dir(target.parent)
            atomic_write_text(target, change.after)
            if mode is not None:
                os.chmod(target, mode)
        except OSError as e:
            raise ConfigurationError(f"cannot write {path} ({e.strerror or e}); check the permissions of {target.parent}, then run setup again") from e
        written.append(str(path))
    return written


# --------------------------------------------------------------------------- skills and agents
def _refuse_symlink(path: Path) -> None:
    """Refuse a planned skill or agent path with a symlink anywhere in it: a link under a model-writable tree could point
    anywhere, and setup must never write through it (the planned paths start from the resolved workspace root)."""
    if os.path.realpath(path) != os.path.abspath(path):
        raise ConfigurationError(
            f"{path} is, or lies under, a symbolic link; setup never writes through links in a host's skills or agents directory. "
            "Remove the link (keep a regular file or directory there) and run setup again"
        )


def _front_matter_list(lines: list[str], i: int) -> tuple[list[str], int]:
    """The items of the front-matter key at ``lines[i]`` (inline ``a, b`` or ``[a, b]``, or indented ``- a`` lines) and the next index."""
    value = lines[i].partition(":")[2].strip()
    i += 1
    if value:
        inner = value[1:-1] if value.startswith("[") and value.endswith("]") else value
        return [v.strip().strip("\"'") for v in inner.split(",") if v.strip()], i
    items: list[str] = []
    while i < len(lines) and lines[i].startswith((" ", "\t")) and lines[i].strip().startswith("- "):
        items.append(lines[i].strip()[2:].strip().strip("\"'"))
        i += 1
    return items, i


def claude_tool_names(names: list[str]) -> list[str]:
    """Edgent tool ids as Claude Code's allowlist names: the host-native tool where one exists, the MCP name otherwise."""
    out: list[str] = []
    for name in names:
        if name.startswith("mcp__") and not name.startswith(MCP_TOOL_PREFIX):
            # an Edgent connector pattern (mcp__perplexity__*) means the host's own connector of that name, server-wide
            server = name.rstrip("*").rstrip("_")
            if server not in out:
                out.append(server)
            continue
        for mapped in CLAUDE_TOOL_NAMES.get(name, (f"{MCP_TOOL_PREFIX}{name}",)):
            if mapped not in out:
                out.append(mapped)
    return out


def host_agent_text(host: str, text: str) -> str:
    """The agent definition as ``host`` must read it. Claude Code gets its ``tools:`` (and ``disallowedTools:``) rewritten
    to its own names and the Edgent-only keys dropped; every other line and the body stay byte for byte."""
    if host != "claude" or not text.startswith("---"):
        return text
    end = text.find("\n---", 3)
    if end == -1:
        return text
    lines = text[:end].split("\n")
    out = [lines[0]]
    i = 1
    while i < len(lines):
        line = lines[i]
        key = line.partition(":")[0].strip() if ":" in line and not line.startswith((" ", "\t")) else ""
        if key not in (*_AGENT_TOOL_KEYS, *_AGENT_EDGENT_ONLY_KEYS):
            out.append(line)
            i += 1
            continue
        items, i = _front_matter_list(lines, i)
        if key in _AGENT_TOOL_KEYS and any(t in ("*", "all") for t in items):
            out.append(f"{key}: {', '.join(items)}")
        elif key in _AGENT_TOOL_KEYS and (mapped := claude_tool_names(items)):
            out.append(f"{key}: {', '.join(mapped)}")
    return "\n".join(out) + text[end:]


def _skill_dirs(source: Path) -> list[Path]:
    if not source.is_dir():
        return []
    return [d for d in sorted(source.iterdir()) if d.is_dir() and (d / SKILL_FILE).is_file()]


def _skill_files(skill_dir: Path) -> list[str]:
    """``SKILL.md`` plus every file under references/ and scripts/, hidden files and caches skipped."""
    out = [SKILL_FILE]
    for sub in _SKILL_SUBDIRS:
        base = skill_dir / sub
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            rel = p.relative_to(skill_dir)
            if not p.is_file() or any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
                continue
            out.append(rel.as_posix())
            if len(out) > SKILL_FILES_MAX:
                raise ConfigurationError(f"skill {skill_dir.name!r} has more than {SKILL_FILES_MAX} files; trim references/ and scripts/ before installing it")
    return out


def _file_change(src: Path, dst: Path, what: str, transform: Callable[[str], str] | None = None) -> Change:
    _refuse_symlink(dst)  # before reading: a link's target is not the plan's business
    try:
        after = src.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        return Change(str(dst), "noop", None, "", f"{what}: skipped, not UTF-8 text (skills ship Markdown, text references and scripts)", follow_symlinks=False)
    except OSError as e:
        raise ConfigurationError(f"cannot read {src} ({e.strerror or e}); fix its permissions and run setup again") from e
    if transform is not None:
        after = transform(after)
    before = _read_text(dst)
    if before == after:
        return Change(str(dst), "noop", before, after, f"{what}: up to date", follow_symlinks=False)
    if before is None:
        return Change(str(dst), "create", None, after, f"{what}: install", follow_symlinks=False)
    return Change(str(dst), "update", before, after, f"{what}: update (content differs; a backup is kept)", follow_symlinks=False)


def plan_skills(host: str, workspace: Any, source_dir: str | os.PathLike[str] | None = None, agents_dir: str | os.PathLike[str] | None = None) -> list[Change]:
    """Per-file changes that install the skill folders (and, for hosts with subagents, the agent files) into the host's directories."""
    paths = host_paths(host, workspace)
    out: list[Change] = []
    if paths.skills is not None:
        source = Path(source_dir) if source_dir is not None else builtin_skills_dir()
        for skill_dir in _skill_dirs(source):
            if source_dir is None and skill_dir.name in HOST_EXCLUDED_SKILLS:
                continue
            for rel in _skill_files(skill_dir):
                out.append(_file_change(skill_dir / rel, paths.skills / skill_dir.name / rel, f"skill {skill_dir.name} ({rel})"))
    if paths.agents is not None:
        if agents_dir is None:
            from .subagents import builtin_agents_dir

            agents_dir = builtin_agents_dir()
        agents = Path(agents_dir)
        if agents.is_dir():
            for f in sorted(agents.glob("*.md")):
                if f.is_file() and not f.name.startswith("."):
                    out.append(_file_change(f, paths.agents / f.name, f"agent {f.stem}", transform=lambda text, h=host: host_agent_text(h, text)))
    return out


def install_skills(
    host: str,
    workspace: Any,
    source_dir: str | os.PathLike[str] | None = None,
    agents_dir: str | os.PathLike[str] | None = None,
    backup: bool = True,
) -> list[Change]:
    """Copy the skills (and agents) into the host's directories; identical files are noops, changed files are backed up first."""
    changes = plan_skills(host, workspace, source_dir, agents_dir)
    apply(changes, backup=backup)
    return changes


# --------------------------------------------------------------------------- doctor
def _existing_entry(paths: HostPaths) -> tuple[dict[str, Any] | None, str | None]:
    """The registered entry and, when the file cannot be parsed, the problem as a hint."""
    if paths.config is None:
        return None, None
    try:
        text = _read_text(paths.config)
        entry = _json_entry(text, paths.config) if paths.format == "json" else _toml_entry(text, paths.config)
    except ConfigurationError as e:
        return None, str(e)
    return entry, None


def doctor(
    host: str,
    workspace: Any,
    home: str | os.PathLike[str] | None = None,
    command: list[str] | str | None = None,
    source_dir: str | os.PathLike[str] | None = None,
    agents_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Diagnose a host's registration and skills without changing anything."""
    spec = _spec(host)
    root = _root(workspace)
    paths = host_paths(host, root, home)
    hints: list[str] = []
    report: dict[str, Any] = {
        "host": host,
        "registered": False,
        "stale": False,
        "skills_installed": [],
        "skills_missing": [],
        "skills_stale": [],
        "agents_missing": [],
        "path": str(paths.config) if paths.config else "",
        "hints": hints,
    }
    if spec.format == "none":
        hints.append("generic host: nothing to check; paste the snippet printed by `edgent setup --host generic` into the client's configuration")
        return report
    entry = server_entry(root, command)
    existing, problem = _existing_entry(paths)
    if problem:
        hints.append(problem)
    report["registered"] = existing is not None
    report["stale"] = existing is not None and not _same_entry(existing, entry)
    if existing is None:
        hints.append(f"{SERVER_NAME} is not registered with {host}; run `edgent setup --host {host}`")
    elif report["stale"]:
        hints.append(f"the registered command or args differ from this installation; run `edgent setup --host {host}` to update them (a backup is kept)")
    else:
        cmd = str(existing.get("command") or "")
        if cmd and not (shutil.which(cmd) or Path(cmd).exists()):
            hints.append(f"the registered command {cmd!r} is not on PATH and does not exist; reinstall edgent or run `edgent setup --host {host}`")
    _doctor_skills(report, host, root, source_dir, agents_dir)
    return report


def _doctor_skills(report: dict[str, Any], host: str, root: Path, source_dir: Any, agents_dir: Any) -> None:
    paths = host_paths(host, root)
    if paths.skills is None:
        return
    by_skill: dict[str, list[Change]] = {}
    agents_missing: list[str] = []
    try:
        changes = plan_skills(host, root, source_dir, agents_dir)
    except ConfigurationError as e:
        report["hints"].append(str(e))
        return
    for change in changes:
        target = Path(change.path)
        if paths.agents is not None and target.parent == paths.agents:
            if change.action != "noop":
                agents_missing.append(target.stem)
            continue
        by_skill.setdefault(target.relative_to(paths.skills).parts[0], []).append(change)
    for name, changes in sorted(by_skill.items()):
        if not (paths.skills / name / SKILL_FILE).is_file():
            report["skills_missing"].append(name)
        elif any(c.action != "noop" for c in changes):
            report["skills_stale"].append(name)
        else:
            report["skills_installed"].append(name)
    report["agents_missing"] = agents_missing
    if report["skills_missing"] or report["skills_stale"] or agents_missing:
        report["hints"].append(f"skills or agents are missing or outdated under {paths.skills.parent}; run `edgent setup --host {host}`")


__all__ = [
    "BACKUPS_PER_FILE_MAX",
    "BACKUP_SUFFIX",
    "CLAUDE_TOOL_NAMES",
    "HOSTS",
    "HOST_SPECS",
    "MCP_TOOL_PREFIX",
    "SERVER_NAME",
    "Change",
    "HostPaths",
    "HostSpec",
    "apply",
    "claude_tool_names",
    "doctor",
    "edgent_command",
    "host_agent_text",
    "host_paths",
    "install_skills",
    "next_backup_path",
    "plan",
    "plan_skills",
    "server_entry",
    "snippet",
]
