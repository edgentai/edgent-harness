"""The system prompt, composed from Markdown files: identity, behaviour, role and mode.

``build_system_prompt`` returns one string. Its first line is the role marker ``[edgent-role: <role>]``
(providers and mocks route by it, and it is the stable cache prefix), then ``system.md``, ``behaviour.md``,
``roles/<role>.md`` when it exists, the lines for the mode, a capability note when the provider warrants one,
and ``extra``. Nothing volatile goes in: no timestamps, no counters, no paths that differ between runs, so
the composed prompt is byte-identical across a session.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..domain.errors import ConfigurationError, Limit

PROMPTS_DIR = Path(__file__).resolve().parent
ROLES_DIRNAME = "roles"
ROLE_MARKER_FORMAT = "[edgent-role: {role}]"  # must match adapters.providers.base.role_marker exactly
MODES: tuple[str, ...] = ("chat", "build", "subagent")
MAIN_PROMPT_TOKENS_LIMIT = Limit("composed main prompt", 2500, "tokens", does_not_raise=("the skill and agent indexes, which are added by the context assembler",))
PROMPT_FILE_MAX_CHARS = 12_000  # one Markdown part; larger means the prompt is doing a skill's job

_NAME_RE = re.compile(r"^(?:roles/)?[a-z0-9][a-z0-9_-]*$")
_ROLE_RE = re.compile(r"^[A-Za-z0-9_-]+$")  # what the providers' marker regex can route

MODE_TEXT: dict[str, str] = {
    "chat": (
        "## Mode: chat\n"
        "You are talking with the learner or user. Teach the way the socratic-tutoring skill says: diagnose from their "
        "words, ask before you tell, one step at a time, and simply answer when they want the answer. Answer from the "
        "sources with citations that resolve; say plainly when a question goes beyond them. Keep turns short; end with "
        "the next step or the next question, not a summary."
    ),
    "build": (
        "## Mode: build\n"
        "You are one session of a long-running build. Build exactly one unit: orient, start it, build it, run its checks, "
        "call book_mark_done, fix what stayed open, call it again, then stop. Never start another unit, never edit "
        "book/outline.json statuses by hand, and never claim done in prose when the gate has not passed. Partial with "
        "the list of what is open is a legal end; record it in progress.md through the gate."
    ),
    "subagent": (
        "## Mode: subagent\n"
        "A parent agent sees only your final message. Return findings, the paths you wrote, evidence ids and locators, "
        "your confidence (observed, derived or inferred) and the gaps; nothing else. No narration of your process, no "
        "raw tool output, no questions back (state the assumption instead), no work beyond the task, no marking units done."
    ),
}


def _validate_name(name: str) -> str:
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValueError(f"prompt name must be a slug such as 'system' or 'roles/worker'; got {name!r}")
    return name


def load(name: str) -> str:
    """Text of ``edgent/prompts/<name>.md`` (``roles/<role>`` for a role), stripped. Only slugs are accepted."""
    path = PROMPTS_DIR / f"{_validate_name(name)}.md"
    if not path.is_file():
        raise FileNotFoundError(f"no prompt file {name}.md under {PROMPTS_DIR}")
    text = path.read_text(encoding="utf-8").strip()
    if len(text) > PROMPT_FILE_MAX_CHARS:
        raise ConfigurationError(f"prompt file {name}.md is {len(text)} chars; the ceiling is {PROMPT_FILE_MAX_CHARS} chars. Move the material into a skill.")
    return text


def roles() -> list[str]:
    """Role names that have a ``roles/<role>.md`` file, sorted."""
    base = PROMPTS_DIR / ROLES_DIRNAME
    if not base.is_dir():
        return []
    return sorted(p.stem for p in base.glob("*.md") if p.is_file() and not p.name.startswith("."))


def role_marker(role: str) -> str:
    if not isinstance(role, str) or not _ROLE_RE.match(role):
        raise ConfigurationError(f"role must be letters, digits, '-' or '_' so providers can route by it; got {role!r}")
    return ROLE_MARKER_FORMAT.format(role=role)


def capability_text(caps: Any) -> str:
    """Stable notes about what this provider cannot do; empty when nothing needs saying. Duck-typed on Capabilities."""
    lines: list[str] = []
    if caps is None:
        return ""
    if not getattr(caps, "images", True):
        lines.append(
            "- This provider cannot show you images: attached frames and screenshots arrive as text stand-ins. Judge "
            "rendered output by its measurements and the described frames, and say when a look was required but not possible."
        )
    ceiling = getattr(caps, "max_prompt_bytes", None)
    if ceiling:
        lines.append(
            f"- This provider refuses prompts over {int(ceiling)} bytes: keep tool outputs small, page through long files "
            "with offset and length, and delegate exploration to subagents."
        )
    if not lines:
        return ""
    return "## Provider\n" + "\n".join(lines)


def build_system_prompt(role: str = "main", *, mode: str = "chat", caps: Any = None, extra: str = "") -> str:
    """The composed system prompt: marker, system.md, behaviour.md, roles/<role>.md if present, mode, provider note, extra."""
    marker = role_marker(role)
    if mode not in MODES:
        raise ConfigurationError(f"mode must be one of {', '.join(MODES)}; got {mode!r}")
    parts = [load("system"), load("behaviour")]
    if role in roles():
        parts.append(load(f"{ROLES_DIRNAME}/{role}"))
    parts.append(MODE_TEXT[mode])
    note = capability_text(caps)
    if note:
        parts.append(note)
    if extra and extra.strip():
        parts.append(extra.strip())
    return marker + "\n" + "\n\n".join(parts)


__all__ = [
    "MAIN_PROMPT_TOKENS_LIMIT",
    "MODES",
    "MODE_TEXT",
    "PROMPTS_DIR",
    "ROLE_MARKER_FORMAT",
    "build_system_prompt",
    "capability_text",
    "load",
    "role_marker",
    "roles",
]
