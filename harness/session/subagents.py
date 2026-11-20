"""Isolated fresh-context subagent definitions (Markdown front matter or code).

General-purpose built-in subagents (modeled on Claude Code / Antigravity):
- ``explore``: fast read-only search/glob/read over workspace; returns concise summary.
- ``coder``: focused implementation/editing worker with files, bash, and tests.
- ``reviewer``: independent fresh-context critic/auditor with read-only + verification tools.
- ``planner``: read-only design and planning subagent.
- ``researcher``: web + workspace research subagent returning sourced briefs.

Any user or project can drop custom ``*.md`` subagents into ``~/.edgent/agents/``
or ``<workspace>/.edgent/agents/`` to extend or override them for any domain.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SubagentSpec:
    name: str
    description: str
    system_prompt: str
    tools: list[str] = field(default_factory=list)  # empty = inherit parent allowlist
    model_role: str = ""  # stage/role key for ModelRouter (defaults to `name`)
    max_turns: int = 16
    effort: Optional[str] = None
    skills: list[str] = field(default_factory=list)
    require_tool_use: bool = False


BUILTIN_SUBAGENTS: dict[str, SubagentSpec] = {
    "explore": SubagentSpec(
        name="explore",
        description="Fast read-only workspace explorer: locate files, grep patterns, inspect code or documents, and return a concise factual summary with file paths and line numbers.",
        system_prompt=(
            "You are a fast, read-only exploration subagent. Search the workspace with glob_files, search_files, "
            "and read_file, then return a concise, structured summary citing exact file paths and line numbers "
            "that you directly observed in tool results. Never modify files or guess uninspected contents."
        ),
        tools=["read_file", "read_numbered", "list_files", "glob_files", "search_files", "fetch_artifact_view", "list_artifacts", "count_matches"],
        max_turns=14,
        require_tool_use=True,
    ),
    "coder": SubagentSpec(
        name="coder",
        description="Focused implementation and editing worker: read, write, surgically edit files, run bash/python/tests, and verify changes before returning a concise summary.",
        system_prompt=(
            "You are a focused worker subagent. Read the relevant files, carry out the assigned edits or creation "
            "cleanly using edit_file or write_file, verify your work by running tests or commands with bash / run_python, "
            "and return a brief factual summary of what changed and what was verified."
        ),
        tools=["read_file", "read_numbered", "write_file", "edit_file", "delete_file", "list_files", "glob_files", "search_files", "bash", "run_command", "run_python", "run_tests", "lint_code", "calculate", "load_skill"],
        max_turns=20,
    ),
    "reviewer": SubagentSpec(
        name="reviewer",
        description="Independent fresh-context reviewer/critic. Audits work against requirements, source files, and executable checks with zero author history; returns a concrete defect list.",
        system_prompt=(
            "You are an independent reviewer with NO shared history from whoever produced the work. "
            "You MUST read every target file and reference file yourself using read_file (with start_line/end_line "
            "for large files) and run real verification checks via bash / run_python / run_tests BEFORE writing "
            "your final review. Prefer read_file over bash cat when reading files. "
            "NEVER fabricate, simulate, or infer file contents, line numbers, or code outputs that were not returned "
            "in an actual tool result in this session. Never edit files. "
            "Return a crisp, prioritized defect list citing only verified observations, plus what you verified as correct."
        ),
        tools=["read_file", "read_numbered", "list_files", "glob_files", "search_files", "bash", "run_command", "run_python", "run_tests", "lint_code", "calculate"],
        max_turns=20,
        require_tool_use=True,
    ),
    "planner": SubagentSpec(
        name="planner",
        description="Read-only architect/planner: inspect the workspace and requirements, identify risks, and propose a concrete step-by-step plan.",
        system_prompt=(
            "You are a read-only planning subagent. Inspect the workspace thoroughly and return a concrete, verifiable "
            "step-by-step plan listing exact target files, tradeoffs, and verification steps. Never modify files."
        ),
        tools=["read_file", "read_numbered", "list_files", "glob_files", "search_files", "list_artifacts", "fetch_artifact_view"],
        max_turns=10,
    ),
    "researcher": SubagentSpec(
        name="researcher",
        description="Research subagent using workspace search and web tools (web_search, web_fetch) to return a concise sourced brief.",
        system_prompt=(
            "You are a research subagent. Gather facts using workspace tools and web_search / web_fetch, distinguish "
            "verified observations from inferences, cite every URL or file path, and return a concise brief."
        ),
        tools=["read_file", "list_files", "glob_files", "search_files", "web_search", "web_fetch", "fetch_artifact_view"],
        max_turns=12,
    ),
}


def _parse_agent_markdown(raw: str, fallback_name: str) -> SubagentSpec:
    text = raw.strip()
    meta: dict[str, str] = {}
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            body = parts[2].strip()
            for line in parts[1].splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip().lower()] = v.strip().strip('"').strip("'")
    tools = [t.strip() for t in meta.get("tools", "").split(",") if t.strip()]
    skills = [s.strip() for s in meta.get("skills", "").split(",") if s.strip()]
    return SubagentSpec(
        name=meta.get("name") or fallback_name,
        description=meta.get("description") or "",
        system_prompt=body,
        tools=tools,
        model_role=meta.get("model_role") or meta.get("model") or meta.get("name") or fallback_name,
        max_turns=int(meta.get("max_turns") or meta.get("maxturns") or 16),
        effort=meta.get("effort") or None,
        skills=skills,
    )


class SubagentRegistry:
    def __init__(self, include_builtins: bool = True) -> None:
        self._specs: dict[str, SubagentSpec] = dict(BUILTIN_SUBAGENTS) if include_builtins else {}

    def register(self, spec: SubagentSpec) -> None:
        self._specs[spec.name] = spec

    def load_dir(self, root: str) -> int:
        if not root or not os.path.isdir(root):
            return 0
        count = 0
        for fn in sorted(os.listdir(root)):
            if fn.endswith(".md") and fn.lower() != "readme.md":
                with open(os.path.join(root, fn), "r", encoding="utf-8", errors="replace") as fh:
                    raw = fh.read()
                spec = _parse_agent_markdown(raw, os.path.splitext(fn)[0])
                self.register(spec)
                count += 1
        return count

    def get(self, name: str) -> Optional[SubagentSpec]:
        return self._specs.get(name)

    def names(self) -> list[str]:
        return sorted(self._specs)

    def all(self) -> list[SubagentSpec]:
        return [self._specs[n] for n in self.names()]

    def render_index(self) -> str:
        if not self._specs:
            return ""
        lines = ["## Subagents available (call spawn_subagent with isolated fresh context)"]
        for n in self.names():
            s = self._specs[n]
            lines.append(f"- `{s.name}`: {s.description}")
        return "\n".join(lines)
