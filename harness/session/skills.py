"""Progressive-disclosure Markdown skill loader (SKILL.md + flat *.md).

Supports Claude Code / Antigravity ``SKILL.md`` folders (YAML front matter
``name`` and ``description`` plus optional ``references/``) as well as single
``*.md`` skill files. Keeps only a compact index in context by default, or
inlines small skill sets automatically when total size is below ``inline_limit``
(avoiding wasted round-trips on non-caching bridges).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Skill:
    name: str
    description: str
    body: str
    source_path: str = ""
    references: list[str] = field(default_factory=list)


def _parse_skill_markdown(raw: str, fallback_name: str) -> tuple[str, str, str]:
    text = raw.strip()
    name = fallback_name
    description = ""
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            fm = parts[1]
            body = parts[2].strip()
            for line in fm.splitlines():
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                k = k.strip().lower()
                v = v.strip().strip('"').strip("'")
                if k == "name" and v:
                    name = v
                elif k == "description" and v:
                    description = v
    elif text.lower().startswith("description:"):
        first, _, rest = text.partition("\n")
        description = first.split(":", 1)[1].strip()
        body = rest.strip()
    return name, description, body


class SkillRegistry:
    def __init__(self, inline_limit_bytes: int = 8_000) -> None:
        self._skills: dict[str, Skill] = {}
        self.inline_limit_bytes = inline_limit_bytes

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def load_dir(self, root: str) -> int:
        """Discover ``<dir>/<name>/SKILL.md`` and ``<dir>/<name>.md`` files."""
        if not root or not os.path.isdir(root):
            return 0
        count = 0
        for entry in sorted(os.listdir(root)):
            full = os.path.join(root, entry)
            if os.path.isdir(full):
                sfile = os.path.join(full, "SKILL.md")
                if os.path.isfile(sfile):
                    with open(sfile, "r", encoding="utf-8", errors="replace") as fh:
                        raw = fh.read()
                    name, desc, body = _parse_skill_markdown(raw, entry)
                    refs_dir = os.path.join(full, "references")
                    refs = sorted(os.listdir(refs_dir)) if os.path.isdir(refs_dir) else []
                    self.register(Skill(name=name, description=desc, body=body, source_path=sfile, references=refs))
                    count += 1
            elif entry.endswith(".md") and entry.lower() != "readme.md":
                with open(full, "r", encoding="utf-8", errors="replace") as fh:
                    raw = fh.read()
                stem = os.path.splitext(entry)[0]
                name, desc, body = _parse_skill_markdown(raw, stem)
                self.register(Skill(name=name, description=desc, body=body, source_path=full))
                count += 1
        return count

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def names(self) -> list[str]:
        return sorted(self._skills)

    def all(self) -> list[Skill]:
        return [self._skills[n] for n in self.names()]

    def total_bytes(self) -> int:
        return sum(len(s.body.encode("utf-8")) for s in self._skills.values())

    def render_for_prompt(self, loaded: Optional[list[str]] = None) -> str:
        if not self._skills:
            return ""
        loaded_set = set(loaded or [])
        auto_inline = self.total_bytes() <= self.inline_limit_bytes
        if auto_inline:
            loaded_set |= set(self._skills)
        blocks: list[str] = []
        unloaded = [self._skills[n] for n in self.names() if n not in loaded_set]
        if unloaded:
            lines = ["## Skills available (call load_skill when relevant)"]
            for s in unloaded:
                lines.append(f"- `{s.name}`: {s.description or '(no description)'}")
            blocks.append("\n".join(lines))
        for n in sorted(loaded_set):
            s = self._skills.get(n)
            if s:
                blocks.append(f"## Skill: {s.name}\n{s.body}")
        return "\n\n".join(blocks)
