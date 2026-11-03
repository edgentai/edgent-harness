"""The outline: every unit of a build with a status that only code can move.

``outline.json`` is the feature list of the long-running-agent protocol. The
model proposes units; statuses move forward only through the done-gate; units
are never removed (a unit that should not be built is ``blocked`` with a
reason). Entries carry their dependencies so independent units can run in
parallel and a unit never starts before what it needs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .utils import now_iso, read_json, slugify, write_json

UnitStatus = Literal["pending", "in_progress", "partial", "done", "reviewed", "blocked"]
UnitKind = Literal["chapter", "section", "figure", "film", "widget", "exercise_set", "research", "integration", "assessment", "lesson", "custom"]

TRANSITIONS: dict[str, set[str]] = {
    "pending": {"in_progress", "blocked"},
    "in_progress": {"partial", "done", "blocked", "pending"},
    "partial": {"in_progress", "done", "blocked"},
    "done": {"reviewed", "in_progress"},
    "reviewed": {"in_progress"},
    "blocked": {"pending", "in_progress"},
}
FINISHED = {"done", "reviewed"}


class OutlineError(Exception):
    pass


@dataclass
class Unit:
    id: str
    kind: UnitKind
    title: str
    objective: str = ""
    status: UnitStatus = "pending"
    depends_on: list[str] = field(default_factory=list)
    path: str | None = None  # target file, workspace-relative
    checks: list[str] = field(default_factory=list)  # closer names required beyond the kind's defaults
    attempts: int = 0
    open_obligations: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)  # brief, persona notes, estimated_words, film brief, ...

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "objective": self.objective,
            "status": self.status,
            "depends_on": list(self.depends_on),
            "path": self.path,
            "checks": list(self.checks),
            "attempts": self.attempts,
            "open_obligations": list(self.open_obligations),
            "history": list(self.history),
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Unit:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def finished(self) -> bool:
        return self.status in FINISHED


class Outline:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        data = read_json(self.path, None) or {"version": 1, "title": "", "persona": "", "units": []}
        self.title: str = data.get("title", "")
        self.persona: str = data.get("persona", "")
        self.meta: dict[str, Any] = dict(data.get("meta") or {})
        self._units: dict[str, Unit] = {}
        for d in data.get("units", []):
            u = Unit.from_dict(d)
            self._units[u.id] = u

    # ------------------------------------------------------------------ persistence
    def save(self) -> None:
        write_json(self.path, {"version": 1, "title": self.title, "persona": self.persona, "meta": self.meta, "units": [u.to_dict() for u in self._units.values()]})

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "persona": self.persona, "meta": self.meta, "units": [u.to_dict() for u in self._units.values()]}

    # ------------------------------------------------------------------ units
    @property
    def units(self) -> list[Unit]:
        return list(self._units.values())

    def get(self, unit_id: str) -> Unit:
        u = self._units.get(unit_id)
        if u is None:
            raise OutlineError(f"unknown unit {unit_id!r}")
        return u

    def has(self, unit_id: str) -> bool:
        return unit_id in self._units

    def add(self, unit: Unit) -> Unit:
        if unit.id in self._units:
            raise OutlineError(f"unit {unit.id!r} already exists; units are never replaced, update them instead")
        for dep in unit.depends_on:
            if dep not in self._units and dep != unit.id:
                raise OutlineError(f"unit {unit.id!r} depends on unknown unit {dep!r}")
        self._units[unit.id] = unit
        self.save()
        return unit

    def new_unit(self, kind: UnitKind, title: str, objective: str = "", depends_on: list[str] | None = None, path: str | None = None, **meta: Any) -> Unit:
        base = f"{kind}-{slugify(title)}"
        uid, n = base, 2
        while uid in self._units:
            uid = f"{base}-{n}"
            n += 1
        return self.add(Unit(id=uid, kind=kind, title=title, objective=objective, depends_on=list(depends_on or []), path=path, meta=meta))

    def update(self, unit_id: str, **fields: Any) -> Unit:
        """Edit descriptive fields. Status is not editable here; use set_status."""
        u = self.get(unit_id)
        for k, v in fields.items():
            if k in ("status", "id", "history"):
                raise OutlineError(f"field {k!r} cannot be edited directly")
            if k == "depends_on":
                for dep in v:
                    if dep not in self._units:
                        raise OutlineError(f"unknown dependency {dep!r}")
            if k not in u.__dataclass_fields__:
                u.meta[k] = v
            else:
                setattr(u, k, v)
        self.save()
        return u

    def set_status(self, unit_id: str, status: UnitStatus, reason: str = "", open_obligations: list[dict[str, Any]] | None = None) -> Unit:
        u = self.get(unit_id)
        if status != u.status and status not in TRANSITIONS[u.status]:
            raise OutlineError(f"{u.status} -> {status} is not allowed for {unit_id}")
        if status == "in_progress" and u.status != "in_progress":
            missing = [d for d in u.depends_on if not self._units[d].finished]
            if missing:
                raise OutlineError(f"{unit_id} cannot start: dependencies not finished: {missing}")
            u.attempts += 1
        u.history.append({"from": u.status, "to": status, "at": now_iso(), "reason": reason})
        u.status = status
        if open_obligations is not None:
            u.open_obligations = list(open_obligations)
        if status in FINISHED:
            u.open_obligations = []
        self.save()
        return u

    # ------------------------------------------------------------------ queries
    def ready(self) -> list[Unit]:
        """Pending or partial units whose dependencies are all finished, in outline order."""
        return [u for u in self._units.values() if u.status in ("pending", "partial") and all(self._units[d].finished for d in u.depends_on if d in self._units)]

    def by_status(self, *statuses: str) -> list[Unit]:
        return [u for u in self._units.values() if u.status in statuses]

    def dependants(self, unit_id: str) -> list[Unit]:
        return [u for u in self._units.values() if unit_id in u.depends_on]

    def validate(self) -> list[str]:
        problems: list[str] = []
        for u in self._units.values():
            for d in u.depends_on:
                if d not in self._units:
                    problems.append(f"{u.id}: unknown dependency {d}")
        # cycle detection
        state: dict[str, int] = {}

        def visit(uid: str, stack: list[str]) -> None:
            if state.get(uid) == 1:
                problems.append("cycle: " + " -> ".join(stack + [uid]))
                return
            if state.get(uid) == 2:
                return
            state[uid] = 1
            for d in self._units[uid].depends_on:
                if d in self._units:
                    visit(d, stack + [uid])
            state[uid] = 2

        for uid in self._units:
            visit(uid, [])
        return problems

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for u in self._units.values():
            counts[u.status] = counts.get(u.status, 0) + 1
        return {"title": self.title, "units": len(self._units), "by_status": counts, "ready": [u.id for u in self.ready()][:10]}

    def render_markdown(self, max_units: int = 200) -> str:
        lines = [f"# {self.title or 'Outline'}", ""]
        if self.persona:
            lines += [f"Persona: {self.persona}", ""]
        marks = {"pending": " ", "in_progress": "~", "partial": "!", "done": "x", "reviewed": "X", "blocked": "-"}
        for u in list(self._units.values())[:max_units]:
            dep = f" (after {', '.join(u.depends_on)})" if u.depends_on else ""
            lines.append(f"- [{marks.get(u.status, ' ')}] `{u.id}` {u.kind}: {u.title}{dep}")
        if len(self._units) > max_units:
            lines.append(f"- … {len(self._units) - max_units} more")
        return "\n".join(lines) + "\n"
