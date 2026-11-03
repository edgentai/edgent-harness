"""The residual-unknowns ledger: open questions, contradictions and gaps.

An unknown is never deleted. It is resolved with evidence, withdrawn, or
marked out of scope, always with a rationale. The walkthrough at the end of a
build is this ledger rendered.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .utils import canonical_json, now_iso, read_json, sha256_text, write_json

UnknownStatus = Literal["open", "investigating", "blocked", "contradicted", "resolved", "withdrawn", "out_of_scope"]
Severity = Literal["low", "medium", "high", "critical"]

_TRANSITIONS: dict[str, set[str]] = {
    "open": {"investigating", "blocked", "contradicted", "resolved", "withdrawn", "out_of_scope"},
    "investigating": {"open", "blocked", "contradicted", "resolved", "withdrawn", "out_of_scope"},
    "blocked": {"open", "investigating", "withdrawn", "out_of_scope"},
    "contradicted": {"investigating", "resolved", "withdrawn", "out_of_scope"},
    "resolved": {"open"},  # reopening is allowed with a reason
    "withdrawn": {"open"},
    "out_of_scope": {"open"},
}
TERMINAL = {"resolved", "withdrawn", "out_of_scope"}


class UnknownTransitionError(Exception):
    pass


@dataclass
class Unknown:
    question: str
    domain: str  # e.g. "sources", "prerequisites", "verification", "learner"
    scope: str | None = None  # unit id, session id, source id
    status: UnknownStatus = "open"
    severity: Severity = "medium"
    supporting_evidence_ids: list[str] = field(default_factory=list)
    contradicting_evidence_ids: list[str] = field(default_factory=list)
    required_authority: str | None = None
    required_confidence: str = "observed"
    recommended_probes: list[dict[str, str]] = field(default_factory=list)  # {"operation": ..., "rationale": ...}
    relationships: list[dict[str, str]] = field(default_factory=list)  # {"type": "depends_on"|"contradicts"|"related_to", "unknown_id": ...}
    resolution: dict[str, Any] | None = None  # {"disposition": ..., "rationale": ..., "evidence_ids": [...]}
    history: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    revision: int = 1
    unknown_id: str = ""

    def __post_init__(self) -> None:
        if not self.unknown_id:
            self.unknown_id = self.compute_id()

    def compute_id(self) -> str:
        key = {"question": " ".join(self.question.split()).lower(), "domain": self.domain, "scope": self.scope}
        return "unk_" + sha256_text(canonical_json(key))[:32]

    def to_dict(self) -> dict[str, Any]:
        return {
            "unknown_id": self.unknown_id,
            "question": self.question,
            "domain": self.domain,
            "scope": self.scope,
            "status": self.status,
            "severity": self.severity,
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "contradicting_evidence_ids": list(self.contradicting_evidence_ids),
            "required_authority": self.required_authority,
            "required_confidence": self.required_confidence,
            "recommended_probes": list(self.recommended_probes),
            "relationships": list(self.relationships),
            "resolution": self.resolution,
            "history": list(self.history),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Unknown:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class UnknownsLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        data = read_json(self.path, {"version": 1, "unknowns": []}) or {"version": 1, "unknowns": []}
        self._items: dict[str, Unknown] = {}
        for d in data.get("unknowns", []):
            u = Unknown.from_dict(d)
            self._items[u.unknown_id] = u

    def save(self) -> None:
        write_json(self.path, {"version": 1, "unknowns": [u.to_dict() for u in self._items.values()]})

    def record(self, u: Unknown) -> str:
        if u.unknown_id in self._items:
            return "duplicate"
        self._items[u.unknown_id] = u
        self.save()
        return "added"

    def get(self, unknown_id: str) -> Unknown | None:
        return self._items.get(unknown_id)

    def all(self) -> list[Unknown]:
        return list(self._items.values())

    def open(self, scope: str | None = None) -> list[Unknown]:
        return [u for u in self._items.values() if u.status not in TERMINAL and (scope is None or u.scope == scope)]

    def by_scope(self, scope: str) -> list[Unknown]:
        return [u for u in self._items.values() if u.scope == scope]

    def set_status(self, unknown_id: str, status: UnknownStatus, reason: str = "") -> Unknown:
        u = self._require(unknown_id)
        if status != u.status and status not in _TRANSITIONS[u.status]:
            raise UnknownTransitionError(f"{u.status} -> {status} is not allowed for {unknown_id}")
        if status == "resolved" and not (u.resolution and u.resolution.get("evidence_ids")):
            raise UnknownTransitionError("resolving requires a resolution with at least one evidence id; use resolve()")
        self._bump(u, u.status, status, reason)
        u.status = status
        self.save()
        return u

    def resolve(self, unknown_id: str, disposition: Literal["verified", "withdrawn", "out_of_scope"], rationale: str, evidence_ids: list[str] | None = None) -> Unknown:
        u = self._require(unknown_id)
        evidence_ids = list(evidence_ids or [])
        if disposition == "verified" and not evidence_ids:
            raise UnknownTransitionError("a verified resolution needs at least one evidence id")
        if not rationale.strip():
            raise UnknownTransitionError("a resolution needs a rationale")
        u.resolution = {"disposition": disposition, "rationale": rationale, "evidence_ids": evidence_ids}
        new_status: UnknownStatus = {"verified": "resolved", "withdrawn": "withdrawn", "out_of_scope": "out_of_scope"}[disposition]
        if new_status not in _TRANSITIONS[u.status] and new_status != u.status:
            raise UnknownTransitionError(f"{u.status} -> {new_status} is not allowed for {unknown_id}")
        self._bump(u, u.status, new_status, rationale)
        u.status = new_status
        u.supporting_evidence_ids = sorted(set(u.supporting_evidence_ids) | set(evidence_ids))
        self.save()
        return u

    def add_evidence(self, unknown_id: str, evidence_id: str, contradicting: bool = False) -> Unknown:
        u = self._require(unknown_id)
        target = u.contradicting_evidence_ids if contradicting else u.supporting_evidence_ids
        if evidence_id not in target:
            target.append(evidence_id)
            u.updated_at = now_iso()
            u.revision += 1
            if contradicting and u.status in ("open", "investigating"):
                self._bump(u, u.status, "contradicted", f"contradicting evidence {evidence_id}")
                u.status = "contradicted"
            self.save()
        return u

    def render_markdown(self, scope: str | None = None) -> str:
        items = self.by_scope(scope) if scope else self.all()
        if not items:
            return "No unknowns recorded.\n"
        lines = ["# Unknowns", ""]
        for status in ("critical", "high", "medium", "low"):
            group = [u for u in items if u.severity == status and u.status not in TERMINAL]
            if group:
                lines.append(f"## Open, {status} severity")
                for u in group:
                    lines.append(f"- **{u.question}** ({u.domain}{', ' + u.scope if u.scope else ''}; {u.status})")
                    for p in u.recommended_probes:
                        lines.append(f"  - probe: {p.get('operation')}: {p.get('rationale')}")
                lines.append("")
        closed = [u for u in items if u.status in TERMINAL]
        if closed:
            lines.append("## Closed")
            for u in closed:
                res = u.resolution or {}
                lines.append(f"- {u.question}: {u.status} ({res.get('rationale', '')})")
            lines.append("")
        return "\n".join(lines)

    def _require(self, unknown_id: str) -> Unknown:
        u = self._items.get(unknown_id)
        if u is None:
            raise KeyError(unknown_id)
        return u

    @staticmethod
    def _bump(u: Unknown, old: str, new: str, reason: str) -> None:
        u.history.append({"from": old, "to": new, "at": now_iso(), "reason": reason})
        u.updated_at = now_iso()
        u.revision += 1
