"""Evidence records: immutable, content-addressed facts the work rests on.

Each record says what was observed, from where, with what confidence
(observed, derived, inferred) and what authority (the source itself, a
controlled replay such as code executed in the sandbox, a historical
reference, an external service, or a model's inference), with limitations.
The id is a digest of the semantic projection, so the same fact recorded
twice has one id and a conflicting record with the same id is rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .utils import canonical_json, now_iso, read_json, sha256_text, write_json

Confidence = Literal["observed", "derived", "inferred"]
Authority = Literal["source", "controlled_replay", "historical_reference", "external_service", "model_inference"]

AUTHORITY_RANK: dict[str, int] = {
    "model_inference": 1,
    "historical_reference": 2,
    "external_service": 2,
    "controlled_replay": 3,
    "source": 3,
}
CONFIDENCE_RANK: dict[str, int] = {"inferred": 1, "derived": 2, "observed": 3}


def authority_at_least(have: str, need: str) -> bool:
    return AUTHORITY_RANK.get(have, 0) >= AUTHORITY_RANK.get(need, 0)


def confidence_at_least(have: str, need: str) -> bool:
    return CONFIDENCE_RANK.get(have, 0) >= CONFIDENCE_RANK.get(need, 0)


class EvidenceIntegrityError(Exception):
    pass


@dataclass
class Evidence:
    operation: str
    parameters: dict[str, Any]
    result: Any
    confidence: Confidence
    authority: Authority
    limitations: list[str] = field(default_factory=list)
    locations: list[dict[str, Any]] = field(default_factory=list)  # {"kind": "page"|"time"|"line"|"path"|"url"|"range", "ref": str}
    links: list[str] = field(default_factory=list)  # other evidence ids
    subject: dict[str, Any] | None = None  # {"source_id": ..., "sha256": ...}
    provider: str = "edgent"
    created_at: str = field(default_factory=now_iso)
    evidence_id: str = ""

    def __post_init__(self) -> None:
        if not self.evidence_id:
            self.evidence_id = self.compute_id()

    def semantic_projection(self) -> dict[str, Any]:
        """What identity is computed from: provider name and timestamp are excluded."""
        return {
            "operation": self.operation,
            "parameters": self.parameters,
            "result": self.result,
            "confidence": self.confidence,
            "authority": self.authority,
            "limitations": sorted(self.limitations),
            "locations": self.locations,
            "links": sorted(self.links),
            "subject": self.subject,
        }

    def compute_id(self) -> str:
        return "ev_" + sha256_text(canonical_json(self.semantic_projection()))

    def verify_id(self) -> bool:
        return self.evidence_id == self.compute_id()

    def locator(self) -> str:
        """A compact citable locator, e.g. ``textbook:p.12`` or ``lecture:t=00:14:32``."""
        src = (self.subject or {}).get("source_id", "")
        if self.locations:
            loc = self.locations[0]
            kind, ref = loc.get("kind"), loc.get("ref")
            if kind == "page":
                return f"{src}:p.{ref}"
            if kind == "time":
                return f"{src}:t={ref}"
            if kind == "line":
                return f"{src}:L{ref}"
            return f"{src}:{ref}" if src else str(ref)
        return src or self.evidence_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "operation": self.operation,
            "parameters": self.parameters,
            "result": self.result,
            "confidence": self.confidence,
            "authority": self.authority,
            "limitations": list(self.limitations),
            "locations": list(self.locations),
            "links": list(self.links),
            "subject": self.subject,
            "provider": self.provider,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Evidence:
        ev = cls(
            operation=d["operation"],
            parameters=dict(d.get("parameters") or {}),
            result=d.get("result"),
            confidence=d.get("confidence", "inferred"),
            authority=d.get("authority", "model_inference"),
            limitations=list(d.get("limitations") or []),
            locations=list(d.get("locations") or []),
            links=list(d.get("links") or []),
            subject=d.get("subject"),
            provider=d.get("provider", "edgent"),
            created_at=d.get("created_at", now_iso()),
            evidence_id=d.get("evidence_id", ""),
        )
        return ev


class EvidenceStore:
    """One JSON file per record under ``evidence/``. Idempotent puts; conflicts rejected."""

    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, evidence_id: str) -> Path:
        return self.dir / f"{evidence_id}.json"

    def put(self, ev: Evidence) -> str:
        if not ev.verify_id():
            raise EvidenceIntegrityError(f"evidence id {ev.evidence_id} does not match its content")
        p = self._path(ev.evidence_id)
        if p.exists():
            existing = Evidence.from_dict(read_json(p))
            if canonical_json(existing.semantic_projection()) != canonical_json(ev.semantic_projection()):
                raise EvidenceIntegrityError(f"conflicting evidence record for {ev.evidence_id}")
            return "duplicate"
        write_json(p, ev.to_dict())
        return "added"

    def get(self, evidence_id: str) -> Evidence | None:
        p = self._path(evidence_id)
        if not p.exists():
            return None
        return Evidence.from_dict(read_json(p))

    def exists(self, evidence_id: str) -> bool:
        return self._path(evidence_id).exists()

    def all(self) -> list[Evidence]:
        return [Evidence.from_dict(read_json(p)) for p in sorted(self.dir.glob("ev_*.json"))]

    def find(self, operation: str | None = None, source_id: str | None = None, limit: int = 100) -> list[Evidence]:
        out = []
        for ev in self.all():
            if operation and ev.operation != operation:
                continue
            if source_id and (ev.subject or {}).get("source_id") != source_id:
                continue
            out.append(ev)
            if len(out) >= limit:
                break
        return out

    def count(self) -> int:
        return len(list(self.dir.glob("ev_*.json")))
