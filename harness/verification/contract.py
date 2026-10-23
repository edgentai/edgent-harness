"""Definition-of-done contract (§11.4): machine-checkable, written before work starts."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.types import Check, ProblemType


@dataclass
class DefinitionOfDone:
    required_outputs: list[dict[str, str]] = field(default_factory=lambda: [{"name": "answer", "format": "text"}])
    required_evidence: str = "every factual claim about the inputs cites an artifact locator"
    checks: list[str] = field(default_factory=lambda: ["schema", "evidence", "numbers_grounded", "policy"])
    acceptable_failures: list[str] = field(default_factory=list)
    min_claim_verification_rate: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DefinitionOfDone":
        if not isinstance(d, dict):
            return cls()
        out = cls()
        if isinstance(d.get("required_outputs"), list) and d["required_outputs"]:
            out.required_outputs = [x for x in d["required_outputs"] if isinstance(x, dict) and "name" in x]
        if isinstance(d.get("required_evidence"), str):
            out.required_evidence = d["required_evidence"]
        if isinstance(d.get("checks"), list):
            out.checks = [str(c) for c in d["checks"]] or out.checks
        if isinstance(d.get("acceptable_failures"), list):
            out.acceptable_failures = [str(c) for c in d["acceptable_failures"]]
        if isinstance(d.get("min_claim_verification_rate"), (int, float)):
            out.min_claim_verification_rate = float(d["min_claim_verification_rate"])
        return out

    @classmethod
    def default_for(cls, problem_type: ProblemType, has_artifacts: bool, has_code: bool = False) -> "DefinitionOfDone":
        checks = ["schema", "policy"]
        if has_artifacts:
            checks += ["evidence", "numbers_grounded"]
        if problem_type == ProblemType.CONVERGENT:
            checks.append("consistency")
        if has_code:
            checks.append("tests")
        dod = cls(checks=checks)
        if not has_artifacts:
            dod.required_evidence = "claims that depend on inputs cite evidence; general knowledge is marked as such"
            dod.min_claim_verification_rate = 0.0
        if problem_type in (ProblemType.DIVERGENT, ProblemType.EXPLORATORY):
            dod.min_claim_verification_rate = 0.6
        return dod

    def check_outputs(self, answer: Any) -> list[Check]:
        checks = []
        for spec in self.required_outputs:
            name, fmt = spec.get("name", "answer"), spec.get("format", "text")
            value = answer.get(name) if isinstance(answer, dict) and name in answer else answer
            ok, detail = _format_ok(value, fmt)
            checks.append(Check(f"contract:{name}", ok, detail, "contract"))
        return checks


def _format_ok(value: Any, fmt: str) -> tuple[bool, str]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return False, "missing or empty"
    if fmt == "json":
        if isinstance(value, (dict, list)):
            return True, "json object"
        try:
            json.loads(str(value))
            return True, "json parses"
        except Exception:
            return False, "not valid JSON"
    if fmt == "number":
        try:
            float(str(value).replace(",", ""))
            return True, "numeric"
        except ValueError:
            return False, "not numeric"
    if fmt == "file":
        return isinstance(value, str) and len(value) > 0, "file path present" if value else "no file path"
    return True, f"{fmt} present"
