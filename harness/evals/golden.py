"""Golden-set runner. Treat it like CI: run on every change to prompts, tools, settings or code."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..cognition.sampling import normalise_answer
from ..core.config import EffortLevel
from ..core.utils import now_iso


@dataclass
class GoldenCase:
    id: str
    task: str
    files: list[str] = field(default_factory=list)
    inline_texts: list[str] = field(default_factory=list)
    expected: Optional[str] = None  # exact/normalised answer
    expected_regex: Optional[str] = None
    expected_status: Optional[str] = None  # verified | partially_verified | failed (e.g. adversarial cases expect failed/partial)
    must_cite: bool = True
    modality: str = "text"
    task_type: str = "convergent"
    effort: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GoldenCase":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def check(self, result: Any) -> tuple[bool, str]:
        answer = result.answer if isinstance(result.answer, str) else json.dumps(result.answer, default=str)
        if self.expected_status and result.status != self.expected_status:
            return False, f"status {result.status} != {self.expected_status}"
        if self.expected is not None:
            if normalise_answer(answer) != normalise_answer(self.expected) and self.expected.lower() not in (answer or "").lower():
                return False, f"answer mismatch: {answer[:80]!r}"
        if self.expected_regex and not re.search(self.expected_regex, answer or "", re.I | re.S):
            return False, "regex not matched"
        if self.must_cite and result.status != "failed" and not result.claims:
            return False, "no citations"
        return True, "ok"


@dataclass
class GoldenSet:
    name: str
    cases: list[GoldenCase]

    @classmethod
    def load(cls, path: str) -> "GoldenSet":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        base = os.path.dirname(os.path.abspath(path))
        cases = []
        for c in data["cases"]:
            gc = GoldenCase.from_dict(c)
            gc.files = [p if os.path.isabs(p) else os.path.join(base, p) for p in gc.files]
            cases.append(gc)
        return cls(data.get("name", os.path.basename(path)), cases)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"name": self.name, "cases": [c.to_dict() for c in self.cases]}, f, indent=2)


@dataclass
class EvalResult:
    case_id: str
    passed: bool
    detail: str
    status: str
    citations_valid: Optional[float]
    cost_tokens: int
    latency_s: float
    effort: str
    honest_failure: bool

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class EvalRunner:
    def __init__(self, run_fn: Callable[..., Any], results_path: Optional[str] = None) -> None:
        self.run_fn = run_fn  # harness.run signature
        self.results_path = results_path

    def run_case(self, case: GoldenCase) -> EvalResult:
        t0 = time.monotonic()
        res = self.run_fn(case.task, files=case.files, effort=EffortLevel(case.effort) if case.effort else None, inline_texts=case.inline_texts)
        passed, detail = case.check(res)
        ev = res.evidence or []
        cit = (sum(1 for e in ev if e.get("resolved")) / len(ev)) if ev else None
        tokens = int((res.cost.get("total") or {}).get("prompt_tokens", 0)) + int((res.cost.get("total") or {}).get("completion_tokens", 0))
        honest = res.status in ("failed", "partially_verified") and bool(res.gaps)
        return EvalResult(case.id, passed, detail, res.status, cit, tokens, time.monotonic() - t0, res.effort, honest)

    def run(self, golden: GoldenSet, label: str = "") -> dict[str, Any]:
        results = [self.run_case(c) for c in golden.cases]
        n = len(results) or 1
        summary = {
            "label": label or golden.name,
            "ts": now_iso(),
            "cases": len(results),
            "accuracy": sum(1 for r in results if r.passed) / n,
            "citation_validity": _mean([r.citations_valid for r in results if r.citations_valid is not None]),
            "verified_rate": sum(1 for r in results if r.status == "verified") / n,
            "honest_failure_rate": sum(1 for r in results if r.honest_failure) / n,
            "mean_tokens": sum(r.cost_tokens for r in results) / n,
            "mean_latency_s": sum(r.latency_s for r in results) / n,
            "results": [r.to_dict() for r in results],
        }
        if self.results_path:
            append_results_row(self.results_path, summary)
        return summary


def _mean(xs: list[float]) -> Optional[float]:
    return (sum(xs) / len(xs)) if xs else None


def append_results_row(path: str, summary: dict[str, Any]) -> None:
    """Results table in the repo: one row per milestone/ablation (the article's before/after habit)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write("| ts | label | cases | accuracy | citation validity | verified rate | honest failure | mean tokens | mean latency s |\n|---|---|---|---|---|---|---|---|---|\n")
        cv = summary.get("citation_validity")
        f.write(f"| {summary['ts']} | {summary['label']} | {summary['cases']} | {summary['accuracy']:.2f} | {('%.2f' % cv) if cv is not None else 'n/a'} | "
                f"{summary['verified_rate']:.2f} | {summary['honest_failure_rate']:.2f} | {summary['mean_tokens']:.0f} | {summary['mean_latency_s']:.2f} |\n")
