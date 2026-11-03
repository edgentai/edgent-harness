"""The evaluation gate, adapted from REA's agent evaluation.

A scenario passes only if: the turn ended normally; the correct first tool
was chosen; no identical call was repeated; no tool call failed schema
validation; the required tool subsequence appeared in order; forbidden tools
were not used; the answer satisfies its text constraints; fixture claims are
supported by evidence of the required confidence and authority; and, where
no oracle exists, an epistemic cue ("unverified", "could not verify") is
present when required. Everything is computed from the trace, the final
text and the evidence store, never from the model's self-report.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..domain.evidence import EvidenceStore, authority_at_least, confidence_at_least
from ..domain.utils import now_iso, read_json
from ..observability.tracing import TraceEvent

EPISTEMIC_CUE_RE = re.compile(r"\b(unverified|could not verify|couldn't verify|not verified|cannot confirm|unknown|not (?:yet )?checked|no evidence)\b", re.I)


@dataclass
class FixtureClaim:
    id: str
    expected: Any
    evidence_operation: str
    required_authority: str = "source"
    required_confidence: str = "observed"
    select: str = "result"  # dotted path into the evidence record, e.g. "result.quote"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FixtureClaim:
        return cls(d["id"], d.get("expected"), d["evidence_operation"], d.get("required_authority", "source"), d.get("required_confidence", "observed"), d.get("select", "result"))


@dataclass
class Scenario:
    id: str
    task: str
    title: str = ""
    setup_files: dict[str, str] = field(default_factory=dict)  # workspace-relative path -> content
    scripted_replies: list[dict[str, Any]] = field(default_factory=list)  # fixture environment for the mock provider
    expected_first_tool: str | None = None
    required_subsequence: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    max_tool_calls: int = 50
    max_iterations: int = 30
    answer_must_contain: list[str] = field(default_factory=list)
    answer_must_not_contain: list[str] = field(default_factory=list)
    epistemic_cue_required: bool = False
    claims: list[FixtureClaim] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Scenario:
        return cls(
            id=d["id"],
            task=d["task"],
            title=d.get("title", d["id"]),
            setup_files=dict(d.get("setup_files") or {}),
            scripted_replies=list(d.get("scripted_replies") or []),
            expected_first_tool=d.get("expected_first_tool"),
            required_subsequence=list(d.get("required_subsequence") or []),
            forbidden_tools=list(d.get("forbidden_tools") or []),
            max_tool_calls=int(d.get("max_tool_calls", 50)),
            max_iterations=int(d.get("max_iterations", 30)),
            answer_must_contain=list(d.get("answer_must_contain") or []),
            answer_must_not_contain=list(d.get("answer_must_not_contain") or []),
            epistemic_cue_required=bool(d.get("epistemic_cue_required", False)),
            claims=[FixtureClaim.from_dict(c) for c in d.get("claims") or []],
            tags=list(d.get("tags") or []),
        )


@dataclass
class GateResult:
    scenario_id: str
    passed: bool
    checks: list[dict[str, Any]]
    tool_calls: int
    repeated_calls: int
    validation_failures: int
    prompt_tokens: int
    completion_tokens: int
    stop_reason: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def failed_checks(self) -> list[str]:
        return [c["name"] for c in self.checks if not c["passed"]]


def _dig(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list) and part.isdigit():
            cur = cur[int(part)] if int(part) < len(cur) else None
        else:
            return None
    return cur


def _subsequence(needles: list[str], haystack: list[str]) -> bool:
    it = iter(haystack)
    return all(any(n == h for h in it) for n in needles)


def evaluate(scenario: Scenario, events: list[TraceEvent], final_text: str, evidence: EvidenceStore | None, stop_reason: str = "done") -> GateResult:
    tool_events = [e for e in events if e.kind == "tool_call" and not e.data.get("replayed")]
    names = [e.data.get("name", "?") for e in tool_events]
    # The order the main agent chose its tools in: a subagent's calls finish (and are traced) before the parent's
    # `agent` call does, so the first and the required sequence are read from the main agent's model replies.
    main_names = [str(n) for e in events if e.kind == "model_call" and not e.node_id for n in (e.data.get("tool_calls") or [])]
    if not main_names:
        main_names = list(names)
    seen: set[str] = set()
    repeated = 0
    for e in tool_events:
        key = e.data.get("name", "?") + json.dumps(e.data.get("arguments", {}), sort_keys=True, default=str)
        if key in seen:
            repeated += 1
        seen.add(key)
    validation_failures = sum(1 for e in tool_events if (e.data.get("error") or {}).get("code") == "invalid_input")
    prompt_tokens = sum(int((e.data.get("usage") or {}).get("prompt_tokens", 0)) for e in events if e.kind == "model_call")
    completion_tokens = sum(int((e.data.get("usage") or {}).get("completion_tokens", 0)) for e in events if e.kind == "model_call")

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("exit_ok", stop_reason == "done", f"stop_reason={stop_reason}")
    if scenario.expected_first_tool is not None:
        check("correct_first_tool", bool(main_names) and main_names[0] == scenario.expected_first_tool, f"first={main_names[0] if main_names else None}, expected={scenario.expected_first_tool}")
    check("no_repeated_calls", repeated == 0, f"repeated={repeated}")
    check("no_validation_failures", validation_failures == 0, f"invalid_input={validation_failures}")
    if scenario.required_subsequence:
        check("required_subsequence", _subsequence(scenario.required_subsequence, main_names), f"calls={main_names}")
    if scenario.forbidden_tools:
        used = sorted(set(names) & set(scenario.forbidden_tools))
        check("no_forbidden_tools", not used, f"used={used}")
    check("within_tool_budget", len(names) <= scenario.max_tool_calls, f"calls={len(names)} max={scenario.max_tool_calls}")
    for s in scenario.answer_must_contain:
        check(f"answer_contains:{s}", s.lower() in (final_text or "").lower())
    for s in scenario.answer_must_not_contain:
        check(f"answer_omits:{s}", s.lower() not in (final_text or "").lower())
    if scenario.epistemic_cue_required:
        check("epistemic_cue", bool(EPISTEMIC_CUE_RE.search(final_text or "")))
    for claim in scenario.claims:
        ok = False
        detail = "no evidence store"
        if evidence is not None:
            candidates = evidence.find(operation=claim.evidence_operation, limit=500)
            matches = [ev for ev in candidates if _dig(ev.to_dict(), claim.select) == claim.expected]
            if not matches:
                detail = f"no {claim.evidence_operation} evidence with {claim.select}=={claim.expected!r} among {len(candidates)}"
            else:
                strong = [ev for ev in matches if authority_at_least(ev.authority, claim.required_authority) and confidence_at_least(ev.confidence, claim.required_confidence)]
                ok = bool(strong)
                detail = "supported" if ok else f"evidence found but authority/confidence too weak ({matches[0].authority}/{matches[0].confidence} < {claim.required_authority}/{claim.required_confidence})"
        check(f"claim:{claim.id}", ok, detail)
    passed = all(c["passed"] for c in checks)
    return GateResult(scenario.id, passed, checks, len(names), repeated, validation_failures, prompt_tokens, completion_tokens, stop_reason)


def load_scenarios(directory: str | Path) -> list[Scenario]:
    out = []
    for p in sorted(Path(directory).glob("*.json")):
        data = read_json(p)
        if isinstance(data, list):
            out.extend(Scenario.from_dict(d) for d in data)
        elif isinstance(data, dict):
            out.append(Scenario.from_dict(data))
    return out


def record_results(results: list[GateResult], path: str | Path, label: str = "") -> str:
    """Append one row per run to a Markdown results table (created with a header when absent)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    passed = sum(1 for r in results if r.passed)
    tokens = sum(r.prompt_tokens + r.completion_tokens for r in results)
    calls = sum(r.tool_calls for r in results)
    failed = ", ".join(f"{r.scenario_id}({'; '.join(r.failed_checks())})" for r in results if not r.passed) or "none"
    row = f"| {now_iso()} | {label} | {passed}/{len(results)} | {tokens} | {calls} | {failed} |\n"
    if not p.exists():
        p.write_text("# Eval results\n\n| when | label | passed | tokens | tool calls | failed |\n|---|---|---|---|---|---|\n" + row, encoding="utf-8")
    else:
        with open(p, "a", encoding="utf-8") as f:
            f.write(row)
    return row


RunnerFn = Callable[[Scenario], tuple[list[TraceEvent], str, EvidenceStore | None, str]]


def run_all(scenarios: list[Scenario], runner: RunnerFn) -> list[GateResult]:
    """Run each scenario through a runner that returns (events, final_text, evidence_store, stop_reason)."""
    results = []
    for sc in scenarios:
        events, text, store, stop = runner(sc)
        results.append(evaluate(sc, events, text, store, stop))
    return results
