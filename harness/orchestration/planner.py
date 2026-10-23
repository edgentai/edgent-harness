"""Planner (§8.1, §11.6): structured plan with dependency edges, validated before running."""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from ..core.client import ModelClient, ModelRequest
from ..core.types import Message, Plan, PlanStep
from ..core.utils import extract_json, truncate

VALID_ROLES = {"intake", "executor", "verifier", "synthesiser"}


def validate_plan(plan: Plan, tools: set[str], artifacts: set[str], verifiers: set[str], max_steps: int = 24) -> list[str]:
    errors: list[str] = []
    if not plan.steps:
        errors.append("plan has no steps")
        return errors
    if len(plan.steps) > max_steps:
        errors.append(f"too many steps ({len(plan.steps)} > {max_steps})")
    ids = [s.id for s in plan.steps]
    if len(set(ids)) != len(ids):
        errors.append("duplicate step ids")
    idset = set(ids)
    for s in plan.steps:
        if not re.fullmatch(r"[A-Za-z0-9_\-\.]{1,40}", s.id):
            errors.append(f"bad step id {s.id!r}")
        if s.role not in VALID_ROLES:
            errors.append(f"step {s.id}: unknown role {s.role!r}")
        for d in s.depends_on:
            if d not in idset:
                errors.append(f"step {s.id}: depends on unknown step {d!r}")
        for t in s.tools:
            if t not in tools:
                errors.append(f"step {s.id}: unknown tool {t!r}")
        for a in s.artifacts:
            if not any(x == a or x.startswith(a) for x in artifacts):
                errors.append(f"step {s.id}: unknown artifact {a[:12]!r}")
        if s.verifier and s.verifier not in verifiers:
            errors.append(f"step {s.id}: unknown verifier {s.verifier!r}")
    # cycle detection (Kahn)
    indeg = {i: 0 for i in idset}
    for s in plan.steps:
        for d in s.depends_on:
            if d in indeg:
                indeg[s.id] += 1
    queue = [i for i, n in indeg.items() if n == 0]
    seen = 0
    while queue:
        cur = queue.pop()
        seen += 1
        for s in plan.steps:
            if cur in s.depends_on:
                indeg[s.id] -= 1
                if indeg[s.id] == 0:
                    queue.append(s.id)
    if seen != len(idset):
        errors.append("plan contains a dependency cycle")
    return errors


def _coerce_plan(data: Any, goal: str) -> Optional[Plan]:
    if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
        return None
    steps = []
    for i, s in enumerate(data["steps"]):
        if not isinstance(s, dict):
            continue
        steps.append(PlanStep(
            id=str(s.get("id") or f"s{i + 1}"),
            description=str(s.get("description") or s.get("task") or "")[:1000],
            role=str(s.get("role") or "executor"),
            depends_on=[str(d) for d in (s.get("depends_on") or []) if d],
            tools=[str(t) for t in (s.get("tools") or [])],
            artifacts=[str(a) for a in (s.get("artifacts") or [])],
            verifier=(str(s["verifier"]) if s.get("verifier") else None),
            expected_output=str(s.get("expected_output") or "text"),
        ))
    return Plan(goal=str(data.get("goal") or goal)[:500], steps=steps, rationale=str(data.get("rationale") or "")[:1000])


class Planner:
    def __init__(self, client: ModelClient, system_prompt: Any, tools: set[str], verifiers: set[str], max_steps: int = 24) -> None:
        self.client = client
        self.system_prompt = system_prompt
        self.tools = tools
        self.verifiers = verifiers
        self.max_steps = max_steps

    def _context(self, task: str, index_text: str, artifact_ids: list[str], contract: dict[str, Any], lessons: list[str], extra: str = "") -> str:
        return (
            f"Goal: {task}\n\nDefinition of done: {json.dumps(contract)}\n\nAvailable tools: {sorted(self.tools)}\n"
            f"Available verifiers: {sorted(self.verifiers)}\nArtifacts (use full ids): {artifact_ids}\n\n{truncate(index_text, 6000)}\n"
            + (f"\nLessons from earlier attempts:\n- " + "\n- ".join(lessons) if lessons else "")
            + (f"\n{extra}" if extra else "")
            + "\n\nProduce the plan JSON."
        )

    def plan(self, task: str, index_text: str, artifact_ids: list[str], contract: dict[str, Any], lessons: Optional[list[str]] = None,
             profile: Any = None, metadata: Optional[dict[str, Any]] = None) -> tuple[Plan, list[Any], list[str]]:
        """Returns (plan, usages, validation_errors_of_final_plan). Retries once with the validator's errors; falls back to a one-step plan."""
        usages: list[Any] = []
        errors: list[str] = []
        extra = ""
        for attempt in range(2):
            req = ModelRequest(messages=[Message.text("system", self.system_prompt("planner")), Message.text("user", self._context(task, index_text, artifact_ids, contract, lessons or [], extra))],
                               temperature=0.2 if attempt == 0 else 0.5, max_tokens=(profile.max_tokens if profile else 2048), thinking_budget=(profile.thinking_budget if profile else 0),
                               response_format={"type": "json_object"}, metadata={**(metadata or {}), "role": "planner"})
            resp = self.client.complete(req)
            usages.append(resp.usage)
            plan = _coerce_plan(extract_json(resp.text), task)
            if plan is None:
                errors = ["planner did not return a plan object"]
            else:
                errors = validate_plan(plan, self.tools, set(artifact_ids), self.verifiers, self.max_steps)
                if not errors:
                    return plan, usages, []
            extra = "Your previous plan was rejected: " + "; ".join(errors) + ". Fix these problems."
        fallback = Plan(goal=task[:500], steps=[PlanStep("s1", f"Complete the task directly: {truncate(task, 300)}", "executor", [], [t for t in ("fetch_artifact_view", "calculate") if t in self.tools], list(artifact_ids), "evidence" if "evidence" in self.verifiers else None)], rationale="fallback single-step plan after invalid planner output")
        return fallback, usages, errors

    def replan(self, task: str, index_text: str, artifact_ids: list[str], contract: dict[str, Any], failed: list[dict[str, Any]], done: list[dict[str, Any]],
               lessons: list[str], generation: int, profile: Any = None, metadata: Optional[dict[str, Any]] = None) -> tuple[Plan, list[Any], list[str]]:
        extra = (
            f"This is replan generation {generation}. Completed steps that remain valid (you may depend on their ids): {json.dumps(done)[:3000]}\n"
            f"Failed/stale steps to replace (do NOT reuse their ids; create new ids with suffix _g{generation}): {json.dumps(failed)[:4000]}\n"
            "Return only the replacement sub-graph."
        )
        usages: list[Any] = []
        done_ids = {d["id"] for d in done}
        errors: list[str] = []
        for attempt in range(2):
            req = ModelRequest(messages=[Message.text("system", self.system_prompt("planner")), Message.text("user", self._context(task, index_text, artifact_ids, contract, lessons, extra))],
                               temperature=0.4, max_tokens=(profile.max_tokens if profile else 2048), thinking_budget=(profile.thinking_budget if profile else 0),
                               response_format={"type": "json_object"}, metadata={**(metadata or {}), "role": "planner"})
            resp = self.client.complete(req)
            usages.append(resp.usage)
            plan = _coerce_plan(extract_json(resp.text), task)
            if plan is None:
                errors = ["no plan object"]
                continue
            # dependencies may point at done nodes outside the sub-plan
            pseudo = Plan(plan.goal, plan.steps + [PlanStep(d, "done", "executor") for d in done_ids if d not in {s.id for s in plan.steps}])
            errors = [e for e in validate_plan(pseudo, self.tools, set(artifact_ids), self.verifiers, self.max_steps)]
            failed_ids = {f["id"] for f in failed}
            errors += [f"step id {s.id} reuses a replaced id" for s in plan.steps if s.id in failed_ids]
            if not errors:
                return plan, usages, []
            extra += " Previous attempt rejected: " + "; ".join(errors)
        fallback = Plan(task[:500], [PlanStep(f"retry_g{generation}", f"Retry with a different approach: {truncate(task, 300)}", "executor", [d for d in done_ids][:0], [t for t in ("fetch_artifact_view", "calculate") if t in self.tools], list(artifact_ids), "evidence" if "evidence" in self.verifiers else None)], "fallback replan")
        return fallback, usages, errors
