"""Effort allocation (§6.2): estimate, map to a budget, escalate on failure."""
from __future__ import annotations

import re
from typing import Any, Optional

from ..core.client import ModelClient, ModelRequest
from ..core.config import EFFORT_PROFILES, EffortLevel, EffortProfile
from ..core.types import Budget, Message, Modality
from ..core.utils import extract_json

HARD_WORDS = ("prove", "every", "all of", "comprehensive", "compare", "analyse", "analyze", "reconcile", "audit", "debug", "why")
EASY_WORDS = ("what is", "when", "who", "how many", "yes or no", "list the")


class DifficultyEstimator:
    """Heuristics first; an optional cheap model call refines the guess."""

    def __init__(self, client: Optional[ModelClient] = None, use_model: bool = False) -> None:
        self.client = client
        self.use_model = use_model

    def heuristic(self, task: str, modalities: list[Modality], total_input_tokens: int) -> EffortLevel:
        score = 0
        lowered = task.lower()
        score += min(3, len(task) // 200)
        score += sum(1 for w in HARD_WORDS if w in lowered)
        score -= sum(1 for w in EASY_WORDS if lowered.startswith(w))
        score += max(0, len(set(modalities)) - 1)
        if Modality.VIDEO in modalities:
            score += 2
        if total_input_tokens > 20_000:
            score += 2
        elif total_input_tokens > 5_000:
            score += 1
        if re.search(r"\b(code|implement|script|function|test)\b", lowered):
            score += 1
        if score <= 0:
            return EffortLevel.LOW
        if score <= 2:
            return EffortLevel.MEDIUM
        if score <= 4:
            return EffortLevel.HIGH
        return EffortLevel.MAX

    def estimate(self, task: str, modalities: list[Modality], total_input_tokens: int, system_prompt: Optional[str] = None) -> tuple[EffortLevel, str]:
        guess = self.heuristic(task, modalities, total_input_tokens)
        if self.use_model and self.client is not None and system_prompt:
            try:
                resp = self.client.complete(ModelRequest(messages=[Message.text("system", system_prompt), Message.text("user", task[:2000])], temperature=0.0, max_tokens=100, metadata={"role": "difficulty"}))
                data = extract_json(resp.text) or {}
                lvl = EffortLevel(str(data.get("effort", guess.value)))
                return lvl, f"model:{data.get('reason', '')}"
            except Exception:
                pass
        return guess, "heuristic"


def budget_for(profile: EffortProfile, context_length: int = 32_768, wall_clock_s: Optional[float] = None, cost_cap_usd: float = 1.0) -> Budget:
    mult = {EffortLevel.MINIMAL: 1, EffortLevel.LOW: 2, EffortLevel.MEDIUM: 5, EffortLevel.HIGH: 12, EffortLevel.MAX: 25}[profile.level]
    return Budget(
        max_tokens=context_length * mult,
        max_tool_calls=profile.max_tool_iterations * max(1, profile.max_parallel_workers) * 2,
        max_model_calls=10 + profile.max_tool_iterations * max(1, profile.max_parallel_workers) * 2 + profile.samples * 4,
        max_branches=profile.tree_of_thoughts_breadth * 2,
        max_wall_clock_s=wall_clock_s if wall_clock_s is not None else 120.0 * mult,
        max_cost_usd=cost_cap_usd,
        max_replans=min(3, max(0, mult // 4)),
        max_refine_rounds=profile.refine_rounds,
    )


def escalate(level: EffortLevel, allowed: bool = True) -> Optional[EffortLevel]:
    """Next level after a verification failure; ``None`` when at the top or disabled."""
    if not allowed:
        return None
    return level.next()


def profile_settings(level: EffortLevel) -> dict[str, Any]:
    return EFFORT_PROFILES[level].to_dict()
