"""Budgets: every ceiling on a turn or a session is a named ``Limit`` with a unit.

Exhaustion is never a silent stop. ``BudgetTracker.check`` raises
``BudgetExhausted`` carrying a ``HarnessError`` that names the limit, the
observed value, the next action and what does not raise it.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from typing import Any

from ..domain.errors import ConfigurationError, ErrorCategory, HarnessError, Limit
from ..domain.types import Usage
from ..domain.utils import Clock

EXHAUSTED_ACTION = "Finish with a partial result and record what is left in progress.md."

# name -> (env variable, unit, floor, maximum, does_not_raise)
LIMIT_SPECS: dict[str, tuple[str | None, str, float, float, tuple[str, ...]]] = {
    "max_iterations": ("EDGENT_MAX_ITERATIONS", "model calls per turn", 1, 10_000, ("max_tokens", "the effort level", "the context window")),
    "max_tool_calls": ("EDGENT_MAX_TOOL_CALLS", "tool calls per turn", 1, 100_000, ("tool timeouts", "the number of tools registered")),
    "max_wall_s": ("EDGENT_MAX_WALL_S", "seconds per turn", 1, 604_800, ("tool timeouts", "provider retries")),
    "max_spawns": ("EDGENT_MAX_SPAWNS", "subagent spawns per turn", 0, 1_000, ("the subagent depth limit", "the concurrency cap")),
    "max_cost_usd": ("EDGENT_MAX_COST_USD", "USD per session (0 = unlimited)", 0, 100_000, ("prompt caching", "the effort level")),
    "max_total_tokens": ("EDGENT_MAX_TOTAL_TOKENS", "tokens per session (0 = unlimited)", 0, 1_000_000_000_000, ("compaction", "the context window")),
    "max_depth": (None, "levels of subagent nesting", 0, 8, ("the spawn limit",)),
}
UNLIMITED_WHEN_ZERO = {"max_cost_usd", "max_total_tokens"}


@dataclass
class Budget:
    max_iterations: int = 60
    max_tool_calls: int = 300
    max_wall_s: float = 3600.0
    max_spawns: int = 20
    max_cost_usd: float = 0.0
    max_total_tokens: int = 0
    max_depth: int = 2
    sources: dict[str, str] = field(default_factory=dict)  # name -> default | setting

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> Budget:
        """Read the EDGENT_MAX_* variables; an invalid value raises ConfigurationError, never falls back."""
        base = cls()
        values: dict[str, Any] = {}
        sources: dict[str, str] = {}
        for name, (env, unit, floor, maximum, does_not_raise) in LIMIT_SPECS.items():
            if env is None:
                continue
            lim = Limit.from_env(name, getattr(base, name), unit, env=env, floor=floor, maximum=maximum, does_not_raise=does_not_raise, environ=environ)
            values[name] = _coerce(name, lim.value)
            sources[name] = lim.source
        return replace(base, **values, sources=sources)

    @classmethod
    def from_settings(cls, settings: dict[str, Any], base: Budget | None = None) -> Budget:
        """Apply ``settings['budget']`` (or top-level max_* keys) over ``base``; invalid values raise ConfigurationError."""
        budget = base if base is not None else cls()
        section = settings.get("budget") if isinstance(settings.get("budget"), dict) else {}
        overrides = {k: v for k, v in {**settings, **section}.items() if k in LIMIT_SPECS}
        values: dict[str, Any] = {}
        sources = dict(budget.sources)
        for name, raw in overrides.items():
            _, unit, floor, maximum, _ = LIMIT_SPECS[name]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ConfigurationError(f"settings.budget.{name} must be a number of {unit}; got {raw!r}. Correct it in the settings file.")
            if raw < floor or raw > maximum:
                raise ConfigurationError(f"settings.budget.{name}={raw} is outside [{floor:g}, {maximum:g}] {unit}. Correct it in the settings file.")
            values[name] = _coerce(name, raw)
            sources[name] = "setting"
        return replace(budget, **values, sources=sources)

    def limit(self, name: str) -> Limit:
        env, unit, floor, maximum, does_not_raise = LIMIT_SPECS[name]
        return Limit(
            name=name.replace("_", " "),
            value=float(getattr(self, name)),
            unit=unit,
            source=self.sources.get(name, "default"),
            setting_name=env,
            floor=floor,
            maximum=maximum,
            does_not_raise=does_not_raise,
        )

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in LIMIT_SPECS}


def _coerce(name: str, value: float) -> int | float:
    return float(value) if name in ("max_wall_s", "max_cost_usd") else int(value)


class BudgetExhausted(Exception):
    """A limit was reached. ``error`` is the canonical HarnessError to show the model and the user."""

    def __init__(self, error: HarnessError) -> None:
        super().__init__(error.message)
        self.error = error


class BudgetTracker:
    """Counts what a turn and a session have used and raises when a ceiling is reached."""

    def __init__(self, budget: Budget | None = None, clock: Clock | None = None) -> None:
        self.budget = budget if budget is not None else Budget()
        self.clock = clock or Clock()
        self._lock = threading.Lock()  # subagents charge their usage from tool threads
        self.iterations = 0
        self.tool_calls = 0
        self.spawns = 0
        self.total_tokens = 0
        self.cost_usd = 0.0
        self.turns = 0
        self._turn_started = self.clock.monotonic()

    # ------------------------------------------------------------------ charging
    def start_turn(self) -> None:
        self.iterations = 0
        self.tool_calls = 0
        self.spawns = 0
        self.turns += 1
        self._turn_started = self.clock.monotonic()

    def charge_model(self, usage: Usage) -> None:
        with self._lock:
            self.iterations += 1
            self._add_usage(usage)

    def charge_usage(self, usage: Usage) -> None:
        """Session-level spend without counting an iteration (subagent usage, side calls)."""
        with self._lock:
            self._add_usage(usage)

    def _add_usage(self, usage: Usage) -> None:
        self.total_tokens += usage.total_tokens
        self.cost_usd += usage.cost_usd

    def charge_tools(self, n: int) -> None:
        with self._lock:
            self.tool_calls += max(0, int(n))

    def charge_spawn(self) -> None:
        with self._lock:
            self.spawns += 1

    def elapsed_s(self) -> float:
        return max(0.0, self.clock.monotonic() - self._turn_started)

    # ------------------------------------------------------------------ checking
    def check(self, stage: str) -> None:
        """May the action named by ``stage`` proceed? ``iteration`` (a model call), ``tool`` (more tool calls), ``spawn``."""
        b = self.budget
        self._guard("max_wall_s", self.elapsed_s(), self.elapsed_s() >= b.max_wall_s)
        if b.max_cost_usd > 0:
            self._guard("max_cost_usd", self.cost_usd, self.cost_usd >= b.max_cost_usd)
        if b.max_total_tokens > 0:
            self._guard("max_total_tokens", self.total_tokens, self.total_tokens >= b.max_total_tokens)
        if stage == "iteration":
            self._guard("max_iterations", self.iterations, self.iterations >= b.max_iterations)
        elif stage == "tool":
            self._guard("max_tool_calls", self.tool_calls, self.tool_calls >= b.max_tool_calls)
        elif stage == "spawn":
            self._guard("max_spawns", self.spawns, self.spawns >= b.max_spawns)

    def _guard(self, name: str, observed: float, exhausted: bool) -> None:
        if exhausted:
            lim = self.budget.limit(name)
            raise BudgetExhausted(lim.exceeded(observed, EXHAUSTED_ACTION, ErrorCategory.resource_constraint, stage=name))

    def remaining(self) -> dict[str, Any]:
        b = self.budget
        return {
            "iterations": max(0, b.max_iterations - self.iterations),
            "tool_calls": max(0, b.max_tool_calls - self.tool_calls),
            "wall_s": max(0.0, b.max_wall_s - self.elapsed_s()),
            "spawns": max(0, b.max_spawns - self.spawns),
            "cost_usd": None if b.max_cost_usd <= 0 else max(0.0, b.max_cost_usd - self.cost_usd),
            "total_tokens": None if b.max_total_tokens <= 0 else max(0, b.max_total_tokens - self.total_tokens),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget": self.budget.to_dict(),
            "used": {
                "iterations": self.iterations,
                "tool_calls": self.tool_calls,
                "spawns": self.spawns,
                "wall_s": round(self.elapsed_s(), 3),
                "cost_usd": round(self.cost_usd, 6),
                "total_tokens": self.total_tokens,
                "turns": self.turns,
            },
            "remaining": self.remaining(),
        }
