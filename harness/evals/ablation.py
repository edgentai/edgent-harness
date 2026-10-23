"""Ablations (§14.2.5): run the golden set with one layer disabled at a time."""
from __future__ import annotations

from typing import Any, Callable

from ..core.config import EffortLevel, HarnessConfig

ABLATIONS: dict[str, Callable[[HarnessConfig], None]] = {
    "baseline": lambda cfg: None,
    "no_thinking": lambda cfg: setattr(cfg, "default_effort", EffortLevel.MINIMAL),
    "no_escalation": lambda cfg: setattr(cfg, "allow_effort_escalation", False),
    "no_native_image": lambda cfg: setattr(cfg.capabilities, "native_image", False),
    "small_context": lambda cfg: setattr(cfg.capabilities, "context_length", 8192),
}


def run_ablations(make_harness: Callable[[HarnessConfig], Any], golden: Any, base_config: HarnessConfig, names: list[str] | None = None, results_path: str | None = None) -> dict[str, Any]:
    from copy import deepcopy

    from .golden import EvalRunner

    out = {}
    for name in names or list(ABLATIONS):
        cfg = deepcopy(base_config)
        ABLATIONS[name](cfg)
        h = make_harness(cfg)
        out[name] = EvalRunner(h.run, results_path).run(golden, label=f"ablation:{name}")
    return out
