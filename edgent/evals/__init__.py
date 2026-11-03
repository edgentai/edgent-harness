"""Golden scenarios and the pass gate. A change that regresses the gate does not ship."""
from .gate import FixtureClaim, GateResult, Scenario, evaluate, load_scenarios, record_results

__all__ = ["FixtureClaim", "GateResult", "Scenario", "evaluate", "load_scenarios", "record_results"]
