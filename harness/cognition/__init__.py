"""Cognitive substrate (§6): effort knob, thinking control, search-based decoding."""
from .effort import DifficultyEstimator, escalate, budget_for  # noqa: F401
from .thinking import ThinkingController, strip_scratchpad  # noqa: F401
from .sampling import self_consistency, best_of_n, normalise_answer, diversify  # noqa: F401
