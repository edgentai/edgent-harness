"""Evaluation (§14.2): golden sets, adversarial cases, ablations, results table."""
from .golden import GoldenCase, GoldenSet, EvalRunner, EvalResult  # noqa: F401
from .ablation import run_ablations, ABLATIONS  # noqa: F401
from .adversarial import adversarial_cases  # noqa: F401
