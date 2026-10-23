import json
import unittest

from harness.cognition.effort import DifficultyEstimator, budget_for, escalate
from harness.cognition.sampling import best_of_n, coverage_curve, diversify, knee, normalise_answer, self_consistency
from harness.cognition.thinking import ThinkingController, strip_scratchpad
from harness.core.client import MockModelClient, ModelRequest, ModelResponse
from harness.core.config import EFFORT_PROFILES, EffortLevel
from harness.core.types import Message, Modality


class NormaliseTests(unittest.TestCase):
    def test_forms(self):
        self.assertEqual(normalise_answer("The total is $1,234.50"), "1234.5")
        self.assertEqual(normalise_answer("42"), "42")
        self.assertEqual(normalise_answer("Yes."), "yes")
        self.assertEqual(normalise_answer("TRUE"), "yes")
        self.assertEqual(normalise_answer(json.dumps({"answer": "Paris"})), "paris")
        self.assertIsNone(normalise_answer("a " * 100))


class SamplingTests(unittest.TestCase):
    def test_self_consistency_majority(self):
        m = MockModelClient()
        m.push("41", "42", "42")
        out = self_consistency(m, ModelRequest([Message.text("user", "q")]), n=3)
        self.assertEqual(out.chosen.text, "42")
        self.assertAlmostEqual(out.agreement, 2 / 3)
        self.assertEqual(out.method, "self_consistency")
        reqs = diversify(ModelRequest([Message.text("user", "q")], seed=1), 3, 0.5)
        self.assertEqual(len({r.messages[-1].plain_text() for r in reqs}), 3)
        self.assertEqual([r.seed for r in reqs], [1, 2, 3])

    def test_best_of_n(self):
        m = MockModelClient()
        m.push("short", "a much longer answer", "mid")
        out = best_of_n(m, ModelRequest([Message.text("user", "q")]), 3, scorer=lambda r: len(r.text))
        self.assertEqual(out.chosen.text, "a much longer answer")
        self.assertEqual(len(out.scores), 3)

    def test_coverage_knee(self):
        curve = coverage_curve([[False, True, True], [True, True, True], [False, False, True], [False, False, False]])
        self.assertEqual(curve, [0.25, 0.5, 0.75])
        self.assertEqual(knee([0.3, 0.6, 0.65, 0.66]), 3)


class ThinkingTests(unittest.TestCase):
    def test_force_answer_after_length_cutoff(self):
        m = MockModelClient()
        m.push(ModelResponse(text="", reasoning="thinking...", finish_reason="length"), "final answer")
        tc = ThinkingController(m)
        resp = tc.run(ModelRequest([Message.text("user", "hard")], thinking_budget=100))
        self.assertEqual(resp.text, "final answer")
        self.assertEqual(len(m.calls), 2)
        self.assertEqual(m.calls[1].thinking_budget, 0)
        self.assertIn("thinking", resp.reasoning)

    def test_extend_if_short(self):
        m = MockModelClient()
        m.push(ModelResponse(text="7", reasoning="", finish_reason="stop"), "8")
        resp = ThinkingController(m).run(ModelRequest([Message.text("user", "hard")], thinking_budget=100), extend_if_short=True)
        self.assertEqual(resp.text, "8")
        self.assertEqual(strip_scratchpad("<think>x</think>y"), "y")


class EffortTests(unittest.TestCase):
    def test_estimator_and_escalation(self):
        est = DifficultyEstimator()
        self.assertEqual(est.heuristic("What is the date?", [Modality.TEXT], 100), EffortLevel.LOW)
        hard = est.heuristic("Comprehensively compare and reconcile every figure across these documents and prove the totals", [Modality.VIDEO, Modality.AUDIO], 30000)
        self.assertEqual(hard, EffortLevel.MAX)
        self.assertEqual(escalate(EffortLevel.LOW), EffortLevel.MEDIUM)
        self.assertIsNone(escalate(EffortLevel.MAX))
        self.assertIsNone(escalate(EffortLevel.LOW, allowed=False))
        b_low = budget_for(EFFORT_PROFILES[EffortLevel.LOW])
        b_max = budget_for(EFFORT_PROFILES[EffortLevel.MAX])
        self.assertLess(b_low.max_tokens, b_max.max_tokens)
        self.assertLess(b_low.max_replans, b_max.max_replans)


if __name__ == "__main__":
    unittest.main()
