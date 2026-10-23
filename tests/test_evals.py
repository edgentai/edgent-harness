import os
import tempfile
import unittest

from harness.core.config import EffortLevel, HarnessConfig
from harness.core.utils import SequentialIdGenerator
from harness.evals import EvalRunner, GoldenCase, GoldenSet, adversarial_cases, run_ablations
from harness.orchestration import Harness
from tests.helpers import temp_state

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GoldenTests(unittest.TestCase):
    def setUp(self):
        self.d, self.cfg = temp_state()
        self.h = Harness(self.cfg, ids=SequentialIdGenerator())
        self.addCleanup(self.h.close)

    def test_repo_golden_set_passes_on_mock(self):
        golden = GoldenSet.load(os.path.join(ROOT, "evals", "golden_text.json"))
        results_path = os.path.join(self.d, "results.md")
        summary = EvalRunner(self.h.run, results_path).run(golden)
        self.assertEqual(summary["cases"], 3)
        self.assertEqual(summary["accuracy"], 1.0, [r for r in summary["results"] if not r["passed"]])
        self.assertIsNotNone(summary["citation_validity"])
        with open(results_path) as f:
            table = f.read()
        self.assertIn("| ts | label |", table)
        self.assertEqual(table.count("golden_text_v1"), 1)

    def test_case_checks(self):
        class R:
            answer = "The total is 318.40"
            status = "verified"
            claims = [{"x": 1}]

        self.assertTrue(GoldenCase("c", "t", expected="318.4").check(R())[0])
        self.assertFalse(GoldenCase("c", "t", expected_status="failed").check(R())[0])
        R.claims = []
        self.assertFalse(GoldenCase("c", "t").check(R())[0])
        self.assertTrue(GoldenCase("c", "t", must_cite=False).check(R())[0])

    def test_adversarial_cases(self):
        cases = adversarial_cases(os.path.join(self.d, "adv"))
        summary = EvalRunner(self.h.run).run(GoldenSet("adv", cases), label="adversarial")
        by_id = {r["case_id"]: r for r in summary["results"]}
        self.assertTrue(by_id["adv_injection"]["passed"], by_id["adv_injection"])
        self.assertIn(by_id["adv_corrupt_image"]["status"], ("partially_verified", "failed", "verified"))
        self.assertTrue(by_id["adv_silent_audio"]["passed"], by_id["adv_silent_audio"])

    def test_ablations(self):
        golden = GoldenSet("mini", [GoldenCase("a", "What was the revenue?", inline_texts=["Revenue was 5 dollars."], expected_regex="5", effort="minimal")])
        made = []

        def make(cfg):
            h = Harness(cfg, ids=SequentialIdGenerator())
            made.append(h)
            return h

        out = run_ablations(make, golden, self.cfg, ["baseline", "no_escalation"])
        for h in made:
            h.close()
        self.assertEqual(set(out), {"baseline", "no_escalation"})
        self.assertEqual(out["baseline"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
