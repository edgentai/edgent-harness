import contextlib
import io
import json
import os
import unittest

from harness.__main__ import main
from tests.helpers import temp_state, write_text


class CliTests(unittest.TestCase):
    def setUp(self):
        self.d, self.cfg = temp_state()
        self.state = self.cfg.state_dir

    def _run(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = main(["--state-dir", self.state, *args])
        return code, out.getvalue()

    def test_run_json_and_followups(self):
        code, out = self._run("--json", "run", "What was the revenue in Q3?", "--text", "Revenue was 1,250,000 dollars in Q3.", "--effort", "low")
        self.assertEqual(code, 0, out)
        data = json.loads(out)
        self.assertEqual(data["status"], "verified")
        run_id = data["run_id"]
        code, out = self._run("runs")
        self.assertIn(run_id, out)
        code, out = self._run("trace", run_id, "--kind", "gate")
        self.assertIn("gate", out)
        code, out = self._run("tools")
        self.assertIn("fetch_artifact_view", out)
        code, out = self._run("health")
        self.assertTrue(json.loads(out)["model"]["ok"])
        code, out = self._run("memory", "list")
        self.assertIn("run_id", out)
        code, out = self._run("memory", "forget")
        self.assertIn("deleted", out)

    def test_run_human_readable_and_file(self):
        p = write_text(self.d, "r.txt", "Headcount grew to 48 people.")
        code, out = self._run("-v", "run", "How many people?", "--file", p, "--effort", "minimal")
        self.assertEqual(code, 0, out)
        self.assertIn("VERDICT", out)
        self.assertIn("48", out)

    def test_eval(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        code, out = self._run("eval", os.path.join(root, "evals", "golden_text.json"), "--results", os.path.join(self.d, "res.md"))
        self.assertEqual(code, 0, out)
        self.assertIn("PASS", out)


if __name__ == "__main__":
    unittest.main()
