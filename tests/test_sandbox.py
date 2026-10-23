import os
import tempfile
import unittest

from harness.sandbox import InterpreterSession, Workspace, run_python, run_subprocess
from harness.sandbox.workspace import WorkspaceEscape


class SandboxTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def test_run_python_ok_and_timeout(self):
        r = run_python("print(6*7)", self.d, timeout_s=20)
        self.assertTrue(r.ok)
        self.assertEqual(r.stdout.strip(), "42")
        r = run_python("import time; time.sleep(10)", self.d, timeout_s=1)
        self.assertTrue(r.timed_out)
        self.assertFalse(r.ok)

    def test_env_is_scrubbed(self):
        os.environ["SECRET_TOKEN_FOR_TEST"] = "leak"
        try:
            r = run_python("import os; print(os.environ.get('SECRET_TOKEN_FOR_TEST'))", self.d, timeout_s=20)
        finally:
            del os.environ["SECRET_TOKEN_FOR_TEST"]
        self.assertEqual(r.stdout.strip(), "None")

    def test_output_capped(self):
        r = run_python("print('x'*100000)", self.d, timeout_s=20)
        self.assertTrue(r.truncated)
        self.assertLess(len(r.stdout), 30000)

    def test_allowed_executables(self):
        r = run_subprocess(["ls"], self.d, allowed_executables=("python3",))
        self.assertEqual(r.returncode, 126)

    def test_interpreter_session_snapshot(self):
        s = InterpreterSession(self.d, timeout_s=20)
        self.assertTrue(s.exec("x = 40")["ok"])
        self.assertEqual(s.exec("x += 2; print(x)")["stdout"].strip(), "42")
        snap = os.path.join(self.d, "snap.pkl")
        self.assertIn("x", s.snapshot(snap)["keys"])
        s.close()
        s2 = InterpreterSession(self.d, timeout_s=20)
        s2.restore(snap)
        self.assertEqual(s2.exec("print(x)")["stdout"].strip(), "42")
        bad = s2.exec("1/0")
        self.assertFalse(bad["ok"])
        self.assertIn("ZeroDivisionError", bad["error"])
        s2.close()

    def test_workspace_escape(self):
        ws = Workspace(self.d, "run1")
        ws.write_text("outputs/a.txt", "hello world")
        self.assertEqual(ws.read_text("outputs/a.txt"), "hello world")
        with self.assertRaises(WorkspaceEscape):
            ws.resolve("../../etc/passwd")
        self.assertEqual(ws.search("hello")[0]["path"], "outputs/a.txt")
        self.assertIn("outputs/a.txt", ws.manifest()["files"])


if __name__ == "__main__":
    unittest.main()
