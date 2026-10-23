import unittest

from harness.core.client import MockModelClient
from harness.core.config import HarnessConfig, ServingConfig
from harness.serving import HealthChecker, Supervisor, environment_manifest, sglang_command, vllm_command


class ServingTests(unittest.TestCase):
    def test_commands(self):
        cfg = ServingConfig(model="Qwen3.8-27B", quantisation="fp8", max_context_length=16384, api_key="k")
        cmd = vllm_command(cfg, "/weights", port=8001)
        self.assertIn("--enable-prefix-caching", cmd)
        self.assertIn("fp8", cmd)
        self.assertIn("16384", cmd)
        self.assertIn("--api-key", cmd)
        self.assertIn("127.0.0.1", cmd)  # never bound publicly
        self.assertIn("sglang.launch_server", sglang_command(cfg, "/weights"))

    def test_health_checker_with_mock(self):
        hc = HealthChecker(MockModelClient(), native_image=True, long_context_tokens=500)
        out = hc.run_all()
        self.assertTrue(out["ok"], out)
        self.assertEqual([c["check"] for c in out["checks"]], ["text", "image", "tool_call", "long_context"])

    def test_supervisor_restarts(self):
        class FakeProc:
            def __init__(self, alive):
                self.alive = alive
                self.killed = False

            def poll(self):
                return None if self.alive else 1

            def kill(self):
                self.killed = True

            def terminate(self):
                self.alive = False

        procs = [FakeProc(False), FakeProc(True)]
        restarted = []
        sup = Supervisor(["srv"], health=lambda: True, on_restart=lambda: restarted.append(1), check_interval_s=0.05, popen=lambda cmd: procs.pop(0))
        sup.start()
        import time

        time.sleep(0.3)
        sup.stop()
        self.assertEqual(sup.restarts, 1)
        self.assertEqual(restarted, [1])

    def test_manifest(self):
        m = environment_manifest(HarnessConfig())
        self.assertIn("python", m)
        self.assertIn("ffmpeg", m["tools"])
        self.assertEqual(m["quantisation"], "fp8")


if __name__ == "__main__":
    unittest.main()
