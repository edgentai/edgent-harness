import os
import tempfile
import time
import unittest

from harness.core.config import HarnessConfig
from harness.modality import ArtifactStore, Ingestor
from harness.modality.sidecars import Sidecars
from harness.sandbox import Workspace
from harness.tools import IdempotencyStore, ToolContext, ToolSpec, build_default_registry
from harness.tools.registry import validate_args
from tests.helpers import have_ffmpeg, write_png


class ValidateArgsTests(unittest.TestCase):
    def test_schema(self):
        sch = {"type": "object", "properties": {"a": {"type": "integer", "minimum": 1}, "b": {"type": "string", "enum": ["x", "y"]}}, "required": ["a"], "additionalProperties": False}
        self.assertEqual(validate_args(sch, {"a": 2, "b": "x"}), [])
        errs = validate_args(sch, {"a": 0, "b": "z", "c": 1})
        self.assertEqual(len(errs), 3)
        self.assertTrue(any("required" in e for e in validate_args(sch, {})))
        self.assertTrue(any("expected integer" in e for e in validate_args(sch, {"a": True})))


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.cfg = HarnessConfig(state_dir=os.path.join(self.d, "state"))
        os.makedirs(self.cfg.state_dir)
        self.store = ArtifactStore(self.cfg.artifacts_dir)
        self.addCleanup(self.store.close)
        self.ing = Ingestor(self.store, work_dir=os.path.join(self.d, "w"))
        self.ws = Workspace(self.cfg.workspaces_dir, "run1")
        self.reg = build_default_registry(IdempotencyStore(self.cfg.db_path))
        self.addCleanup(lambda: self.reg._pool.shutdown(wait=False))
        self.ctx = ToolContext("run1", "n1", self.store, self.ws, Sidecars(), self.cfg, approve=lambda t, a: t == "write_file")

    def test_unknown_and_denied(self):
        r = self.reg.call("nope", {}, self.ctx)
        self.assertEqual(r.error.code, "not_found")
        r = self.reg.call("run_python", {"code": "print(1)"}, self.ctx, allowlist={"read_file"})
        self.assertEqual(r.error.code, "permission_denied")
        r = self.reg.call("read_file", {"path": 3}, self.ctx)
        self.assertEqual(r.error.code, "invalid_arguments")
        self.assertIn("read_file", self.reg.failures)

    def test_dangerous_requires_approval(self):
        ctx = ToolContext("run1", "n1", self.store, self.ws, Sidecars(), self.cfg, approve=None)
        r = self.reg.call("write_file", {"path": "outputs/x.txt", "content": "hi"}, ctx)
        self.assertEqual(r.error.code, "permission_denied")
        r = self.reg.call("write_file", {"path": "outputs/x.txt", "content": "hi"}, self.ctx)
        self.assertTrue(r.ok)

    def test_idempotency_replay(self):
        calls = {"n": 0}

        def fn(ctx, a):
            calls["n"] += 1
            return {"n": calls["n"]}

        self.reg.register(ToolSpec("side", "side effect", {"type": "object", "properties": {}}, fn, permission="compute", side_effects=True))
        r1 = self.reg.call("side", {}, self.ctx, idempotency_key="k1")
        r2 = self.reg.call("side", {}, self.ctx, idempotency_key="k1")
        self.assertEqual((r1.output["n"], r2.output["n"], calls["n"]), (1, 1, 1))
        r3 = self.reg.call("side", {}, self.ctx, idempotency_key="k2")
        self.assertEqual(r3.output["n"], 2)

    def test_timeout_typed_error(self):
        self.reg.register(ToolSpec("slow", "slow", {"type": "object", "properties": {}}, lambda c, a: time.sleep(5), timeout_s=0.2))
        r = self.reg.call("slow", {}, self.ctx)
        self.assertEqual(r.error.code, "timeout")
        self.assertTrue(r.error.retryable)

    def test_execution_error_is_typed_not_traceback(self):
        self.reg.register(ToolSpec("boom", "boom", {"type": "object", "properties": {}}, lambda c, a: 1 / 0))
        r = self.reg.call("boom", {}, self.ctx)
        self.assertEqual(r.error.code, "execution_error")
        self.assertNotIn("Traceback", r.error.message)

    def test_compaction(self):
        self.reg.register(ToolSpec("big", "big", {"type": "object", "properties": {}}, lambda c, a: {"data": "x" * 50_000}))
        r = self.reg.call("big", {}, self.ctx)
        self.assertTrue(r.compacted)
        self.assertIn("full_output_artifact_id", r.output)
        full = self.store.get_view(r.output["full_output_artifact_id"], "text").text
        self.assertEqual(len(full), len('{"data": "' + "x" * 50_000 + '"}'))

    def test_calculate_and_count(self):
        r = self.reg.call("calculate", {"expression": "sum([1.5, 2.5]) * 3"}, self.ctx)
        self.assertEqual(r.output["result"], "12.0")
        r = self.reg.call("calculate", {"expression": "__import__('os')"}, self.ctx)
        self.assertFalse(r.ok)
        a = self.ing.ingest_text("cat dog cat bird cat")
        r = self.reg.call("count_matches", {"artifact_id": a.id, "pattern": r"\bcat\b"}, self.ctx)
        self.assertEqual(r.output["count"], 3)

    def test_fetch_view_paging(self):
        a = self.ing.ingest_text("0123456789" * 10)
        r = self.reg.call("fetch_artifact_view", {"artifact_id": a.id, "view": "text", "offset": 10, "length": 5}, self.ctx)
        self.assertEqual(r.output["text"], "01234")
        self.assertTrue(r.output["untrusted"])
        r = self.reg.call("fetch_artifact_view", {"artifact_id": a.id, "view": "nope"}, self.ctx)
        self.assertEqual(r.error.code, "not_found")

    def test_run_python_sandboxed(self):
        r = self.reg.call("run_python", {"code": "import os; print(sorted(os.listdir('.')))"}, self.ctx)
        self.assertTrue(r.ok)
        self.assertIn("scratch", r.output["stdout"])

    def test_unavailable_sidecars_are_honest(self):
        a = self.ing.ingest_path(write_png(self.d))
        r = self.reg.call("transcribe_range", {"artifact_id": a.id}, self.ctx)
        self.assertEqual(r.error.code, "invalid_arguments")
        wav_art = self.ing.ingest_text("not audio")
        r = self.reg.call("diarise", {"artifact_id": wav_art.id}, self.ctx)
        self.assertEqual(r.error.code, "unavailable")

    @unittest.skipUnless(have_ffmpeg(), "ffmpeg required")
    def test_crop_zoom_cached(self):
        a = self.ing.ingest_path(write_png(self.d, w=40, h=30))
        r1 = self.reg.call("crop_zoom", {"artifact_id": a.id, "x": 0, "y": 0, "width": 10, "height": 10, "zoom": 2}, self.ctx)
        self.assertTrue(r1.ok, r1.error)
        r2 = self.reg.call("crop_zoom", {"artifact_id": a.id, "x": 0, "y": 0, "width": 10, "height": 10, "zoom": 2}, self.ctx)
        self.assertEqual(r1.output["artifact_id"], r2.output["artifact_id"])
        self.assertEqual(self.store.get(r1.output["artifact_id"]).parent_id, a.id)
        r = self.reg.call("compare_images", {"artifact_a": a.id, "artifact_b": a.id}, self.ctx)
        self.assertTrue(r.output["identical"])

    def test_mcp_manifest(self):
        m = self.reg.mcp_manifest()
        names = {t["name"] for t in m["tools"]}
        self.assertIn("fetch_artifact_view", names)
        wf = next(t for t in m["tools"] if t["name"] == "write_file")
        self.assertFalse(wf["annotations"]["readOnlyHint"])
        self.reg.improve_description("calculate", "better", "never for text")
        self.assertEqual(self.reg.get("calculate").version, 2)
        self.assertIn("Do not use when", self.reg.get("calculate").openai_schema()["function"]["description"])


if __name__ == "__main__":
    unittest.main()
