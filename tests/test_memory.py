import os
import tempfile
import unittest

from harness.core.types import Claim, Evidence, Message
from harness.memory import BM25Index, MemorySystem, WorkingMemory, rerank
from harness.modality import ArtifactStore, Ingestor


class RetrievalTests(unittest.TestCase):
    def test_bm25_and_rerank(self):
        idx = BM25Index()
        idx.add("d1", "the invoice total is 318 euros")
        idx.add("d2", "video of a cat playing piano")
        idx.add("d3", "invoice due date thirty days")
        hits = idx.search("invoice total")
        self.assertEqual(hits[0][0], "d1")
        idx.remove("d1")
        self.assertEqual(idx.search("invoice total")[0][0], "d3")
        rr = rerank("cat piano", [("a", "a cat plays the piano"), ("b", "dogs")])
        self.assertEqual(rr[0][0], "a")


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.mem = MemorySystem(os.path.join(self.d, "mem"))

    def test_gated_writes_and_bitemporal(self):
        with self.assertRaises(PermissionError):
            self.mem.semantic.add_fact("x", "src", {"run_id": "r"}, verified=False)
        with self.assertRaises(PermissionError):
            self.mem.semantic.add_fact("x", "src", {}, verified=True)
        f = self.mem.semantic.add_fact("The launch is on 3 June", "run:1", {"run_id": "1"}, True, subject="launch", valid_from="2026-05-12")
        self.assertEqual(len(self.mem.semantic.facts(subject="launch")), 1)
        self.assertEqual(len(self.mem.semantic.facts(as_of_learned="2000-01-01T00:00:00")), 0)
        self.assertEqual(len(self.mem.semantic.facts(as_of_learned="2999-01-01T00:00:00")), 1)
        self.assertTrue(self.mem.semantic.retract(f.id))
        self.assertEqual(len(self.mem.semantic.facts()), 0)
        self.assertEqual(len(self.mem.semantic.facts(include_retracted=True)), 1)
        self.assertEqual(self.mem.search("launch"), [])

    def test_commit_verified_only(self):
        c1 = Claim("verified one", [Evidence("a" * 64, "text", {"offset": 0}, "q")], verified=True)
        c2 = Claim("unverified", [Evidence("a" * 64, "text", {}, "q")], verified=False)
        stored = self.mem.commit_verified("run9", [c1, c2], user_id="u1")
        self.assertEqual([f.text for f in stored], ["verified one"])
        self.assertEqual(stored[0].provenance["run_id"], "run9")
        self.assertEqual(self.mem.forget_user("u1"), 1)
        self.assertEqual(self.mem.search("verified"), [])

    def test_index_artifact_and_search_pointer(self):
        store = ArtifactStore(os.path.join(self.d, "art"))
        self.addCleanup(store.close)
        art = Ingestor(store, work_dir=os.path.join(self.d, "w")).ingest_text("Section one talks about pelicans.\n\n" + "filler " * 300 + "\n\nSection two talks about budgets.")
        n = self.mem.index_artifact(store, art.id)
        self.assertGreaterEqual(n, 2)
        hit = self.mem.search("pelicans")[0]
        self.assertEqual(hit["artifact_id"], art.id)
        self.assertIn("offset", hit["locator"])

    def test_procedural_versions(self):
        self.assertEqual(self.mem.procedural.save("recipes", "video qa", {"a": 1}), 1)
        self.assertEqual(self.mem.procedural.save("recipes", "video qa", {"a": 2}), 2)
        self.assertEqual(self.mem.procedural.load("recipes", "video qa")["content"], {"a": 2})
        self.assertEqual(self.mem.procedural.load("recipes", "video qa", 1)["content"], {"a": 1})
        self.assertIsNone(self.mem.procedural.load("recipes", "missing"))

    def test_working_memory_compaction(self):
        wm = WorkingMemory(max_tokens=200, keep_recent=2)
        wm.add(Message.text("system", "sys"))
        for i in range(10):
            wm.add(Message.text("tool", "x" * 200, name=f"t{i}"))
        wm.stance.append("the total is 42")
        self.assertTrue(wm.needs_compaction())
        summary = wm.compact()
        self.assertIn("compacted", summary)
        self.assertIn("the total is 42", summary)
        self.assertEqual(wm.messages[0].role, "system")
        self.assertEqual(len(wm.messages), 4)

    def test_episodic(self):
        self.mem.episodic.record("r1", "n1", "lesson", "check offsets")
        self.assertEqual(self.mem.episodic.lessons(), ["check offsets"])
        self.assertEqual(self.mem.episodic.for_run("r1")[0]["kind"], "lesson")


if __name__ == "__main__":
    unittest.main()
