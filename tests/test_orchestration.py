import json
import os
import tempfile
import threading
import time
import unittest

from harness.core.client import MockModelClient, ModelResponse
from harness.core.config import EFFORT_PROFILES, EffortLevel, HarnessConfig, ModelCapabilities
from harness.core.types import Budget, Modality, NodeStatus, Plan, PlanStep, ToolCall
from harness.core.utils import SequentialIdGenerator, extract_json
from harness.modality import ArtifactStore, Ingestor
from harness.modality.sidecars import Sidecars
from harness.observability.tracing import Tracer
from harness.orchestration import InvalidTransition, Scheduler, StateDB, SubAgent, TaskClassifier, TaskDAG, allowlist_for, validate_plan
from harness.orchestration.planner import Planner
from harness.prompts import system_prompt
from harness.sandbox import Workspace
from harness.tools import ToolContext, build_default_registry


class DagTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.db = StateDB(os.path.join(self.d, "s.sqlite3"))
        self.addCleanup(self.db.close)
        self.db.create_run("r1", "t")
        self.dag = TaskDAG(self.db, "r1")
        self.dag.load_plan(Plan("g", [PlanStep("a", "A"), PlanStep("b", "B", depends_on=["a"]), PlanStep("c", "C", depends_on=["a"]), PlanStep("d", "D", depends_on=["b", "c"])]))

    def test_transitions(self):
        self.assertEqual([n.id for n in self.dag.ready_nodes()], ["a"])
        self.dag.start("a", "w")
        self.dag.complete("a", {"x": 1}, {})
        self.assertEqual({n.id for n in self.dag.ready_nodes()}, {"b", "c"})
        with self.assertRaises(InvalidTransition):
            self.dag.set_status("a", NodeStatus.RUNNING)
        self.dag.start("b", "w")
        self.dag.fail("b", "boom")
        self.assertEqual(self.dag.node("d").status, NodeStatus.BLOCKED)
        self.assertTrue(self.dag.terminal() is False)  # c is still ready
        self.assertEqual(self.dag.history("b")[-1]["to_status"], "failed")

    def test_stale_and_splice(self):
        self.dag.start("a", "w"); self.dag.complete("a", {}, {})
        self.dag.start("b", "w"); self.dag.fail("b", "bad")
        self.dag.start("c", "w"); self.dag.complete("c", {}, {})
        stale = self.dag.mark_stale("b")
        self.assertEqual(set(stale), {"b", "d"})
        self.assertEqual(self.dag.node("c").status, NodeStatus.DONE)  # unaffected branch untouched
        new = self.dag.splice(Plan("g", [PlanStep("b2", "B again", depends_on=["a"])]), ["b", "d"], 2)
        self.assertEqual(new, ["b2"])
        self.assertEqual(self.dag.node("b").status, NodeStatus.SKIPPED)
        self.assertIn("b2", self.dag.ready_nodes()[0].id)
        self.assertEqual(self.dag.node("b2").generation, 2)

    def test_recover_running(self):
        self.dag.start("a", "w")
        self.assertEqual(self.dag.recover_running(), ["a"])
        self.assertEqual(self.dag.node("a").status, NodeStatus.READY)
        self.assertEqual(self.db.in_flight(), ["r1"])
        self.db.update_run("r1", status="running")
        self.db.checkpoint("r1", {"x": 1})
        self.assertEqual(self.db.get_run("r1")["status"], "running")


class PlanValidationTests(unittest.TestCase):
    def test_errors(self):
        tools, arts, vers = {"fetch_artifact_view"}, {"a" * 64}, {"evidence"}
        cyc = Plan("g", [PlanStep("a", "A", depends_on=["b"]), PlanStep("b", "B", depends_on=["a"])])
        self.assertTrue(any("cycle" in e for e in validate_plan(cyc, tools, arts, vers)))
        bad = Plan("g", [PlanStep("a", "A", tools=["nope"], artifacts=["zzz"], verifier="x", role="wizard", depends_on=["q"])])
        errs = validate_plan(bad, tools, arts, vers)
        self.assertEqual(len(errs), 5)
        self.assertEqual(validate_plan(Plan("g", []), tools, arts, vers), ["plan has no steps"])
        ok = Plan("g", [PlanStep("a", "A", tools=["fetch_artifact_view"], artifacts=["a" * 12], verifier="evidence")])
        self.assertEqual(validate_plan(ok, tools, arts, vers), [])


class PlannerTests(unittest.TestCase):
    def test_retry_then_fallback(self):
        m = MockModelClient()
        m.push("garbage", '{"steps": [{"id": "x", "depends_on": ["missing"]}]}')
        p = Planner(m, system_prompt, {"fetch_artifact_view", "calculate"}, {"evidence"})
        plan, usages, errors = p.plan("do it", "No input artifacts.", [], {})
        self.assertEqual(len(usages), 2)
        self.assertTrue(errors)
        self.assertEqual(len(plan.steps), 1)
        self.assertIn("fallback", plan.rationale)
        self.assertIn("rejected", m.calls[1].messages[-1].plain_text())

    def test_valid_plan(self):
        m = MockModelClient()
        p = Planner(m, system_prompt, {"fetch_artifact_view"}, {"evidence"})
        plan, _, errors = p.plan("q", "artifact_id=" + "a" * 64, ["a" * 64], {})
        self.assertEqual(errors, [])
        self.assertEqual([s.id for s in plan.steps], ["s1", "s_final"])
        sub, _, errors = p.replan("q", "artifact_id=" + "a" * 64, ["a" * 64], {}, [{"id": "s1", "description": "d", "error": "e"}], [], [], 2)
        self.assertEqual(errors, [])
        self.assertTrue(all(s.id.endswith("_g2") for s in sub.steps))


class ClassifierTests(unittest.TestCase):
    def test_heuristics(self):
        c = TaskClassifier(None, use_model=False)
        self.assertEqual(c.heuristic("Suggest ten titles for this video", [Modality.VIDEO]).problem_type.value, "divergent")
        self.assertEqual(c.heuristic("Convert this slide deck into a report", [Modality.IMAGE]).problem_type.value, "structural")
        self.assertEqual(c.heuristic("What's interesting in this recording?", [Modality.AUDIO]).problem_type.value, "exploratory")
        conv = c.heuristic("What is the total on this invoice?", [Modality.IMAGE])
        self.assertEqual(conv.problem_type.value, "convergent")
        self.assertIn("calculate", conv.expected_tools)
        self.assertIn("crop_zoom", conv.expected_tools)
        self.assertTrue(c.heuristic("fix the bug in this python function", []).code_task)
        self.assertNotIn("spawn_subagent", allowlist_for("executor", False, False))
        self.assertNotIn("write_file", allowlist_for("verifier", True, True))


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.cfg = HarnessConfig(state_dir=os.path.join(self.d, "state"))
        os.makedirs(self.cfg.state_dir)
        self.store = ArtifactStore(self.cfg.artifacts_dir)
        self.addCleanup(self.store.close)
        self.art = Ingestor(self.store, work_dir=os.path.join(self.d, "w")).ingest_text("The secret number is 7731.\nSecond line.")
        self.reg = build_default_registry(ids=SequentialIdGenerator())
        self.ws = Workspace(self.cfg.workspaces_dir, "r")
        self.tracer = Tracer(self.cfg.traces_dir, "r")
        self.ctx = ToolContext("r", "n1", self.store, self.ws, Sidecars(), self.cfg, tracer=self.tracer)
        self.budget = Budget()

    def _agent(self, client, caps=None, allow=None, agent_budget=None):
        return SubAgent("executor", client, self.reg, self.ctx, allow if allow is not None else {"fetch_artifact_view", "calculate"}, EFFORT_PROFILES[EffortLevel.LOW],
                        caps or ModelCapabilities(), system_prompt, self.budget, self.tracer, ids=SequentialIdGenerator(), agent_budget=agent_budget)

    def test_react_loop_and_output_discipline(self):
        m = MockModelClient(ids=SequentialIdGenerator())
        m.push(ModelResponse(text="", tool_calls=[ToolCall("c1", "fetch_artifact_view", {"artifact_id": self.art.id, "view": "text"})], finish_reason="tool_calls"),
               json.dumps({"answer": "7731", "claims": [{"text": "secret is 7731", "evidence": [{"artifact_id": self.art.id, "kind": "text", "locator": {"offset": 0}, "quote": "The secret number is 7731."}]}], "confidence": 0.9, "gaps": []}))
        res = self._agent(m).run("find the secret number", "artifact_id=" + self.art.id)
        self.assertEqual(res.answer, "7731")
        self.assertEqual(res.tool_calls_used, 1)
        self.assertEqual(res.iterations, 2)
        self.assertEqual(len(res.claims), 1)
        self.assertEqual(self.budget.used_model_calls, 2)
        tool_msg = m.calls[1].messages[-1]
        self.assertEqual(tool_msg.role, "tool")
        self.assertIn("UNTRUSTED_DATA", tool_msg.plain_text())
        kinds = [e.kind for e in self.tracer.events()]
        self.assertIn("model_call", kinds)
        self.assertIn("tool_call", kinds)

    def test_disallowed_tool_is_reported_to_model(self):
        m = MockModelClient(ids=SequentialIdGenerator())
        m.push(ModelResponse(text="", tool_calls=[ToolCall("c1", "run_python", {"code": "print(1)"})], finish_reason="tool_calls"), '{"answer": "gave up", "claims": [], "gaps": ["no python"]}')
        res = self._agent(m).run("x", "ctx")
        self.assertIn("permission_denied", m.calls[1].messages[-1].plain_text())
        self.assertEqual(res.answer, "gave up")

    def test_text_fallback_tool_calls(self):
        m = MockModelClient(ids=SequentialIdGenerator())
        m.push('{"tool": "calculate", "arguments": {"expression": "6*7"}}', '{"answer": "42", "claims": [], "gaps": []}')
        res = self._agent(m, caps=ModelCapabilities(native_tool_calls=False)).run("compute", "ctx")
        self.assertEqual(res.answer, "42")
        self.assertIn('"result": "42"', m.calls[1].messages[-1].plain_text())

    def test_iteration_budget_forces_final(self):
        m = MockModelClient(ids=SequentialIdGenerator())
        looping = lambda req: ModelResponse(text="", tool_calls=[ToolCall("c", "calculate", {"expression": "1+1"})], finish_reason="tool_calls") if req.tools else ModelResponse(text='{"answer": "forced", "claims": [], "gaps": []}')
        m.handlers["executor"] = looping
        res = self._agent(m, agent_budget=__import__("harness.orchestration.agent", fromlist=["AgentBudget"]).AgentBudget(max_iterations=3, max_tool_calls=2)).run("loop", "ctx")
        self.assertEqual(res.answer, "forced")
        self.assertEqual(res.tool_calls_used, 2)
        self.assertTrue(any("budget" in g for g in res.gaps))
        # with no room for a tool-free turn, the loop ends and the forced final answer is marked as budget_exhausted
        m2 = MockModelClient(ids=SequentialIdGenerator())
        m2.handlers["executor"] = looping
        res2 = self._agent(m2, agent_budget=__import__("harness.orchestration.agent", fromlist=["AgentBudget"]).AgentBudget(max_iterations=2, max_tool_calls=5)).run("loop", "ctx")
        self.assertEqual(res2.status, "budget_exhausted")
        self.assertEqual(res2.answer, "forced")

    def test_unstructured_answer_gets_gap(self):
        m = MockModelClient(ids=SequentialIdGenerator())
        m.push("just prose")
        res = self._agent(m, allow=set()).run("x", "ctx")
        self.assertEqual(res.answer, "just prose")
        self.assertTrue(res.gaps)
        self.assertLess(res.confidence, 0.5)

    def test_reflexion(self):
        m = MockModelClient(ids=SequentialIdGenerator())
        m.push("Always check the cited offset against the view.")
        lesson = self._agent(m).reflect("task", __import__("harness.core.types", fromlist=["SubAgentResult"]).SubAgentResult(answer="x"), ["quote not found"])
        self.assertIn("offset", lesson)


class SchedulerTests(unittest.TestCase):
    def test_parallel_wave(self):
        d = tempfile.mkdtemp()
        db = StateDB(os.path.join(d, "s.sqlite3"))
        self.addCleanup(db.close)
        db.create_run("r", "t")
        dag = TaskDAG(db, "r")
        dag.load_plan(Plan("g", [PlanStep("a", "A"), PlanStep("b", "B"), PlanStep("c", "C", depends_on=["a", "b"])]))
        active = {"n": 0, "max": 0}
        lock = threading.Lock()

        def worker(node):
            with lock:
                active["n"] += 1
                active["max"] = max(active["max"], active["n"])
            time.sleep(0.2)
            with lock:
                active["n"] -= 1
            if node.id == "b":
                return False, {}, {}, "nope"
            return True, {"answer": node.id}, {"tokens": 1}, None

        sched = Scheduler(db, dag, max_parallel=4)
        done = sched.step(worker, lambda: False)
        self.assertEqual({n.id for n in done}, {"a", "b"})
        self.assertEqual(active["max"], 2)
        self.assertEqual(dag.node("a").status, NodeStatus.DONE)
        self.assertEqual(dag.node("b").status, NodeStatus.FAILED)
        self.assertEqual(dag.node("c").status, NodeStatus.BLOCKED)
        self.assertEqual(sched.step(worker, lambda: False), [])


if __name__ == "__main__":
    unittest.main()
