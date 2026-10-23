"""The master loop (§13.2) and the ``Harness`` facade that wires every layer together.

Receive task and artifacts -> ingest -> classify, set effort and budgets ->
definition of done -> plan (DAG) -> dispatch ready nodes in parallel ->
verify node outputs -> rollback/replan on failure -> synthesise -> trust
gate -> answer + evidence + verdict, or an honest failure report.

Loop invariants: budgets and stop conditions are checked first in every
iteration; every state change is persisted before the next step; every
model call is traced.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..cognition.effort import DifficultyEstimator, budget_for, escalate
from ..cognition.sampling import normalise_answer
from ..core.client import MockModelClient, ModelClient, ModelRequest, OpenAICompatibleClient
from ..core.config import EFFORT_PROFILES, EffortLevel, EffortProfile, HarnessConfig
from ..core.types import Budget, Claim, Message, Modality, NodeStatus, Plan, ProblemType, SubAgentResult, Usage, Verdict, VerdictStatus
from ..core.utils import Clock, IdGenerator, ensure_dir, estimate_tokens, extract_json, truncate
from ..memory.tiers import MemorySystem
from ..modality.artifacts import Artifact, ArtifactStore
from ..modality.budgeter import BudgetDecision, ModalityBudgeter
from ..modality.index import ModalityIndex
from ..modality.ingest import Ingestor
from ..modality.sidecars import Sidecars
from ..observability.cost import CostAccountant
from ..observability.logging import StructuredLogger
from ..observability.metrics import Metrics
from ..observability.tracing import Tracer
from ..prompts.loader import PROMPT_VERSION, system_prompt
from ..sandbox.workspace import Workspace
from ..tools.builtin import build_default_registry
from ..tools.registry import IdempotencyStore, ToolContext
from ..verification.contract import DefinitionOfDone
from ..verification.gate import EvidenceBundle, TrustGate
from ..verification.verifiers import VerifyContext, build_default_verifiers
from .agent import AgentBudget, SubAgent
from .classifier import Classification, TaskClassifier
from .dag import Node, StateDB, TaskDAG
from .planner import Planner
from .roles import allowlist_for
from .scheduler import Scheduler


@dataclass
class RunResult:
    run_id: str
    status: str  # verified | partially_verified | failed | error
    answer: Any
    claims: list[dict[str, Any]] = field(default_factory=list)
    verdict: Optional[dict[str, Any]] = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    effort: str = ""
    problem_type: str = ""
    cost: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)
    trace_path: str = ""
    plan: Optional[dict[str, Any]] = None
    nodes: dict[str, int] = field(default_factory=dict)
    replans: int = 0
    escalations: int = 0
    artifacts: list[str] = field(default_factory=list)
    wall_clock_s: float = 0.0
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class RunState:
    run_id: str
    task: str
    user_id: str
    artifacts: list[Artifact]
    classification: Classification
    level: EffortLevel
    profile: EffortProfile
    budget: Budget
    contract: DefinitionOfDone
    decision: BudgetDecision
    index_text: str
    tracer: Tracer
    workspace: Workspace
    cost: CostAccountant
    dag: TaskDAG
    plan: Optional[Plan] = None
    lessons: list[str] = field(default_factory=list)
    generation: int = 1
    replans: int = 0
    escalations: int = 0
    started: float = 0.0
    stop_reason: Optional[str] = None  # hard stop: budget exhausted or session draining
    blocked_reason: Optional[str] = None  # soft stop: graph cannot progress (may be fixed by escalation)


class Harness:
    def __init__(self, config: Optional[HarnessConfig] = None, client: Optional[ModelClient] = None, backend: str = "mock",
                 sidecars: Optional[Sidecars] = None, ids: Optional[IdGenerator] = None, clock: Optional[Clock] = None,
                 ask_user: Optional[Callable[[str], str]] = None, approve: Optional[Callable[[str, dict[str, Any]], bool]] = None,
                 echo: Optional[Callable[[Any], None]] = None) -> None:
        self.config = config or HarnessConfig()
        ensure_dir(self.config.state_dir)
        self.ids = ids or IdGenerator()
        self.clock = clock or Clock()
        self.client = client or self._make_client(backend)
        self.sidecars = sidecars or Sidecars()
        self.db = StateDB(self.config.db_path)
        self.store = ArtifactStore(self.config.artifacts_dir)
        self.ingestor = Ingestor(self.store, self.config.modality, self.sidecars, os.path.join(self.config.state_dir, "ingest_tmp"))
        self.memory = MemorySystem(self.config.memory_dir, embed=self.sidecars.embed, ids=self.ids)
        self.metrics = Metrics()
        self.registry = build_default_registry(IdempotencyStore(self.config.db_path), self.ids, self.metrics)
        self.verifiers = build_default_verifiers()
        self.gate = TrustGate(self.verifiers)
        self.budgeter = ModalityBudgeter(self.store, self.config.capabilities, self.config.modality)
        self.index = ModalityIndex(self.store)
        self.log = StructuredLogger("harness")
        self.ask_user = ask_user
        self.approve = approve or self._default_approve
        self.echo = echo
        self.session_started = self.clock.monotonic()
        self._lock = threading.Lock()

    # -- wiring -------------------------------------------------------------
    def _make_client(self, backend: str) -> ModelClient:
        if backend == "mock":
            return MockModelClient(ids=self.ids)
        s = self.config.serving
        c = self.config.capabilities
        return OpenAICompatibleClient(s.base_url, s.model, s.api_key, s.timeout_s, s.max_retries, s.price_per_million_prompt, s.price_per_million_completion,
                                      c.native_image, c.native_audio, c.native_video, c.thinking_switch, self.ids)

    def _default_approve(self, tool: str, args: dict[str, Any]) -> bool:
        # Writes stay inside the sandboxed workspace, so they are approved; network/delete need an explicit policy.
        spec = self.registry.get(tool)
        return bool(spec and spec.permission == "write")

    def close(self) -> None:
        """Release database connections and the tool thread pool."""
        self.store.close()
        self.db.close()
        try:
            self.registry._pool.shutdown(wait=False)
        except Exception:
            pass

    def session_draining(self) -> bool:
        """§4.4: near the expected session limit, stop admitting long jobs."""
        elapsed = self.clock.monotonic() - self.session_started
        return elapsed >= self.config.session_time_limit_s * self.config.drain_fraction

    def health(self) -> dict[str, Any]:
        return {"model": self.client.health(), "sidecars": self.sidecars.status(), "state_dir": self.config.state_dir, "tools": len(self.registry.names()),
                "verifiers": self.verifiers.names(), "in_flight_runs": self.db.in_flight(), "metrics": self.metrics.snapshot()}

    # -- public API ----------------------------------------------------------
    def run(self, task: str, files: Optional[list[str]] = None, effort: Optional[EffortLevel] = None, user_id: str = "default",
            run_id: Optional[str] = None, inline_texts: Optional[list[str]] = None) -> RunResult:
        if self.session_draining():
            return RunResult(run_id or "", "error", None, message="session is draining before the expected time limit; not admitting new jobs")
        run_id = run_id or self.ids.new("run")
        t0 = self.clock.monotonic()
        tracer = Tracer(self.config.traces_dir, run_id, self.clock, self.echo)
        log = self.log.bind(run_id)
        self.db.create_run(run_id, task, user_id, {"prompt_version": PROMPT_VERSION, "effort_override": effort.value if effort else None})
        workspace = Workspace(self.config.workspaces_dir, run_id)
        tracer.record("note", {"event": "run_started", "task": truncate(task, 500), "files": files or []})
        try:
            # 1. Ingest and build the modality index.
            artifacts = self._ingest(files or [], inline_texts or [], workspace, tracer, user_id)
            # 2. Classify, set effort and budgets.
            classification, level, profile, budget = self._classify_and_budget(task, artifacts, effort, tracer)
            # 3. Definition of done.
            contract = self._contract(task, classification, artifacts, profile, tracer, run_id)
            decision = self.budgeter.decide(artifacts, self.config.capabilities.context_length // 2)
            index_text = self.index.render(artifacts, decision)
            tracer.record("note", {"modality_budget": decision.to_dict()})
            dag = TaskDAG(self.db, run_id)
            state = RunState(run_id, task, user_id, artifacts, classification, level, profile, budget, contract, decision, index_text, tracer, workspace, CostAccountant(), dag, started=t0)
            self.db.update_run(run_id, status="planning", effort=level.value, problem_type=classification.problem_type.value, contract=contract.to_dict(),
                               budget=budget.to_dict(), artifacts=[a.id for a in artifacts])
            # 4. Plan.
            self._plan(state)
            self.db.update_run(run_id, status="running", plan=state.plan.to_dict() if state.plan else None, budget=budget.to_dict())
            # 5-9. Execute, verify, replan, synthesise, gate.
            return self._drive(state, log)
        except Exception as e:  # noqa: BLE001
            tracer.record("error", {"error": f"{type(e).__name__}: {e}"})
            log.error("run_crashed", error=str(e))
            self.db.update_run(run_id, status="error", result={"error": str(e)})
            return RunResult(run_id, "error", None, message=f"{type(e).__name__}: {e}", trace_path=tracer.path, wall_clock_s=self.clock.monotonic() - t0)

    def resume(self, run_id: str) -> RunResult:
        """Re-attach to an interrupted run from its stored graph (§4.4)."""
        rec = self.db.get_run(run_id)
        if rec is None:
            return RunResult(run_id, "error", None, message="unknown run")
        if rec["status"] in ("verified", "partially_verified", "failed"):
            res = rec.get("result") or {}
            return RunResult(**{**res, "run_id": run_id}) if isinstance(res, dict) and res.get("status") else RunResult(run_id, rec["status"], None, message="run already finished")
        t0 = self.clock.monotonic()
        tracer = Tracer(self.config.traces_dir, run_id, self.clock, self.echo)
        tracer.record("note", {"event": "resumed", "previous_status": rec["status"]})
        log = self.log.bind(run_id)
        workspace = Workspace(self.config.workspaces_dir, run_id)
        artifacts = [a for a in (self.store.get(x) for x in (rec.get("artifacts") or [])) if a]
        task = rec["task"]
        level = EffortLevel(rec.get("effort") or self.config.default_effort.value)
        profile = EFFORT_PROFILES[level]
        budget = Budget.from_dict(rec["budget"]) if isinstance(rec.get("budget"), dict) else budget_for(profile, self.config.capabilities.context_length)
        contract = DefinitionOfDone.from_dict(rec.get("contract") or {})
        classification = TaskClassifier(None, False).heuristic(task, [a.modality for a in artifacts])
        if rec.get("problem_type"):
            classification.problem_type = ProblemType(rec["problem_type"])
        decision = self.budgeter.decide(artifacts, self.config.capabilities.context_length // 2)
        index_text = self.index.render(artifacts, decision)
        dag = TaskDAG(self.db, run_id)
        state = RunState(run_id, task, rec.get("user_id") or "default", artifacts, classification, level, profile, budget, contract, decision, index_text, tracer, workspace, CostAccountant(), dag, started=t0, replans=int(rec.get("replans") or 0))
        state.plan = Plan.from_dict(rec["plan"]) if isinstance(rec.get("plan"), dict) else None
        state.lessons = [l if isinstance(l, str) else json.dumps(l) for l in self.memory.episodic.lessons(5)]
        try:
            if state.plan is None or not dag.nodes():
                self._plan(state)
                self.db.update_run(run_id, plan=state.plan.to_dict() if state.plan else None)
            recovered = dag.recover_running()
            if recovered:
                tracer.record("note", {"recovered_nodes": recovered})
            dag.refresh_ready()
            self.db.update_run(run_id, status="running")
            return self._drive(state, log)
        except Exception as e:  # noqa: BLE001
            tracer.record("error", {"error": f"{type(e).__name__}: {e}"})
            self.db.update_run(run_id, status="error", result={"error": str(e)})
            return RunResult(run_id, "error", None, message=f"{type(e).__name__}: {e}", trace_path=tracer.path)

    def resume_all(self) -> list[RunResult]:
        """Supervisor behaviour on start-up: which runs were in flight? Resume them."""
        return [self.resume(r) for r in self.db.in_flight()]

    # -- phases -------------------------------------------------------------
    def _ingest(self, files: list[str], inline_texts: list[str], workspace: Workspace, tracer: Tracer, user_id: str) -> list[Artifact]:
        artifacts: list[Artifact] = []
        for path in files:
            art = self.ingestor.ingest_path(path)
            try:
                workspace.copy_in(path)
            except Exception:
                pass
            artifacts.append(art)
        for text in inline_texts:
            artifacts.append(self.ingestor.ingest_text(text))
        for a in artifacts:
            try:
                self.memory.index_artifact(self.store, a.id, user_id)
            except Exception:
                pass
            tracer.record("note", {"ingested": a.id, "modality": a.modality.value, "mime": a.mime, "views": [v.name for v in self.store.views(a.id)]})
        return artifacts

    def _classify_and_budget(self, task: str, artifacts: list[Artifact], effort: Optional[EffortLevel], tracer: Tracer) -> tuple[Classification, EffortLevel, EffortProfile, Budget]:
        modalities = [a.modality for a in artifacts]
        classifier = TaskClassifier(self.client, use_model=True, system_prompt=system_prompt)
        classification, usage = classifier.classify(task, modalities)
        tokens = sum(estimate_tokens((self.store.get_view(a.id, "text") or type("v", (), {"text": ""})).text or "") for a in artifacts)
        if effort is not None:
            level, how = effort, "user override"
        elif classification.effort_hint is not None:
            level, how = classification.effort_hint, "classifier"
        else:
            level, how = DifficultyEstimator().estimate(task, modalities, tokens)
        profile = EFFORT_PROFILES[level]
        budget = budget_for(profile, self.config.capabilities.context_length)
        if usage:
            budget.charge(usage)
            self.metrics.record_model_call(usage, 0.0)
        tracer.record("plan", {"phase": "classify", "classification": classification.to_dict(), "effort": level.value, "effort_source": how, "profile": profile.to_dict(), "budget": budget.to_dict()})
        return classification, level, profile, budget

    def _contract(self, task: str, cls: Classification, artifacts: list[Artifact], profile: EffortProfile, tracer: Tracer, run_id: str) -> DefinitionOfDone:
        dod = DefinitionOfDone.default_for(cls.problem_type, bool(artifacts), cls.code_task)
        if profile.level in (EffortLevel.HIGH, EffortLevel.MAX):
            dod.checks = list(dict.fromkeys(dod.checks + ["critic"]))
        if profile.thinking_budget > 0:
            try:
                resp = self.client.complete(ModelRequest(messages=[Message.text("system", system_prompt("contract")),
                                                                   Message.text("user", f"Task: {task[:3000]}\nProblem type: {cls.problem_type.value}\nVerifiers available: {self.verifiers.names()}\nDefault contract: {json.dumps(dod.to_dict())}")],
                                                         temperature=0.0, max_tokens=400, response_format={"type": "json_object"}, metadata={"role": "contract", "run_id": run_id}))
                self.metrics.record_model_call(resp.usage, resp.latency_s)
                data = extract_json(resp.text)
                if isinstance(data, dict):
                    refined = DefinitionOfDone.from_dict(data)
                    refined.checks = [c for c in refined.checks if self.verifiers.has(c)] or dod.checks
                    for must in ("schema", "policy"):
                        if must not in refined.checks:
                            refined.checks.append(must)
                    if artifacts and "evidence" not in refined.checks:
                        refined.checks.append("evidence")
                    refined.min_claim_verification_rate = dod.min_claim_verification_rate
                    dod = refined
            except Exception as e:  # noqa: BLE001
                tracer.record("note", {"contract_refinement_failed": str(e)})
        tracer.record("plan", {"phase": "contract", "contract": dod.to_dict()})
        return dod

    def _planner(self) -> Planner:
        return Planner(self.client, system_prompt, set(self.registry.names()), set(self.verifiers.names()))

    def _plan(self, state: RunState) -> None:
        planner = self._planner()
        plan, usages, errors = planner.plan(state.task, state.index_text, [a.id for a in state.artifacts], state.contract.to_dict(), state.lessons, state.profile, {"run_id": state.run_id})
        for u in usages:
            state.budget.charge(u)
            state.cost.charge(u, "_plan", "planner", state.level.value)
            self.metrics.record_model_call(u, 0.0)
        state.plan = plan
        state.dag.load_plan(plan, state.generation)
        state.tracer.record("plan", {"phase": "plan", "generation": state.generation, "plan": plan.to_dict(), "validation_errors": errors})

    # -- execution ----------------------------------------------------------
    def _tick_budget(self, state: RunState) -> Optional[str]:
        state.budget.used_wall_clock_s = self.clock.monotonic() - state.started
        ex = state.budget.exhausted()
        if ex:
            return "budget exhausted: " + ", ".join(ex)
        if self.session_draining():
            return "session draining"
        return None

    def _upstream_context(self, state: RunState, node: Node) -> str:
        parts = []
        for dep in state.dag.deps(node.id):
            d = state.dag.node(dep)
            if d.output:
                parts.append(f"Result of step {dep} ({truncate(d.description, 100)}):\n" + truncate(json.dumps({"answer": d.output.get("answer"), "claims": d.output.get("claims", [])[:12], "gaps": d.output.get("gaps", [])}, default=str), 3000))
        return "\n\n".join(parts)

    def _tool_ctx(self, state: RunState, node_id: str, role: str) -> ToolContext:
        def spawn(sub_role: str, sub_task: str, opts: dict[str, Any]) -> dict[str, Any]:
            if state.budget.used_branches >= state.budget.max_branches:
                return {"ok": False, "error": "branch budget exhausted"}
            state.budget.used_branches += 1
            res = self._run_agent(state, f"{node_id}.sub{state.budget.used_branches}", sub_role, sub_task, AgentBudget(max(2, opts.get("max_tool_calls", 8) // 2), opts.get("max_tool_calls", 8)))
            return {"answer": res.answer, "claims": [c.to_dict() for c in res.claims], "confidence": res.confidence, "gaps": res.gaps, "cost": res.usage.to_dict()}

        return ToolContext(state.run_id, node_id, self.store, state.workspace, self.sidecars, self.config, self.memory, state.tracer,
                           self.ask_user, spawn if state.profile.sub_agents_allowed else None, self.approve, role)

    def _run_agent(self, state: RunState, node_id: str, role: str, task: str, agent_budget: Optional[AgentBudget] = None, lessons: Optional[list[str]] = None) -> SubAgentResult:
        ctx = self._tool_ctx(state, node_id, role)
        allow = allowlist_for(role, state.profile.sub_agents_allowed, self.ask_user is not None)
        agent = SubAgent(role, self.client, self.registry, ctx, allow, state.profile, self.config.capabilities, system_prompt, state.budget, state.tracer, self.metrics, state.cost, self.ids, agent_budget)
        image_parts = self.index.content_parts(state.artifacts, state.decision, self.config.capabilities.native_image)
        context = f"Definition of done: {json.dumps(state.contract.to_dict())}\n\n{state.index_text}"
        result = agent.run(task, context, image_parts, lessons if lessons is not None else state.lessons)
        result.usage = result.usage  # already charged per call
        state.tracer.record("note", {"agent_result": {"role": role, "status": result.status, "confidence": result.confidence, "claims": len(result.claims), "gaps": result.gaps[:5], "tool_calls": result.tool_calls_used}}, node_id=node_id)
        return result, agent

    def _verify_node(self, state: RunState, node: Node, result: SubAgentResult, extra: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
        names = ["schema"]
        if node.verifier:
            names.append(node.verifier)
        if state.artifacts and result.claims and "evidence" not in names:
            names.append("evidence")
        if state.profile.verification_passes >= 2 and "critic" not in names:
            names.append("critic")
        vctx = VerifyContext(self.store, state.workspace, self.registry, self._tool_ctx(state, node.id, "verifier"), self.client, system_prompt,
                             {"expected_output": node.expected_output, **extra})
        checks = self.verifiers.run_many(names, result, vctx)
        for u in vctx.extra.get("usage", []):
            state.budget.charge(u)
            state.cost.charge(u, node.id, "verifier", state.level.value)
        # Node-level pass: hard verifiers must pass; evidence failures are tolerated when the result already declares them as gaps.
        failed = [c for c in checks if not c.passed]
        hard = [c for c in failed if c.verifier in ("schema", "computation", "tests", "critic")]
        evidence_fail = [c for c in failed if c.verifier == "evidence"]
        claim_checks = [c for c in checks if c.name.startswith("evidence:claim")]
        rate = (sum(1 for c in claim_checks if c.passed) / len(claim_checks)) if claim_checks else 1.0
        passed = not hard and rate >= min(0.5, state.contract.min_claim_verification_rate)
        state.tracer.record("verdict", {"scope": "node", "node": node.id, "passed": passed, "checks": [c.to_dict() for c in checks], "claim_rate": rate}, node_id=node.id)
        return passed, [c.to_dict() for c in checks]

    def _worker(self, state: RunState) -> Callable[[Node], tuple[bool, dict[str, Any], dict[str, Any], Optional[str]]]:
        def work(node: Node) -> tuple[bool, dict[str, Any], dict[str, Any], Optional[str]]:
            upstream = self._upstream_context(state, node)
            task = f"Overall goal: {state.task}\n\nThis step: {node.description}" + (f"\n\nInputs from earlier steps:\n{upstream}" if upstream else "")
            if node.artifacts:
                task += f"\n\nArtifacts for this step: {node.artifacts}"
            samples = state.profile.samples if (state.classification.problem_type == ProblemType.CONVERGENT and not state.dag.dependents(node.id)) else 1
            results: list[SubAgentResult] = []
            agent = None
            if samples > 1:
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=min(samples, state.profile.max_parallel_workers)) as pool:
                    futs = [pool.submit(self._run_agent, state, f"{node.id}#{i}", node.role, task) for i in range(samples)]
                    for f in futs:
                        r, agent = f.result()
                        results.append(r)
                keys = [normalise_answer(r.answer if isinstance(r.answer, str) else json.dumps(r.answer, default=str)) for r in results]
                voteable = [k for k in keys if k]
                if voteable:
                    from collections import Counter

                    winner, votes = Counter(voteable).most_common(1)[0]
                    result = results[keys.index(winner)]
                    agreement = votes / len(results)
                else:
                    result = max(results, key=lambda r: (len([c for c in r.claims if c.evidence]), r.confidence))
                    agreement = 1.0 / len(results)
                state.tracer.record("note", {"self_consistency": {"samples": samples, "agreement": agreement, "keys": keys}}, node_id=node.id)
            else:
                result, agent = self._run_agent(state, node.id, node.role, task)
                agreement = None
            extra: dict[str, Any] = {"tool_outputs": agent.tool_outputs if agent else [], "samples": samples}
            if agreement is not None:
                extra["agreement"] = agreement
            passed, checks = self._verify_node(state, node, result, extra)
            output = {"answer": result.answer, "claims": [c.to_dict() for c in result.claims], "confidence": result.confidence, "gaps": result.gaps,
                      "status": result.status, "checks": checks, "agreement": agreement, "artifacts_produced": result.artifacts_produced}
            cost = {"usage": result.usage.to_dict(), "tool_calls": result.tool_calls_used, "iterations": result.iterations}
            self.memory.episodic.record(state.run_id, node.id, "node_result", {"passed": passed, "answer": truncate(json.dumps(result.answer, default=str), 500)})
            if not passed:
                issues = [c["detail"] for c in checks if not c["passed"]]
                lesson = agent.reflect(task, result, issues) if agent else None
                if lesson:
                    state.lessons.append(lesson)
                    self.memory.episodic.record(state.run_id, node.id, "lesson", lesson)
                return False, output, cost, "verification failed: " + "; ".join(issues)[:500]
            return True, output, cost, None

        return work

    MAX_ESCALATIONS = 2

    def _replan(self, state: RunState, log: StructuredLogger, force: bool = False) -> bool:
        if state.replans >= state.budget.max_replans and not force:
            return False
        failed = [n for n in state.dag.nodes() if n.status in (NodeStatus.FAILED, NodeStatus.STALE, NodeStatus.BLOCKED)]
        if not failed:
            return False
        state.replans += 1
        state.generation += 1
        state.budget.used_replans = state.replans
        self.db.update_run(state.run_id, status="replanning", replans=state.replans)
        for n in failed:
            if n.status == NodeStatus.FAILED:
                state.dag.mark_stale(n.id, "failed verification")
        stale = [n for n in state.dag.nodes() if n.status in (NodeStatus.STALE, NodeStatus.BLOCKED)]
        done = [{"id": n.id, "description": n.description, "answer": truncate(json.dumps((n.output or {}).get("answer"), default=str), 300)} for n in state.dag.nodes(NodeStatus.DONE)]
        failed_desc = [{"id": n.id, "description": n.description, "error": n.error, "gaps": (n.output or {}).get("gaps", [])} for n in stale]
        sub, usages, errors = self._planner().replan(state.task, state.index_text, [a.id for a in state.artifacts], state.contract.to_dict(), failed_desc, done, state.lessons, state.generation, state.profile, {"run_id": state.run_id})
        for u in usages:
            state.budget.charge(u)
            state.cost.charge(u, "_replan", "planner", state.level.value)
        new_ids = state.dag.splice(sub, [n.id for n in stale], state.generation)
        state.blocked_reason = None
        state.tracer.record("plan", {"phase": "replan", "generation": state.generation, "replaced": [n.id for n in stale], "new": new_ids, "subplan": sub.to_dict(), "errors": errors})
        log.info("replanned", generation=state.generation, replaced=[n.id for n in stale], new=new_ids)
        self.db.update_run(state.run_id, status="running", plan={**(state.plan.to_dict() if state.plan else {}), "generation": state.generation, "splices": new_ids})
        return bool(new_ids)

    def _execute(self, state: RunState, log: StructuredLogger) -> None:
        scheduler = Scheduler(self.db, state.dag, state.profile.max_parallel_workers, state.tracer, self._dispatch_hook)
        worker = self._worker(state)
        while True:
            reason = self._tick_budget(state)
            if reason:
                state.stop_reason = reason
                state.tracer.record("budget", {"stop": reason, "budget": state.budget.to_dict()})
                return
            if state.dag.all_done():
                return
            if not state.dag.ready_nodes():
                if state.dag.terminal():
                    if not self._replan(state, log):
                        state.blocked_reason = "nodes failed verification and no replan budget is left"
                        return
                    continue
                state.dag.refresh_ready()
                if not state.dag.ready_nodes():
                    state.blocked_reason = "no runnable nodes"
                    return
            scheduler.step(worker, lambda: self._tick_budget(state) is not None)
            self.db.update_run(state.run_id, budget=state.budget.to_dict())

    _dispatch_hook: Optional[Callable[[Node], None]] = None  # tests inject crashes here

    # -- synthesis and gate --------------------------------------------------
    def _synthesise(self, state: RunState) -> SubAgentResult:
        done = state.dag.nodes(NodeStatus.DONE)
        if len(done) == 1 and done[0].output:  # delegation cost (§10.4): one node, no synthesis call needed
            o = done[0].output
            return SubAgentResult(o.get("answer"), _dedupe_claims([Claim.from_dict(c) for c in o.get("claims", [])]), confidence=float(o.get("confidence", 0.5)), gaps=list(o.get("gaps", [])))
        if not done:
            failed = state.dag.nodes(NodeStatus.FAILED)
            best = max(failed, key=lambda n: float((n.output or {}).get("confidence", 0)), default=None)
            if best and best.output:
                o = best.output
                return SubAgentResult(o.get("answer"), [Claim.from_dict(c) for c in o.get("claims", [])], confidence=min(0.3, float(o.get("confidence", 0.3))),
                                      gaps=list(o.get("gaps", [])) + [f"best available result failed verification: {best.error}"], status="failed")
            return SubAgentResult(None, [], confidence=0.0, gaps=["no step produced a result"], status="failed")
        payload = {"task": state.task, "results": [{"step": n.id, "description": n.description, "answer": n.output.get("answer"), "claims": n.output.get("claims", []), "gaps": n.output.get("gaps", [])} for n in done if n.output]}
        result, _ = self._run_agent(state, "_synthesis", "synthesiser", json.dumps(payload, default=str)[:20_000], AgentBudget(3, 4), lessons=[])
        if not result.claims:  # keep upstream citations if the synthesiser dropped them
            result.claims = [Claim.from_dict(c) for n in done if n.output for c in n.output.get("claims", [])]
        for n in done:
            result.gaps += [g for g in (n.output or {}).get("gaps", []) if g not in result.gaps]
        result.claims = _dedupe_claims(result.claims)
        return result

    def _probe(self, state: RunState, result: SubAgentResult) -> None:
        """Adversarial self-probe (§9.3): what would contradict this, and is it in the material?"""
        if state.profile.level not in (EffortLevel.HIGH, EffortLevel.MAX) or not result.claims:
            return
        try:
            resp = self.client.complete(ModelRequest(messages=[Message.text("system", system_prompt("probe")), Message.text("user", json.dumps({"answer": result.answer, "claims": [c.to_dict() for c in result.claims][:20], "index": truncate(state.index_text, 6000)}, default=str))],
                                                     temperature=0.0, max_tokens=500, metadata={"role": "probe", "run_id": state.run_id}))
            state.budget.charge(resp.usage)
            data = extract_json(resp.text) or {}
            state.tracer.record("verdict", {"scope": "probe", "data": data})
            if data.get("found"):
                result.gaps.append(f"self-probe found contradicting evidence: {truncate(str(data.get('detail', '')), 300)}")
        except Exception as e:  # noqa: BLE001
            state.tracer.record("note", {"probe_failed": str(e)})

    def _gate(self, state: RunState, result: SubAgentResult) -> tuple[Verdict, EvidenceBundle]:
        agreement = None
        samples = 1
        for n in state.dag.nodes(NodeStatus.DONE):
            if n.output and n.output.get("agreement") is not None:
                agreement = n.output["agreement"]
                samples = state.profile.samples
        vctx = VerifyContext(self.store, state.workspace, self.registry, self._tool_ctx(state, "_gate", "verifier"), self.client, system_prompt,
                             {"expected_output": "text", "agreement": agreement, "samples": samples, "tool_outputs": []})
        verdict, bundle = self.gate.evaluate(result, state.contract, vctx)
        for u in vctx.extra.get("usage", []):
            state.budget.charge(u)
        state.tracer.record("gate", {"status": verdict.status.value, "summary": verdict.summary, "checks": [c.to_dict() for c in verdict.checks], "unverified": verdict.unverified})
        return verdict, bundle

    def _drive(self, state: RunState, log: StructuredLogger) -> RunResult:
        while True:
            self._execute(state, log)
            self.db.update_run(state.run_id, status="synthesising", budget=state.budget.to_dict())
            result = self._synthesise(state)
            self._probe(state, result)
            verdict, bundle = self._gate(state, result)
            if verdict.status == VerdictStatus.FAILED and state.stop_reason is None and self.config.allow_effort_escalation:
                nxt = escalate(state.level)
                if nxt is not None and state.escalations < self.MAX_ESCALATIONS and not state.budget.near_exhaustion():
                    # Escalate effort and replan the whole graph with the lessons learned (§6.2).
                    state.escalations += 1
                    state.level, state.profile = nxt, EFFORT_PROFILES[nxt]
                    bigger = budget_for(state.profile, self.config.capabilities.context_length)
                    for k in ("used_tokens", "used_tool_calls", "used_model_calls", "used_cost_usd", "used_replans"):
                        setattr(bigger, k, getattr(state.budget, k))
                    state.budget = bigger
                    state.lessons.append(f"Previous attempt at effort {state.level.value} failed the trust gate: {verdict.summary}")
                    for n in state.dag.nodes(NodeStatus.DONE):
                        state.dag.mark_stale(n.id, "effort escalation")
                    state.tracer.record("plan", {"phase": "escalate", "to": nxt.value, "verdict": verdict.summary})
                    self.db.update_run(state.run_id, effort=nxt.value, budget=state.budget.to_dict())
                    if self._replan(state, log, force=True):
                        continue
            break
        return self._finish(state, result, verdict, bundle, log)

    def _finish(self, state: RunState, result: SubAgentResult, verdict: Verdict, bundle: EvidenceBundle, log: StructuredLogger) -> RunResult:
        status = verdict.status.value
        if state.stop_reason and verdict.status != VerdictStatus.VERIFIED:
            result.gaps.append(f"stopped early: {state.stop_reason}")
        if state.blocked_reason and verdict.status != VerdictStatus.VERIFIED:
            result.gaps.append(f"stopped: {state.blocked_reason}")
        if verdict.status != VerdictStatus.FAILED:
            try:
                stored = self.memory.commit_verified(state.run_id, result.claims, state.user_id)
                if stored:
                    state.tracer.record("note", {"memory_commit": [f.id for f in stored]})
                if verdict.status == VerdictStatus.VERIFIED and state.plan:
                    self.memory.procedural.save("plans", state.classification.problem_type.value + "_" + "_".join(m.value for m in sorted(set(state.classification.modalities), key=lambda m: m.value)), state.plan.to_dict(), f"verified plan from run {state.run_id}")
            except Exception as e:  # noqa: BLE001
                state.tracer.record("note", {"memory_commit_failed": str(e)})
        if self.registry.failures:
            self.memory.procedural.save("tool_failures", "latest", {k: v[-5:] for k, v in self.registry.failures.items()}, "tool misuse log for description improvement")
        wall = self.clock.monotonic() - state.started
        message = verdict.summary if verdict.status != VerdictStatus.FAILED else f"Honest failure report: {verdict.summary}. Missing: {verdict.unverified[:5]}"
        rr = RunResult(state.run_id, status, result.answer, [c.to_dict() for c in result.claims], verdict.to_dict(), bundle.items, list(dict.fromkeys(result.gaps)), state.level.value,
                       state.classification.problem_type.value, state.cost.report(), state.budget.to_dict(), state.tracer.path, state.plan.to_dict() if state.plan else None,
                       state.dag.summary(), state.replans, state.escalations, [a.id for a in state.artifacts], wall, message)
        self.db.update_run(state.run_id, status=status, result=rr.to_dict(), budget=state.budget.to_dict(), replans=state.replans)
        state.tracer.record("note", {"event": "run_finished", "status": status, "wall_clock_s": round(wall, 2), "cost": state.cost.total.to_dict()})
        log.info("run_finished", status=status, effort=state.level.value, replans=state.replans, wall_clock_s=round(wall, 2))
        return rr


def _dedupe_claims(claims: list[Claim]) -> list[Claim]:
    seen: set[str] = set()
    out: list[Claim] = []
    for c in claims:
        key = json.dumps({"t": c.text.strip().lower(), "e": [(e.artifact_id, e.kind, sorted(e.locator.items())) for e in c.evidence]}, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out
