"""The agent loop: send the context and the tool schemas; run the tool calls; stop on plain text.

No planner, no classifier, no gate. Everything else here is what goes into
the request (``ContextAssembler``), what the model may do (``ToolRegistry``,
``Permissions``, hooks), and what happens when a budget runs out
(``BudgetTracker``). State goes to disk before the next model call.
"""
from __future__ import annotations

import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from ..adapters.providers.base import ROLE_MARKER_PREFIX, Provider, ProviderError, role_marker
from ..adapters.toolchain import Toolchain
from ..domain.errors import ErrorCategory, HarnessError, ToolFailure
from ..domain.evidence import EvidenceStore
from ..domain.types import ContentPart, Message, Reply, ToolCall, Usage
from ..domain.utils import Clock, truncate
from ..domain.workspace import Workspace
from ..tools.registry import PermissionPolicy, ToolContext, ToolRegistry, ToolResult
from .budgets import Budget, BudgetExhausted, BudgetTracker
from .compaction import Compactor
from .context import ContextAssembler, ContextTooLarge
from .hooks import HookOutcome, HookRunner
from .permissions import Permissions
from .sessions import Session
from .skills import SkillIndex
from .subagents import AgentIndex, SubagentRunner

MAX_PROVIDER_RETRIES = 2
RETRY_BACKOFF_S = (1.0, 2.0)
TURN_ERRORS_MAX = 50
EVENT_PREVIEW_CHARS = 300
MAX_ITERATIONS_NOTE = (
    "Stopped: the per-turn limit on model calls was reached before the task was finished. "
    "What was done is recorded above; start a new turn to continue the rest."
)
PARSE_ERROR_FEEDBACK = "[harness] Your tool block could not be parsed: {errors}. Emit one valid ```tool block, or reply in plain prose if you are done."

_toolchain_cache: Toolchain | None = None


def detect_toolchain() -> Toolchain:
    global _toolchain_cache
    if _toolchain_cache is None:
        _toolchain_cache = Toolchain.detect()
    return _toolchain_cache


@dataclass
class TurnResult:
    text: str
    tool_calls: int = 0
    iterations: int = 0
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = "done"  # done | max_iterations | budget | error | cancelled
    errors: list[str] = field(default_factory=list)
    attachments: list[ContentPart] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tool_calls": self.tool_calls,
            "iterations": self.iterations,
            "usage": self.usage.to_dict(),
            "stop_reason": self.stop_reason,
            "errors": list(self.errors),
            "attachments": [a.to_dict() for a in self.attachments],
        }


class AgentLoop:
    def __init__(
        self,
        provider: Provider,
        registry: ToolRegistry,
        workspace: Workspace,
        session: Session,
        *,
        system_prompt: str,
        permissions: PermissionPolicy | None = None,
        hooks: HookRunner | None = None,
        budget: BudgetTracker | None = None,
        tracer: Any = None,
        metrics: Any = None,
        skills: SkillIndex | None = None,
        agents: AgentIndex | None = None,
        allowlist: set[str] | None = None,
        effort: str = "medium",
        max_tokens: int = 4096,
        compactor: Compactor | None = None,
        context: ContextAssembler | None = None,
        ask_user: Callable[[str], str] | None = None,
        ask_permission: Callable[..., bool] | None = None,
        role: str = "main",
        depth: int = 0,
        subagents: SubagentRunner | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        clock: Clock | None = None,
        evidence_store: EvidenceStore | None = None,
        toolchain: Toolchain | None = None,
        settings: dict[str, Any] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.workspace = workspace
        self.session = session
        self.system_prompt = system_prompt if ROLE_MARKER_PREFIX in system_prompt else f"{role_marker(role)}\n{system_prompt}"
        self.permissions: PermissionPolicy = permissions if permissions is not None else Permissions("ask", workspace=workspace)
        self.settings: dict[str, Any] = settings if settings is not None else _workspace_settings(workspace)
        self.tracer = tracer
        self.metrics = metrics
        self.hooks = hooks if hooks is not None else (HookRunner(self.settings, workspace, tracer) if self.settings.get("hooks") else None)
        self.clock = clock or Clock()
        self.budget = budget if budget is not None else BudgetTracker(Budget(), clock=self.clock)
        self.skills = skills
        self.agents = agents
        self.allowlist = set(allowlist) if allowlist is not None else None
        self.effort = effort
        self.max_tokens = max_tokens
        self.compactor = compactor if compactor is not None else Compactor(provider)
        self.context = context if context is not None else ContextAssembler()
        self.ask_user = ask_user
        self.ask_permission = ask_permission
        self.role = role
        self.depth = depth
        self.on_event = on_event
        self.evidence = evidence_store
        self._toolchain = toolchain
        self._sleep = sleep or time.sleep
        self.subagents = subagents if subagents is not None else SubagentRunner(self._child_loop, max_depth=self.budget.budget.max_depth)
        self.node_id: str = str(session.meta.get("unit_id") or (session.session_id if depth > 0 else ""))
        self._cancel = threading.Event()
        self._session_started = False
        self._overhead_tokens = 0
        self._reset_turn()

    @classmethod
    def resume(cls, workspace: Workspace, session_id: str, *, provider: Provider, registry: ToolRegistry, **kw: Any) -> AgentLoop:
        return cls(provider, registry, workspace, Session.load(workspace, session_id), **kw)

    # ------------------------------------------------------------------ wiring
    def tool_context(self) -> ToolContext:
        if self.evidence is None:
            self.evidence = EvidenceStore(self.workspace.path("evidence"))
        if self._toolchain is None:
            self._toolchain = detect_toolchain()
        return ToolContext(
            workspace=self.workspace,
            session_id=self.session.session_id,
            run_id=str(self.session.meta.get("run_id", "")),
            node_id=self.node_id,
            role=self.role,
            provider=self.provider,
            evidence=self.evidence,
            settings=self.settings,
            toolchain=self._toolchain,
            permissions=self.permissions,
            ask_user=self.ask_user,
            ask_permission=self.ask_permission,
            spawn=self.spawn,
            tracer=self.tracer,
            metrics=self.metrics,
            session=self.session,
            skills=self.skills,
        )

    def cancel(self) -> None:
        self._cancel.set()

    # ------------------------------------------------------------------ the turn
    def run_turn(self, user_text: str, parts: list[ContentPart] | None = None, extra_context: dict[str, Any] | None = None) -> TurnResult:
        self.budget.start_turn()
        self._cancel.clear()
        self._reset_turn()
        self._session_start()
        self._resolve_dangling_tool_calls("the previous turn was interrupted before the result was recorded")
        self.session.append(Message.user(user_text, parts))
        stop_reason = "done"
        try:
            while True:
                if self._cancel.is_set():
                    stop_reason = "cancelled"
                    break
                self.budget.check("iteration")
                self._compact_if_under_pressure()
                reply = self._complete(extra_context)
                self._record_reply(reply)
                if reply.tool_calls:
                    self._run_tools(reply.tool_calls)
                    self._feedback_parse_errors(reply)  # a sibling call that did not parse is reported next to the results
                elif self._feedback_parse_errors(reply):
                    pass
                else:
                    if reply.finish_reason == "length":
                        self._note_error("the reply was cut by the output limit (finish_reason=length); the answer may be incomplete")
                        self._emit({"type": "warning", "session_id": self.session.session_id, "text": "reply cut by the output limit"})
                    break
                if self.budget.iterations >= self.budget.budget.max_iterations:
                    stop_reason = "max_iterations"
                    self._say(MAX_ITERATIONS_NOTE)
                    break
        except BudgetExhausted as e:
            stop_reason = "budget"
            self._fail(e.error)
        except (ContextTooLarge, ProviderError) as e:
            stop_reason = "error"
            self._fail(e.error)
        except Exception as e:  # noqa: BLE001 - the session is persisted and the failure is reported, never swallowed
            stop_reason = "error"
            self._fail(HarnessError("loop_error", f"{type(e).__name__}: {e}"[:600], ErrorCategory.execution_failure, remediation="Inspect the trace; this is a harness fault, not a model fault.", details={"traceback": traceback.format_exc()[-2000:]}))
        finally:
            self._stop_hook(stop_reason)
            self.session.save()
        return TurnResult(
            text=self._turn_text,
            tool_calls=self._turn_tool_calls,
            iterations=self.budget.iterations,
            usage=self._turn_usage,
            stop_reason=stop_reason,
            errors=list(self._turn_errors),
            attachments=list(self._turn_attachments),
        )

    # ------------------------------------------------------------------ stage: the request
    def _complete(self, extra_context: dict[str, Any] | None) -> Reply:
        request = self._build_request(extra_context, allow_compaction=True)
        attempt = 0
        compacted = False
        while True:
            try:
                return self.provider.complete(request)
            except ProviderError as e:
                compactable = bool((getattr(e.error, "details", None) or {}).get("compactable", True))  # a media or system-prompt ceiling is not helped by dropping history
                if e.category == ErrorCategory.truncated and not compacted and compactable:
                    compacted = True
                    self._compact(force=True, reason="provider refused the prompt size")
                    request = self._build_request(extra_context, allow_compaction=False)
                    continue
                if e.retryable and attempt < MAX_PROVIDER_RETRIES:
                    delay = RETRY_BACKOFF_S[min(attempt, len(RETRY_BACKOFF_S) - 1)]
                    self._trace("retry", {"attempt": attempt + 1, "delay_s": delay, "error": e.error.to_dict()})
                    self._sleep(delay)
                    attempt += 1
                    continue
                raise

    def _build_request(self, extra_context: dict[str, Any] | None, allow_compaction: bool) -> Any:
        caps = self.provider.caps
        # compact schemas only where bytes are scarce: parameter descriptions are how the model learns units and ranges
        compact = self.context.config.compact_schemas and bool(caps.max_prompt_bytes)
        try:
            request = self.context.build_request(
                self.session,
                self.workspace,
                system_prompt=self.system_prompt,
                tool_schemas=self.registry.openai_schemas(self.allowlist, compact=compact),
                effort=self.effort,  # type: ignore[arg-type]
                max_tokens=self.max_tokens,
                role=self.role,
                skills_index=self.skills.index_text() if self.skills is not None else "",
                agents_index=self.agents.index_text() if self.agents is not None and "agent" in self._visible_tools() else "",
                extra_context=extra_context,
                caps=caps,
            )
        except ContextTooLarge:
            if not allow_compaction:
                raise
            self._compact(force=True, reason="context too large for the provider")
            return self._build_request(extra_context, allow_compaction=False)
        self._overhead_tokens = sum(m.tokens() for m in request.messages if m.role == "system") + 60 * len(request.tools)
        return request

    def _visible_tools(self) -> set[str]:
        names = set(self.registry.names())
        return names if self.allowlist is None else names & self.allowlist

    # ------------------------------------------------------------------ stage: the reply
    def _record_reply(self, reply: Reply) -> None:
        usage = reply.usage
        self.budget.charge_model(usage)
        self.session.add_usage(usage)
        self._turn_usage = self._turn_usage + usage
        names = [tc.name for tc in reply.tool_calls]
        self._trace("model_call", {"role": self.role, "usage": usage.to_dict(), "tool_calls": names, "finish_reason": reply.finish_reason, "latency_s": round(reply.latency_s, 3), "request_id": reply.request_id, "model": reply.model})
        if self.metrics is not None:
            self.metrics.record_model_call(usage, reply.latency_s)
        message = reply.as_message()
        signatures = reply.raw.get("thought_signatures") if isinstance(reply.raw, dict) else None
        if isinstance(signatures, dict) and signatures:  # Gemini must see its own thought signatures again after a resume
            message.meta["thought_signatures"] = dict(signatures)
        self.session.append(message)
        if reply.text:
            self._turn_text = reply.text
        self._emit({"type": "model_call", "session_id": self.session.session_id, "role": self.role, "iteration": self.budget.iterations, "text": truncate(reply.text, EVENT_PREVIEW_CHARS), "tool_calls": names, "usage": usage.to_dict()})

    def _feedback_parse_errors(self, reply: Reply) -> bool:
        """A text-protocol reply with a malformed tool block gets one chance to fix it (bounded by the iteration budget)."""
        errors = (reply.raw or {}).get("tool_parse_errors") if reply.raw else None
        if not errors:
            return False
        self.session.append(Message.user(PARSE_ERROR_FEEDBACK.format(errors="; ".join(str(e) for e in errors)[:500])))
        self._turn_errors.append("tool block parse error: " + "; ".join(str(e) for e in errors)[:200])
        return True

    # ------------------------------------------------------------------ stage: the tools
    def _run_tools(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        self.budget.check("tool")
        ctx = self.tool_context()
        planned = [self._pre_tool(tc) for tc in tool_calls]
        calls = [(tc.name, args, tc.id) for tc, args, blocked in planned if blocked is None]
        ran = iter(self.registry.call_parallel(calls, ctx, self.allowlist, key_scope=f"turn{self.session.seq}:") if calls else [])
        results: list[ToolResult] = []
        for tc, args, blocked in planned:
            result = ToolResult(tc.id, tc.name, False, None, blocked) if blocked is not None else next(ran)
            self._post_tool(tc, args, result)
            results.append(result)
        self.budget.charge_tools(len(tool_calls))
        self._turn_tool_calls += len(tool_calls)
        return results

    def _pre_tool(self, tc: ToolCall) -> tuple[ToolCall, dict[str, Any], HarnessError | None]:
        if self.hooks is None or not self.hooks.has("PreToolUse"):
            return tc, tc.arguments, None
        outcome = self.hooks.run("PreToolUse", {"tool_name": tc.name, "arguments": tc.arguments, "call_id": tc.id, "session_id": self.session.session_id, "role": self.role})
        if not outcome.allow:
            error = HarnessError(
                "permission_denied",
                f"call to {tc.name!r} was blocked by a PreToolUse hook: {outcome.message or 'no reason given'}",
                ErrorCategory.permission_denied,
                remediation="Do not repeat the call unchanged; follow the hook's message or choose another approach.",
            )
            return tc, tc.arguments, error
        if outcome.updated_arguments is not None:
            return tc, dict(outcome.updated_arguments), None
        return tc, tc.arguments, None

    def _post_tool(self, tc: ToolCall, args: dict[str, Any], result: ToolResult) -> None:
        text = result.as_text()
        if self.hooks is not None and self.hooks.has("PostToolUse"):
            outcome = self.hooks.run("PostToolUse", {"tool_name": tc.name, "arguments": args, "call_id": tc.id, "ok": result.ok, "output_preview": truncate(text, 1000), "error": result.error.to_dict() if result.error else None, "session_id": self.session.session_id, "role": self.role})
            if outcome.message:
                text += f"\n[hook] {outcome.message}"
        meta: dict[str, Any] = {}
        if result.compacted and isinstance(result.output, dict) and result.output.get("full_output_path"):
            meta["persisted_path"] = result.output["full_output_path"]
        if args is not tc.arguments:
            meta["arguments_updated_by_hook"] = True
        if not result.ok and result.error is not None:
            self._note_error(f"{tc.name}: {result.error.code}: {result.error.message}")
        self._turn_attachments.extend(result.attachments)
        if self.metrics is not None and getattr(self.registry, "metrics", None) is None:
            self.metrics.record_tool_call(tc.name, result.ok, result.duration_s)
        self.session.append(Message.tool(tc.id, tc.name, text, self._attachment_parts(result.attachments), **meta))
        self._emit({"type": "tool_call", "session_id": self.session.session_id, "name": tc.name, "call_id": tc.id, "ok": result.ok, "duration_s": round(result.duration_s, 3), "preview": truncate(text, EVENT_PREVIEW_CHARS)})

    def _attachment_parts(self, attachments: list[ContentPart]) -> list[ContentPart]:
        caps = self.provider.caps
        return [p if caps.accepts(p.type) else ContentPart.of_text(p.text or f"[{p.type}: {p.path}]") for p in attachments]

    # ------------------------------------------------------------------ stage: compaction
    def _compact_if_under_pressure(self) -> None:
        caps = self.provider.caps
        if self.compactor.pressure(self.session, caps.context_tokens, self._overhead_tokens) >= self.compactor.limits.pressure_threshold:
            self._compact(force=False, reason="context pressure")

    def _compact(self, force: bool, reason: str) -> None:
        report = self.compactor.maybe_compact(self.session, self.provider.caps.context_tokens, force=force, overhead_tokens=self._overhead_tokens)
        if not (report.tool_results_cleared or report.notes_added or report.rolled_up):
            return  # nothing could be freed (the recent window is protected): nothing to report
        data = {"reason": reason, "forced": force, **report.to_dict()}
        self._trace("compaction", data)
        if self.hooks is not None and self.hooks.has("Compaction"):
            self.hooks.run("Compaction", {"session_id": self.session.session_id, **data})
        self._emit({"type": "compaction", "session_id": self.session.session_id, **data})

    # ------------------------------------------------------------------ subagents
    def spawn(self, agent_name: str, task: str, **opts: Any) -> dict[str, Any]:
        definition = self.agents.get(agent_name) if self.agents is not None else None
        if definition is None:
            available = ", ".join(self.agents.names()) if self.agents is not None else "none"
            raise ToolFailure("not_found", f"unknown agent {agent_name!r}; available: {available}", ErrorCategory.not_found, remediation="Choose one of the listed agents, or do the task here.")
        try:
            self.budget.check("spawn")
        except BudgetExhausted as e:  # the tool contract: a typed error with the limit's remediation, not a stack trace
            raise ToolFailure(e.error.code, e.error.message, e.error.category, remediation=e.error.remediation, **e.error.details) from e
        self.budget.charge_spawn()
        if opts.get("max_turns"):
            definition = replace(definition, max_turns=int(opts["max_turns"]))
        return self.subagents.spawn(definition, task, parent=self, depth=self.depth).to_dict()

    def _child_loop(self, *, session: Session, system_prompt: str, allowlist: set[str], role: str, effort: str, max_iterations: int, depth: int, permission_mode: str | None = None, model: str | None = None) -> AgentLoop:
        budget = BudgetTracker(replace(self.budget.budget, max_iterations=max_iterations), clock=self.clock)
        permissions = self.permissions
        if permission_mode and isinstance(permissions, Permissions):
            permissions = permissions.with_mode(permission_mode)
        return AgentLoop(
            self.provider, self.registry, self.workspace, session,
            system_prompt=system_prompt, permissions=permissions, hooks=self.hooks, budget=budget, tracer=self.tracer, metrics=self.metrics,
            skills=self.skills, agents=self.agents, allowlist=allowlist, effort=effort, max_tokens=self.max_tokens, compactor=self.compactor,
            context=self.context, ask_user=None, ask_permission=self.ask_permission, role=role, depth=depth, subagents=self.subagents,
            on_event=self.on_event, clock=self.clock, evidence_store=self.evidence, toolchain=self._toolchain, settings=self.settings, sleep=self._sleep,
        )

    # ------------------------------------------------------------------ hooks and bookkeeping
    def _session_start(self) -> None:
        if self._session_started:
            return
        self._session_started = True
        source = "resume" if self.session.meta.get("started") else "startup"
        self.session.meta["started"] = True
        if self.hooks is not None and self.hooks.has("SessionStart"):
            outcome = self.hooks.run("SessionStart", {"session_id": self.session.session_id, "mode": self.session.mode, "role": self.role, "source": source})
            if outcome.message:
                self.session.meta.setdefault("hook_messages", []).append(outcome.message)

    def _stop_hook(self, stop_reason: str) -> HookOutcome | None:
        if self.hooks is None or not self.hooks.has("Stop"):
            return None
        return self.hooks.run("Stop", {"session_id": self.session.session_id, "role": self.role, "stop_reason": stop_reason, "text": truncate(self.session.last_text("assistant"), 2000)})

    def _fail(self, error: HarnessError) -> None:
        text = error.message + (f" {error.remediation}" if error.remediation else "")
        self._note_error(error.message)
        self._trace("error", error.to_dict())
        self._resolve_dangling_tool_calls(error.message)
        self._say(text)

    def _say(self, text: str) -> None:
        """The harness speaks as the assistant (a limit note, an error); persisted like any other message."""
        self.session.append(Message.assistant(text))
        self._turn_text = text

    def _resolve_dangling_tool_calls(self, reason: str) -> int:
        """Every tool call gets a result before the next model call: providers reject a conversation without one."""
        missing = self.session.dangling_tool_calls()
        for tc in missing:
            error = HarnessError("interrupted", f"no result was recorded for this call: {reason}", ErrorCategory.cancelled, remediation="Repeat the call if the result is still needed.")
            self.session.append(Message.tool(tc.id, tc.name, ToolResult(tc.id, tc.name, False, None, error).as_text(), interrupted=True))
        return len(missing)

    def _note_error(self, text: str) -> None:
        if len(self._turn_errors) < TURN_ERRORS_MAX:
            self._turn_errors.append(truncate(text, 300))

    def _reset_turn(self) -> None:
        self._turn_text = ""
        self._turn_errors: list[str] = []
        self._turn_attachments: list[ContentPart] = []
        self._turn_tool_calls = 0
        self._turn_usage = Usage()

    def _trace(self, kind: str, data: dict[str, Any]) -> None:
        if self.tracer is None:
            return
        try:
            self.tracer.record(kind, data, node_id=self.node_id or None)
        except Exception:  # noqa: BLE001
            pass

    def _emit(self, event: dict[str, Any]) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(event)
        except Exception:  # noqa: BLE001 - a UI callback must never break the loop
            pass


def _workspace_settings(workspace: Workspace) -> dict[str, Any]:
    try:
        return workspace.settings()
    except Exception:  # noqa: BLE001
        return {}
