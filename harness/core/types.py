"""Typed schemas shared by every layer.

Plain dataclasses with ``to_dict``/``from_dict`` so they serialise to JSON
for the state database and the trace without any third-party library.
"""
from __future__ import annotations

import dataclasses
import enum
from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = [
    "Modality",
    "ContentPart",
    "Message",
    "ToolCall",
    "ToolResult",
    "ToolError",
    "Usage",
    "Evidence",
    "Claim",
    "PlanStep",
    "Plan",
    "VerdictStatus",
    "Verdict",
    "Check",
    "Budget",
    "NodeStatus",
    "ProblemType",
    "SubAgentResult",
    "asdict",
]


def asdict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: asdict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: asdict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [asdict(v) for v in obj]
    return obj


class _Serialisable:
    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)  # type: ignore[arg-type]
        return _enum_to_value(d)

    @classmethod
    def from_dict(cls, d: dict[str, Any]):  # type: ignore[no-untyped-def]
        names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
        return cls(**{k: v for k, v in d.items() if k in names})  # type: ignore[call-arg]


def _enum_to_value(x: Any) -> Any:
    if isinstance(x, enum.Enum):
        return x.value
    if isinstance(x, dict):
        return {k: _enum_to_value(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_enum_to_value(v) for v in x]
    return x


class Modality(str, enum.Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    UNKNOWN = "unknown"


@dataclass
class ContentPart(_Serialisable):
    """One part of a message: text, or a reference to a media artifact.

    ``artifact_id`` points into the artifact store. ``data_url`` is filled
    only when the part is actually sent to a model that accepts the media
    natively; otherwise the derived text view is used.
    """

    type: str  # "text" | "image" | "audio" | "video" | "artifact_ref"
    text: Optional[str] = None
    artifact_id: Optional[str] = None
    data_url: Optional[str] = None
    mime: Optional[str] = None
    untrusted: bool = False  # content extracted from media or files

    @staticmethod
    def of_text(text: str, untrusted: bool = False) -> "ContentPart":
        return ContentPart(type="text", text=text, untrusted=untrusted)


@dataclass
class ToolCall(_Serialisable):
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolError(_Serialisable):
    """Typed, model-readable tool failure. Never a stack trace."""

    code: str  # "invalid_arguments" | "not_found" | "timeout" | "permission_denied" | "unavailable" | "execution_error"
    message: str
    retryable: bool = False
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult(_Serialisable):
    call_id: str
    name: str
    ok: bool
    output: Any = None
    error: Optional[ToolError] = None
    artifact_ids: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    compacted: bool = False  # output was summarised/truncated; full result stored as artifact

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ToolResult":
        d = dict(d)
        if d.get("error") and isinstance(d["error"], dict):
            d["error"] = ToolError.from_dict(d["error"])
        return super().from_dict(d)  # type: ignore[return-value]


@dataclass
class Message(_Serialisable):
    role: str  # system | user | assistant | tool
    content: list[ContentPart] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    reasoning: Optional[str] = None  # thinking channel, stored but not shown to users

    @staticmethod
    def text(role: str, text: str, **kw: Any) -> "Message":
        return Message(role=role, content=[ContentPart.of_text(text)], **kw)

    def plain_text(self) -> str:
        return "\n".join(p.text for p in self.content if p.type == "text" and p.text)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Message":
        d = dict(d)
        d["content"] = [ContentPart.from_dict(p) for p in d.get("content", [])]
        d["tool_calls"] = [ToolCall.from_dict(p) for p in d.get("tool_calls", [])]
        return super().from_dict(d)  # type: ignore[return-value]


@dataclass
class Usage(_Serialisable):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.reasoning_tokens + other.reasoning_tokens,
            self.cached_tokens + other.cached_tokens,
            self.cost_usd + other.cost_usd,
        )


@dataclass
class Evidence(_Serialisable):
    """A citation that the trust gate can resolve against real content."""

    artifact_id: str
    kind: str = "text"  # text | frame | transcript | ocr | tool_output | file
    locator: dict[str, Any] = field(default_factory=dict)  # {"offset": .., "length": ..} | {"t": 12.3} | {"page":1,"box":[..]}
    quote: Optional[str] = None  # exact text the claim relies on, when text-like

    def cite(self) -> str:
        loc = ",".join(f"{k}={v}" for k, v in sorted(self.locator.items()))
        return f"[{self.artifact_id[:12]}:{self.kind}:{loc}]"


@dataclass
class Claim(_Serialisable):
    text: str
    evidence: list[Evidence] = field(default_factory=list)
    confidence: float = 0.5
    verified: Optional[bool] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Claim":
        d = dict(d)
        d["evidence"] = [Evidence.from_dict(e) for e in d.get("evidence", [])]
        return super().from_dict(d)  # type: ignore[return-value]


@dataclass
class PlanStep(_Serialisable):
    id: str
    description: str
    role: str = "executor"  # which sub-agent role runs it
    depends_on: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)  # tools the step expects to use
    artifacts: list[str] = field(default_factory=list)  # input artifact IDs
    verifier: Optional[str] = None  # verifier ID to run on the output
    expected_output: str = "text"  # text | json | number | file


@dataclass
class Plan(_Serialisable):
    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    rationale: str = ""
    version: int = 1

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Plan":
        d = dict(d)
        d["steps"] = [PlanStep.from_dict(s) for s in d.get("steps", [])]
        return super().from_dict(d)  # type: ignore[return-value]

    def step(self, step_id: str) -> PlanStep:
        for s in self.steps:
            if s.id == step_id:
                return s
        raise KeyError(step_id)


class VerdictStatus(str, enum.Enum):
    VERIFIED = "verified"
    PARTIAL = "partially_verified"
    FAILED = "failed"


@dataclass
class Check(_Serialisable):
    name: str
    passed: bool
    detail: str = ""
    verifier: str = ""


@dataclass
class Verdict(_Serialisable):
    status: VerdictStatus
    checks: list[Check] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)
    summary: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Verdict":
        d = dict(d)
        d["status"] = VerdictStatus(d["status"])
        d["checks"] = [Check.from_dict(c) for c in d.get("checks", [])]
        return super().from_dict(d)  # type: ignore[return-value]

    @property
    def passed(self) -> bool:
        return self.status == VerdictStatus.VERIFIED


@dataclass
class Budget(_Serialisable):
    """Hard caps for one run. Checked at the top of every loop iteration."""

    max_tokens: int = 200_000
    max_tool_calls: int = 60
    max_model_calls: int = 60
    max_branches: int = 4
    max_wall_clock_s: float = 600.0
    max_cost_usd: float = 1.0
    max_replans: int = 2
    max_refine_rounds: int = 2

    # Consumption (mutable counters)
    used_tokens: int = 0
    used_tool_calls: int = 0
    used_model_calls: int = 0
    used_branches: int = 0
    used_wall_clock_s: float = 0.0
    used_cost_usd: float = 0.0
    used_replans: int = 0

    def exhausted(self) -> list[str]:
        out = []
        if self.used_tokens >= self.max_tokens:
            out.append("tokens")
        if self.used_tool_calls >= self.max_tool_calls:
            out.append("tool_calls")
        if self.used_model_calls >= self.max_model_calls:
            out.append("model_calls")
        if self.used_wall_clock_s >= self.max_wall_clock_s:
            out.append("wall_clock")
        if self.used_cost_usd >= self.max_cost_usd:
            out.append("cost")
        return out

    def near_exhaustion(self, fraction: float = 0.9) -> bool:
        return (
            self.used_tokens >= self.max_tokens * fraction
            or self.used_tool_calls >= self.max_tool_calls * fraction
            or self.used_model_calls >= self.max_model_calls * fraction
            or self.used_wall_clock_s >= self.max_wall_clock_s * fraction
            or self.used_cost_usd >= self.max_cost_usd * fraction
        )

    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.used_tokens)

    def charge(self, usage: Usage) -> None:
        self.used_tokens += usage.total_tokens
        self.used_model_calls += 1
        self.used_cost_usd += usage.cost_usd


class NodeStatus(str, enum.Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"
    STALE = "stale"


class ProblemType(str, enum.Enum):
    CONVERGENT = "convergent"
    DIVERGENT = "divergent"
    EXPLORATORY = "exploratory"
    STRUCTURAL = "structural"


@dataclass
class SubAgentResult(_Serialisable):
    """The only thing a sub-agent returns to its parent (§7.5)."""

    answer: Any
    claims: list[Claim] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    confidence: float = 0.5
    gaps: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    tool_calls_used: int = 0
    iterations: int = 0
    artifacts_produced: list[str] = field(default_factory=list)
    lesson: Optional[str] = None  # Reflexion note after a failure
    status: str = "ok"  # ok | failed | budget_exhausted

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SubAgentResult":
        d = dict(d)
        d["claims"] = [Claim.from_dict(c) for c in d.get("claims", [])]
        d["evidence"] = [Evidence.from_dict(e) for e in d.get("evidence", [])]
        d["usage"] = Usage.from_dict(d.get("usage", {}))
        return super().from_dict(d)  # type: ignore[return-value]
