"""Problem-type classification (§11.1) with modality and tool-use dimensions."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.client import ModelClient, ModelRequest
from ..core.config import EffortLevel
from ..core.types import Message, Modality, ProblemType
from ..core.utils import extract_json

STRATEGY = {
    ProblemType.CONVERGENT: {"verifier": "evidence", "sampling": "self_consistency", "topology": "plan_execute"},
    ProblemType.DIVERGENT: {"verifier": "judge", "sampling": "diverse_rank", "topology": "mixture_of_agents"},
    ProblemType.EXPLORATORY: {"verifier": "evidence", "sampling": "single", "topology": "map_reduce"},
    ProblemType.STRUCTURAL: {"verifier": "schema", "sampling": "single", "topology": "architect_editor"},
}


@dataclass
class Classification:
    problem_type: ProblemType
    modalities: list[Modality]
    expected_tools: list[str]
    effort_hint: Optional[EffortLevel] = None
    reason: str = ""
    strategy: dict[str, str] = field(default_factory=dict)
    code_task: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"problem_type": self.problem_type.value, "modalities": [m.value for m in self.modalities], "expected_tools": self.expected_tools,
                "effort_hint": self.effort_hint.value if self.effort_hint else None, "reason": self.reason, "strategy": self.strategy, "code_task": self.code_task}


class TaskClassifier:
    def __init__(self, client: Optional[ModelClient] = None, use_model: bool = True, system_prompt: Optional[Any] = None) -> None:
        self.client = client
        self.use_model = use_model
        self.system_prompt = system_prompt

    def heuristic(self, task: str, modalities: list[Modality]) -> Classification:
        t = task.lower()
        if re.search(r"\b(suggest|brainstorm|ideas|titles|names|options|alternatives)\b", t):
            pt = ProblemType.DIVERGENT
        elif re.search(r"\b(convert|rewrite|restructure|turn .* into|format|translate|summari[sz]e .* into a report)\b", t):
            pt = ProblemType.STRUCTURAL
        elif re.search(r"\b(what'?s interesting|explore|overview|anything notable|what is in|describe everything)\b", t):
            pt = ProblemType.EXPLORATORY
        else:
            pt = ProblemType.CONVERGENT
        tools = ["fetch_artifact_view"] if modalities else []
        if re.search(r"\b(total|sum|count|how many|average|percent|calculate)\b", t):
            tools += ["calculate", "count_matches"]
        if Modality.VIDEO in modalities:
            tools += ["video_scenes", "frame_at", "transcript_between"]
        if Modality.IMAGE in modalities:
            tools += ["crop_zoom", "ocr_region"]
        if Modality.AUDIO in modalities:
            tools += ["transcript_between", "transcribe_range"]
        code = bool(re.search(r"\b(code|implement|script|function|bug|test|refactor|python)\b", t))
        if code:
            tools += ["run_python", "run_tests"]
        return Classification(pt, modalities, list(dict.fromkeys(tools)), None, "heuristic", dict(STRATEGY[pt]), code)

    def classify(self, task: str, modalities: list[Modality]) -> tuple[Classification, Any]:
        base = self.heuristic(task, modalities)
        usage = None
        if self.use_model and self.client is not None and self.system_prompt is not None:
            try:
                resp = self.client.complete(ModelRequest(messages=[Message.text("system", self.system_prompt("classifier")),
                                                                   Message.text("user", f"Task: {task[:3000]}\nModalities present: {[m.value for m in modalities]}")],
                                                         temperature=0.0, max_tokens=200, metadata={"role": "classifier"}))
                usage = resp.usage
                data = extract_json(resp.text) or {}
                if data.get("problem_type") in {p.value for p in ProblemType}:
                    base.problem_type = ProblemType(data["problem_type"])
                    base.strategy = dict(STRATEGY[base.problem_type])
                if data.get("effort") in {e.value for e in EffortLevel}:
                    base.effort_hint = EffortLevel(data["effort"])
                if isinstance(data.get("expected_tools"), list):
                    base.expected_tools = list(dict.fromkeys(base.expected_tools + [str(x) for x in data["expected_tools"]]))
                base.reason = f"model: {data.get('reason', '')}"
            except Exception as e:  # noqa: BLE001
                base.reason = f"heuristic (model classification failed: {e})"
        return base, usage
