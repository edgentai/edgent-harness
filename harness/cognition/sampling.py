"""Search-based decoding (§6.3, §9.7): self-consistency, best-of-N, diversity."""
from __future__ import annotations

import concurrent.futures
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..core.client import ModelClient, ModelRequest, ModelResponse
from ..core.types import Usage
from ..core.utils import extract_json

FRAMINGS = [
    "",
    "Approach this as a sceptical auditor who double-checks every figure.",
    "Approach this as a domain expert explaining to a careful colleague.",
    "Approach this by first listing what could go wrong, then answering.",
    "Approach this from the evidence outward: cite first, conclude second.",
]


def normalise_answer(text: str) -> Optional[str]:
    """Canonical form for voting: numbers, short labels, yes/no, JSON 'answer' field."""
    if text is None:
        return None
    data = extract_json(text)
    if isinstance(data, dict) and "answer" in data:
        text = str(data["answer"])
    t = text.strip().lower()
    m = re.fullmatch(r"[^\d\-]*(-?\d[\d,]*\.?\d*)\s*%?[^\d]*", t)
    if m and len(t) < 64:
        num = m.group(1).replace(",", "")
        try:
            f = float(num)
            return str(int(f)) if f.is_integer() else f"{f:.6g}"
        except ValueError:
            pass
    if t in ("yes", "no", "true", "false"):
        return {"true": "yes", "false": "no"}.get(t, t)
    if len(t) <= 80:
        return re.sub(r"[^a-z0-9 ]+", "", t).strip()
    return None  # free text: not voteable


def diversify(req: ModelRequest, n: int, base_temperature: float) -> list[ModelRequest]:
    """N differently framed, differently seeded requests. Identical samples are wasted compute."""
    out = []
    for i in range(n):
        msgs = [m for m in req.messages]
        framing = FRAMINGS[i % len(FRAMINGS)]
        if framing and msgs and msgs[-1].role == "user":
            from ..core.types import ContentPart, Message

            last = msgs[-1]
            msgs[-1] = Message(role="user", content=last.content + [ContentPart.of_text("\n" + framing)], tool_calls=last.tool_calls)
        out.append(ModelRequest(messages=msgs, tools=req.tools, tool_choice=req.tool_choice,
                                temperature=min(1.0, base_temperature + 0.15 * (i % 3)), max_tokens=req.max_tokens, thinking_budget=req.thinking_budget,
                                response_format=req.response_format, seed=(req.seed or 0) + i if req.seed is not None else None, metadata=dict(req.metadata)))
    return out


@dataclass
class SampleOutcome:
    chosen: ModelResponse
    candidates: list[ModelResponse]
    votes: dict[str, int] = field(default_factory=dict)
    scores: list[float] = field(default_factory=list)
    agreement: float = 1.0
    usage: Usage = field(default_factory=Usage)
    method: str = "single"


def _run_all(client: ModelClient, reqs: list[ModelRequest], parallel: int = 4) -> list[ModelResponse]:
    if len(reqs) == 1:
        return [client.complete(reqs[0])]
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(parallel, len(reqs))) as ex:
        return list(ex.map(client.complete, reqs))


def self_consistency(client: ModelClient, req: ModelRequest, n: int, base_temperature: float = 0.7, parallel: int = 4) -> SampleOutcome:
    """Majority vote over normalised answers. Falls back to the first sample for free text."""
    reqs = diversify(req, n, base_temperature) if n > 1 else [req]
    responses = _run_all(client, reqs, parallel)
    usage = Usage()
    for r in responses:
        usage = usage.add(r.usage)
    keys = [normalise_answer(r.text) for r in responses]
    voteable = [k for k in keys if k]
    if not voteable:
        return SampleOutcome(responses[0], responses, {}, [], 1.0 / max(1, n), usage, "first_of_n_unvoteable")
    counts = Counter(voteable)
    winner, votes = counts.most_common(1)[0]
    idx = keys.index(winner)
    return SampleOutcome(responses[idx], responses, dict(counts), [], votes / len(responses), usage, "self_consistency")


def best_of_n(client: ModelClient, req: ModelRequest, n: int, scorer: Callable[[ModelResponse], float], base_temperature: float = 0.7, parallel: int = 4) -> SampleOutcome:
    """Generate N candidates and keep the one the verifier scores highest (§6.3.2)."""
    reqs = diversify(req, n, base_temperature) if n > 1 else [req]
    responses = _run_all(client, reqs, parallel)
    usage = Usage()
    for r in responses:
        usage = usage.add(r.usage)
    scores = [scorer(r) for r in responses]
    best = max(range(len(responses)), key=scores.__getitem__)
    spread = (max(scores) - min(scores)) if scores else 0.0
    return SampleOutcome(responses[best], responses, {}, scores, 1.0 - spread, usage, "best_of_n")


def coverage_curve(results: list[list[bool]]) -> list[float]:
    """pass@k style coverage from per-task sample outcomes (§10.2). results[task][sample]."""
    if not results:
        return []
    max_n = max(len(r) for r in results)
    curve = []
    for k in range(1, max_n + 1):
        passed = sum(1 for r in results if any(r[:k]))
        curve.append(passed / len(results))
    return curve


def knee(curve: list[float], min_gain: float = 0.02) -> int:
    """Smallest N after which an extra sample adds less than ``min_gain``."""
    for i in range(1, len(curve)):
        if curve[i] - curve[i - 1] < min_gain:
            return i
    return max(1, len(curve))
