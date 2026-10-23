"""Model clients.

``OpenAICompatibleClient`` talks to any OpenAI-compatible server (vLLM,
SGLang) over plain ``urllib`` with retries, timeouts, token accounting and
request IDs. ``MockModelClient`` runs the whole harness offline: it
recognises the role marker in the system prompt and produces well-formed
plans, tool calls, verdicts and answers so that tests and demos need no GPU.
"""
from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .types import ContentPart, Message, ToolCall, Usage
from .utils import IdGenerator, estimate_tokens, extract_json

ROLE_MARKER_PREFIX = "[harness-role:"


def role_marker(role: str) -> str:
    """Stable marker placed at the top of each system prompt (cache friendly)."""
    return f"{ROLE_MARKER_PREFIX} {role}]"


def find_role(messages: list[Message]) -> Optional[str]:
    for m in messages:
        if m.role == "system":
            mt = re.search(r"\[harness-role:\s*([a-z_\-]+)\]", m.plain_text())
            if mt:
                return mt.group(1)
    return None


class ModelError(Exception):
    def __init__(self, message: str, status: Optional[int] = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


@dataclass
class ModelRequest:
    messages: list[Message]
    tools: list[dict[str, Any]] = field(default_factory=list)  # OpenAI tool schemas
    tool_choice: Optional[Any] = None  # "auto" | "none" | {"type":"function","function":{"name":..}}
    temperature: float = 0.2
    max_tokens: int = 2048
    thinking_budget: int = 0  # 0 = thinking off
    response_format: Optional[dict[str, Any]] = None  # {"type":"json_object"} or json_schema
    seed: Optional[int] = None
    stop: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)  # role, run_id, node_id (for traces)


@dataclass
class ModelResponse:
    text: str
    reasoning: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    request_id: str = ""
    finish_reason: str = "stop"
    latency_s: float = 0.0
    model: str = ""
    raw: Optional[dict[str, Any]] = None

    def json(self) -> Any:
        return extract_json(self.text)

    def to_message(self) -> Message:
        return Message(
            role="assistant",
            content=[ContentPart.of_text(self.text)] if self.text else [],
            tool_calls=list(self.tool_calls),
            reasoning=self.reasoning,
        )


class ModelClient:
    """Interface. Subclasses implement ``complete``."""

    name = "base"

    def complete(self, request: ModelRequest) -> ModelResponse:  # pragma: no cover - interface
        raise NotImplementedError

    def health(self) -> dict[str, Any]:
        return {"ok": True, "backend": self.name}


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def split_thinking(text: str) -> tuple[Optional[str], str]:
    """Separate a ``<think>…</think>`` scratchpad from the answer (§6.1)."""
    if not text:
        return None, text
    m = _THINK_RE.search(text)
    if not m:
        # Unterminated think block: treat everything after <think> as reasoning.
        if "<think>" in text:
            head, _, tail = text.partition("<think>")
            return tail.strip(), head.strip()
        return None, text
    reasoning = m.group(1).strip()
    answer = (text[: m.start()] + text[m.end() :]).strip()
    return reasoning, answer


class OpenAICompatibleClient(ModelClient):
    name = "openai_compatible"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout_s: float = 120.0,
        max_retries: int = 3,
        price_per_million_prompt: float = 0.0,
        price_per_million_completion: float = 0.0,
        native_image: bool = True,
        native_audio: bool = False,
        native_video: bool = False,
        thinking_switch: bool = True,
        ids: Optional[IdGenerator] = None,
        opener: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.price_prompt = price_per_million_prompt
        self.price_completion = price_per_million_completion
        self.native_image = native_image
        self.native_audio = native_audio
        self.native_video = native_video
        self.thinking_switch = thinking_switch
        self.ids = ids or IdGenerator()
        self._open = opener or urllib.request.urlopen

    # -- request building -------------------------------------------------
    def _convert_part(self, p: ContentPart) -> Optional[dict[str, Any]]:
        if p.type == "text":
            return {"type": "text", "text": p.text or ""}
        if p.type == "image" and p.data_url and self.native_image:
            return {"type": "image_url", "image_url": {"url": p.data_url}}
        if p.type == "audio" and p.data_url and self.native_audio:
            fmt = (p.mime or "audio/wav").split("/")[-1]
            b64 = p.data_url.split(",", 1)[-1]
            return {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}}
        if p.type == "video" and p.data_url and self.native_video:
            return {"type": "video_url", "video_url": {"url": p.data_url}}
        # Non-native media: fall back to the text reference that ingestion attached.
        if p.text:
            return {"type": "text", "text": p.text}
        return None

    def _convert_message(self, m: Message) -> dict[str, Any]:
        out: dict[str, Any] = {"role": m.role}
        parts = [cp for cp in (self._convert_part(p) for p in m.content) if cp]
        if m.role in ("system", "tool") or all(pp["type"] == "text" for pp in parts):
            out["content"] = "\n".join(pp["text"] for pp in parts)
        else:
            out["content"] = parts
        if m.tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in m.tool_calls
            ]
        if m.tool_call_id:
            out["tool_call_id"] = m.tool_call_id
        if m.name:
            out["name"] = m.name
        return out

    def build_payload(self, req: ModelRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [self._convert_message(m) for m in req.messages],
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
        }
        if req.tools:
            payload["tools"] = req.tools
            if req.tool_choice is not None:
                payload["tool_choice"] = req.tool_choice
        if req.response_format:
            payload["response_format"] = req.response_format
        if req.seed is not None:
            payload["seed"] = req.seed
        if req.stop:
            payload["stop"] = req.stop
        if self.thinking_switch:
            # vLLM/SGLang expose Qwen3's thinking switch through chat_template_kwargs.
            payload["chat_template_kwargs"] = {"enable_thinking": req.thinking_budget > 0}
        return payload

    # -- transport --------------------------------------------------------
    def _post(self, path: str, payload: dict[str, Any], request_id: str) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "X-Request-Id": request_id}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        http_req = urllib.request.Request(self.base_url + path, data=data, headers=headers, method="POST")
        delay = 1.0
        last: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                with self._open(http_req, timeout=self.timeout_s) as resp:
                    body = resp.read().decode("utf-8")
                return json.loads(body)
            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8", "replace")[:500]
                except Exception:
                    pass
                retryable = e.code in (408, 409, 425, 429, 500, 502, 503, 504)
                last = ModelError(f"HTTP {e.code}: {body}", status=e.code, retryable=retryable)
                if not retryable:
                    raise last
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last = ModelError(f"transport error: {e}", retryable=True)
            if attempt < self.max_retries:
                time.sleep(delay)
                delay = min(delay * 2, 16.0)
        assert last is not None
        raise last

    def _usage(self, raw: dict[str, Any]) -> Usage:
        u = raw.get("usage") or {}
        pt = int(u.get("prompt_tokens", 0) or 0)
        ct = int(u.get("completion_tokens", 0) or 0)
        details = u.get("completion_tokens_details") or {}
        ptd = u.get("prompt_tokens_details") or {}
        return Usage(
            prompt_tokens=pt,
            completion_tokens=ct,
            reasoning_tokens=int(details.get("reasoning_tokens", 0) or 0),
            cached_tokens=int(ptd.get("cached_tokens", 0) or 0),
            cost_usd=(pt * self.price_prompt + ct * self.price_completion) / 1e6,
        )

    def complete(self, req: ModelRequest) -> ModelResponse:
        request_id = self.ids.new("req")
        payload = self.build_payload(req)
        t0 = time.monotonic()
        raw = self._post("/chat/completions", payload, request_id)
        latency = time.monotonic() - t0
        choices = raw.get("choices") or []
        if not choices:
            raise ModelError("empty choices in response")
        choice = choices[0]
        msg = choice.get("message") or {}
        text = msg.get("content") or ""
        if isinstance(text, list):  # some servers return content parts
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        if not reasoning:
            reasoning, text = split_thinking(text)
        tool_calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments") or "{}"
            if isinstance(args, str):
                parsed = extract_json(args)
                args = parsed if isinstance(parsed, dict) else {"_raw": args}
            tool_calls.append(ToolCall(id=tc.get("id") or self.ids.new("call"), name=fn.get("name", ""), arguments=args))
        return ModelResponse(
            text=text,
            reasoning=reasoning,
            tool_calls=tool_calls,
            usage=self._usage(raw),
            request_id=request_id,
            finish_reason=choice.get("finish_reason") or "stop",
            latency_s=latency,
            model=raw.get("model", self.model),
            raw=raw,
        )

    def health(self) -> dict[str, Any]:
        try:
            http_req = urllib.request.Request(self.base_url + "/models", headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})
            with self._open(http_req, timeout=min(self.timeout_s, 10.0)) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            models = [m.get("id") for m in body.get("data", [])]
            return {"ok": True, "backend": self.name, "models": models, "base_url": self.base_url}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "backend": self.name, "error": str(e), "base_url": self.base_url}


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------

Handler = Callable[[ModelRequest], ModelResponse]


class MockModelClient(ModelClient):
    """Offline model.

    Resolution order for each request:
    1. a scripted response queued with ``push``;
    2. a handler registered for the request's role marker;
    3. the built-in heuristic policy for that role.
    """

    name = "mock"

    def __init__(self, handlers: Optional[dict[str, Handler]] = None, ids: Optional[IdGenerator] = None,
                 fail_on_roles: Optional[set[str]] = None) -> None:
        self.handlers: dict[str, Handler] = dict(handlers or {})
        self.ids = ids or IdGenerator()
        self.script: list[ModelResponse | Handler] = []
        self.calls: list[ModelRequest] = []
        self.fail_on_roles = set(fail_on_roles or ())
        self._lock = threading.Lock()

    def push(self, *responses: ModelResponse | Handler | str) -> None:
        for r in responses:
            self.script.append(ModelResponse(text=r) if isinstance(r, str) else r)

    def _usage(self, req: ModelRequest, text: str) -> Usage:
        pt = sum(estimate_tokens(m.plain_text()) for m in req.messages) + 50 * len(req.tools)
        return Usage(prompt_tokens=pt, completion_tokens=estimate_tokens(text), cached_tokens=pt // 2)

    def complete(self, req: ModelRequest) -> ModelResponse:
        with self._lock:
            self.calls.append(req)
            role = find_role(req.messages) or req.metadata.get("role") or "generic"
            if role in self.fail_on_roles:
                raise ModelError(f"mock failure for role {role}", retryable=False)
            if self.script:
                item = self.script.pop(0)
                resp = item(req) if callable(item) else item
            elif role in self.handlers:
                resp = self.handlers[role](req)
            else:
                resp = self._policy(role, req)
        if not resp.request_id:
            resp.request_id = self.ids.new("req")
        if resp.usage.total_tokens == 0:
            resp.usage = self._usage(req, resp.text)
        resp.model = "mock"
        return resp

    # -- heuristic policy ------------------------------------------------
    @staticmethod
    def _last_user_text(req: ModelRequest) -> str:
        for m in reversed(req.messages):
            if m.role == "user":
                return m.plain_text()
        return ""

    @staticmethod
    def _tool_results(req: ModelRequest) -> list[Message]:
        return [m for m in req.messages if m.role == "tool"]

    @staticmethod
    def _artifact_ids(req: ModelRequest) -> list[str]:
        text = "\n".join(m.plain_text() for m in req.messages)
        return list(dict.fromkeys(re.findall(r"artifact_id=([0-9a-f]{12,64})", text)))

    def _policy(self, role: str, req: ModelRequest) -> ModelResponse:
        task = self._last_user_text(req)
        if role == "classifier":
            lowered = task.lower()
            if any(w in lowered for w in ("suggest", "ideas", "titles", "brainstorm")):
                pt = "divergent"
            elif any(w in lowered for w in ("interesting", "explore", "what's in", "overview")):
                pt = "exploratory"
            elif any(w in lowered for w in ("convert", "rewrite", "report", "restructure")):
                pt = "structural"
            else:
                pt = "convergent"
            return ModelResponse(text=json.dumps({"problem_type": pt, "effort": "medium", "expected_tools": ["fetch_artifact_view"], "reason": "mock heuristic"}))
        if role == "contract":
            return ModelResponse(text=json.dumps({
                "required_outputs": [{"name": "answer", "format": "text"}],
                "required_evidence": "every claim cites an artifact locator",
                "checks": ["contract", "evidence", "consistency", "policy"],
                "acceptable_failures": ["sidecar unavailable: report as unverified"],
            }))
        if role == "planner":
            arts = self._artifact_ids(req)
            gen = re.search(r"replan generation (\d+)", task)
            sfx = f"_g{gen.group(1)}" if gen else ""
            steps = []
            if arts:
                for i, a in enumerate(arts, 1):
                    steps.append({"id": f"s{i}{sfx}", "description": f"Inspect artifact {a[:12]} to answer: {task[:200]}", "role": "executor", "depends_on": [], "tools": ["fetch_artifact_view"], "artifacts": [a], "verifier": "evidence"})
                steps.append({"id": f"s_final{sfx}", "description": f"Combine findings into the answer for: {task[:200]}", "role": "executor", "depends_on": [s["id"] for s in steps], "tools": [], "artifacts": [], "verifier": "evidence"})
            else:
                steps.append({"id": f"s1{sfx}", "description": "Answer the task directly", "role": "executor", "depends_on": [], "tools": [], "artifacts": [], "verifier": "evidence"})
            return ModelResponse(text=json.dumps({"goal": task[:200], "steps": steps, "rationale": "mock planner"}))
        if role in ("executor", "intake", "perception"):
            return self._executor_policy(req, task)
        if role in ("verifier", "critic"):
            return ModelResponse(text=json.dumps({"passed": True, "issues": [], "confidence": 0.8, "detail": "mock verifier: evidence resolved by code checks"}))
        if role == "judge":
            return ModelResponse(text=json.dumps({"score": 0.8, "reason": "mock judge"}))
        if role == "synthesiser":
            return self._synth_policy(req, task)
        if role == "probe":
            return ModelResponse(text=json.dumps({"contradicting_evidence_needed": [], "found": False}))
        if role == "difficulty":
            return ModelResponse(text=json.dumps({"effort": "medium"}))
        return ModelResponse(text=f"Mock response to: {task[:200]}")

    @staticmethod
    def _question(task: str) -> str:
        """The step text only: never let untrusted context steer the mock's line choice."""
        step = task.split("Your step:", 1)[-1]
        for marker in ("\n\nInputs from earlier steps:", "\n\nArtifacts for this step:"):
            step = step.split(marker, 1)[0]
        return step.strip()

    def _executor_policy(self, req: ModelRequest, task: str) -> ModelResponse:
        task = self._question(task)
        tool_names = [t["function"]["name"] for t in req.tools if t.get("type") == "function"]
        results = self._tool_results(req)
        arts = self._artifact_ids(req)
        fetched = {m.name for m in results}
        if arts and "fetch_artifact_view" in tool_names and not results:
            return ModelResponse(
                text="",
                tool_calls=[ToolCall(id=self.ids.new("call"), name="fetch_artifact_view", arguments={"artifact_id": arts[0], "view": "text"})],
                finish_reason="tool_calls",
            )
        claims = []
        answer_bits = []
        for m in results:
            data = extract_json(m.plain_text())
            if isinstance(data, dict) and data.get("ok") and isinstance(data.get("output"), dict):
                out = data["output"]
                text = str(out.get("text") or out.get("content") or "")
                aid = out.get("artifact_id") or (arts[0] if arts else None)
                if text and aid:
                    snippet = _best_line(text, task)
                    off = text.find(snippet)
                    claims.append({
                        "text": f"The source states: \"{snippet}\"",
                        "evidence": [{"artifact_id": aid, "kind": out.get("kind", "text"), "locator": {"offset": off, "length": len(snippet)}, "quote": snippet}],
                        "confidence": 0.85,
                    })
                    answer_bits.append(snippet)
        if claims:
            answer = "Based on the source material: " + " | ".join(answer_bits)
            gaps: list[str] = []
            conf = 0.8
        else:
            answer = f"[mock backend] No real model is attached. Task received: {task[:160]}"
            gaps = ["mock backend: answer is a placeholder, not a real analysis"]
            conf = 0.3
        payload = {"answer": answer, "claims": claims, "confidence": conf, "gaps": gaps}
        return ModelResponse(text=json.dumps(payload), finish_reason="stop")

    def _synth_policy(self, req: ModelRequest, task: str) -> ModelResponse:
        # The step payload follows the stable context; parse only that part.
        step = task.split("Your step:", 1)[-1]
        data = extract_json(step)
        claims = []
        answers = []
        if isinstance(data, dict):
            for node in data.get("results", []):
                if isinstance(node, dict):
                    if node.get("answer"):
                        answers.append(str(node["answer"]))
                    claims.extend(node.get("claims", []))
        answer = "\n".join(dict.fromkeys(answers)) or "No verified findings were produced."
        return ModelResponse(text=json.dumps({"answer": answer, "claims": claims, "confidence": 0.75 if claims else 0.3, "gaps": [] if claims else ["no evidence-backed findings"]}))


_STOP = {"the", "a", "an", "of", "to", "and", "in", "on", "is", "are", "was", "were", "what", "which", "who", "when", "did", "does", "do", "this", "that", "for", "it", "about"}


def _best_line(text: str, task: str) -> str:
    """Mock heuristic: the source line sharing the most words with the question (first line otherwise)."""
    q = {w for w in re.findall(r"[a-z0-9]+", task.lower()) if w not in _STOP}
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    if not lines:
        return ""
    # overlap with the question first, then value-like lines ("Total: 318.40", digits), then document order
    best = max(lines, key=lambda ln: (len(q & set(re.findall(r"[a-z0-9]+", ln.lower()))), bool(re.search(r"\d", ln)), ":" in ln or "=" in ln, -lines.index(ln)))
    return best[:120]
