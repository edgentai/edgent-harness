"""Thinking channel control (§6.1) and budget forcing (§6.3)."""
from __future__ import annotations

from typing import Optional

from ..core.client import ModelClient, ModelRequest, ModelResponse, split_thinking
from ..core.types import Message

CONTINUE_PROMPT = "Your reasoning stopped early. Continue thinking carefully, then give the final answer."
FORCE_ANSWER_PROMPT = "Stop reasoning now. Give the final answer immediately in the required format."


def strip_scratchpad(text: str) -> str:
    return split_thinking(text)[1]


class ThinkingController:
    """Runs a request with the thinking budget of the active effort profile.

    - ``thinking_budget == 0`` disables the thinking channel entirely.
    - If the model stops with ``length`` before producing an answer, it is
      asked to finish (forced answer).
    - If the task is hard and the model answers with little reasoning, one
      "continue" extension is allowed when ``extend_if_short`` is set.
    """

    def __init__(self, client: ModelClient) -> None:
        self.client = client

    def run(self, req: ModelRequest, extend_if_short: bool = False, min_reasoning_tokens: int = 64) -> ModelResponse:
        resp = self.client.complete(req)
        if resp.finish_reason == "length" and not resp.tool_calls and not resp.text.strip():
            forced = ModelRequest(messages=req.messages + [resp.to_message(), Message.text("user", FORCE_ANSWER_PROMPT)], tools=req.tools,
                                  temperature=0.0, max_tokens=max(256, req.max_tokens // 2), thinking_budget=0, response_format=req.response_format, metadata=req.metadata)
            second = self.client.complete(forced)
            second.usage = resp.usage.add(second.usage)
            second.reasoning = (resp.reasoning or "") + ("\n" + second.reasoning if second.reasoning else "")
            return second
        if extend_if_short and req.thinking_budget > 0 and not resp.tool_calls:
            reasoning_len = len(resp.reasoning or "") // 4
            if reasoning_len < min_reasoning_tokens:
                ext = ModelRequest(messages=req.messages + [resp.to_message(), Message.text("user", CONTINUE_PROMPT)], tools=req.tools,
                                   temperature=req.temperature, max_tokens=req.max_tokens, thinking_budget=req.thinking_budget, response_format=req.response_format, metadata=req.metadata)
                third = self.client.complete(ext)
                third.usage = resp.usage.add(third.usage)
                if third.text.strip():
                    return third
        return resp

    @staticmethod
    def user_visible(resp: ModelResponse) -> str:
        """Never show raw reasoning to end users by default."""
        return strip_scratchpad(resp.text)
