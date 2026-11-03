"""What a provider can do. The loop and the context assembler branch on these, never on vendor names."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EffortControl = Literal["thinking_budget", "switch", "model_suffix", "none"]


@dataclass(frozen=True)
class Capabilities:
    native_tools: bool = True  # structured tool calls; otherwise the text protocol is used
    system_channel: bool = True  # a real system message; otherwise markers inside the prompt
    images: bool = False
    audio: bool = False
    video: bool = False
    streaming: bool = False
    json_schema: bool = False  # structured output by schema
    max_prompt_bytes: int | None = None  # hard ceiling to refuse at, not approach
    context_tokens: int = 128_000
    effort_control: EffortControl = "none"
    concurrency: int = 4
    prompt_caching: bool = False
    multi_turn: bool = True  # False for single-turn print modes: the conversation is written into one prompt

    def accepts(self, part_type: str) -> bool:
        return {"text": True, "image": self.images, "audio": self.audio, "video": self.video, "file": False}.get(part_type, False)
