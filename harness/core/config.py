"""Configuration: effort profiles (§6.2) and the harness-wide config object."""
from __future__ import annotations

import enum
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


class EffortLevel(str, enum.Enum):
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"

    def next(self) -> Optional["EffortLevel"]:
        order = list(EffortLevel)
        i = order.index(self)
        return order[i + 1] if i + 1 < len(order) else None


@dataclass
class EffortProfile:
    """One bundle of settings per effort level (the table in §6.2)."""

    level: EffortLevel
    thinking_budget: int  # 0 disables the thinking channel
    samples: int  # self-consistency samples
    verification_passes: int
    max_tool_iterations: int
    sub_agents_allowed: bool
    max_parallel_workers: int
    temperature: float
    max_tokens: int  # per model call
    refine_rounds: int
    tree_of_thoughts_breadth: int  # 1 = disabled

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["level"] = self.level.value
        return d


EFFORT_PROFILES: dict[EffortLevel, EffortProfile] = {
    EffortLevel.MINIMAL: EffortProfile(EffortLevel.MINIMAL, 0, 1, 0, 3, False, 1, 0.0, 1024, 0, 1),
    EffortLevel.LOW: EffortProfile(EffortLevel.LOW, 512, 1, 1, 6, False, 1, 0.2, 2048, 1, 1),
    EffortLevel.MEDIUM: EffortProfile(EffortLevel.MEDIUM, 2048, 3, 1, 12, True, 3, 0.5, 4096, 1, 1),
    EffortLevel.HIGH: EffortProfile(EffortLevel.HIGH, 8192, 5, 2, 25, True, 4, 0.7, 8192, 2, 3),
    EffortLevel.MAX: EffortProfile(EffortLevel.MAX, 16384, 9, 3, 40, True, 6, 0.8, 8192, 3, 4),
}


@dataclass
class ModelCapabilities:
    """Answers to the §2 capability audit. The harness branches on these."""

    native_image: bool = True
    native_audio: bool = False
    native_video: bool = False
    thinking_switch: bool = True
    native_tool_calls: bool = True
    structured_output: bool = True
    context_length: int = 32_768
    image_token_cost: int = 1_000  # planning estimate per image/frame


@dataclass
class ServingConfig:
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "Qwen3.8-27B"
    api_key: str = ""
    timeout_s: float = 120.0
    max_retries: int = 3
    quantisation: str = "fp8"
    max_context_length: int = 32_768
    max_concurrent_sequences: int = 16
    prefix_caching: bool = True
    price_per_million_prompt: float = 0.0  # self-hosted: 0 by default
    price_per_million_completion: float = 0.0


@dataclass
class SandboxConfig:
    cpu_seconds: int = 30
    memory_bytes: int = 1 << 30
    timeout_s: float = 60.0
    allow_network: bool = False
    allowed_executables: tuple[str, ...] = ("python3", "ffmpeg", "ffprobe", "tesseract")


@dataclass
class ModalityLimits:
    max_file_bytes: int = 512 << 20
    max_video_seconds: float = 3 * 3600
    max_audio_seconds: float = 3 * 3600
    max_image_pixels: int = 50_000_000
    keyframe_interval_s: float = 10.0
    scene_threshold: float = 0.4
    preview_max_side: int = 768
    inline_text_tokens: int = 2_000  # per derived view before it becomes a reference


@dataclass
class HarnessConfig:
    state_dir: str = ".harness"
    serving: ServingConfig = field(default_factory=ServingConfig)
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    modality: ModalityLimits = field(default_factory=ModalityLimits)
    default_effort: EffortLevel = EffortLevel.MEDIUM
    allow_effort_escalation: bool = True
    prompt_version: str = "v1"
    session_time_limit_s: float = 20 * 3600  # Colab session expectation (§4.4)
    drain_fraction: float = 0.9

    @property
    def db_path(self) -> str:
        return os.path.join(self.state_dir, "state.sqlite3")

    @property
    def artifacts_dir(self) -> str:
        return os.path.join(self.state_dir, "artifacts")

    @property
    def traces_dir(self) -> str:
        return os.path.join(self.state_dir, "traces")

    @property
    def workspaces_dir(self) -> str:
        return os.path.join(self.state_dir, "workspaces")

    @property
    def memory_dir(self) -> str:
        return os.path.join(self.state_dir, "memory")

    def profile(self, level: Optional[EffortLevel] = None) -> EffortProfile:
        return EFFORT_PROFILES[level or self.default_effort]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["default_effort"] = self.default_effort.value
        d["sandbox"]["allowed_executables"] = list(self.sandbox.allowed_executables)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HarnessConfig":
        d = dict(d)
        if "serving" in d:
            d["serving"] = ServingConfig(**d["serving"])
        if "capabilities" in d:
            d["capabilities"] = ModelCapabilities(**d["capabilities"])
        if "sandbox" in d:
            sb = dict(d["sandbox"])
            if "allowed_executables" in sb:
                sb["allowed_executables"] = tuple(sb["allowed_executables"])
            d["sandbox"] = SandboxConfig(**sb)
        if "modality" in d:
            d["modality"] = ModalityLimits(**d["modality"])
        if "default_effort" in d:
            d["default_effort"] = EffortLevel(d["default_effort"])
        return cls(**d)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "HarnessConfig":
        """Load from a JSON file, then apply ``HARNESS_*`` environment overrides."""
        cfg = cls()
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                cfg = cls.from_dict(json.load(f))
        env = os.environ
        if env.get("HARNESS_STATE_DIR"):
            cfg.state_dir = env["HARNESS_STATE_DIR"]
        if env.get("HARNESS_BASE_URL"):
            cfg.serving.base_url = env["HARNESS_BASE_URL"]
        if env.get("HARNESS_MODEL"):
            cfg.serving.model = env["HARNESS_MODEL"]
        if env.get("HARNESS_API_KEY"):
            cfg.serving.api_key = env["HARNESS_API_KEY"]
        if env.get("HARNESS_EFFORT"):
            cfg.default_effort = EffortLevel(env["HARNESS_EFFORT"])
        return cfg

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
