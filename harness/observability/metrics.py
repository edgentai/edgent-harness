"""In-process counters, gauges and histograms (latency percentiles, cache-hit rate)."""
from __future__ import annotations

import threading
from typing import Any


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: dict[str, float] = {}
        self.gauges: dict[str, float] = {}
        self.histograms: dict[str, list[float]] = {}

    def inc(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0.0) + value

    def set(self, name: str, value: float) -> None:
        with self._lock:
            self.gauges[name] = value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self.histograms.setdefault(name, []).append(value)

    def percentile(self, name: str, p: float) -> float:
        vals = sorted(self.histograms.get(name, []))
        if not vals:
            return 0.0
        k = max(0, min(len(vals) - 1, int(round((p / 100.0) * (len(vals) - 1)))))
        return vals[k]

    def cache_hit_rate(self) -> float:
        prompt = self.counters.get("tokens.prompt", 0.0)
        cached = self.counters.get("tokens.cached", 0.0)
        return (cached / prompt) if prompt else 0.0

    def record_model_call(self, usage: Any, latency_s: float) -> None:
        self.inc("model.calls")
        self.inc("tokens.prompt", usage.prompt_tokens)
        self.inc("tokens.completion", usage.completion_tokens)
        self.inc("tokens.cached", usage.cached_tokens)
        self.inc("tokens.reasoning", usage.reasoning_tokens)
        self.inc("cost.usd", usage.cost_usd)
        self.observe("model.latency_s", latency_s)
        if latency_s > 0:
            self.observe("model.tokens_per_s", usage.completion_tokens / latency_s)

    def record_tool_call(self, name: str, ok: bool, duration_s: float) -> None:
        self.inc("tool.calls")
        self.inc(f"tool.calls.{name}")
        if not ok:
            self.inc("tool.errors")
            self.inc(f"tool.errors.{name}")
        self.observe("tool.duration_s", duration_s)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "latency_p50_s": self.percentile("model.latency_s", 50),
                "latency_p95_s": self.percentile("model.latency_s", 95),
                "cache_hit_rate": self.cache_hit_rate(),
                "tool_error_rate": (self.counters.get("tool.errors", 0.0) / self.counters["tool.calls"]) if self.counters.get("tool.calls") else 0.0,
            }
