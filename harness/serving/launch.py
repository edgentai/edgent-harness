"""Serving-engine command lines and a crash-restarting supervisor (§4.2, §4.4)."""
from __future__ import annotations

import subprocess
import threading
import time
from typing import Any, Callable, Optional

from ..core.config import ServingConfig


def vllm_command(cfg: ServingConfig, model_path: str, port: int = 8000, gpu_memory_utilization: float = 0.90, extra: Optional[list[str]] = None) -> list[str]:
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path, "--served-model-name", cfg.model, "--host", "127.0.0.1", "--port", str(port),
        "--max-model-len", str(cfg.max_context_length), "--max-num-seqs", str(cfg.max_concurrent_sequences),
        "--gpu-memory-utilization", str(gpu_memory_utilization), "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
        "--reasoning-parser", "qwen3", "--guided-decoding-backend", "xgrammar",
    ]
    if cfg.prefix_caching:
        cmd.append("--enable-prefix-caching")
    if cfg.quantisation.lower() == "fp8":
        cmd += ["--quantization", "fp8", "--kv-cache-dtype", "fp8"]
    if cfg.api_key:
        cmd += ["--api-key", cfg.api_key]
    return cmd + list(extra or [])


def sglang_command(cfg: ServingConfig, model_path: str, port: int = 8000, mem_fraction: float = 0.88, extra: Optional[list[str]] = None) -> list[str]:
    cmd = ["python3", "-m", "sglang.launch_server", "--model-path", model_path, "--served-model-name", cfg.model, "--host", "127.0.0.1", "--port", str(port),
           "--context-length", str(cfg.max_context_length), "--max-running-requests", str(cfg.max_concurrent_sequences), "--mem-fraction-static", str(mem_fraction),
           "--tool-call-parser", "qwen25", "--reasoning-parser", "qwen3"]
    if cfg.quantisation.lower() == "fp8":
        cmd += ["--quantization", "fp8"]
    if cfg.api_key:
        cmd += ["--api-key", cfg.api_key]
    return cmd + list(extra or [])


class Supervisor:
    """Keeps the model server alive and re-attaches interrupted runs after a restart."""

    def __init__(self, command: list[str], health: Callable[[], bool], on_restart: Optional[Callable[[], Any]] = None,
                 check_interval_s: float = 15.0, max_restarts: int = 10, popen: Callable[..., Any] = subprocess.Popen) -> None:
        self.command = command
        self.health = health
        self.on_restart = on_restart
        self.check_interval_s = check_interval_s
        self.max_restarts = max_restarts
        self._popen = popen
        self.proc: Optional[Any] = None
        self.restarts = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.proc = self._popen(self.command)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(self.check_interval_s)
            dead = self.proc is None or self.proc.poll() is not None
            if dead or not self._safe_health():
                if self.restarts >= self.max_restarts:
                    break
                self.restarts += 1
                try:
                    if self.proc and self.proc.poll() is None:
                        self.proc.kill()
                except Exception:
                    pass
                self.proc = self._popen(self.command)
                if self.on_restart:
                    try:
                        self.on_restart()  # e.g. harness.resume_all()
                    except Exception:
                        pass

    def _safe_health(self) -> bool:
        try:
            return bool(self.health())
        except Exception:
            return False

    def stop(self) -> None:
        self._stop.set()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
