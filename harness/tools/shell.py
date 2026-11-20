"""Sandboxed shell (`bash`) tool with persistent cwd, output offloading & background tasks.

Modeled directly on Claude Code / Antigravity shell tools:
- Executes shell commands inside the workspace (refusing cwd escape).
- Scrubs ambient credentials; isolates network/writes with `unshare` (Linux) or
  `sandbox-exec` (macOS) when `allow_network` is False.
- Tracks persistent working directory per session/workspace.
- Offloads large stdout/stderr (> inline cap) to `.edgent/tool_outputs/<id>.log`
  and returns head + tail + file path so context never overflows.
- Supports `background=True` for long-running processes with `task_status`,
  `task_output`, and `task_stop`.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.utils import ensure_dir
from ..sandbox.runner import _can_sandbox_exec, _can_unshare, _limits, _macos_sandbox_profile, _scrubbed_env, which
from .registry import ToolContext, ToolFailure

INLINE_STDOUT_CHARS = 3_500
INLINE_STDERR_CHARS = 1_500


def _shell_executable() -> str:
    return shutil.which("bash") or "/bin/sh"


def _offload_output(workspace_path: str, call_id: str, text: str, limit: int = INLINE_STDOUT_CHARS) -> tuple[str, Optional[str]]:
    if len(text) <= limit:
        return text, None
    out_dir = ensure_dir(os.path.join(workspace_path, ".edgent", "tool_outputs"))
    fname = f"shell_{call_id}.log"
    full_path = os.path.join(out_dir, fname)
    with open(full_path, "w", encoding="utf-8", errors="replace") as f:
        f.write(text)
    rel = os.path.relpath(full_path, workspace_path)
    half = limit // 2
    omitted = len(text) - limit
    preview = text[:half] + f"\n\n... [{omitted:,} chars omitted — full log saved at {rel}] ...\n\n" + text[-half:]
    return preview, rel


@dataclass
class BackgroundTask:
    task_id: str
    command: str
    cwd: str
    log_path: str
    proc: subprocess.Popen[Any]
    started_at: float = field(default_factory=time.monotonic)
    finished_at: Optional[float] = None
    _log_file: Any = None

    def poll_status(self) -> dict[str, Any]:
        rc = self.proc.poll()
        if rc is not None and self.finished_at is None:
            self.finished_at = time.monotonic()
            if self._log_file is not None:
                try:
                    self._log_file.flush()
                    self._log_file.close()
                except Exception:
                    pass
                self._log_file = None
        elapsed = (self.finished_at or time.monotonic()) - self.started_at
        return {
            "task_id": self.task_id,
            "command": self.command,
            "running": rc is None,
            "returncode": rc,
            "elapsed_s": round(elapsed, 2),
            "log_path": self.log_path,
        }


class BackgroundTaskManager:
    """Tracks background shell processes started by the agent."""

    def __init__(self, workspace_path: str) -> None:
        self.workspace_path = workspace_path
        self._tasks: dict[str, BackgroundTask] = {}
        self._lock = threading.Lock()

    def start(self, command: str, cwd: str, *, allow_network: bool = False, cpu_seconds: int = 300, memory_bytes: int = 1 << 30) -> dict[str, Any]:
        tid = f"bg_{uuid.uuid4().hex[:8]}"
        log_dir = ensure_dir(os.path.join(self.workspace_path, ".edgent", "tasks"))
        log_abs = os.path.join(log_dir, f"{tid}.log")
        log_rel = os.path.relpath(log_abs, self.workspace_path)
        argv = [_shell_executable(), "-c", command]
        if not allow_network:
            if _can_unshare():
                argv = [which("unshare") or "unshare", "-n", *argv]
            elif _can_sandbox_exec():
                argv = [which("sandbox-exec") or "sandbox-exec", "-p", _macos_sandbox_profile(self.workspace_path), *argv]
        lf = open(log_abs, "w+", encoding="utf-8", errors="replace")
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=_scrubbed_env(self.workspace_path),
            preexec_fn=_limits(cpu_seconds, memory_bytes) if os.name == "posix" else None,
        )
        task = BackgroundTask(tid, command, cwd, log_rel, proc, _log_file=lf)
        with self._lock:
            self._tasks[tid] = task
        return task.poll_status()

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [t.poll_status() for t in self._tasks.values()]

    def status(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            t = self._tasks.get(task_id)
        if t is None:
            raise ToolFailure("not_found", f"no background task '{task_id}'")
        return t.poll_status()

    def output(self, task_id: str, *, tail_chars: int = 6_000) -> dict[str, Any]:
        info = self.status(task_id)
        abs_log = os.path.join(self.workspace_path, info["log_path"])
        text = ""
        if os.path.exists(abs_log):
            with open(abs_log, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        truncated = len(text) > tail_chars
        return {**info, "output": text[-tail_chars:] if truncated else text, "truncated": truncated, "total_chars": len(text)}

    def stop(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            t = self._tasks.get(task_id)
        if t is None:
            raise ToolFailure("not_found", f"no background task '{task_id}'")
        if t.proc.poll() is None:
            try:
                os.killpg(t.proc.pid, signal.SIGTERM)
                t.proc.wait(timeout=2.0)
            except Exception:
                try:
                    os.killpg(t.proc.pid, signal.SIGKILL)
                except Exception:
                    t.proc.kill()
                try:
                    t.proc.wait(timeout=2.0)
                except Exception:
                    pass
        return t.poll_status()

    def stop_all(self) -> None:
        with self._lock:
            ids = list(self._tasks)
        for tid in ids:
            try:
                self.stop(tid)
            except Exception:
                pass


def _task_mgr(ctx: ToolContext) -> BackgroundTaskManager:
    mgr = ctx.extra.get("task_manager")
    if mgr is None:
        mgr = BackgroundTaskManager(ctx.workspace.path)
        ctx.extra["task_manager"] = mgr
    return mgr


def bash_tool(ctx: ToolContext, a: dict[str, Any]) -> Any:
    """Execute a shell command inside the sandboxed workspace."""
    cmd = str(a["command"]).strip()
    if not cmd:
        raise ToolFailure("invalid_arguments", "command must be non-empty")

    # Resolve cwd inside workspace (persisting relative cwd across calls in ctx.extra["shell_cwd"]).
    cur_rel = str(a.get("cwd") or ctx.extra.get("shell_cwd") or ".").strip() or "."
    cwd = ctx.workspace.resolve(cur_rel)
    if not os.path.isdir(cwd):
        raise ToolFailure("not_found", f"working directory does not exist: {cur_rel}")

    sb = ctx.config.sandbox
    allow_net = bool(a.get("allow_network", sb.allow_network))
    if bool(a.get("background", False)):
        return _task_mgr(ctx).start(cmd, cwd, allow_network=allow_net, cpu_seconds=max(sb.cpu_seconds, 300), memory_bytes=sb.memory_bytes)

    timeout = min(float(a.get("timeout_s", sb.timeout_s)), 600.0)
    # Track cwd changes via a sentinel marker appended after the user command.
    sentinel = f"__EDGENT_CWD_{uuid.uuid4().hex[:8]}__"
    wrapped = f"{cmd}\n__edg_rc=$?\nprintf '\\n{sentinel}%s\\n' \"$PWD\"\nexit $__edg_rc"
    argv = [_shell_executable(), "-c", wrapped]
    if not allow_net:
        if _can_unshare():
            argv = [which("unshare") or "unshare", "-n", *argv]
        elif _can_sandbox_exec():
            argv = [which("sandbox-exec") or "sandbox-exec", "-p", _macos_sandbox_profile(ctx.workspace.path), *argv]

    env_extra: dict[str, str] = {}
    if sb.python_executable:
        py_dir = os.path.dirname(os.path.abspath(sb.python_executable))
        env_extra["PATH"] = py_dir + os.pathsep + os.environ.get("PATH", "/usr/bin:/bin")

    t0 = time.monotonic()
    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_scrubbed_env(ctx.workspace.path, env_extra),
        preexec_fn=_limits(max(sb.cpu_seconds, int(timeout) + 5), sb.memory_bytes) if os.name == "posix" else None,
    )
    timed_out = False
    try:
        out_b, err_b = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        out_b, err_b = proc.communicate()

    duration = round(time.monotonic() - t0, 3)
    stdout = out_b.decode("utf-8", "replace")
    stderr = err_b.decode("utf-8", "replace")

    # Extract updated cwd if still inside workspace.
    marker = f"\n{sentinel}"
    if marker in stdout:
        head, tail = stdout.rsplit(marker, 1)
        new_cwd = tail.splitlines()[0].strip() if tail.splitlines() else ""
        stdout = head
        if new_cwd:
            try:
                resolved = ctx.workspace.resolve(new_cwd)
                if os.path.isdir(resolved):
                    ctx.extra["shell_cwd"] = os.path.relpath(resolved, ctx.workspace.path)
            except Exception:
                pass

    call_tag = uuid.uuid4().hex[:10]
    stdout_preview, stdout_log = _offload_output(ctx.workspace.path, call_tag + "_out", stdout, INLINE_STDOUT_CHARS)
    stderr_preview, stderr_log = _offload_output(ctx.workspace.path, call_tag + "_err", stderr, INLINE_STDERR_CHARS)
    rc = -9 if timed_out else proc.returncode
    return {
        "ok": rc == 0 and not timed_out,
        "returncode": rc,
        "timed_out": timed_out,
        "duration_s": duration,
        "cwd": ctx.extra.get("shell_cwd", "."),
        "stdout": stdout_preview,
        "stderr": stderr_preview,
        "stdout_log": stdout_log,
        "stderr_log": stderr_log,
        "untrusted": True,
    }


def task_status_tool(ctx: ToolContext, a: dict[str, Any]) -> Any:
    mgr = _task_mgr(ctx)
    if a.get("task_id"):
        return mgr.status(str(a["task_id"]))
    return {"tasks": mgr.list()}


def task_output_tool(ctx: ToolContext, a: dict[str, Any]) -> Any:
    return _task_mgr(ctx).output(str(a["task_id"]), tail_chars=int(a.get("tail_chars", 6_000)))


def task_stop_tool(ctx: ToolContext, a: dict[str, Any]) -> Any:
    return _task_mgr(ctx).stop(str(a["task_id"]))
