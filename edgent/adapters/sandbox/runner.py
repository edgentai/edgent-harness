"""Subprocess sandbox for everything model-written.

Limits: CPU time, address space, wall-clock timeout, a scrubbed environment
with no ambient credentials, cwd pinned to the workspace, optional network
namespace isolation when ``unshare`` is usable. Output is capped so a runaway
process cannot flood the context. ``InterpreterSession`` keeps a Python child
alive across steps and can snapshot and restore its picklable globals.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

MAX_OUTPUT = 20_000

_unshare_ok: bool | None = None


def which(name: str) -> str | None:
    return shutil.which(name)


def _can_unshare() -> bool:
    global _unshare_ok
    if _unshare_ok is None:
        exe = which("unshare")
        if not exe:
            _unshare_ok = False
        else:
            try:
                r = subprocess.run([exe, "-n", "true"], capture_output=True, timeout=5)
                _unshare_ok = r.returncode == 0
            except Exception:
                _unshare_ok = False
    return _unshare_ok


def _limits(cpu_seconds: int, memory_bytes: int):
    def apply() -> None:
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 5))
            soft = max(memory_bytes, 512 << 20)  # ffmpeg and Chromium map large regions
            resource.setrlimit(resource.RLIMIT_AS, (soft, soft))
            resource.setrlimit(resource.RLIMIT_NOFILE, (512, 512))
        except Exception:
            pass
        try:
            os.setsid()
        except Exception:
            pass

    return apply


def scrubbed_env(cwd: str, allow: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": cwd,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "TMPDIR": cwd,
        "MPLBACKEND": "Agg",
    }
    if allow:
        env.update(allow)
    return env


@dataclass
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False
    truncated: bool = False
    argv: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_s": round(self.duration_s, 3),
            "timed_out": self.timed_out,
            "truncated": self.truncated,
        }


def _cap(s: bytes, limit: int = MAX_OUTPUT) -> tuple[str, bool]:
    text = s.decode("utf-8", "replace")
    if len(text) > limit:
        return text[:limit] + "\n…[output truncated]", True
    return text, False


def run_subprocess(
    argv: list[str],
    cwd: str,
    timeout_s: float = 60.0,
    cpu_seconds: int = 30,
    memory_bytes: int = 1 << 30,
    allow_network: bool = False,
    stdin: bytes | None = None,
    env_extra: dict[str, str] | None = None,
    allowed_executables: tuple[str, ...] | None = None,
    max_output: int = MAX_OUTPUT,
) -> SandboxResult:
    if not argv:
        raise ValueError("empty argv")
    exe_name = os.path.basename(argv[0])
    if allowed_executables is not None and exe_name not in allowed_executables and argv[0] not in allowed_executables:
        return SandboxResult(126, "", f"executable not allowed: {exe_name}", 0.0, argv=argv)
    full = list(argv)
    if not allow_network and _can_unshare():
        full = [which("unshare") or "unshare", "-n", *full]
    t0 = time.monotonic()
    try:
        proc = subprocess.Popen(
            full,
            cwd=cwd,
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=scrubbed_env(cwd, env_extra),
            preexec_fn=_limits(cpu_seconds, memory_bytes) if os.name == "posix" else None,
        )
    except FileNotFoundError as e:
        return SandboxResult(127, "", f"executable not found: {e}", time.monotonic() - t0, argv=argv)
    timed_out = False
    try:
        out, err = proc.communicate(input=stdin, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, 9)
        except Exception:
            proc.kill()
        out, err = proc.communicate()
    so, t1 = _cap(out, max_output)
    se, t2 = _cap(err, max_output)
    return SandboxResult(proc.returncode if not timed_out else -9, so, se, time.monotonic() - t0, timed_out, t1 or t2, argv)


def run_python(
    code: str,
    cwd: str,
    timeout_s: float = 60.0,
    cpu_seconds: int = 30,
    memory_bytes: int = 1 << 30,
    allow_network: bool = False,
    filename: str | None = None,
) -> SandboxResult:
    """Run a Python snippet as an isolated process (``-I``) inside ``cwd``."""
    scratch = os.path.join(cwd, ".edgent", "scratch")
    os.makedirs(scratch, exist_ok=True)
    script = os.path.join(scratch, filename or f"cell_{uuid.uuid4().hex[:10]}.py")
    with open(script, "w", encoding="utf-8") as f:
        f.write(code)
    try:
        return run_subprocess([sys.executable, "-I", script], cwd, timeout_s, cpu_seconds, memory_bytes, allow_network)
    finally:
        try:
            os.remove(script)
        except OSError:
            pass


_CHILD = textwrap.dedent(
    r"""
    import sys, json, io, traceback, pickle, contextlib, os
    try:
        import resource
        cpu = int(os.environ.get("SBX_CPU", "30")); mem = int(os.environ.get("SBX_MEM", str(1 << 30)))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 5))
        resource.setrlimit(resource.RLIMIT_AS, (max(mem, 512 << 20), max(mem, 512 << 20)))
    except Exception:
        pass
    G = {"__name__": "__sandbox__"}
    def snapshot(path):
        keep = {}
        for k, v in G.items():
            if k.startswith("__"):
                continue
            try:
                pickle.dumps(v); keep[k] = v
            except Exception:
                pass
        with open(path, "wb") as f:
            pickle.dump(keep, f)
        return sorted(keep)
    def restore(path):
        with open(path, "rb") as f:
            G.update(pickle.load(f))
        return True
    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except Exception:
            continue
        op = msg.get("op")
        out, err = io.StringIO(), io.StringIO()
        resp = {"ok": True}
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                if op == "exec":
                    exec(compile(msg["code"], "<cell>", "exec"), G)
                elif op == "snapshot":
                    resp["keys"] = snapshot(msg["path"])
                elif op == "restore":
                    restore(msg["path"])
                elif op == "exit":
                    break
        except SystemExit:
            pass
        except BaseException:
            resp["ok"] = False
            resp["error"] = traceback.format_exc(limit=5)
        resp["stdout"] = out.getvalue()[:20000]
        resp["stderr"] = err.getvalue()[:20000]
        sys.__stdout__.write(json.dumps(resp) + "\n"); sys.__stdout__.flush()
    """
)


class InterpreterSession:
    """A persistent Python interpreter in a sandboxed child process."""

    def __init__(self, cwd: str, cpu_seconds: int = 30, memory_bytes: int = 1 << 30, timeout_s: float = 60.0) -> None:
        self.cwd = cwd
        self.timeout_s = timeout_s
        self.history: list[str] = []
        env = scrubbed_env(cwd, {"SBX_CPU": str(cpu_seconds), "SBX_MEM": str(memory_bytes)})
        self._proc = subprocess.Popen(
            [sys.executable, "-I", "-c", _CHILD],
            cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True,
        )
        self._lock = threading.Lock()

    def _rpc(self, msg: dict[str, Any]) -> dict[str, Any]:
        if self._proc.poll() is not None:
            return {"ok": False, "error": "interpreter died", "stdout": "", "stderr": ""}
        with self._lock:
            assert self._proc.stdin and self._proc.stdout
            self._proc.stdin.write(json.dumps(msg) + "\n")
            self._proc.stdin.flush()
            result: dict[str, Any] = {}

            def reader() -> None:
                line = self._proc.stdout.readline()  # type: ignore[union-attr]
                try:
                    result.update(json.loads(line) if line else {"ok": False, "error": "no response", "stdout": "", "stderr": ""})
                except Exception:
                    result.update({"ok": False, "error": "bad response", "stdout": "", "stderr": ""})

            t = threading.Thread(target=reader, daemon=True)
            t.start()
            t.join(self.timeout_s)
            if t.is_alive():
                self.close()
                return {"ok": False, "error": f"timeout after {self.timeout_s}s; interpreter killed", "stdout": "", "stderr": ""}
            return result

    def exec(self, code: str) -> dict[str, Any]:
        r = self._rpc({"op": "exec", "code": code})
        if r.get("ok"):
            self.history.append(code)
        return r

    def snapshot(self, path: str) -> dict[str, Any]:
        return self._rpc({"op": "snapshot", "path": path})

    def restore(self, path: str) -> dict[str, Any]:
        return self._rpc({"op": "restore", "path": path})

    def rebuild(self, history: list[str]) -> None:
        for code in history:
            self.exec(code)

    def alive(self) -> bool:
        return self._proc.poll() is None

    def close(self) -> None:
        try:
            if self._proc.poll() is None:
                self._proc.kill()
            self._proc.wait(timeout=5)
        except Exception:
            pass
        for stream in (self._proc.stdin, self._proc.stdout, self._proc.stderr):
            try:
                if stream:
                    stream.close()
            except Exception:
                pass
