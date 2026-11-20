"""Read-only environment & toolchain doctor (never mutates the host)."""
from __future__ import annotations

import platform
import shutil
import sys
from typing import Any

from ..sandbox.runner import _can_sandbox_exec, _can_unshare

BINARIES = ("git", "bash", "python3", "rg", "ffmpeg", "ffprobe", "tesseract", "node", "docker")


def run_doctor() -> dict[str, Any]:
    bins = {name: shutil.which(name) for name in BINARIES}
    sandbox_mode = "unshare" if _can_unshare() else ("sandbox-exec" if _can_sandbox_exec() else "rlimits+scrubbed_env")
    hints: list[str] = []
    if not bins["git"]:
        hints.append("git missing: automatic git checkpointing unavailable (file snapshots still work)")
    if not bins["bash"]:
        hints.append("bash missing: falling back to /bin/sh for shell tool")
    if not bins["ffmpeg"]:
        hints.append("ffmpeg missing: optional video/audio tools unavailable")
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "sandbox_isolation": sandbox_mode,
        "binaries": {k: {"available": bool(v), "path": v} for k, v in bins.items()},
        "hints": hints,
        "ok": True,
    }
