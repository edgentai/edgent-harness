"""Narration to audio with whatever on-device speech synthesiser is installed.

Backends in order of preference: ``piper`` (needs a voice model: ``voice`` or ``EDGENT_PIPER_VOICE`` pointing at
a ``.onnx`` file), macOS ``say``, ``espeak-ng``, ``espeak``. All run in the sandbox with the network off and
write a WAV file. When none is installed the caller gets the typed ``unavailable`` error that lists the options.
Tests inject ``which`` (PATH lookup) and ``runner`` (the subprocess runner).
"""
from __future__ import annotations

import math
import os
import shutil
import sys
import wave
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ...domain.errors import ErrorCategory, Limit, ToolFailure
from ...domain.utils import ensure_dir
from ..sandbox import SandboxResult, run_subprocess
from ..toolchain import Toolchain

TTS_TEXT_LIMIT = Limit("tts text", 20_000, "chars", does_not_raise=("splitting the narration into several calls is the way around it",))
TTS_TIMEOUT_LIMIT = Limit("tts timeout", 300, "s", floor=5)
TTS_MEMORY_BYTES = 2 << 30
BACKENDS: tuple[str, ...] = ("piper", "say", "espeak-ng", "espeak")
PLATFORM = sys.platform  # `say` is the macOS synthesiser; a `say` on another platform's PATH is something else
PIPER_VOICE_ENV = "EDGENT_PIPER_VOICE"
PIPER_VOICE_ENV_FALLBACK = "PIPER_VOICE"
SAY_FORMAT = "LEI16@22050"
INSTALL_OPTIONS = (
    "piper (pip install piper-tts, download a voice .onnx and set EDGENT_PIPER_VOICE to it)",
    "espeak-ng (Debian: sudo apt install espeak-ng; macOS: brew install espeak-ng)",
    "the built-in `say` command on macOS",
)

Runner = Callable[..., SandboxResult]
Which = Callable[[str], str | None]


def _piper_model(voice: str | None, environ: Mapping[str, str]) -> str | None:
    candidates = [voice, environ.get(PIPER_VOICE_ENV), environ.get(PIPER_VOICE_ENV_FALLBACK)]
    for c in candidates:
        if c and c.endswith(".onnx") and Path(c).is_file():
            return c
    return None


def available_backends(voice: str | None = None, which: Which | None = None, environ: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """Which synthesisers are on PATH, with the reason a present one still cannot be used."""
    look = which or shutil.which
    env = os.environ if environ is None else environ
    rows: list[dict[str, Any]] = []
    for name in BACKENDS:
        if name == "say" and PLATFORM != "darwin":  # a `say` on another platform's PATH is some other program
            rows.append({"backend": name, "path": None, "usable": False, "note": f"say is the macOS synthesiser; not looked for on {PLATFORM}"})
            continue
        exe = look(name)
        row: dict[str, Any] = {"backend": name, "path": exe, "usable": bool(exe), "note": None}
        if exe and name == "piper":
            model = _piper_model(voice, env)
            row["model"] = model
            if not model:
                row["usable"] = False
                row["note"] = f"piper is installed but no voice model was found; pass voice=/path/to/voice.onnx or set {PIPER_VOICE_ENV}"
        rows.append(row)
    return rows


def _argv(name: str, exe: str, out: Path, voice: str | None, model: str | None) -> tuple[list[str], bytes | None, str]:
    """(argv, stdin, text-mode) for one backend; text goes through stdin so no narration is parsed as an option."""
    if name == "piper":
        return [exe, "--model", str(model), "--output_file", str(out)], None, "stdin"
    if name == "say":
        argv = [exe, "-o", str(out), f"--data-format={SAY_FORMAT}"]
        if voice:
            argv += ["-v", voice]
        return [*argv, "-f", "-"], None, "stdin"
    argv = [exe, "-w", str(out)]
    if voice:
        argv += ["-v", voice]
    return [*argv, "--stdin"], None, "stdin"


def wav_duration_s(path: str | os.PathLike[str]) -> float | None:
    """Duration from the WAV header (stdlib); None when the file is not a readable WAV."""
    try:
        with wave.open(str(path), "rb") as w:
            rate = w.getframerate()
            return round(w.getnframes() / rate, 3) if rate else None
    except (wave.Error, OSError, EOFError):
        return None


def synthesize(
    text: str,
    out_path: str | os.PathLike[str],
    voice: str | None = None,
    toolchain: Toolchain | None = None,
    runner: Runner | None = None,
    which: Which | None = None,
    environ: Mapping[str, str] | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Speak ``text`` into ``out_path`` (WAV). Returns ``{"path", "backend", "duration_s", "voice"}``."""
    text = (text or "").strip()
    if not text:
        raise ToolFailure("invalid_input", "there is no text to speak", ErrorCategory.invalid_input, remediation="Pass the narration text.")
    if len(text) > TTS_TEXT_LIMIT.value:
        err = TTS_TEXT_LIMIT.exceeded(len(text), "Split the narration by beat and synthesise each part.")
        raise ToolFailure(err.code, err.message, err.category, remediation=err.remediation)
    env = os.environ if environ is None else environ
    timeout = float(TTS_TIMEOUT_LIMIT.value if timeout_s is None else min(max(timeout_s, TTS_TIMEOUT_LIMIT.floor or 0), TTS_TIMEOUT_LIMIT.value))
    out = Path(out_path)
    ensure_dir(out.parent)
    run = runner or run_subprocess
    attempts: list[str] = []
    for row in available_backends(voice, which, env):
        name = row["backend"]
        if not row["path"]:
            continue
        if not row["usable"]:
            attempts.append(str(row["note"]))
            continue
        argv, stdin, _mode = _argv(name, row["path"], out, voice, row.get("model"))
        if out.exists():
            out.unlink()
        r = run(argv, str(out.parent), timeout_s=timeout, cpu_seconds=int(math.ceil(timeout)), memory_bytes=TTS_MEMORY_BYTES, allow_network=False, stdin=text.encode("utf-8"))
        if r.timed_out:
            attempts.append(f"{name} timed out after {timeout:g} s")
            continue
        if r.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
            attempts.append(f"{name} failed (exit {r.returncode}): {(r.stderr or r.stdout).strip()[-300:] or 'no output written'}")
            continue
        duration = wav_duration_s(out)
        if duration is None and _toolchain(toolchain).has("ffprobe"):
            duration = _probe_duration(out, toolchain, runner)
        return {"path": str(out), "backend": name, "duration_s": duration, "voice": row.get("model") or voice, "chars": len(text)}
    if attempts:
        raise ToolFailure(
            "tts_failed",
            "every installed speech backend failed: " + "; ".join(attempts),
            ErrorCategory.execution_failure,
            remediation="Fix the backend named in the message (a missing voice model is the usual cause) or install another: " + "; ".join(INSTALL_OPTIONS) + ".",
            attempts=attempts,
        )
    raise ToolFailure(
        "unavailable",
        "no speech synthesiser is installed (looked for " + ", ".join(BACKENDS) + ").",
        ErrorCategory.unavailable,
        remediation="Install one of: " + "; ".join(INSTALL_OPTIONS) + ". Then run `edgent doctor`. Until then keep the narration as text (captions) and record the audio as an unknown.",
        toolchain="tts",
    )


def _toolchain(toolchain: Toolchain | None) -> Toolchain:
    return toolchain if toolchain is not None else Toolchain.detect()


def _probe_duration(path: Path, toolchain: Toolchain | None, runner: Runner | None) -> float | None:
    from .av import probe  # local import: av needs ffmpeg knowledge this module does not otherwise carry

    try:
        return probe(path, toolchain, runner)["duration_s"]
    except ToolFailure:
        return None


__all__ = ["BACKENDS", "INSTALL_OPTIONS", "PIPER_VOICE_ENV", "TTS_TEXT_LIMIT", "TTS_TIMEOUT_LIMIT", "available_backends", "synthesize", "wav_duration_s"]
