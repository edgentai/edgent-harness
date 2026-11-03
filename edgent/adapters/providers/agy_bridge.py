"""The Antigravity CLI (``agy``) in print mode as a completion endpoint.

Measured facts (2026-08, release dependent; everything below is configurable or
probed, never assumed):

- no ``--system-prompt``: the system text travels inside labelled markers in
  the prompt (the base ``prepare`` folds it because ``system_channel`` is False);
- the prompt is the value of ``--print`` (argv), no stdin; above about 128 KB
  the CLI truncates silently and still reports success, so the bridge refuses
  above a named limit (``EDGENT_AGY_MAX_PROMPT_BYTES``) instead;
- tools cannot be switched off, but headless mode auto-denies tools that are
  not allow-listed, so a plain completion is effectively tool-free; the bridge
  never passes ``--dangerously-skip-permissions``;
- an empty response with a stderr line like ``a tool required the "read_file"
  permission`` is a denial; the fix is one settings line, which is explained,
  never written;
- images are written to a temp directory and referenced as ``@/abs/path``
  lines (that needs the ``read_file(*)`` permission);
- effort is baked into the model id (``-low``/``-medium``/``-high``) and
  ``--effort`` must not be sent: the requested effort resolves to a sibling id
  the CLI lists (``agy models``, cached; a fallback list when the CLI cannot
  be asked), and the id stays unchanged when no sibling exists;
- ``--output-format json`` prints one envelope ``{conversation_id, status,
  response, duration_seconds, num_turns, usage{input_tokens, output_tokens,
  thinking_tokens, cache_read_tokens, total_tokens}}``; ``max_tokens`` is
  ignored; about three processes run side by side.
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import threading
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from ...domain.capabilities import Capabilities
from ...domain.errors import ErrorCategory, Limit
from ...domain.types import Usage
from ...domain.utils import extract_json
from .base import ProviderError
from .cli_bridge import (
    DEFAULT_CONCURRENCY,
    DEFAULT_TIMEOUT_S,
    EFFORT_LEVELS,
    CliBridgeProvider,
    Runner,
    _int,
    bridge_error,
    decode_output,
    envelope_text,
    parse_json_envelope,
)

DEFAULT_MODEL = "gemini-3.1-pro-high"
ENV_MODEL = "EDGENT_AGY_MODEL"
ENV_MODELS = "EDGENT_AGY_MODELS"  # comma-separated ids; skips asking the CLI
ENV_MAX_PROMPT_BYTES = "EDGENT_AGY_MAX_PROMPT_BYTES"
PROMPT_BYTES_DEFAULT = 128 * 1024  # bytes: the measured ceiling above which agy truncates silently
PROMPT_BYTES_FLOOR = 4096  # bytes
PROMPT_BYTES_MAX = 1 << 20  # bytes
CONTEXT_TOKENS = 1_000_000  # tokens: the model's window; the prompt ceiling above is what binds
MODELS_TIMEOUT_S = 60.0  # seconds for `agy models`
SETTINGS_PATH = "~/.gemini/antigravity-cli/settings.json"
REPLY_TRAILER = "Reply with the requested output only."
MODELS_SUBCOMMAND: tuple[str, ...] = ("models",)
FALLBACK_MODELS: tuple[str, ...] = (
    "gemini-3.1-pro-low",
    "gemini-3.1-pro-medium",
    "gemini-3.1-pro-high",
    "gemini-3-pro-low",
    "gemini-3-pro-high",
    "gemini-3-flash",
    "claude-sonnet-4.5",
    "claude-opus-4.1-thinking",
    "gpt-oss-120b-medium",
)

_MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*[a-z0-9]$")
_SEPARATOR_RE = re.compile(r"^[-=+|:\s]+$")
_HEADER_WORDS = {"model", "models", "id", "name", "available", "default", "provider", "description"}
_OK_STATUSES = {"", "success", "succeeded", "ok", "complete", "completed", "done"}


def prompt_size_limit(environ: Mapping[str, str] | None = None) -> Limit:
    """The agy prompt ceiling: refused, never approached. ``EDGENT_AGY_MAX_PROMPT_BYTES`` overrides within bounds."""
    return Limit.from_env(
        "agy prompt size",
        PROMPT_BYTES_DEFAULT,
        "bytes",
        env=ENV_MAX_PROMPT_BYTES,
        floor=PROMPT_BYTES_FLOOR,
        maximum=PROMPT_BYTES_MAX,
        does_not_raise=("the model's context window", "max_tokens", "the effort level", "--print-timeout"),
        environ=dict(environ) if environ is not None else None,
    )


def split_effort_suffix(model: str) -> tuple[str, str | None]:
    """``('gemini-3.1-pro', 'high')`` for ``gemini-3.1-pro-high``; ``(model, None)`` without a suffix."""
    for level in EFFORT_LEVELS:
        if model.endswith("-" + level):
            return model[: -len(level) - 1], level
    return model, None


def parse_models_output(text: str) -> list[str]:
    """Model ids from ``agy models``: the first column of a table, or ids from a JSON list."""
    stripped = (text or "").strip()
    ids: list[str] = []
    if stripped[:1] in "[{":
        data = extract_json(stripped)
        items = data if isinstance(data, list) else (data.get("models") if isinstance(data, dict) else None)
        for item in items or []:
            if isinstance(item, str):
                ids.append(item)
            elif isinstance(item, dict):
                for key in ("id", "name", "model"):
                    if isinstance(item.get(key), str):
                        ids.append(item[key])
                        break
        if ids:
            return _dedupe(ids)
    for raw in stripped.splitlines():
        line = raw.strip()
        if not line or _SEPARATOR_RE.match(line) or line[0] in "#=+":
            continue
        line = line.lstrip("|*•>- ").strip()
        first = re.split(r"[\s|,]+", line)[0].strip("*`'\"") if line else ""
        if not first or first.lower() in _HEADER_WORDS or not _MODEL_ID_RE.match(first):
            continue
        ids.append(first)
    return _dedupe(ids)


def _dedupe(ids: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


class AgyProvider(CliBridgeProvider):
    """``agy --print`` as a bare completion endpoint behind Edgent's own loop (bridge mode)."""

    name = "agy"
    caps = Capabilities(
        native_tools=False,
        system_channel=False,
        images=True,
        audio=False,
        video=False,
        streaming=False,
        json_schema=False,
        max_prompt_bytes=PROMPT_BYTES_DEFAULT,
        context_tokens=CONTEXT_TOKENS,
        effort_control="model_suffix",
        concurrency=DEFAULT_CONCURRENCY,
        prompt_caching=True,
        multi_turn=False,
    )
    install_hint = "Install the Antigravity CLI (agy) and complete its login flow once, then make sure `agy` is on PATH (see https://codelabs.developers.google.com/antigravity-cli-hands-on)."
    login_hint = "Run `agy` once interactively and complete its login flow."
    settings_path = SETTINGS_PATH
    forbidden_flags = ("--dangerously-skip-permissions", "--effort")

    def __init__(
        self,
        model: str | None = None,
        *,
        cli: str = "agy",
        runner: Runner | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        concurrency: int = DEFAULT_CONCURRENCY,
        cwd: str | os.PathLike[str] | None = None,
        env_extra: dict[str, str] | None = None,
        effort_default: str = "medium",
        prices: dict[str, float] | None = None,
        known_models: Iterable[str] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        env = os.environ if environ is None else environ
        self.prompt_size_limit = prompt_size_limit(env)
        super().__init__(cli, model or env.get(ENV_MODEL) or DEFAULT_MODEL, runner=runner, timeout_s=timeout_s, concurrency=concurrency, cwd=cwd, env_extra=env_extra, effort_default=effort_default, prices=prices)
        self.caps = replace(self.caps, max_prompt_bytes=int(self.prompt_size_limit.value), concurrency=self.concurrency)
        self._models: list[str] | None = None
        self._models_source = "unset"
        self._models_lock = threading.Lock()
        env_models = env.get(ENV_MODELS, "")
        if known_models is not None:
            self._models, self._models_source = _dedupe(known_models), "constructor"
        elif env_models.strip():
            self._models, self._models_source = _dedupe(m.strip() for m in env_models.split(",") if m.strip()), "environment"

    # ------------------------------------------------------------------ capabilities
    def prompt_limit(self) -> Limit | None:
        return self.prompt_size_limit

    def default_env(self) -> dict[str, str]:
        return {"AGY_ENTRYPOINT": "edgent"}

    # ------------------------------------------------------------------ models and effort
    def list_models(self, force: bool = False) -> list[str]:
        """Ids the CLI lists (first column of ``agy models``), cached; the fallback list when it cannot be asked."""
        with self._models_lock:
            if self._models is not None and not force:
                return list(self._models)
            ids: list[str] = []
            try:
                r = self._runner([self.cli, *MODELS_SUBCOMMAND], None, MODELS_TIMEOUT_S, self.child_env())
                if _int(r.returncode) == 0:
                    ids = parse_models_output(decode_output(r.stdout))
            except (OSError, subprocess.TimeoutExpired, ValueError):
                ids = []
            self._models = ids or list(FALLBACK_MODELS)
            self._models_source = "cli" if ids else "fallback"
            return list(self._models)

    @property
    def models_source(self) -> str:
        return self._models_source

    def resolve_model(self, model: str, effort: str, known: Iterable[str] | None = None) -> str:
        """The sibling id for this effort when the CLI lists one; otherwise the id unchanged (``--effort`` is never sent)."""
        if not effort:
            return model
        base, current = split_effort_suffix(model)
        if current == effort:
            return model
        candidate = f"{base}-{effort}"
        ids = set(known) if known is not None else set(self.list_models())
        return candidate if candidate in ids else model

    # ------------------------------------------------------------------ argv and envelope
    def render_prompt(self, prompt_text: str, image_paths: list[str]) -> str:
        return super().render_prompt(prompt_text, image_paths) + "\n\n" + REPLY_TRAILER

    def build_argv(self, prompt_text: str, image_paths: list[str], effort: str, model: str, *, system_text: str = "", workdir: str | None = None) -> tuple[list[str], bytes | None]:
        # system_text is always empty here: the caps have no system channel, so prepare folded it into the prompt.
        argv = [
            self.cli,
            "--print",
            prompt_text,
            "--model",
            model,
            "--output-format",
            "json",
            "--disable-slash-commands",
            "--print-timeout",
            f"{max(1, math.ceil(self.timeout.value))}s",
        ]
        return argv, None

    def parse_envelope(self, stdout: str) -> tuple[str, Usage, dict[str, Any]]:
        env = parse_json_envelope(stdout, ("response", "status", "conversation_id"))
        text = envelope_text(env.get("response"))
        u = env.get("usage") if isinstance(env.get("usage"), dict) else {}
        usage = Usage(
            prompt_tokens=_int(u.get("input_tokens")),
            completion_tokens=_int(u.get("output_tokens")),
            reasoning_tokens=_int(u.get("thinking_tokens")),
            cached_tokens=_int(u.get("cache_read_tokens")),
        )
        status = str(env.get("status") or "").strip()
        extra: dict[str, Any] = {
            "request_id": env.get("conversation_id") or "",
            "status": status,
            "num_turns": env.get("num_turns"),
            "duration_seconds": env.get("duration_seconds"),
            "total_tokens": u.get("total_tokens"),
        }
        self._check_status(status, text)
        return text, usage, extra

    def _check_status(self, status: str, text: str) -> None:
        s = status.lower()
        if s in _OK_STATUSES:
            return
        detail = f" {text.strip()[:300]}" if text.strip() else ""
        if "timeout" in s or "timed_out" in s or "timed out" in s:
            raise bridge_error("cli_timeout", f"agy reported status {status!r}.{detail}", ErrorCategory.timeout, "Split the task or ask for a shorter answer, then retry once.", retryable=True, status=status)
        if "cancel" in s:
            raise bridge_error("cli_cancelled", f"agy reported status {status!r}.{detail}", ErrorCategory.cancelled, "Retry once if the cancellation was not yours.", retryable=True, status=status)
        if "error" in s or "fail" in s or "denied" in s:
            raise bridge_error("cli_failed", f"agy reported status {status!r}.{detail}", ErrorCategory.execution_failure, "Read the message and change the request; do not repeat the unchanged call.", status=status)
        # an unknown but not obviously failing status: accept the response and keep the status in raw

    # ------------------------------------------------------------------ health
    def health(self) -> dict[str, Any]:
        out = super().health()
        out["prompt_limit_bytes"] = self.prompt_size_limit.value
        out["prompt_limit_source"] = self.prompt_size_limit.source
        out["models_source"] = self._models_source
        return out


__all__ = [
    "DEFAULT_MODEL",
    "ENV_MAX_PROMPT_BYTES",
    "ENV_MODEL",
    "ENV_MODELS",
    "FALLBACK_MODELS",
    "PROMPT_BYTES_DEFAULT",
    "PROMPT_BYTES_FLOOR",
    "PROMPT_BYTES_MAX",
    "REPLY_TRAILER",
    "SETTINGS_PATH",
    "AgyProvider",
    "ProviderError",
    "parse_models_output",
    "prompt_size_limit",
    "split_effort_suffix",
]
