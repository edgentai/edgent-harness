"""Claude Code (``claude -p``) as a completion endpoint.

Measured facts (2026-08, release dependent; flag spellings are probed from
``claude --help`` once per process and cached):

- ``claude -p`` reads the prompt from stdin and takes ``--system-prompt``,
  ``--model``, ``--effort low|medium|high`` and ``--output-format json``;
- the JSON envelope carries ``result`` (text), ``usage{input_tokens,
  output_tokens, cache_read_input_tokens, cache_creation_input_tokens}``,
  ``total_cost_usd``, ``num_turns``, ``session_id`` and ``is_error``;
- tools are restricted with ``--tools ""`` (removes them) or, on releases
  without it, ``--allowedTools ""``; the probe picks whichever exists so the
  text tool protocol is the only tool channel;
- images reach the model through ``@path`` references in the prompt;
- ``max_tokens`` is ignored; about three processes run side by side.

A system prompt is passed on argv, which Linux caps at 128 KiB per element;
a larger one goes through ``--system-prompt-file`` when the release has it
and is refused otherwise (never truncated).
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from ...domain.capabilities import Capabilities
from ...domain.errors import ConfigurationError, ErrorCategory, Limit
from ...domain.types import Reply, Usage
from .base import ProviderError
from .cli_bridge import (
    DEFAULT_CONCURRENCY,
    DEFAULT_TIMEOUT_S,
    CliBridgeProvider,
    CompletedLike,
    ProbeResult,
    Runner,
    _int,
    envelope_text,
    parse_json_envelope,
)

DEFAULT_MODEL = "sonnet"
ENV_MODEL = "EDGENT_CLAUDE_MODEL"
CONTEXT_TOKENS = 200_000  # tokens
TOOL_RESTRICTION_FLAGS: tuple[str, ...] = ("--tools", "--allowedTools")  # preferred first: --tools removes the tools outright
# sent when the release has them: no transcript on disk, no slash commands, and none of the user's own MCP servers
# started for every completion (--strict-mcp-config with no --mcp-config loads none). Never --bare: it skips the login.
OPTIONAL_FLAGS: tuple[str, ...] = ("--no-session-persistence", "--disable-slash-commands", "--strict-mcp-config")
SYSTEM_PROMPT_ARGV_MAX_BYTES = 120_000  # bytes: below the 131072-byte kernel ceiling for one argv element
SYSTEM_PROMPT_ARGV_LIMIT = Limit("claude system prompt argv size", SYSTEM_PROMPT_ARGV_MAX_BYTES, "bytes", does_not_raise=("max_tokens", "the effort level", "the model's context window"))
DEFAULT_SYSTEM_PROMPT = "You are a careful, direct assistant. Answer the request itself; do not narrate your process."


class ClaudeCliProvider(CliBridgeProvider):
    """``claude -p`` as a bare completion endpoint behind Edgent's own loop (bridge mode).

    ``tool_restriction``: ``"auto"`` (probe ``--help`` and prefer ``--tools``), an explicit flag such as
    ``"--allowedTools"``, or ``None`` to send no restriction (the CLI's own tools then stay reachable).
    """

    name = "claude-cli"
    caps = Capabilities(
        native_tools=False,
        system_channel=True,
        images=True,
        audio=False,
        video=False,
        streaming=False,
        json_schema=False,
        max_prompt_bytes=None,
        context_tokens=CONTEXT_TOKENS,
        effort_control="switch",
        concurrency=DEFAULT_CONCURRENCY,
        prompt_caching=True,
        multi_turn=False,
    )
    install_hint = "Install Claude Code (npm install -g @anthropic-ai/claude-code, or curl -fsSL https://claude.ai/install.sh | bash), run `claude` once to log in, and make sure it is on PATH."
    login_hint = "Run `claude` once interactively and log in (or `claude auth login`)."
    settings_path = "~/.claude/settings.json"
    forbidden_flags = ("--dangerously-skip-permissions", "--allow-dangerously-skip-permissions")

    def __init__(
        self,
        model: str | None = None,
        *,
        cli: str = "claude",
        runner: Runner | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        concurrency: int = DEFAULT_CONCURRENCY,
        cwd: str | os.PathLike[str] | None = None,
        env_extra: dict[str, str] | None = None,
        effort_default: str = "medium",
        prices: dict[str, float] | None = None,
        tool_restriction: str | None = "auto",
        environ: Mapping[str, str] | None = None,
    ) -> None:
        env = os.environ if environ is None else environ
        if tool_restriction not in (None, "auto") and not str(tool_restriction).startswith("--"):
            raise ConfigurationError(f"tool_restriction must be 'auto', None or a flag such as '--tools'; got {tool_restriction!r}.")
        self.tool_restriction = tool_restriction
        super().__init__(cli, model or env.get(ENV_MODEL) or DEFAULT_MODEL, runner=runner, timeout_s=timeout_s, concurrency=concurrency, cwd=cwd, env_extra=env_extra, effort_default=effort_default, prices=prices)

    # ------------------------------------------------------------------ release-dependent flags
    def default_env(self) -> dict[str, str]:
        # a completion endpoint must not auto-update or phone home mid-run
        return {"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}

    def tool_restriction_flag(self) -> str | None:
        """Which flag switches the CLI's own tools off for this release, from the cached probe."""
        if self.tool_restriction != "auto":
            return self.tool_restriction or None
        probe = self.probe()
        for flag in TOOL_RESTRICTION_FLAGS:
            if probe.has_flag(flag):
                return flag
        # an unprobeable CLI gets the modern spelling; an available one that has neither cannot be restricted
        return TOOL_RESTRICTION_FLAGS[0] if not probe.available else None

    # ------------------------------------------------------------------ argv and envelope
    def build_argv(self, prompt_text: str, image_paths: list[str], effort: str, model: str, *, system_text: str = "", workdir: str | None = None) -> tuple[list[str], bytes | None]:
        probe = self.probe()
        argv = [self.cli, "-p", "--output-format", "json", "--model", model]
        if effort and probe.has_flag("--effort"):
            argv += ["--effort", effort]
        argv += self._system_prompt_args(system_text.strip() or DEFAULT_SYSTEM_PROMPT, probe, workdir)
        flag = self.tool_restriction_flag()
        if flag:
            argv += [flag, ""]
        argv += [f for f in OPTIONAL_FLAGS if probe.has_flag(f)]
        return argv, prompt_text.encode("utf-8")

    def _system_prompt_args(self, system_text: str, probe: ProbeResult, workdir: str | None) -> list[str]:
        size = len(system_text.encode("utf-8"))
        if size <= SYSTEM_PROMPT_ARGV_MAX_BYTES:
            return ["--system-prompt", system_text]
        if probe.has_flag("--system-prompt-file") and workdir:
            path = os.path.join(workdir, "system-prompt.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(system_text)
            return ["--system-prompt-file", path]
        err = SYSTEM_PROMPT_ARGV_LIMIT.exceeded(
            size,
            "Shorten the system prompt (fewer tools in the catalogue, shorter skills), or update claude to a release with --system-prompt-file.",
            ErrorCategory.truncated,
        )
        raise ProviderError(err.message, retryable=False, category=ErrorCategory.truncated, error=err)

    def parse_envelope(self, stdout: str) -> tuple[str, Usage, dict[str, Any]]:
        env = parse_json_envelope(stdout, ("result", "is_error", "type"))
        text = envelope_text(env.get("result"))
        u = env.get("usage") if isinstance(env.get("usage"), dict) else {}
        cache_read = _int(u.get("cache_read_input_tokens"))
        cache_created = _int(u.get("cache_creation_input_tokens"))
        usage = Usage(
            prompt_tokens=_int(u.get("input_tokens")) + cache_read + cache_created,
            completion_tokens=_int(u.get("output_tokens")),
            cached_tokens=cache_read,
        )
        extra: dict[str, Any] = {
            "request_id": env.get("session_id") or "",
            "subtype": env.get("subtype"),
            "num_turns": env.get("num_turns"),
            "duration_ms": env.get("duration_ms"),
            "reported_cost_usd": env.get("total_cost_usd"),
        }
        if env.get("is_error"):
            detail = text.strip() or envelope_text(env.get("error")) or str(env.get("subtype") or "unknown error")
            raise self.classify_failure(1, "", detail, f"is_error (subtype {env.get('subtype') or 'unknown'})")
        return text, usage, extra

    def interpret(self, result: CompletedLike, argv: list[str], model: str, latency_s: float, had_images: bool = False) -> Reply:
        reply = super().interpret(result, argv, model, latency_s, had_images)
        if self.tool_restriction_flag() is None and self.tool_restriction == "auto":
            reply.raw = dict(reply.raw or {})
            reply.raw.setdefault("warnings", []).append("this claude release has neither --tools nor --allowedTools; its own tools stayed reachable")
        return reply

    # ------------------------------------------------------------------ health
    def health(self) -> dict[str, Any]:
        out = super().health()
        p = self.probe()
        out["tool_flag"] = self.tool_restriction_flag()
        out["effort_flag"] = p.has_flag("--effort")
        out["system_prompt_file"] = p.has_flag("--system-prompt-file")
        if p.available and out["tool_flag"] is None:
            out["warning"] = "no flag to switch the CLI's tools off; completions may run the CLI's own tools"
        return out


__all__ = ["DEFAULT_MODEL", "DEFAULT_SYSTEM_PROMPT", "ENV_MODEL", "OPTIONAL_FLAGS", "SYSTEM_PROMPT_ARGV_MAX_BYTES", "TOOL_RESTRICTION_FLAGS", "ClaudeCliProvider"]
