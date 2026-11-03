"""Run a subscription CLI in print mode as a completion endpoint.

A user with a Claude Code or Antigravity subscription and no API key can still
drive Edgent's own loop: the CLI is started once per completion in its print
mode, the prepared request travels as one prompt, and the JSON envelope the CLI
prints becomes a ``Reply``. Everything release dependent (flag spellings, model
ids, ceilings) is configured or probed, never hard-wired, and every ceiling is
refused rather than approached: a silently truncated prompt answers a different
question with a success code.

This module owns the shared machinery: the concurrency semaphore, the temporary
directory that holds image attachments for one call, prompt assembly from the
prepared request, the size check, envelope and usage mapping, stderr
classification, and the ``--version``/``--help`` probe that records which flags
the installed release has. Subclasses set ``caps``, build the argv and parse
the envelope. ``max_tokens``, ``temperature``, ``stop`` and ``json_schema``
are accepted and ignored: print modes expose none of them.
"""
from __future__ import annotations

import base64
import errno
import json
import mimetypes
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol

from ...domain.capabilities import Capabilities
from ...domain.errors import ConfigurationError, ErrorCategory, HarnessError, Limit
from ...domain.types import CompletionRequest, ContentPart, Reply, Usage
from ...domain.utils import extract_json, json_dumps
from .base import Provider, ProviderError

DEFAULT_TIMEOUT_S = 1800.0  # seconds: one completion, including the CLI's own start-up
DEFAULT_CONCURRENCY = 3  # processes: what the subscription CLIs tolerate side by side
PROBE_TIMEOUT_S = 30.0  # seconds: --version and --help must answer within this
TIMEOUT_GRACE_S = 30.0  # seconds the child may outlive the completion timeout before it is killed
STDERR_TAIL_CHARS = 800  # characters of stderr kept in errors and traces
ARGV_TRACE_MAX_CHARS = 200  # characters: longer argv elements are elided in traces
IMAGES_PER_CALL_MAX = 16  # images staged for one call (a sanity cap, not a measured CLI limit)
ARGV_STRING_MAX_BYTES = 131_072  # bytes: Linux MAX_ARG_STRLEN, the kernel ceiling for one argv element
EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high")

ATTACHED_IMAGES_HEADER = "ATTACHED IMAGES (each @path below is part of the request)"
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}

_FLAG_RE = re.compile(r"(?<![\w-])(--[a-zA-Z][\w-]*)")
DENIAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r'required the "(?P<tool>[^"]+)" permission', re.IGNORECASE),
    re.compile(r'permission (?:for|to use) (?:the )?(?:tool )?"?(?P<tool>[\w.-]+)"? (?:was|is|has been) (?:denied|refused|required)', re.IGNORECASE),
)
_AUTH_RE = re.compile(
    r"not (?:logged|signed) in|please (?:log|sign) ?in|unauthenticated|unauthorized|authentication (?:failed|required|error)"
    r"|invalid (?:api key|credentials)|expired (?:token|credentials)|\b401\b",
    re.IGNORECASE,
)
_RATE_RE = re.compile(r"rate.?limit|too many requests|\b429\b|quota|overloaded|over capacity|usage limit|\b503\b|\b529\b", re.IGNORECASE)
_UNKNOWN_FLAG_RE = re.compile(r"unknown (?:option|argument|flag)|unrecognized (?:option|argument)|unexpected argument|no such option|invalid option", re.IGNORECASE)
_TIMEOUT_RE = re.compile(r"timed out|timeout", re.IGNORECASE)


# ----------------------------------------------------------------------------- runner
class CompletedLike(Protocol):
    """What a runner returns: the three fields of ``subprocess.CompletedProcess`` that matter."""

    returncode: int
    stdout: bytes | str
    stderr: bytes | str


Runner = Callable[[list[str], bytes | None, float, dict[str, str]], CompletedLike]
"""``runner(argv, stdin_bytes_or_None, timeout_s, env) -> CompletedLike``. Raises ``FileNotFoundError``
when the executable is missing and ``subprocess.TimeoutExpired`` on timeout; tests inject fakes."""


def subprocess_runner(cwd: str | None = None) -> Runner:
    """The default runner: one child per call in its own session, so a timeout kills its whole tree."""

    def run(argv: list[str], stdin: bytes | None, timeout_s: float, env: dict[str, str]) -> CompletedLike:
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            out, err = proc.communicate(stdin, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            raise
        return subprocess.CompletedProcess(argv, proc.returncode, out, err)

    return run


def _kill_tree(proc: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:  # noqa: BLE001 - no process group (Windows) or already gone: fall back to the child alone
        proc.kill()
    try:
        proc.communicate(timeout=5)
    except Exception:  # noqa: BLE001 - reaping is best effort once the tree is killed
        pass


# ----------------------------------------------------------------------------- probe and helpers
@dataclass(frozen=True)
class ProbeResult:
    """What ``<cli> --version`` and ``<cli> --help`` revealed about the installed release."""

    available: bool
    executable: str | None
    version: str = ""
    flags: frozenset[str] = frozenset()
    error: str | None = None

    def has_flag(self, flag: str) -> bool:
        return flag in self.flags

    def to_dict(self) -> dict[str, Any]:
        return {"available": self.available, "executable": self.executable, "version": self.version, "flags": sorted(self.flags), "error": self.error}


class EnvelopeError(ValueError):
    """The CLI's stdout held no usable JSON envelope."""


def decode_output(data: bytes | str | None) -> str:
    if data is None:
        return ""
    return data if isinstance(data, str) else data.decode("utf-8", "replace")


def tail(text: str, limit: int = STDERR_TAIL_CHARS) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else "…" + text[-limit:]


def strip_nul(text: str) -> str:
    """Argv and environment strings cannot carry NUL; in model text it is binary noise, so it is replaced, never a failure."""
    return text.replace("\x00", "\ufffd") if "\x00" in text else text


def redact_argv(argv: list[str]) -> list[str]:
    """Argv for traces: long elements (the prompt, the system text) are replaced by their size."""
    return [a if len(a) <= ARGV_TRACE_MAX_CHARS else f"<{len(a.encode('utf-8'))} bytes>" for a in argv]


def parse_help_flags(help_text: str) -> frozenset[str]:
    return frozenset(_FLAG_RE.findall(help_text or ""))


def parse_json_envelope(stdout: str, required_keys: tuple[str, ...]) -> dict[str, Any]:
    """The JSON object in stdout that carries one of ``required_keys``; the last one wins for line streams."""
    text = (stdout or "").strip()
    if not text:
        raise EnvelopeError("the CLI printed nothing on stdout")
    candidates: list[Any] = []
    try:
        candidates.append(json.loads(text))
    except ValueError:
        for line in reversed(text.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    candidates.append(json.loads(line))
                except ValueError:
                    continue
        found = extract_json(text)
        if found is not None:
            candidates.append(found)
    for cand in candidates:
        items = cand if isinstance(cand, list) else [cand]
        for item in reversed(items):
            if isinstance(item, dict) and any(k in item for k in required_keys):
                return item
    raise EnvelopeError(f"stdout holds no JSON object with any of {', '.join(required_keys)}")


def bridge_error(code: str, message: str, category: ErrorCategory, remediation: str, retryable: bool = False, **details: Any) -> ProviderError:
    err = HarnessError(code, message, category, retryable, remediation, details=details)
    return ProviderError(message, retryable=retryable, category=category, error=err)


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def envelope_text(value: Any) -> str:
    """The reply text of an envelope field: a string as is, nothing for null, JSON for anything structured."""
    if value is None:
        return ""
    return value if isinstance(value, str) else json_dumps(value)


def _image_extension(part: ContentPart) -> str:
    if part.path:
        ext = os.path.splitext(part.path)[1].lower()
        if ext in _IMAGE_EXTENSIONS:
            return ext
    guessed = mimetypes.guess_extension(part.mime or "") if part.mime else None
    return guessed if guessed in _IMAGE_EXTENSIONS else ".png"


# ----------------------------------------------------------------------------- the bridge
class CliBridgeProvider(Provider):
    """Base for print-mode CLI bridges. Subclasses define ``caps``, ``build_argv`` and ``parse_envelope``.

    ``runner`` is injectable (tests never spawn a real CLI); the default runs a subprocess in ``cwd``.
    ``prices`` maps ``input``/``output``/``cached``/``reasoning`` to USD per million tokens; without it the
    cost is what the CLI reports, or zero.
    """

    name = "cli-bridge"
    caps = Capabilities(native_tools=False, system_channel=False, images=False, streaming=False, json_schema=False, concurrency=DEFAULT_CONCURRENCY, multi_turn=False)
    install_hint = "Install the CLI, log in once, and make sure it is on PATH."
    login_hint = "Run the CLI once interactively and complete its login flow."
    settings_path = "the CLI's settings file"
    forbidden_flags: tuple[str, ...] = ()
    denial_patterns: tuple[re.Pattern[str], ...] = DENIAL_PATTERNS

    def __init__(
        self,
        cli: str,
        model: str,
        *,
        runner: Runner | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        concurrency: int = DEFAULT_CONCURRENCY,
        cwd: str | os.PathLike[str] | None = None,
        env_extra: dict[str, str] | None = None,
        effort_default: str = "medium",
        prices: dict[str, float] | None = None,
    ) -> None:
        if not cli:
            raise ConfigurationError("cli must name the executable to run, for example 'agy' or 'claude'.")
        if int(concurrency) < 1:
            raise ConfigurationError(f"concurrency must be at least 1 process; got {concurrency}.")
        if float(timeout_s) <= 0:
            raise ConfigurationError(f"timeout_s must be a positive number of seconds; got {timeout_s}.")
        if effort_default not in EFFORT_LEVELS:
            raise ConfigurationError(f"effort_default must be one of {', '.join(EFFORT_LEVELS)}; got {effort_default!r}.")
        self.cli = cli
        self.model = model
        self.cwd = os.fspath(cwd) if cwd else None
        if self.cwd is not None and not os.path.isdir(self.cwd):
            raise ConfigurationError(f"cwd must be an existing directory for {cli} to run in; got {self.cwd!r}.")
        self.env_extra = dict(env_extra or {})
        self.effort_default = effort_default
        self.prices = dict(prices) if prices else None
        self.concurrency = int(concurrency)
        self.timeout = Limit(
            f"{self.name} completion timeout",
            float(timeout_s),
            "seconds",
            source="default" if float(timeout_s) == DEFAULT_TIMEOUT_S else "setting",
            does_not_raise=("max_tokens", "the effort level"),
        )
        self.caps = replace(self.caps, concurrency=self.concurrency)
        self._runner: Runner = runner or subprocess_runner(self.cwd)
        self._semaphore = threading.Semaphore(self.concurrency)
        self._probe: ProbeResult | None = None
        self._probe_lock = threading.Lock()

    # ------------------------------------------------------------------ subclass contract
    @abstractmethod
    def build_argv(self, prompt_text: str, image_paths: list[str], effort: str, model: str, *, system_text: str = "", workdir: str | None = None) -> tuple[list[str], bytes | None]:
        """The argv to run and the bytes to feed on stdin (``None`` for no stdin)."""

    @abstractmethod
    def parse_envelope(self, stdout: str) -> tuple[str, Usage, dict[str, Any]]:
        """``(reply text, usage, extra)`` from the CLI's stdout. ``extra['request_id']`` names the CLI's own id.

        Raises ``EnvelopeError`` when stdout holds no envelope and ``ProviderError`` when the envelope reports a failure.
        """

    def default_env(self) -> dict[str, str]:
        """Variables every child of this bridge gets; ``env_extra`` overrides them."""
        return {}

    def resolve_model(self, model: str, effort: str) -> str:
        """The model id to send for this effort. The base keeps it unchanged."""
        return model

    def render_prompt(self, prompt_text: str, image_paths: list[str]) -> str:
        """The final prompt: the prepared text plus one ``@path`` line per staged image."""
        parts = [prompt_text.rstrip()] if prompt_text.strip() else []
        if image_paths:
            parts.append(ATTACHED_IMAGES_HEADER + "\n" + "\n".join(f"@{p}" for p in image_paths))
        return "\n\n".join(parts)

    # ------------------------------------------------------------------ completion
    def complete(self, request: CompletionRequest) -> Reply:
        req = self.prepare(request)
        system_text, prompt_text, images = self.split_request(req)
        system_text, prompt_text = strip_nul(system_text), strip_nul(prompt_text)
        effort = self.normalise_effort(req.effort)
        model = self.resolve_model(self.model, effort)
        workdir = tempfile.mkdtemp(prefix=f"edgent-{self.name}-")
        try:
            image_paths = self.stage_images(images, workdir)
            if not prompt_text.strip() and not image_paths:
                raise bridge_error(
                    "empty_prompt",
                    "the request has no user text and no image to send.",
                    ErrorCategory.invalid_input,
                    "Add a user message (or an image) to the request; a system prompt alone is not a completion.",
                )
            prompt = self.render_prompt(prompt_text, image_paths)
            self.check_prompt_size(prompt)
            argv, stdin = self.build_argv(prompt, image_paths, effort, model, system_text=system_text, workdir=workdir)
            self.guard_argv(argv)
            result, latency_s = self.run_cli(argv, stdin)
            reply = self.interpret(result, argv, model, latency_s, had_images=bool(image_paths))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        return self.finalize(reply)

    def split_request(self, req: CompletionRequest) -> tuple[str, str, list[ContentPart]]:
        """``(system text, prompt text, image parts)`` of a prepared request.

        ``prepare`` has already folded tools, system text and turns as the caps demand, so the prompt is the
        text of the remaining message(s); media stand-ins are already inside that text and images are staged
        separately.
        """
        system_text = req.system_text()
        rest = req.non_system()
        prompt_text = "\n\n".join(m.text() for m in rest if m.text())
        images = [p for m in rest for p in m.media() if p.type == "image"]
        return system_text, prompt_text, images

    def normalise_effort(self, effort: str | None) -> str:
        return effort if effort in EFFORT_LEVELS else self.effort_default

    def stage_images(self, images: list[ContentPart], workdir: str) -> list[str]:
        """Copy image parts into the per-call directory under safe names; returns absolute paths."""
        if len(images) > IMAGES_PER_CALL_MAX:
            raise bridge_error(
                "too_many_images",
                f"{len(images)} images attached; this bridge stages at most {IMAGES_PER_CALL_MAX} per call.",
                ErrorCategory.resource_constraint,
                f"Attach at most {IMAGES_PER_CALL_MAX} images per call, or split the request.",
                observed=len(images),
                limit=IMAGES_PER_CALL_MAX,
            )
        paths: list[str] = []
        for i, part in enumerate(images, 1):
            dest = os.path.join(workdir, f"image-{i}{_image_extension(part)}")
            try:
                if part.data_b64:
                    with open(dest, "wb") as f:
                        f.write(base64.b64decode(part.data_b64))
                elif part.path:
                    shutil.copyfile(part.path, dest)
                else:
                    raise bridge_error("image_missing", "an image part has neither a path nor inline data", ErrorCategory.invalid_input, "Attach images by path or as inline base64 data.")
            except FileNotFoundError as e:
                raise bridge_error("image_not_found", f"image attachment not found: {part.path}", ErrorCategory.not_found, "Attach an existing image file; check the path first.", path=part.path) from e
            except PermissionError as e:
                raise bridge_error("image_unreadable", f"image attachment cannot be read: {e}", ErrorCategory.permission_denied, "Attach an image file this process may read.", path=part.path) from e
            except (OSError, ValueError) as e:  # a directory, a failed copy, or inline data that is not base64
                raise bridge_error("image_invalid", f"image attachment could not be staged: {e}", ErrorCategory.invalid_input, "Attach a readable image file, or inline data as valid base64.", path=part.path) from e
            paths.append(dest)
        return paths

    def check_prompt_size(self, prompt: str) -> None:
        """Refuse, never truncate, a prompt over the provider's byte ceiling (the exact bytes that will be sent)."""
        limit = self.prompt_limit()
        if limit is None:
            return
        size = len(prompt.encode("utf-8"))
        if size > limit.value:
            err = limit.exceeded(size, "Send less: compact the session, drop old tool results, or split the task.", ErrorCategory.truncated)
            raise ProviderError(err.message, retryable=False, category=ErrorCategory.truncated, error=err)

    def guard_argv(self, argv: list[str]) -> None:
        bad = [a for a in argv[1:] if a in self.forbidden_flags]
        if bad:
            raise ConfigurationError(f"{self.name} must never pass {', '.join(bad)}; refusing to start {self.cli}.")

    def child_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(self.default_env())
        env.update(self.env_extra)
        return env

    def run_cli(self, argv: list[str], stdin: bytes | None, timeout_s: float | None = None) -> tuple[CompletedLike, float]:
        """Run one CLI process under the concurrency semaphore; OS-level failures become typed errors."""
        budget = self.timeout.value + TIMEOUT_GRACE_S if timeout_s is None else timeout_s
        env = self.child_env()
        with self._semaphore:
            t0 = time.monotonic()  # the CLI's own time, not the wait for a slot
            try:
                result = self._runner(argv, stdin, budget, env)
            except FileNotFoundError as e:
                raise self.unavailable_error(str(e)) from e
            except subprocess.TimeoutExpired as e:
                raise self.timeout_error(budget) from e
            except PermissionError as e:
                raise self.unavailable_error(f"not executable: {e}") from e
            except ValueError as e:  # Popen refuses an argv or environment string with an embedded NUL
                raise bridge_error("cli_argv_invalid", f"could not start {self.cli}: {e}.", ErrorCategory.invalid_input, "Remove the NUL bytes from the request text or this provider's env_extra and retry.") from e
            except OSError as e:
                if e.errno == errno.E2BIG:
                    raise bridge_error(
                        "argv_too_long",
                        f"the operating system refused the command line: one argument is over {ARGV_STRING_MAX_BYTES} bytes ({e}).",
                        ErrorCategory.truncated,
                        "Send less: compact the session or split the task, and keep this provider's prompt ceiling below 131072 bytes.",
                    ) from e
                raise bridge_error("cli_start_failed", f"could not start {self.cli}: {e}", ErrorCategory.execution_failure, f"{self.install_hint} Then run `edgent doctor`.") from e
        return result, time.monotonic() - t0

    def interpret(self, result: CompletedLike, argv: list[str], model: str, latency_s: float, had_images: bool = False) -> Reply:
        """Turn the finished process into a ``Reply`` or a typed ``ProviderError``."""
        stdout, stderr = decode_output(result.stdout), decode_output(result.stderr)
        returncode = _int(result.returncode)
        denied = self.detect_denial(stderr)
        try:
            text, usage, extra = self.parse_envelope(stdout)
        except EnvelopeError as e:
            if denied:
                raise self.denial_error(denied, stderr) from e
            raise self.classify_failure(returncode, stdout, stderr, str(e)) from e
        except ProviderError as e:  # a failing envelope status caused by the denial: the settings line is the fix
            if denied:
                raise self.denial_error(denied, stderr) from e
            raise
        warnings: list[str] = []
        if denied:
            if not text.strip() or had_images:
                raise self.denial_error(denied, stderr)
            warnings.append(f'the CLI denied the "{denied}" tool during this call; the answer was produced without it')
        if returncode != 0:
            if not text.strip():
                raise self.classify_failure(returncode, stdout, stderr, f"exit status {returncode}")
            warnings.append(f"the CLI exited with status {returncode} but still returned a response")
        usage = self.price(usage, extra)
        raw: dict[str, Any] = {"cli": self.cli, "argv": redact_argv(argv), "returncode": returncode, "envelope": extra}
        if stderr.strip():
            raw["stderr_tail"] = tail(stderr)
        if warnings:
            raw["warnings"] = warnings
        return Reply(text=text, usage=usage, model=model, latency_s=latency_s, request_id=str(extra.get("request_id") or ""), raw=raw)

    def price(self, usage: Usage, extra: dict[str, Any]) -> Usage:
        """Cost from the price table when there is one, else what the CLI reported, else zero."""
        if self.prices:
            per = {k: float(self.prices.get(k, 0.0)) / 1_000_000 for k in ("input", "output", "cached", "reasoning")}
            if "reasoning" not in self.prices:
                per["reasoning"] = per["output"]
            uncached = max(usage.prompt_tokens - usage.cached_tokens, 0)
            usage.cost_usd = uncached * per["input"] + usage.cached_tokens * per["cached"] + usage.completion_tokens * per["output"] + usage.reasoning_tokens * per["reasoning"]
        elif extra.get("reported_cost_usd") is not None:
            try:
                usage.cost_usd = float(extra["reported_cost_usd"])
            except (TypeError, ValueError):
                usage.cost_usd = 0.0
        return usage

    # ------------------------------------------------------------------ classification
    def detect_denial(self, stderr: str) -> str | None:
        """The tool the CLI auto-denied in headless mode, when stderr says so."""
        for rx in self.denial_patterns:
            m = rx.search(stderr or "")
            if m:
                return m.group("tool")
        return None

    def denial_error(self, tool: str, stderr: str) -> ProviderError:
        line = f'"permissions": {{"allow": ["{tool}(*)"]}}'
        return bridge_error(
            "cli_permission_denied",
            f'{self.cli} auto-denied the "{tool}" tool in headless mode, so the request could not be completed.',
            ErrorCategory.permission_denied,
            f"Add this to {self.settings_path} and retry: {line}. Edgent explains this line and never edits that file.",
            tool=tool,
            settings_path=self.settings_path,
            settings_line=line,
            stderr_tail=tail(stderr),
        )

    def unavailable_error(self, detail: str) -> ProviderError:
        return bridge_error(
            "cli_unavailable",
            f"{self.cli} is not installed or not on PATH ({detail}).",
            ErrorCategory.unavailable,
            f"{self.install_hint} Then run `edgent doctor` to confirm. Until then, use another provider or an API key.",
            cli=self.cli,
        )

    def timeout_error(self, timeout_s: float) -> ProviderError:
        return bridge_error(
            "cli_timeout",
            f"{self.cli} did not finish within {timeout_s:g} s and was killed. {self.timeout.describe()}",
            ErrorCategory.timeout,
            "Split the task or ask for a shorter answer, then retry once. To allow longer completions, raise timeout_s for this provider.",
            retryable=True,
            timeout_s=timeout_s,
        )

    def classify_failure(self, returncode: int, stdout: str, stderr: str, reason: str) -> ProviderError:
        blob = f"{stderr}\n{stdout}"
        detail = tail(stderr) or tail(stdout) or reason
        if _AUTH_RE.search(blob):
            return bridge_error("cli_not_logged_in", f"{self.cli} refused the request: not logged in or credentials expired. {detail}", ErrorCategory.permission_denied, f"{self.login_hint} Then retry.", returncode=returncode)
        if _RATE_RE.search(blob):
            return bridge_error("cli_rate_limited", f"{self.cli} is rate limited or out of quota. {detail}", ErrorCategory.resource_constraint, "Wait and retry with backoff, or lower this provider's concurrency.", retryable=True, returncode=returncode)
        if _UNKNOWN_FLAG_RE.search(blob):
            return bridge_error("cli_flag_unsupported", f"{self.cli} rejected a flag; this release differs from the measured one. {detail}", ErrorCategory.configuration_invalid, "Run `edgent doctor` to re-probe the CLI, update it, or configure the flag spelling for this provider.", returncode=returncode)
        if _TIMEOUT_RE.search(blob):
            return bridge_error("cli_timeout", f"{self.cli} reported a timeout. {detail}", ErrorCategory.timeout, "Split the task or ask for a shorter answer, then retry once.", retryable=True, returncode=returncode)
        return bridge_error("cli_failed", f"{self.cli} failed ({reason}): {detail}", ErrorCategory.execution_failure, "Read the CLI's message and change the request; run `edgent doctor` if the installation looks broken. Do not repeat the unchanged call.", returncode=returncode)

    # ------------------------------------------------------------------ probe and health
    def probe(self, force: bool = False) -> ProbeResult:
        """Run ``--version`` and ``--help`` once (cached) and record which flags this release has."""
        with self._probe_lock:
            if self._probe is None or force:
                self._probe = self._run_probe()
            return self._probe

    def _run_probe(self) -> ProbeResult:
        executable = self.cli if os.path.isabs(self.cli) else shutil.which(self.cli)
        env = self.child_env()
        version, help_text, errors = "", "", []
        for flag in ("--version", "--help"):
            try:
                r = self._runner([self.cli, flag], None, PROBE_TIMEOUT_S, env)
            except FileNotFoundError as e:
                return ProbeResult(False, executable, error=f"{self.cli} is not on PATH ({e})")
            except subprocess.TimeoutExpired:
                errors.append(f"{flag} did not answer within {PROBE_TIMEOUT_S:g} s")
                continue
            except OSError as e:
                errors.append(f"{flag}: {e}")
                continue
            out = decode_output(r.stdout)
            if flag == "--version":
                version = next((ln.strip() for ln in out.splitlines() if ln.strip()), "")[:120]
            else:
                help_text = out + "\n" + decode_output(r.stderr)
        flags = parse_help_flags(help_text)
        return ProbeResult(bool(version or flags), executable, version, flags, "; ".join(errors) or None)

    def health(self) -> dict[str, Any]:
        p = self.probe()
        out: dict[str, Any] = {
            "ok": p.available,
            "provider": self.name,
            "model": self.model,
            "cli": self.cli,
            "executable": p.executable,
            "version": p.version,
            "flags": len(p.flags),
            "concurrency": self.concurrency,
            "timeout_s": self.timeout.value,
        }
        if p.error:
            out["error"] = p.error
        if not p.available:
            out["remediation"] = self.install_hint
        return out


__all__ = [
    "ARGV_STRING_MAX_BYTES",
    "CliBridgeProvider",
    "CompletedLike",
    "DEFAULT_CONCURRENCY",
    "DEFAULT_TIMEOUT_S",
    "EnvelopeError",
    "IMAGES_PER_CALL_MAX",
    "PROBE_TIMEOUT_S",
    "ProbeResult",
    "Runner",
    "bridge_error",
    "decode_output",
    "envelope_text",
    "parse_help_flags",
    "parse_json_envelope",
    "redact_argv",
    "strip_nul",
    "subprocess_runner",
    "tail",
]
