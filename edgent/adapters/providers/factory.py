"""Choose and build the provider from settings, the environment and what is on PATH.

Order: an explicit kind (the ``kind`` argument, ``settings['provider']['kind']``,
or ``EDGENT_PROVIDER``); else a Gemini key in the environment; else an
OpenAI-compatible base URL (or an OpenAI key); else the ``agy`` CLI on PATH;
else the ``claude`` CLI on PATH; else the offline ``MockProvider`` carrying a
warning in ``provider.meta``. Keys come from the environment, either by the
default names or through ``settings['provider']['api_key_env']`` (the name of
a variable). A plain ``api_key`` in a settings file is accepted with a
warning. Nothing here logs or echoes a key: ``ProviderSpec`` hides it from
``repr`` and ``describe``.
"""
from __future__ import annotations

import importlib
import inspect
import os
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ...domain.errors import ConfigurationError
from .base import Provider
from .http import Opener

KINDS = ("gemini", "openai", "agy", "claude", "mock")
PROVIDER_ENV = "EDGENT_PROVIDER"
MODEL_ENV = "EDGENT_MODEL"
GEMINI_KEY_ENVS = ("EDGENT_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")
GEMINI_MODEL_ENVS = ("EDGENT_GEMINI_MODEL",)
OPENAI_BASE_URL_ENVS = ("EDGENT_OPENAI_BASE_URL", "OPENAI_BASE_URL")
OPENAI_KEY_ENVS = ("EDGENT_OPENAI_API_KEY", "OPENAI_API_KEY")
OPENAI_MODEL_ENVS = ("EDGENT_OPENAI_MODEL", "OPENAI_MODEL")
BRIDGES = {
    "agy": ("edgent.adapters.providers.agy_bridge", "AgyProvider"),
    "claude": ("edgent.adapters.providers.claude_bridge", "ClaudeCliProvider"),
}
NO_PROVIDER_WARNING = "no provider configured"

Which = Callable[[str], str | None]


@dataclass
class ProviderSpec:
    kind: str
    model: str | None = None
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    source: str = ""  # how the kind was chosen, for `edgent doctor`
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        """A loggable view: the key is reported as present or absent, never by value."""
        return {
            "kind": self.kind,
            "model": self.model,
            "base_url": self.base_url,
            "api_key": "[set]" if self.api_key else "[not set]",
            "options": sorted(self.options),
            "source": self.source,
            "warnings": list(self.warnings),
        }


def _first(env: Mapping[str, str], names: tuple[str, ...]) -> tuple[str | None, str | None]:
    for n in names:
        v = env.get(n)
        if v:
            return v, n
    return None, None


def _provider_settings(settings: Mapping[str, Any] | None) -> dict[str, Any]:
    section = (settings or {}).get("provider")
    if section is None:
        return {}
    if not isinstance(section, Mapping):
        raise ConfigurationError("settings.provider must be an object such as {\"kind\": \"openai\", \"model\": \"...\"}.")
    return dict(section)


def _resolve_key(conf: Mapping[str, Any], env: Mapping[str, str], default_envs: tuple[str, ...], warnings: list[str]) -> str | None:
    env_name = conf.get("api_key_env")
    if env_name:
        value = env.get(str(env_name))
        if not value:
            raise ConfigurationError(f"settings.provider.api_key_env names {env_name}, which is not set in the environment that starts Edgent.")
        return value
    plain = conf.get("api_key")
    if plain:
        warnings.append("settings.provider.api_key is stored in plain text; prefer api_key_env naming an environment variable.")
        return str(plain)
    value, _ = _first(env, default_envs)
    return value


def resolve_spec(
    settings: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
    which: Which | str | None = None,
    kind: str | None = None,
) -> ProviderSpec:
    """Decide which provider to build without building it. ``which`` is the PATH lookup (default ``shutil.which``)."""
    env: Mapping[str, str] = _environ(environ)
    if isinstance(which, str):  # tolerate make_provider(which="gemini") as the explicit kind
        kind, which = which, None
    lookup: Which = which or shutil.which
    conf = _provider_settings(settings)
    warnings: list[str] = []
    chosen, source = _choose_kind(kind, conf, env, lookup)
    options = dict(conf.get("options") or {})
    if options.get("api_key"):  # a key tucked into options gets the same plain-text handling as a top-level one
        conf = {**conf, "api_key": conf.get("api_key") or options.pop("api_key")}
    model = conf.get("model") or None
    base_url = conf.get("base_url") or None
    api_key: str | None = None
    if chosen == "gemini":
        api_key = _resolve_key(conf, env, GEMINI_KEY_ENVS, warnings)
        if not api_key:
            raise ConfigurationError("provider gemini needs an API key: set EDGENT_GEMINI_API_KEY (or GEMINI_API_KEY), or settings.provider.api_key_env.")
        model = model or _first(env, GEMINI_MODEL_ENVS)[0] or env.get(MODEL_ENV) or None
    elif chosen == "openai":
        api_key = _resolve_key(conf, env, OPENAI_KEY_ENVS, warnings)
        base_url = base_url or _first(env, OPENAI_BASE_URL_ENVS)[0] or None
        model = model or _first(env, OPENAI_MODEL_ENVS)[0] or env.get(MODEL_ENV) or None
        if not base_url:
            if not api_key:
                raise ConfigurationError("provider openai needs a base URL: set EDGENT_OPENAI_BASE_URL (for example http://localhost:8000/v1) or settings.provider.base_url.")
            from .openai_compat import DEFAULT_BASE_URL

            base_url = DEFAULT_BASE_URL
        if not model:
            raise ConfigurationError("provider openai needs a model id: set EDGENT_OPENAI_MODEL (or OPENAI_MODEL, EDGENT_MODEL) or settings.provider.model; `edgent doctor` lists what the server serves.")
    else:
        model = model or env.get(MODEL_ENV) or None
    return ProviderSpec(kind=chosen, model=model, api_key=api_key, base_url=base_url, options=options, source=source, warnings=warnings)


def _choose_kind(kind: str | None, conf: Mapping[str, Any], env: Mapping[str, str], lookup: Which) -> tuple[str, str]:
    explicit = kind or conf.get("kind") or env.get(PROVIDER_ENV)
    if explicit:
        explicit = str(explicit).lower()
        if explicit not in KINDS:
            raise ConfigurationError(f"unknown provider kind {explicit!r}; choose one of {', '.join(KINDS)}.")
        origin = "argument" if kind else ("settings.provider.kind" if conf.get("kind") else PROVIDER_ENV)
        return explicit, origin
    _, key_env = _first(env, GEMINI_KEY_ENVS)
    if key_env:
        return "gemini", f"env:{key_env}"
    _, url_env = _first(env, OPENAI_BASE_URL_ENVS)
    if url_env:
        return "openai", f"env:{url_env}"
    _, openai_key_env = _first(env, OPENAI_KEY_ENVS)
    if openai_key_env:
        return "openai", f"env:{openai_key_env}"
    for cli in ("agy", "claude"):
        if lookup(cli):
            return cli, f"path:{cli}"
    return "mock", "fallback"


def make_provider(
    settings: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
    which: Which | str | None = None,
    kind: str | None = None,
    opener: Opener | None = None,
) -> Provider:
    """Build the provider for this run. Every provider gets a ``meta`` dict (kind, source, warnings)."""
    spec = resolve_spec(settings, environ, which, kind)
    provider = build(spec, opener=opener)
    meta: dict[str, Any] = {"kind": spec.kind, "source": spec.source, "warnings": list(spec.warnings)}
    if spec.kind == "mock" and spec.source == "fallback":
        meta["warning"] = NO_PROVIDER_WARNING
        meta["hint"] = "Set a Gemini key, an OpenAI-compatible base URL, or install agy or claude; see describe_options()."
    provider.meta = meta  # type: ignore[attr-defined]  # a plain attribute; the port does not define it
    return provider


def build(spec: ProviderSpec, opener: Opener | None = None) -> Provider:
    candidates: dict[str, Any] = {"model": spec.model, "opener": opener}
    if spec.kind == "gemini":
        from .gemini import GeminiProvider

        candidates.update(api_key=spec.api_key, base_url=spec.base_url)
        return _construct(GeminiProvider, "gemini", candidates, spec.options)
    if spec.kind == "openai":
        from .openai_compat import OpenAICompatProvider

        candidates.update(api_key=spec.api_key or "", base_url=spec.base_url)
        return _construct(OpenAICompatProvider, "openai", candidates, spec.options)
    if spec.kind in BRIDGES:
        module_name, class_name = BRIDGES[spec.kind]
        try:
            cls = getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError) as e:
            raise ConfigurationError(
                f"provider {spec.kind} needs {module_name}.{class_name}, which this build does not provide ({e}). "
                f"Choose another provider with {PROVIDER_ENV} or settings.provider.kind."
            ) from e
        return _construct(cls, spec.kind, candidates, spec.options)
    if spec.kind == "mock":
        from .mock import MockProvider

        return _construct(MockProvider, "mock", {}, spec.options)
    raise ConfigurationError(f"unknown provider kind {spec.kind!r}; choose one of {', '.join(KINDS)}.")


def _construct(cls: type, kind: str, candidates: Mapping[str, Any], options: Mapping[str, Any]) -> Any:
    """Call a constructor with only the keyword arguments it declares; unknown settings options are an error."""
    try:
        params = inspect.signature(cls).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins without signatures
        return cls(**{**candidates, **options})
    takes_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    unknown = sorted(k for k in options if k not in params and not takes_kwargs)
    if unknown:
        raise ConfigurationError(f"settings.provider.options for {kind} has unknown keys {unknown}; accepted: {sorted(params)}.")
    kwargs = {k: v for k, v in candidates.items() if v is not None and (k in params or takes_kwargs)}
    kwargs.update(options)
    try:
        return cls(**kwargs)
    except TypeError as e:
        raise ConfigurationError(f"cannot build provider {kind}: {e}") from e


def describe_options() -> str:
    """How to configure each provider, for `edgent doctor` and the README."""
    rows = [
        ("kind", "selects when", "model", "credentials"),
        ("gemini", "EDGENT_GEMINI_API_KEY or GEMINI_API_KEY is set", "EDGENT_GEMINI_MODEL (default gemini-2.5-pro)", "the key above; never in a settings file"),
        ("openai", "EDGENT_OPENAI_BASE_URL or OPENAI_BASE_URL is set (…/v1), or OPENAI_API_KEY for api.openai.com", "EDGENT_OPENAI_MODEL or OPENAI_MODEL (required)", "OPENAI_API_KEY when the server needs one"),
        ("agy", "the agy CLI is on PATH and nothing above applies", "EDGENT_MODEL", "the CLI's own login"),
        ("claude", "the claude CLI is on PATH and nothing above applies", "EDGENT_MODEL", "the CLI's own login"),
        ("mock", "nothing applies (offline; tests and demos)", "mock", "none"),
    ]
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    lines = ["  ".join(c.ljust(widths[i]) for i, c in enumerate(r)).rstrip() for r in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    lines += [
        "",
        f"Force a kind with {PROVIDER_ENV}=<kind> or settings.provider.kind. Settings file (.edgent/settings.json or ~/.edgent/settings.json):",
        '  {"provider": {"kind": "openai", "model": "qwen3-32b", "base_url": "http://localhost:8000/v1",',
        '                "api_key_env": "MY_SERVER_KEY", "options": {"thinking_switch": true}}}',
        "Put keys in environment variables and name the variable in api_key_env; a plain api_key in the file is accepted with a warning.",
        "options are passed to the provider constructor (for example thinking_switch, reasoning_effort_param, timeout_s, thinking_budgets, prices).",
    ]
    return "\n".join(lines)


def _environ(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environ is None else environ
