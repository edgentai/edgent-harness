"""HTTP plumbing shared by the API providers.

``request_json`` (with the ``post_json`` and ``get_json`` wrappers) sends one
JSON request through an injectable opener and returns the decoded body. It
retries transport errors and the retryable statuses with exponential backoff
(honouring ``Retry-After`` within a cap), tags every request with an
``X-Request-Id`` header, refuses bodies over the named response-size limit,
and never lets an auth header value reach an error message. Tests inject an
opener and a sleep function, so nothing here touches the network or the clock.
"""
from __future__ import annotations

import http.client
import json
import re
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

from ...domain.errors import ConfigurationError, ErrorCategory, HarnessError, Limit
from ...domain.types import Usage
from ...domain.utils import IdGenerator
from .base import ProviderError

DEFAULT_TIMEOUT_S = 120.0
DEFAULT_RETRIES = 3  # additional attempts after the first one
DEFAULT_BACKOFF_S = 0.5  # first delay; doubles per attempt
BACKOFF_CAP_S = 20.0
RETRY_AFTER_CAP_S = 30.0  # a server may ask for an hour; we will not block a session that long
ERROR_BODY_SNIPPET_CHARS = 500
MIN_SECRET_CHARS = 4  # shorter header values are not worth scrubbing and would mangle messages
USER_AGENT = "edgent/0.1"
RETRYABLE_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
AUTH_HEADERS = frozenset({"authorization", "proxy-authorization", "x-goog-api-key", "x-api-key", "api-key"})
TOKENS_PER_PRICE_UNIT = 1_000_000  # prices are quoted per million tokens

RESPONSE_SIZE_LIMIT = Limit(
    name="provider response size",
    value=16 * 1024 * 1024,
    unit="bytes",
    does_not_raise=("max_tokens", "timeout_s", "max_retries"),
)

_ids = IdGenerator()
_BEARER_RE = re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=\-]+")
_QUERY_SECRET_RE = re.compile(r"(?i)([?&](?:key|api_key|apikey|access_token|token)=)[^&\s]+")


@dataclass
class HttpRequest:
    """What an opener receives. Plain data, so fakes can assert on it without urllib quirks."""

    method: str
    url: str
    headers: dict[str, str]
    body: bytes | None
    timeout_s: float
    max_response_bytes: int

    def json(self) -> Any:
        return json.loads(self.body or b"null")


@dataclass
class HttpResponse:
    status: int
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)

    def header(self, name: str) -> str | None:
        low = name.lower()
        for k, v in self.headers.items():
            if k.lower() == low:
                return v
        return None

    def text(self) -> str:
        return self.body.decode("utf-8", "replace")

    def json(self) -> Any:
        return json.loads(self.text())


Opener = Callable[[HttpRequest], HttpResponse]
Sleep = Callable[[float], None]


def urllib_opener(req: HttpRequest) -> HttpResponse:
    """The production opener: urllib, proxy settings from the environment, a hard read cap."""
    import urllib.error
    import urllib.request

    r = urllib.request.Request(req.url, data=req.body, method=req.method)
    for k, v in req.headers.items():
        r.add_header(k, v)
    cap = req.max_response_bytes + 1  # one extra byte tells us the body was over the limit
    try:
        with urllib.request.urlopen(r, timeout=req.timeout_s) as resp:  # noqa: S310 - scheme is fixed by the provider
            return HttpResponse(int(resp.status), resp.read(cap), dict(resp.headers.items()))
    except urllib.error.HTTPError as e:
        try:
            body = e.read(cap)
        except Exception:  # noqa: BLE001 - an error without a body is still an error
            body = b""
        return HttpResponse(int(e.code), body, dict(e.headers.items()) if e.headers else {})


# ----------------------------------------------------------------------------- redaction
def secret_values(headers: Mapping[str, str]) -> list[str]:
    """Header values that must never appear in logs or error text."""
    out: list[str] = []
    for k, v in headers.items():
        if k.lower() in AUTH_HEADERS and v:
            # keep the scheme word readable; only the token is secret
            out.append(v[7:].strip() if v.lower().startswith("bearer ") else v)
    return out


def redact_text(text: str, secrets: Iterable[str] = ()) -> str:
    out = text
    for s in sorted({s for s in secrets if s and len(s) >= MIN_SECRET_CHARS}, key=len, reverse=True):
        out = out.replace(s, "[REDACTED]")
    out = _BEARER_RE.sub(r"\1[REDACTED]", out)
    return _QUERY_SECRET_RE.sub(r"\1[REDACTED]", out)


def validate_base_url(url: str, example: str) -> str:
    """The API root a provider builds on: an absolute http(s) URL, returned without its trailing slash.

    A scheme-less value such as ``localhost:8000/v1`` is refused here because urllib turns it into a
    transport error, which would be retried and reported as a network problem instead of a setting.
    """
    if not isinstance(url, str) or not url.strip():
        raise ConfigurationError(f"a base URL is required (for example {example}).")
    url = url.strip().rstrip("/")
    if not url.lower().startswith(("http://", "https://")):
        raise ConfigurationError(f"base URL {redact_text(url)!r} must start with http:// or https:// (for example {example}).")
    return url


# ----------------------------------------------------------------------------- errors
def status_category(status: int) -> ErrorCategory:
    if status in (400, 422):
        return ErrorCategory.invalid_input
    if status in (401, 403):
        return ErrorCategory.permission_denied
    if status == 404:
        return ErrorCategory.not_found
    if status == 413:
        return ErrorCategory.truncated
    if status == 429:
        return ErrorCategory.resource_constraint
    if status in (408, 504):
        return ErrorCategory.timeout
    if status >= 500:
        return ErrorCategory.unavailable
    return ErrorCategory.execution_failure


def status_remediation(status: int) -> str:
    if status in (401, 403):
        return "Check the API key or credentials for this provider (see `edgent doctor`); do not retry with the same ones."
    if status == 404:
        return "Check the base URL and the model id; an OpenAI-compatible base URL usually ends with /v1."
    if status in (400, 422):
        return "Read the message and change the request (schema, media type, model id); do not repeat the unchanged call."
    if status == 413:
        return "Send less: compact the session, drop old tool results, or shrink the attachments."
    if status == 429:
        return "The provider's rate or quota limit was hit; lower concurrency or wait before retrying."
    if status >= 500 or status in (408,):
        return "The provider is unavailable right now; retry later or switch provider."
    return "Do not repeat the unchanged call."


def error_text(body: bytes) -> str:
    """The human part of an error body: ``error.message`` when the body is JSON, else a snippet."""
    text = body.decode("utf-8", "replace").strip()
    try:
        data = json.loads(text)
    except ValueError:
        return text[:ERROR_BODY_SNIPPET_CHARS]
    msg: Any = None
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            msg = err.get("message") or err.get("status")
        elif isinstance(err, str):
            msg = err
        msg = msg or data.get("message") or data.get("detail")
    return (str(msg) if msg else text)[:ERROR_BODY_SNIPPET_CHARS]


def error_status(err: Any) -> int | None:
    """The HTTP-like status an error object carries in ``code``, when it is one."""
    code = err.get("code") if isinstance(err, Mapping) else None
    if isinstance(code, str) and code.isdigit():
        code = int(code)
    if isinstance(code, int) and not isinstance(code, bool) and 100 <= code <= 599:
        return code
    return None


def unreadable_response(provider: str, data: Any, exc: Exception) -> ProviderError:
    """A 2xx JSON body the provider cannot read: the wrong endpoint or API version, most likely."""
    shape = sorted(str(k) for k in data)[:12] if isinstance(data, Mapping) else type(data).__name__
    msg = f"{provider} returned a response this provider cannot read ({type(exc).__name__}: {str(exc)[:200]}); top-level keys: {shape}"
    err = HarnessError(
        "provider_bad_response",
        msg,
        ErrorCategory.execution_failure,
        False,
        "Check that the base URL points at the right API and version and that the model id is served there; do not repeat the unchanged call.",
    )
    return ProviderError(msg, retryable=False, category=ErrorCategory.execution_failure, error=err)


def retry_delay_s(backoff_s: float, attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            return max(0.0, min(float(retry_after), RETRY_AFTER_CAP_S))
        except ValueError:
            pass  # an HTTP-date Retry-After falls back to the backoff schedule
    return min(backoff_s * (2 ** (attempt - 1)), BACKOFF_CAP_S)


def _provider_error(
    message: str,
    *,
    status: int | None,
    retryable: bool,
    category: ErrorCategory,
    remediation: str,
    secrets: list[str],
    details: dict[str, Any],
) -> ProviderError:
    msg = redact_text(message, secrets)
    clean_details = {k: (redact_text(v, secrets) if isinstance(v, str) else v) for k, v in details.items()}
    code = f"provider_http_{status}" if status else "provider_transport"
    err = HarnessError(code, msg, category, retryable, remediation, details=clean_details)
    return ProviderError(msg, status=status, retryable=retryable, category=category, error=err)


# ----------------------------------------------------------------------------- requests
def request_json(
    method: str,
    url: str,
    payload: Any | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    opener: Opener | None = None,
    retries: int = DEFAULT_RETRIES,
    backoff_s: float = DEFAULT_BACKOFF_S,
    request_id: str | None = None,
    sleep: Sleep | None = None,
    response_limit: Limit = RESPONSE_SIZE_LIMIT,
) -> dict[str, Any]:
    """Send JSON, return the decoded JSON object. Raises ``ProviderError`` with the canonical error shape."""
    opener = opener or urllib_opener
    sleep = sleep or time.sleep
    request_id = request_id or _ids.new("req")
    hdrs: dict[str, str] = {"Accept": "application/json", "User-Agent": USER_AGENT, "X-Request-Id": request_id}
    if payload is not None:
        hdrs["Content-Type"] = "application/json; charset=utf-8"
    hdrs.update(headers or {})
    secrets = secret_values(hdrs)
    # "replace": a lone surrogate in a tool result (a file decoded with surrogateescape) must not crash the call
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8", "replace") if payload is not None else None
    attempts = max(0, int(retries)) + 1
    base_details: dict[str, Any] = {"url": url, "request_id": request_id, "method": method}
    last_failure = ""
    for attempt in range(1, attempts + 1):
        req = HttpRequest(method, url, dict(hdrs), body, timeout_s, int(response_limit.value))
        try:
            resp = opener(req)
        except ValueError as e:  # a malformed URL: refused before any connection, so never retried
            raise _provider_error(
                f"{method} {url} is not a request this client can send: {e}",
                status=None,
                retryable=False,
                category=ErrorCategory.configuration_invalid,
                remediation="Check the base URL: it must start with http:// or https:// and name the API root (…/v1 for an OpenAI-compatible server).",
                secrets=secrets,
                details=dict(base_details),
            ) from e
        except (OSError, http.client.HTTPException) as e:
            last_failure = f"{type(e).__name__}: {e}"
            if attempt < attempts:
                sleep(retry_delay_s(backoff_s, attempt, None))
                continue
            timed_out = isinstance(e, TimeoutError) or "timed out" in str(e).lower()
            return_category = ErrorCategory.timeout if timed_out else ErrorCategory.unavailable
            raise _provider_error(
                f"{method} {url} failed after {attempts} attempt(s): {last_failure}",
                status=None,
                retryable=True,
                category=return_category,
                remediation=(
                    "Check the network and the base URL, then retry; the request was tagged with "
                    f"X-Request-Id {request_id}."
                ),
                secrets=secrets,
                details={**base_details, "attempts": attempts, "transport_error": last_failure},
            ) from e
        if len(resp.body) > response_limit.value:
            # resource_constraint, not truncated: the loop answers "truncated" by compacting the prompt, which cannot help here
            err = response_limit.exceeded(len(resp.body), "Ask for less output (smaller max_tokens, fewer results).")
            raise ProviderError(err.message, status=resp.status, retryable=False, category=ErrorCategory.resource_constraint, error=err)
        if resp.status in RETRYABLE_STATUSES:
            last_failure = f"HTTP {resp.status}: {error_text(resp.body)}"
            if attempt < attempts:
                sleep(retry_delay_s(backoff_s, attempt, resp.header("Retry-After")))
                continue
            raise _provider_error(
                f"{method} {url} failed after {attempts} attempt(s): {last_failure}",
                status=resp.status,
                retryable=True,
                category=status_category(resp.status),
                remediation=status_remediation(resp.status),
                secrets=secrets,
                details={**base_details, "status": resp.status, "attempts": attempts, "body": error_text(resp.body)},
            )
        if resp.status < 200 or resp.status >= 300:
            snippet = error_text(resp.body)
            raise _provider_error(
                f"HTTP {resp.status} from {method} {url}: {snippet}",
                status=resp.status,
                retryable=False,
                category=status_category(resp.status),
                remediation=status_remediation(resp.status),
                secrets=secrets,
                details={**base_details, "status": resp.status, "attempts": attempt, "body": snippet},
            )
        return _decode(resp, method, url, secrets, base_details)
    raise AssertionError("unreachable")  # pragma: no cover


def _decode(resp: HttpResponse, method: str, url: str, secrets: list[str], details: dict[str, Any]) -> dict[str, Any]:
    try:
        data = resp.json()
    except ValueError as e:
        raise _provider_error(
            f"{method} {url} returned HTTP {resp.status} with a non-JSON body: {resp.text()[:ERROR_BODY_SNIPPET_CHARS]}",
            status=resp.status,
            retryable=False,
            category=ErrorCategory.execution_failure,
            remediation="Check the base URL: this endpoint did not answer with JSON (a proxy or a wrong path, most likely).",
            secrets=secrets,
            details={**details, "status": resp.status, "body": resp.text()[:ERROR_BODY_SNIPPET_CHARS]},
        ) from e
    if isinstance(data, dict):
        if data.get("error"):  # some gateways deliver the error object with a 2xx status
            status = error_status(data["error"])
            snippet = error_text(resp.body)
            raise _provider_error(
                f"{method} {url} answered HTTP {resp.status} with an error body: {snippet}",
                status=status,
                retryable=False,
                category=status_category(status) if status else ErrorCategory.execution_failure,
                remediation=status_remediation(status) if status else "Read the message and change the request; do not repeat the unchanged call.",
                secrets=secrets,
                details={**details, "status": resp.status, "body": snippet},
            )
        return data
    return {"_value": data}  # a bare list or scalar is unusual but not an error; keep it reachable


def post_json(
    url: str,
    payload: Any,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    opener: Opener | None = None,
    retries: int = DEFAULT_RETRIES,
    backoff_s: float = DEFAULT_BACKOFF_S,
    request_id: str | None = None,
    sleep: Sleep | None = None,
    response_limit: Limit = RESPONSE_SIZE_LIMIT,
) -> dict[str, Any]:
    return request_json("POST", url, payload, headers, timeout_s, opener, retries, backoff_s, request_id, sleep, response_limit)


def get_json(
    url: str,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    opener: Opener | None = None,
    retries: int = DEFAULT_RETRIES,
    backoff_s: float = DEFAULT_BACKOFF_S,
    request_id: str | None = None,
    sleep: Sleep | None = None,
    response_limit: Limit = RESPONSE_SIZE_LIMIT,
) -> dict[str, Any]:
    return request_json("GET", url, None, headers, timeout_s, opener, retries, backoff_s, request_id, sleep, response_limit)


def with_query(url: str, params: Mapping[str, Any] | None) -> str:
    if not params:
        return url
    return url + ("&" if "?" in url else "?") + urlencode({k: v for k, v in params.items() if v is not None})


# ----------------------------------------------------------------------------- cost
def priced(usage: Usage, prices: Mapping[str, float] | None, reasoning_in_completion: bool = True) -> Usage:
    """Fill ``cost_usd`` from prices per million tokens: ``input``, ``output``, optional ``cached_input`` and ``reasoning``.

    ``reasoning_in_completion`` says whether ``completion_tokens`` already counts the reasoning tokens (OpenAI shape)
    or not (Gemini reports thoughts separately).
    """
    if not prices:
        return usage
    inp = float(prices.get("input", 0.0))
    out = float(prices.get("output", 0.0))
    cached = float(prices.get("cached_input", inp))
    reasoning = float(prices.get("reasoning", out))
    uncached = max(usage.prompt_tokens - usage.cached_tokens, 0)
    cost = uncached * inp + usage.cached_tokens * cached + usage.completion_tokens * out
    if not reasoning_in_completion:
        cost += usage.reasoning_tokens * reasoning
    usage.cost_usd = cost / TOKENS_PER_PRICE_UNIT
    return usage
