"""Stdlib web tools: ``web_fetch`` and ``web_search`` (permission='network').

All fetched content is tagged ``untrusted=True``, SSRF-checked against private
loopback/link-local hosts by default, stripped of scripts/styles via stdlib
``html.parser``, and cached cleanly in ``.edgent/web_cache/``.
"""
from __future__ import annotations

import hashlib
import html
import html.parser
import ipaddress
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Optional

from ..core.utils import ensure_dir, truncate
from .registry import ToolContext, ToolFailure

MAX_FETCH_BYTES = 2 * 1024 * 1024  # 2 MB raw download cap
DEFAULT_MAX_CHARS = 12_000


class _HTMLToText(html.parser.HTMLParser):
    _SKIP_TAGS = {"script", "style", "noscript", "svg", "iframe"}
    _BLOCK_TAGS = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "pre", "blockquote", "section", "article"}

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._in_title = False
        self.title = ""
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        t = tag.lower()
        if t in self._SKIP_TAGS:
            self._skip_depth += 1
        elif t == "title":
            self._in_title = True
        elif t in self._BLOCK_TAGS and self._skip_depth == 0:
            self._parts.append("\n")
            if t in ("h1", "h2", "h3"):
                self._parts.append("#" * int(t[1]) + " ")
            elif t == "li":
                self._parts.append("- ")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif t == "title":
            self._in_title = False
        elif t in self._BLOCK_TAGS and self._skip_depth == 0:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._in_title:
            self.title += data
            return
        self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        lines = [" ".join(ln.split()) for ln in raw.splitlines()]
        cleaned = "\n".join(ln for ln in lines if ln)
        return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def html_to_text(html_body: str) -> tuple[str, str]:
    """Return ``(title, plain_text)`` from raw HTML using only stdlib."""
    parser = _HTMLToText()
    parser.feed(html_body or "")
    parser.close()
    return " ".join(parser.title.split()), parser.text()


def _check_url_safe(url: str, *, allow_private: bool = False) -> urllib.parse.ParseResult:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ToolFailure("invalid_arguments", f"unsupported URL scheme '{parsed.scheme}'; only http/https allowed")
    host = (parsed.hostname or "").strip()
    if not host:
        raise ToolFailure("invalid_arguments", "URL has no host")
    if allow_private:
        return parsed
    if host.lower() in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        raise ToolFailure("permission_denied", f"refusing to fetch loopback host '{host}'")
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
        for _fam, _type, _proto, _canon, sockaddr in infos:
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                raise ToolFailure("permission_denied", f"refusing to fetch private/internal address {ip} for '{host}'")
    except ToolFailure:
        raise
    except Exception:
        pass
    return parsed


def web_fetch(ctx: ToolContext, a: dict[str, Any]) -> Any:
    """Fetch a URL, extract readable text, persist full text under .edgent/web_cache, return bounded view."""
    url = str(a["url"]).strip()
    allow_private = bool(ctx.extra.get("allow_private_urls", False))
    _check_url_safe(url, allow_private=allow_private)
    max_chars = int(a.get("max_chars", DEFAULT_MAX_CHARS))
    timeout_s = min(float(a.get("timeout_s", 20.0)), 60.0)
    opener: Callable[..., Any] = ctx.extra.get("web_opener") or urllib.request.urlopen

    req = urllib.request.Request(url, headers={"User-Agent": "EdgentHarness/0.2 (+https://localhost)"})
    try:
        with opener(req, timeout=timeout_s) as resp:
            ctype = str(resp.headers.get("Content-Type") or "text/plain") if hasattr(resp, "headers") and resp.headers else "text/plain"
            raw = resp.read(MAX_FETCH_BYTES + 1)
            status = getattr(resp, "status", 200)
    except urllib.error.HTTPError as e:
        raise ToolFailure("execution_error", f"HTTP {e.code} fetching {url}", retryable=e.code >= 500)
    except Exception as e:
        raise ToolFailure("unavailable", f"could not fetch {url}: {e}", retryable=True)

    decoded = raw[:MAX_FETCH_BYTES].decode("utf-8", "replace")
    title = ""
    if "html" in ctype.lower() or "<html" in decoded[:500].lower():
        title, text = html_to_text(decoded)
    else:
        text = decoded

    cache_dir = ensure_dir(os.path.join(ctx.workspace.path, ".edgent", "web_cache"))
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    saved_abs = os.path.join(cache_dir, f"{digest}.txt")
    with open(saved_abs, "w", encoding="utf-8") as f:
        f.write(f"URL: {url}\nTITLE: {title}\n\n{text}\n")
    saved_rel = os.path.relpath(saved_abs, ctx.workspace.path)
    return {
        "url": url,
        "status": status,
        "content_type": ctype,
        "title": title,
        "total_chars": len(text),
        "truncated": len(text) > max_chars,
        "saved_path": saved_rel,
        "content": truncate(text, max_chars),
        "untrusted": True,
    }


_DDG_LINK_RE = re.compile(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL | re.IGNORECASE)
_DDG_SNIP_RE = re.compile(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>|<div[^>]+class="result__snippet"[^>]*>(.*?)</div>', re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_tags(s: str) -> str:
    return " ".join(html.unescape(_TAG_RE.sub(" ", s or "")).split())


def web_search(ctx: ToolContext, a: dict[str, Any]) -> Any:
    """Search the web via an injected search provider or DuckDuckGo HTML lite."""
    query = str(a["query"]).strip()
    if not query:
        raise ToolFailure("invalid_arguments", "query must be non-empty")
    limit = min(int(a.get("limit", 8)), 20)

    custom = ctx.extra.get("web_search_fn")
    if callable(custom):
        return {"query": query, "results": list(custom(query, limit))[:limit], "untrusted": True}

    opener: Callable[..., Any] = ctx.extra.get("web_opener") or urllib.request.urlopen
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    req = urllib.request.Request(url, headers={"User-Agent": "EdgentHarness/0.2"})
    try:
        with opener(req, timeout=15.0) as resp:
            body = resp.read(512 * 1024).decode("utf-8", "replace")
    except Exception as e:
        raise ToolFailure("unavailable", f"web search unreachable: {e}", retryable=True)

    links = _DDG_LINK_RE.findall(body)
    snips = [s1 or s2 for s1, s2 in _DDG_SNIP_RE.findall(body)]
    results = []
    for idx, (href, raw_title) in enumerate(links[:limit]):
        # Unwrap DuckDuckGo redirect URL if present
        parsed = urllib.parse.urlparse(href)
        qs = urllib.parse.parse_qs(parsed.query)
        target_url = qs.get("uddg", [href])[0]
        results.append({
            "title": _strip_tags(raw_title),
            "url": target_url,
            "snippet": _strip_tags(snips[idx]) if idx < len(snips) else "",
        })
    return {"query": query, "results": results, "untrusted": True}
