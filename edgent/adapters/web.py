"""Fetch web pages, turn them into Markdown, and search the web with the standard library.

``fetch`` wraps ``urllib`` with the protections a harness needs: only http(s); never localhost or a
private network address, also after a redirect; a byte ceiling; a timeout; charset detection; and an
injectable opener and resolver so tests never touch the network. ``to_markdown`` turns a result into
reading material (HTML to Markdown, JSON pretty-printed, PDFs saved for ``sources_add``). ``search``
scrapes the DuckDuckGo HTML endpoint, which needs no API key. Everything that comes back is untrusted
data: the callers label it so.
"""
from __future__ import annotations

import codecs
import http.client
import ipaddress
import json
import mimetypes
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from ..domain.errors import ErrorCategory, Limit, ToolFailure
from ..domain.utils import ensure_dir, sha256_text, slugify
from .html2md import html_to_markdown

DEFAULT_USER_AGENT = "edgent/0.1 (+https://github.com/edgentai/edgent-harness)"
DEFAULT_TIMEOUT_S = 30.0
FETCH_BYTES_LIMIT = Limit("web fetch body size", 5_000_000, "bytes", does_not_raise=("max_chars of web_fetch",))
SEARCH_RESULTS_LIMIT = Limit("web search results", 10, "results")
SEARCH_URL = "https://html.duckduckgo.com/html/"
SEARCH_TIMEOUT_S = 20.0
SEARCH_MAX_BYTES = 2_000_000
SEARCH_REMEDIATION = "Searching is blocked here: try web_fetch on a known URL (documentation, Wikipedia, the project's site) instead."

_READ_CHUNK_BYTES = 64 * 1024
_SNIFF_BYTES = 4096
_ACCEPT = "text/html,application/xhtml+xml,application/json;q=0.9,text/plain;q=0.8,*/*;q=0.5"
_LOCAL_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback", "broadcasthost"}
_LOCAL_SUFFIXES = (".localhost", ".local", ".internal", ".home.arpa", ".lan")
_HTML_TYPES = {"text/html", "application/xhtml+xml"}
_JSON_TYPES = {"application/json", "text/json", "application/ld+json"}
_MARKDOWN_TYPES = {"text/markdown", "text/x-markdown"}
_TEXT_TYPES = {
    "application/xml", "application/javascript", "application/x-javascript", "application/x-yaml", "application/yaml",
    "application/toml", "application/x-sh", "application/x-python", "application/sql", "image/svg+xml",
}
_SNIFFABLE_TYPES = {"", "application/octet-stream", "text/plain"}
_META_CHARSET_RE = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?\s*([-\w.:]+)""", re.I)
_XML_ENCODING_RE = re.compile(rb"""<\?xml[^>]+encoding\s*=\s*["']([-\w.:]+)""", re.I)
_HTML_SNIFF_RE = re.compile(r"<(!doctype\s+html|html|head|body)\b", re.I)
_VOID_TAGS = {"br", "img", "wbr", "hr", "input", "meta", "link", "source", "area", "base", "col", "embed", "param", "track"}
_BLOCK_MARKERS = ("anomaly", "bots use duckduckgo too", "challenge-form", "captcha", "verify you are human", "unusual traffic")
_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_URL_PATH_SAFE = "/%:@!$&'()*+,;=~-._"
_URL_QUERY_SAFE = _URL_PATH_SAFE + "?"

Resolver = Callable[[str], list[str]]
Opener = Any  # anything with .open(request, timeout=...) -> response; urllib.request.OpenerDirector in production
ContentKind = str  # html | markdown | json | text | pdf | binary


@dataclass
class FetchResult:
    """One fetched document. ``text`` is decoded; ``data`` holds the raw bytes (never serialised)."""

    url: str
    final_url: str
    status: int
    content_type: str  # media type only, lower case, e.g. text/html
    text: str
    bytes_read: int
    truncated: bool = False
    charset: str = "utf-8"
    note: str = ""  # set when truncated: the limit that cut the body
    data: bytes = field(default=b"", repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "final_url": self.final_url,
            "status": self.status,
            "content_type": self.content_type,
            "text": self.text,
            "bytes_read": self.bytes_read,
            "truncated": self.truncated,
            "charset": self.charset,
            "note": self.note,
        }


# ----------------------------------------------------------------------------- URL safety
def _invalid(message: str, remediation: str, **details: Any) -> ToolFailure:
    return ToolFailure("invalid_input", message, ErrorCategory.invalid_input, remediation=remediation, **details)


def _embedded_ipv4(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    """The IPv4 address an IPv6 one carries (mapped, NAT64, 6to4), so 127.0.0.1 cannot hide inside it."""
    if not isinstance(ip, ipaddress.IPv6Address):
        return None
    if ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    if ip in _NAT64:
        return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    return ip.sixtofour


def _is_private_host(host: str) -> bool:
    """True for names and addresses that must never be fetched: loopback, link-local, private ranges, local names."""
    h = host.lower().rstrip(".")
    if h in _LOCAL_NAMES or h.endswith(_LOCAL_SUFFIXES):
        return True
    try:
        ip = ipaddress.ip_address(h.strip("[]"))
    except ValueError:
        return False
    embedded = _embedded_ipv4(ip)
    if embedded is not None:
        ip = embedded
    return ip.is_multicast or not ip.is_global


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


def _resolve_addresses(host: str) -> list[str]:
    """Addresses a name resolves to; empty when it cannot be resolved (the fetch itself will then fail honestly)."""
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except (OSError, UnicodeError):
        return []
    return sorted({str(info[4][0]) for info in infos})


def check_url(url: str, resolver: Resolver | None = None) -> urllib.parse.SplitResult:
    """Refuse anything but http(s) to a public host. Returns the split URL."""
    parts = urllib.parse.urlsplit(url.strip())
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        raise _invalid(
            f"only http and https URLs can be fetched; got {scheme or 'no'} scheme in {url!r}.",
            "Use an http(s) URL. Local files are read with read_file, not fetched.",
            url=url,
        )
    host = parts.hostname
    if not host:
        raise _invalid(f"the URL {url!r} has no host.", "Give a full URL such as https://example.com/page.", url=url)
    try:
        _ = parts.port  # raises for a port outside 0-65535, which http.client would turn into an OverflowError
    except ValueError as e:
        raise _invalid(f"the URL {url!r} has an unusable port: {e}.", "Use a port between 1 and 65535, or none.", url=url) from e
    if _is_private_host(host):
        raise _invalid(
            f"refusing to fetch {host}: localhost and private network addresses are off limits.",
            "Fetch public URLs only. Local files are read with read_file; a local service is reached with bash when the policy allows network.",
            url=url,
        )
    if not _is_ip_literal(host):
        for addr in (resolver or _resolve_addresses)(host):
            if _is_private_host(addr):
                raise _invalid(
                    f"refusing to fetch {host}: it resolves to {addr}, a private network address.",
                    "Fetch public URLs only; a host that resolves to a private address is not reachable from here.",
                    url=url,
                )
    return parts


def _request_url(parts: urllib.parse.SplitResult) -> str:
    """The URL as http.client can send it: an IDNA host and percent-encoded non-ASCII or blank characters in the
    path and query. An ASCII URL that needs nothing comes back unchanged (minus any fragment)."""
    raw = urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))
    if raw.isascii() and not any(c <= " " for c in raw):
        return raw
    host = parts.hostname or ""
    try:
        host_ascii = host.encode("idna").decode("ascii")
    except UnicodeError as e:
        raise _invalid(f"the host {host!r} cannot be encoded for DNS: {e}.", "Check the host name.", url=raw) from e
    if ":" in host_ascii:
        host_ascii = f"[{host_ascii}]"
    userinfo, _, _ = parts.netloc.rpartition("@")
    netloc = (userinfo + "@" if userinfo else "") + host_ascii + (f":{parts.port}" if parts.port is not None else "")
    path = urllib.parse.quote(parts.path, safe=_URL_PATH_SAFE)
    query = urllib.parse.quote(parts.query, safe=_URL_QUERY_SAFE)
    return urllib.parse.urlunsplit((parts.scheme.lower(), netloc, path, query, ""))


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """urllib follows redirects; this makes every hop pass the same check as the first URL."""

    def __init__(self, resolver: Resolver | None = None) -> None:
        super().__init__()
        self.resolver = resolver

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        check_url(newurl, self.resolver)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def build_opener(resolver: Resolver | None = None) -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_SafeRedirectHandler(resolver))


# ----------------------------------------------------------------------------- fetching
def _header(headers: Any, name: str) -> str:
    if headers is None:
        return ""
    getter = getattr(headers, "get", None)
    if getter is not None:
        value = getter(name)
        if value is None and hasattr(headers, "items"):
            value = next((v for k, v in headers.items() if str(k).lower() == name.lower()), None)
        return str(value) if value is not None else ""
    return ""


def _parse_content_type(value: str) -> tuple[str, str | None]:
    """'text/html; charset=ISO-8859-1' -> ('text/html', 'iso-8859-1')."""
    mime, _, rest = value.partition(";")
    charset = None
    for param in rest.split(";"):
        key, _, val = param.strip().partition("=")
        if key.strip().lower() == "charset" and val.strip():
            charset = val.strip().strip("\"'").lower()
    return mime.strip().lower(), charset


def _read_capped(resp: Any, max_bytes: int) -> tuple[bytes, bool]:
    chunks: list[bytes] = []
    total = 0
    while total <= max_bytes:
        chunk = resp.read(_READ_CHUNK_BYTES)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    data = b"".join(chunks)
    return data[:max_bytes], len(data) > max_bytes


def _decompress(data: bytes, encoding: str, max_bytes: int) -> tuple[bytes, bool]:
    """gzip or deflate bodies, decompressed under the same ceiling so a bomb cannot expand past it."""
    candidates = [16 + zlib.MAX_WBITS] if encoding in ("gzip", "x-gzip") else [zlib.MAX_WBITS, -zlib.MAX_WBITS]
    for wbits in candidates:
        try:
            d = zlib.decompressobj(wbits)
            out = d.decompress(data, max_bytes)
            return out, bool(d.unconsumed_tail)
        except zlib.error:
            continue
    return data, False


def _meta_charset(head: bytes) -> str | None:
    m = _META_CHARSET_RE.search(head) or _XML_ENCODING_RE.search(head)
    return m.group(1).decode("ascii", "ignore").lower() if m else None


def decode_body(data: bytes, charset: str | None = None) -> tuple[str, str]:
    """Decode with the declared charset, a BOM, a meta tag, or UTF-8 with replacement. Returns (text, charset)."""
    if data.startswith(codecs.BOM_UTF8):
        return data[len(codecs.BOM_UTF8) :].decode("utf-8", "replace"), "utf-8"
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16", "replace"), "utf-16"
    enc = charset or _meta_charset(data[:_SNIFF_BYTES]) or "utf-8"
    try:
        codecs.lookup(enc)
    except LookupError:
        enc = "utf-8"
    return data.decode(enc, "replace"), enc


def _http_failure(e: urllib.error.HTTPError, url: str) -> ToolFailure:
    status = int(e.code)
    if status in (404, 410):
        category, retryable = ErrorCategory.not_found, False
        remediation = "Check the URL (the page may have moved) or find it again with web_search."
    elif status in (401, 403, 451):
        category, retryable = ErrorCategory.permission_denied, False
        remediation = "The site refuses automated access; use another source or ask the user for the content."
    elif status == 429 or status >= 500:
        category, retryable = ErrorCategory.unavailable, True
        remediation = "Retry once after a pause; if it persists, use another source."
    else:
        category, retryable = ErrorCategory.invalid_input, False
        remediation = "Check the URL and the request; do not repeat the unchanged call."
    return ToolFailure("http_error", f"HTTP {status} {e.reason} for {url}", category, retryable=retryable, remediation=remediation, status=status, url=url)


def fetch(
    url: str,
    *,
    max_bytes: int = int(FETCH_BYTES_LIMIT.value),
    timeout_s: float = DEFAULT_TIMEOUT_S,
    opener: Opener | None = None,
    user_agent: str = DEFAULT_USER_AGENT,
    resolver: Resolver | None = None,
) -> FetchResult:
    """GET a public http(s) URL, following redirects, reading at most ``max_bytes`` (a named limit)."""
    if max_bytes <= 0:
        raise _invalid(f"max_bytes must be positive; got {max_bytes}.", "Pass a positive byte ceiling or leave the default.")
    if timeout_s <= 0:
        raise _invalid(f"timeout_s must be positive; got {timeout_s}.", "Pass a positive timeout in seconds or leave the default.")
    url = url.strip()
    parts = check_url(url, resolver)
    limit = FETCH_BYTES_LIMIT if max_bytes == FETCH_BYTES_LIMIT.value else Limit(FETCH_BYTES_LIMIT.name, max_bytes, "bytes", does_not_raise=FETCH_BYTES_LIMIT.does_not_raise)
    try:
        req = urllib.request.Request(_request_url(parts), headers={"User-Agent": user_agent, "Accept": _ACCEPT, "Accept-Language": "en-US,en;q=0.8"})
    except ValueError as e:
        raise _invalid(f"malformed URL {url!r}: {e}", "Give a full, well-formed http(s) URL.", url=url) from e
    op = opener if opener is not None else build_opener(resolver)
    try:
        resp = op.open(req, timeout=timeout_s)
    except urllib.error.HTTPError as e:
        raise _http_failure(e, url) from e
    except urllib.error.URLError as e:
        if isinstance(e.reason, TimeoutError):
            raise ToolFailure("timeout", f"no response from {url} within {timeout_s:g} s", ErrorCategory.timeout, retryable=True, remediation="Retry once with a longer timeout_s, then use another source.", url=url) from e
        raise ToolFailure("unavailable", f"could not reach {url}: {e.reason}", ErrorCategory.unavailable, retryable=True, remediation="Check the host name and the network; retry once, then use another source.", url=url) from e
    except TimeoutError as e:
        raise ToolFailure("timeout", f"no response from {url} within {timeout_s:g} s", ErrorCategory.timeout, retryable=True, remediation="Retry once with a longer timeout_s, then use another source.", url=url) from e
    except (OSError, http.client.HTTPException) as e:
        raise ToolFailure("unavailable", f"could not fetch {url}: {type(e).__name__}: {e}", ErrorCategory.unavailable, retryable=True, remediation="Retry once; if it persists, use another source.", url=url) from e
    except ValueError as e:  # http.client rejects what urllib accepted: control characters, an unusable tunnel host
        raise _invalid(f"the URL {url!r} cannot be requested: {e}", "Give a well-formed http(s) URL without blank or control characters.", url=url) from e
    try:
        final_url = str(resp.geturl() or url) if hasattr(resp, "geturl") else url
        if final_url != url:
            check_url(final_url, resolver)  # an injected opener may not run the safe redirect handler
        status = int(getattr(resp, "status", None) or (resp.getcode() if hasattr(resp, "getcode") else 200) or 200)
        headers = getattr(resp, "headers", None)
        try:
            data, truncated = _read_capped(resp, max_bytes)
        except (OSError, http.client.HTTPException) as e:
            raise ToolFailure("unavailable", f"the connection to {url} broke while reading: {type(e).__name__}: {e}", ErrorCategory.unavailable, retryable=True, remediation="Retry once; if it persists, use another source.", url=url) from e
    finally:
        close = getattr(resp, "close", None)
        if callable(close):
            close()
    encoding = _header(headers, "Content-Encoding").strip().lower()
    if encoding in ("gzip", "x-gzip", "deflate"):
        data, cut = _decompress(data, encoding, max_bytes)
        truncated = truncated or cut
    mime, charset = _parse_content_type(_header(headers, "Content-Type"))
    text, used = decode_body(data, charset)
    note = limit.exceeded(len(data), "Read the saved file in pages, or fetch a narrower URL.", ErrorCategory.truncated).message if truncated else ""
    return FetchResult(url, final_url, status, mime, text, len(data), truncated, used, note, data)


# ----------------------------------------------------------------------------- to markdown
def _url_ext(url: str) -> str:
    return os.path.splitext(urllib.parse.urlsplit(url).path)[1].lower()


def content_kind(result: FetchResult) -> ContentKind:
    """html | markdown | json | text | pdf | binary, from the media type with a sniff for mislabelled bodies."""
    ct = result.content_type
    head = result.data[:_SNIFF_BYTES]
    if ct == "application/pdf" or head.startswith(b"%PDF-"):
        return "pdf"
    if ct in _HTML_TYPES:
        return "html"
    if ct in _JSON_TYPES or ct.endswith("+json"):
        return "json"
    if ct in _MARKDOWN_TYPES or (ct in _SNIFFABLE_TYPES and _url_ext(result.final_url) in (".md", ".markdown")):
        return "markdown"
    if ct in _SNIFFABLE_TYPES and _HTML_SNIFF_RE.search(head[:1024].decode("latin-1")):
        return "html"
    if b"\x00" in head:
        return "binary"
    if ct.startswith("text/") or ct in _TEXT_TYPES or ct.endswith("+xml") or ct in _SNIFFABLE_TYPES:
        return "text"
    return "binary"


def canonical_url(url: str) -> str:
    parts = urllib.parse.urlsplit(url.strip())
    return urllib.parse.urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or "/", parts.query, ""))


def raw_file_stem(result: FetchResult) -> str:
    """``<slug>-<shorthash>``: readable, stable per URL, so a refetch overwrites instead of piling up."""
    url = canonical_url(result.final_url)
    parts = urllib.parse.urlsplit(url)
    base = f"{parts.netloc}{parts.path}".rstrip("/") or parts.netloc
    return f"{slugify(base, 60)}-{sha256_text(url)[:10]}"


def raw_file_path(result: FetchResult, save_dir: str | os.PathLike[str]) -> Path:
    kind = content_kind(result)
    if kind == "pdf":
        ext = ".pdf"
    else:
        ext = mimetypes.guess_extension(result.content_type) if result.content_type else None
        ext = ext or ".bin"
    return Path(save_dir) / f"{raw_file_stem(result)}{ext}"


def save_raw(result: FetchResult, save_dir: str | os.PathLike[str]) -> Path:
    """Write the raw bytes atomically under ``save_dir`` and return the path."""
    path = raw_file_path(result, save_dir)
    ensure_dir(path.parent)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:6]}")
    with open(tmp, "wb") as f:
        f.write(result.data)
    os.replace(tmp, path)
    return path


def _first_heading(markdown: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return ""


def _pretty_json(text: str) -> str:
    try:
        return json.dumps(json.loads(text), indent=2, ensure_ascii=False) + "\n"
    except ValueError:
        return text


def _binary_note(result: FetchResult, kind: ContentKind, save_dir: str | os.PathLike[str] | None) -> str:
    label = "PDF" if kind == "pdf" else f"binary content ({result.content_type or 'unknown type'})"
    head = f"[{label}, {result.bytes_read} bytes, from {result.final_url}]"
    if save_dir is None:
        return f"{head} Not converted to text and not saved. Fetch again with save enabled, or pass the URL to sources_add, which derives a greppable text view.\n"
    path = save_raw(result, save_dir)
    return f"{head} The bytes were saved to {path}. read_file cannot read this format: run sources_add on that path to derive a text view with locators, then grep and read it.\n"


def to_markdown(result: FetchResult, save_dir: str | os.PathLike[str] | None = None) -> tuple[str, str]:
    """(markdown, title). HTML is converted; text and Markdown pass through; JSON is pretty-printed;
    PDFs and other binaries become a note, with the bytes saved under ``save_dir`` when one is given."""
    kind = content_kind(result)
    if kind == "html":
        markdown, title = html_to_markdown(result.text)
    elif kind == "json":
        markdown, title = _pretty_json(result.text), ""
    elif kind == "markdown":
        markdown, title = result.text, _first_heading(result.text)
    elif kind == "text":
        markdown, title = result.text, ""
    else:
        return _binary_note(result, kind, save_dir), ""
    if result.truncated:
        markdown = markdown.rstrip("\n") + f"\n\n[fetch truncated: {result.note}]\n"
    return markdown, title.strip()


# ----------------------------------------------------------------------------- search
def _unwrap_ddg_href(href: str) -> str:
    """DuckDuckGo wraps result links as //duckduckgo.com/l/?uddg=<url>; return the real URL or ''."""
    href = href.strip()
    if href.startswith("//"):
        href = "https:" + href
    parts = urllib.parse.urlsplit(href)
    if parts.scheme not in ("http", "https"):
        return ""
    if parts.netloc.endswith("duckduckgo.com"):
        if parts.path.startswith("/l/"):
            target = urllib.parse.parse_qs(parts.query).get("uddg", [""])[0]
            return target if target.startswith(("http://", "https://")) else ""
        if parts.path.startswith("/y.js"):
            return ""  # an advertisement
    return href


class _DDGParser(HTMLParser):
    """Walks the DuckDuckGo HTML results page: one record per ``div.result``, title from ``a.result__a``,
    snippet from ``.result__snippet``. No regex over the page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, Any]] = []
        self.page_text: list[str] = []
        self._current: dict[str, Any] | None = None
        self._capture: str | None = None
        self._depth = 0

    def _start_result(self, ad: bool = False) -> dict[str, Any]:
        self._current = {"title": "", "url": "", "snippet": "", "ad": ad}
        self.results.append(self._current)
        return self._current

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _VOID_TAGS:
            if tag == "br" and self._capture and self._current is not None:
                self._current[self._capture] += " "  # a line break inside a snippet separates words
            return
        if self._capture:
            self._depth += 1
            return
        a = dict(attrs)
        classes = (a.get("class") or "").split()
        if tag == "div" and ("result" in classes or "web-result" in classes):
            self._start_result(ad="result--ad" in classes or any(c.startswith("result--ad") for c in classes))
            return
        if tag == "a" and "result__a" in classes:
            current = self._current or self._start_result()
            current["url"] = _unwrap_ddg_href(a.get("href") or "")
            self._capture, self._depth = "title", 0
            return
        if "result__snippet" in classes:
            if self._current is None:
                self._start_result()
            self._capture, self._depth = "snippet", 0

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID_TAGS or not self._capture:
            return
        if self._depth == 0:
            self._capture = None
        else:
            self._depth -= 1

    def handle_data(self, data: str) -> None:
        self.page_text.append(data)
        if self._capture and self._current is not None:
            self._current[self._capture] += data


def parse_search_page(html: str) -> tuple[list[dict[str, str]], str]:
    """(results, page text). Results are {'title','url','snippet'} with ads and unparsable entries dropped."""
    p = _DDGParser()
    try:
        p.feed(html)
        p.close()
    except Exception:  # noqa: BLE001 - keep what was parsed from a malformed page
        pass
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for r in p.results:
        title = " ".join(r["title"].split())
        url = r["url"]
        if r["ad"] or not url or not title or url in seen:
            continue
        seen.add(url)
        out.append({"title": title, "url": url, "snippet": " ".join(r["snippet"].split())})
    return out, " ".join(p.page_text)


def _blocked(message: str) -> ToolFailure:
    return ToolFailure("search_unavailable", message, ErrorCategory.unavailable, retryable=False, remediation=SEARCH_REMEDIATION)


def search(
    query: str,
    max_results: int = 5,
    opener: Opener | None = None,
    *,
    resolver: Resolver | None = None,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout_s: float = SEARCH_TIMEOUT_S,
) -> list[dict[str, str]]:
    """Web search through the DuckDuckGo HTML endpoint. Blocks and captchas become a typed ``unavailable`` error."""
    q = " ".join(query.split())
    if not q:
        raise _invalid("the search query is empty.", "Give two to eight specific keywords, in English where possible.")
    n = max(1, min(int(max_results), int(SEARCH_RESULTS_LIMIT.value)))
    url = SEARCH_URL + "?" + urllib.parse.urlencode({"q": q})
    try:
        page = fetch(url, max_bytes=SEARCH_MAX_BYTES, timeout_s=timeout_s, opener=opener, user_agent=user_agent, resolver=resolver)
    except ToolFailure as e:
        if e.error.category in (ErrorCategory.permission_denied, ErrorCategory.unavailable, ErrorCategory.timeout, ErrorCategory.not_found):
            raise _blocked(f"the search endpoint did not return results: {e.error.message}") from e
        raise
    results, page_text = parse_search_page(page.text)
    if results:
        return results[:n]
    low = page_text.lower()
    if any(marker in low for marker in _BLOCK_MARKERS):
        raise _blocked("the search endpoint returned a bot challenge instead of results.")
    if "no results" in low:
        return []
    raise _blocked("no results could be parsed from the search page; its format may have changed.")


__all__ = [
    "DEFAULT_USER_AGENT",
    "FETCH_BYTES_LIMIT",
    "FetchResult",
    "SEARCH_RESULTS_LIMIT",
    "build_opener",
    "canonical_url",
    "check_url",
    "content_kind",
    "decode_body",
    "fetch",
    "parse_search_page",
    "raw_file_path",
    "raw_file_stem",
    "save_raw",
    "search",
    "to_markdown",
]
