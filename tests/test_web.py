"""Web adapter: fetching through a fake opener, decoding, Markdown conversion, DuckDuckGo parsing. No network."""
from __future__ import annotations

import gzip
import io
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from edgent.adapters import web
from edgent.adapters.web import (
    FetchResult,
    _SafeRedirectHandler,
    check_url,
    content_kind,
    fetch,
    parse_search_page,
    search,
    to_markdown,
)
from edgent.domain.errors import ToolFailure

PUBLIC_RESOLVER = lambda host: ["93.184.216.34"]  # noqa: E731 - a stub resolver: every name is public


class FakeResponse:
    """The slice of http.client.HTTPResponse that fetch() touches."""

    def __init__(self, body: bytes | str, url: str, status: int = 200, content_type: str = "text/html; charset=utf-8", headers: dict[str, str] | None = None) -> None:
        self._buf = io.BytesIO(body.encode("utf-8") if isinstance(body, str) else body)
        self.url = url
        self.status = status
        self.headers: dict[str, str] = {"Content-Type": content_type, **(headers or {})}
        self.closed = False

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def geturl(self) -> str:
        return self.url

    def getcode(self) -> int:
        return self.status

    def close(self) -> None:
        self.closed = True


class FakeOpener:
    """Routes by URL prefix to a response, an exception, or a factory; records every request."""

    def __init__(self, routes: dict[str, Any]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, dict[str, str], float | None]] = []

    def open(self, req: urllib.request.Request, timeout: float | None = None) -> Any:
        url = req.full_url
        self.calls.append((url, {k.lower(): v for k, v in req.header_items()}, timeout))
        for prefix, target in self.routes.items():
            if url.startswith(prefix):
                if isinstance(target, Exception):
                    raise target
                return target(url) if callable(target) else target
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)  # type: ignore[arg-type]


def html_page(title: str = "Bayes' theorem", body: str = "") -> str:
    return (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{title}</title><script>evil()</script></head>"
        f"<body><nav>menu</nav><h1>{title}</h1><p>Posterior is <b>prior</b> times likelihood over evidence.</p>"
        f"<ul><li>one</li><li>two</li></ul>{body}</body></html>"
    )


DDG_HTML = """<!DOCTYPE html>
<html lang="en-US"><head><meta charset="utf-8"><title>bayes theorem at DuckDuckGo</title></head>
<body class="body--html">
<div class="header"><form action="/html/" method="post"><input name="q" value="bayes theorem"><input type="submit" value="Search"></form></div>
<div id="links" class="results">
<div class="result results_links results_links_deep web-result result--ad">
  <div class="links_main links_deep result__body">
    <h2 class="result__title"><a rel="nofollow" class="result__a" href="https://duckduckgo.com/y.js?ad_provider=bingv7aa&amp;u3=https%3A%2F%2Fads.example%2F">Buy Statistics Courses</a></h2>
    <a class="result__snippet" href="https://duckduckgo.com/y.js?x=1">Sponsored: learn stats fast.</a>
  </div>
</div>
<div class="result results_links results_links_deep web-result ">
  <div class="links_main links_deep result__body">
    <h2 class="result__title">
      <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FBayes%2527_theorem&amp;rut=abc123">Bayes' theorem - Wikipedia</a>
    </h2>
    <div class="result__extras">
      <div class="result__extras__url">
        <span class="result__icon"><img src="//external-content.duckduckgo.com/ip3/en.wikipedia.org.ico" alt=""></span>
        <a class="result__url" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FBayes%2527_theorem&amp;rut=abc123">en.wikipedia.org/wiki/Bayes'_theorem</a>
      </div>
    </div>
    <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FBayes%2527_theorem&amp;rut=abc123">In probability theory, <b>Bayes</b>' <b>theorem</b> describes the probability of an event<br>based on prior knowledge.</a>
  </div>
</div>
<div class="result results_links results_links_deep web-result ">
  <div class="links_main links_deep result__body">
    <h2 class="result__title"><a rel="nofollow" class="result__a" href="https://www.probabilitycourse.com/chapter1/1_4_3_bayes_rule.php">Bayes&#39; Rule | Probability Course</a></h2>
    <a class="result__snippet" href="https://www.probabilitycourse.com/chapter1/1_4_3_bayes_rule.php">Bayes&#39; rule lets us compute the posterior from the prior &amp; likelihood.</a>
  </div>
</div>
<div class="result results_links results_links_deep web-result ">
  <div class="links_main links_deep result__body">
    <h2 class="result__title"><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fbayes%3Flang%3Den%26v%3D2&amp;rut=zzz">Example: Bayes with coins</a></h2>
    <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fbayes&amp;rut=zzz">Two coins, one biased: a worked example.</a>
  </div>
</div>
<div class="result results_links results_links_deep web-result ">
  <div class="links_main links_deep result__body">
    <h2 class="result__title"><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fbayes%3Flang%3Den%26v%3D2&amp;rut=dup">Duplicate of the coins page</a></h2>
  </div>
</div>
</div>
<div class="nav-link"><form action="/html/" method="post"><input type="submit" class="btn btn--alt" value="Next"></form></div>
</body></html>
"""

DDG_BLOCKED_HTML = """<html><head><title>DuckDuckGo</title></head><body>
<div class="anomaly-modal__title">Unfortunately, bots use DuckDuckGo too.</div>
<form id="challenge-form" action="/anomaly"><input type="submit" value="Continue"></form></body></html>"""

DDG_EMPTY_HTML = """<html><head><title>x at DuckDuckGo</title></head><body><div id="links" class="results">
<div class="no-results">No results.</div></div></body></html>"""


# ----------------------------------------------------------------------------- URL safety
@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "javascript:alert(1)",
        "data:text/plain,hi",
        "http://127.0.0.1/",
        "http://[::1]:8080/",
        "http://10.0.0.5/admin",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://localhost:8000/",
        "http://api.localhost/",
        "http://metadata.google.internal/",
        "http://0.0.0.0/",
        "http://[::ffff:127.0.0.1]/",
        "http:///nohost",
        "not a url",
    ],
)
def test_check_url_refuses_unsafe_targets(url: str) -> None:
    with pytest.raises(ToolFailure) as ei:
        check_url(url, resolver=PUBLIC_RESOLVER)
    assert ei.value.error.category.value == "invalid_input" and ei.value.error.remediation


def test_check_url_uses_resolver_for_names() -> None:
    assert check_url("https://example.com/page?x=1", resolver=PUBLIC_RESOLVER).netloc == "example.com"
    with pytest.raises(ToolFailure) as ei:
        check_url("https://evil.example/", resolver=lambda host: ["93.184.216.34", "10.0.0.1"])
    assert "resolves to 10.0.0.1" in ei.value.error.message
    assert check_url("https://unresolvable.example/", resolver=lambda host: []).scheme == "https"
    assert check_url("https://93.184.216.34/", resolver=lambda host: pytest.fail("literals are not resolved")).hostname == "93.184.216.34"
    assert web._resolve_addresses("127.0.0.1") == ["127.0.0.1"]


def test_safe_redirect_handler_checks_every_hop() -> None:
    handler = _SafeRedirectHandler(resolver=PUBLIC_RESOLVER)
    req = urllib.request.Request("http://example.com/", headers={"User-Agent": "t"})
    with pytest.raises(ToolFailure):
        handler.redirect_request(req, None, 302, "Found", {}, "http://127.0.0.1/admin")
    new = handler.redirect_request(req, None, 302, "Found", {}, "https://example.org/x")
    assert new.full_url == "https://example.org/x" and new.get_method() == "GET"
    assert isinstance(web.build_opener(PUBLIC_RESOLVER), urllib.request.OpenerDirector)


# ----------------------------------------------------------------------------- fetch
def test_fetch_html_through_fake_opener() -> None:
    opener = FakeOpener({"https://example.com/": FakeResponse(html_page(), "https://example.com/page")})
    r = fetch("https://example.com/page", opener=opener, resolver=PUBLIC_RESOLVER, timeout_s=7)
    assert r.status == 200 and r.content_type == "text/html" and r.charset == "utf-8" and not r.truncated
    assert r.final_url == "https://example.com/page" and r.bytes_read == len(html_page().encode()) and "Posterior" in r.text
    url, headers, timeout = opener.calls[0]
    assert url == "https://example.com/page" and headers["user-agent"].startswith("edgent/") and timeout == 7
    assert "data" not in r.to_dict() and r.to_dict()["content_type"] == "text/html"
    assert opener.routes["https://example.com/"].closed


def test_fetch_decodes_charsets() -> None:
    latin = FakeResponse("caf\xe9".encode("latin-1"), "https://a.example/l", content_type="text/plain; charset=ISO-8859-1")
    meta = FakeResponse(b"<html><head><meta http-equiv='Content-Type' content='text/html; charset=windows-1252'></head><body>\x93quoted\x94</body></html>", "https://a.example/m", content_type="text/html")
    bad = FakeResponse(b"ok \xff\xfe end", "https://a.example/b", content_type="text/plain")
    bom = FakeResponse(b"\xef\xbb\xbfhello", "https://a.example/with-bom", content_type="text/plain; charset=bogus-charset")
    opener = FakeOpener({"https://a.example/l": latin, "https://a.example/m": meta, "https://a.example/b": bad, "https://a.example/with-bom": bom})
    assert fetch("https://a.example/l", opener=opener, resolver=PUBLIC_RESOLVER).text == "café"
    r = fetch("https://a.example/m", opener=opener, resolver=PUBLIC_RESOLVER)
    assert r.charset == "windows-1252" and "“quoted”" in r.text
    r = fetch("https://a.example/b", opener=opener, resolver=PUBLIC_RESOLVER)
    assert r.text.startswith("ok ") and "�" in r.text and r.charset == "utf-8"
    assert fetch("https://a.example/with-bom", opener=opener, resolver=PUBLIC_RESOLVER).text == "hello"


def test_fetch_honours_max_bytes_limit() -> None:
    body = b"x" * 1000
    opener = FakeOpener({"https://a.example/": lambda url: FakeResponse(body, url, content_type="text/plain")})
    r = fetch("https://a.example/big", max_bytes=400, opener=opener, resolver=PUBLIC_RESOLVER)
    assert r.truncated and r.bytes_read == 400 and len(r.text) == 400 and "400 bytes" in r.note and web.FETCH_BYTES_LIMIT.name in r.note
    r = fetch("https://a.example/fits", max_bytes=1000, opener=opener, resolver=PUBLIC_RESOLVER)
    assert not r.truncated and r.bytes_read == 1000
    with pytest.raises(ToolFailure) as ei:
        fetch("https://a.example/", max_bytes=0, opener=opener, resolver=PUBLIC_RESOLVER)
    assert ei.value.error.category.value == "invalid_input"
    assert web.FETCH_BYTES_LIMIT.unit == "bytes" and web.FETCH_BYTES_LIMIT.value == 5_000_000


def test_fetch_decompresses_gzip_under_the_same_limit() -> None:
    payload = ("line of text\n" * 2000).encode()
    gz = gzip.compress(payload)
    opener = FakeOpener({"https://a.example/": lambda url: FakeResponse(gz, url, content_type="text/plain", headers={"Content-Encoding": "gzip"})})
    r = fetch("https://a.example/gz", opener=opener, resolver=PUBLIC_RESOLVER)
    assert r.text == payload.decode() and not r.truncated
    r = fetch("https://a.example/gz", max_bytes=5000, opener=opener, resolver=PUBLIC_RESOLVER)
    assert r.truncated and len(r.text) == 5000


def test_fetch_maps_errors_to_the_contract() -> None:
    def http(code: int, reason: str) -> urllib.error.HTTPError:
        return urllib.error.HTTPError("https://a.example/", code, reason, {}, None)  # type: ignore[arg-type]

    opener = FakeOpener({
        "https://a.example/404": http(404, "Not Found"),
        "https://a.example/403": http(403, "Forbidden"),
        "https://a.example/503": http(503, "Unavailable"),
        "https://a.example/418": http(418, "Teapot"),
        "https://a.example/dns": urllib.error.URLError(OSError("Name or service not known")),
        "https://a.example/slow": urllib.error.URLError(TimeoutError("timed out")),
        "https://a.example/slow2": TimeoutError("timed out"),
        "https://a.example/reset": ConnectionResetError("reset"),
        "https://a.example/redirect": FakeResponse("secret", "http://127.0.0.1/admin", content_type="text/plain"),
    })
    expect = {"404": ("not_found", False), "403": ("permission_denied", False), "503": ("unavailable", True), "418": ("invalid_input", False),
              "dns": ("unavailable", True), "slow": ("timeout", True), "slow2": ("timeout", True), "reset": ("unavailable", True), "redirect": ("invalid_input", False)}
    for path, (category, retryable) in expect.items():
        with pytest.raises(ToolFailure) as ei:
            fetch(f"https://a.example/{path}", opener=opener, resolver=PUBLIC_RESOLVER)
        err = ei.value.error
        assert err.category.value == category and err.retryable is retryable and err.remediation, path
    assert "HTTP 404" in str(expect) or True  # keeps the table readable; the assertions above are the test


# ----------------------------------------------------------------------------- to_markdown
def make_result(body: bytes | str, content_type: str, url: str = "https://a.example/doc", truncated: bool = False) -> FetchResult:
    data = body.encode("utf-8") if isinstance(body, str) else body
    mime, declared = web._parse_content_type(content_type)
    text, charset = web.decode_body(data, declared)
    return FetchResult(url, url, 200, mime, text, len(data), truncated, charset, "the limit note" if truncated else "", data)


def test_to_markdown_html_text_markdown_json(tmp_path: Path) -> None:
    md, title = to_markdown(make_result(html_page(), "text/html"))
    assert title == "Bayes' theorem" and "# Bayes' theorem" in md and "**prior**" in md and "- one" in md and "evil()" not in md and "menu" not in md
    md, title = to_markdown(make_result("plain words\n", "text/plain"))
    assert md == "plain words\n" and title == ""
    md, title = to_markdown(make_result("# Notes\n\nbody\n", "text/markdown"))
    assert md.startswith("# Notes") and title == "Notes"
    md, title = to_markdown(make_result("# Readme\n", "text/plain", url="https://a.example/README.md"))
    assert title == "Readme"
    md, title = to_markdown(make_result('{"b":[1,2],"a":{"c":null}}', "application/json; charset=utf-8"))
    assert md == '{\n  "b": [\n    1,\n    2\n  ],\n  "a": {\n    "c": null\n  }\n}\n' and title == ""
    md, _ = to_markdown(make_result("{not json", "application/json"))
    assert md == "{not json"
    md, title = to_markdown(make_result("<html><body><h2>Sniffed</h2></body></html>", "application/octet-stream"))
    assert "## Sniffed" in md
    md, _ = to_markdown(make_result("cut here", "text/plain", truncated=True))
    assert md.endswith("[fetch truncated: the limit note]\n")
    assert content_kind(make_result(b"\x00\x01\x02", "application/octet-stream")) == "binary"
    assert content_kind(make_result("<svg/>", "image/svg+xml")) == "text"
    assert content_kind(make_result("<?xml version='1.0'?><feed/>", "application/atom+xml")) == "text"


def test_to_markdown_pdf_saved_for_sources_add(tmp_path: Path) -> None:
    pdf = make_result(b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog >>\nendobj\n", "application/pdf", url="https://a.example/papers/bayes.pdf")
    assert content_kind(pdf) == "pdf"
    md, title = to_markdown(pdf)
    assert title == "" and "not saved" in md and "sources_add" in md
    md, _ = to_markdown(pdf, save_dir=tmp_path / "raw")
    saved = web.raw_file_path(pdf, tmp_path / "raw")
    assert saved.exists() and saved.suffix == ".pdf" and saved.read_bytes().startswith(b"%PDF-") and str(saved) in md and "sources_add" in md
    assert saved.name.startswith("a-example-papers-bayes-pdf-") and len(saved.stem.rsplit("-", 1)[1]) == 10
    sniffed = make_result(b"%PDF-1.4 binary", "application/octet-stream", url="https://a.example/x")
    assert content_kind(sniffed) == "pdf" and web.raw_file_path(sniffed, tmp_path).suffix == ".pdf"
    png = make_result(b"\x89PNG\r\n\x1a\n\x00\x00", "image/png", url="https://a.example/img.png")
    md, _ = to_markdown(png, save_dir=tmp_path / "raw")
    assert "binary content (image/png)" in md and web.raw_file_path(png, tmp_path / "raw").exists()


def test_raw_file_stem_is_stable_per_url() -> None:
    a = make_result("x", "text/html", url="https://Example.com/Docs/Guide/?v=1#top")
    b = make_result("y", "text/html", url="https://example.com/Docs/Guide/?v=1#other")
    assert web.raw_file_stem(a) == web.raw_file_stem(b) and web.raw_file_stem(a).startswith("example-com-docs-guide-")
    c = make_result("y", "text/html", url="https://example.com/Docs/Guide/?v=2")
    assert web.raw_file_stem(c) != web.raw_file_stem(a)
    assert web.canonical_url("HTTPS://Example.com") == "https://example.com/"


# ----------------------------------------------------------------------------- search
def test_parse_search_page_fixture() -> None:
    results, page_text = parse_search_page(DDG_HTML)
    assert [r["url"] for r in results] == [
        "https://en.wikipedia.org/wiki/Bayes%27_theorem",
        "https://www.probabilitycourse.com/chapter1/1_4_3_bayes_rule.php",
        "https://example.com/bayes?lang=en&v=2",
    ]
    assert results[0]["title"] == "Bayes' theorem - Wikipedia"
    assert results[0]["snippet"] == "In probability theory, Bayes' theorem describes the probability of an event based on prior knowledge."
    assert results[1]["title"] == "Bayes' Rule | Probability Course" and "prior & likelihood" in results[1]["snippet"]
    assert results[2]["snippet"] == "Two coins, one biased: a worked example."
    assert "Buy Statistics Courses" not in str(results) and "bayes theorem at DuckDuckGo" in page_text
    assert parse_search_page("<div class='result'><a class='result__a' href='/relative'>x</a>")[0] == []
    assert parse_search_page("")[0] == []


def test_search_through_fake_opener() -> None:
    opener = FakeOpener({web.SEARCH_URL: lambda url: FakeResponse(DDG_HTML, url)})
    results = search("bayes  theorem", max_results=2, opener=opener, resolver=PUBLIC_RESOLVER)
    assert len(results) == 2 and results[0]["title"].startswith("Bayes' theorem") and set(results[0]) == {"title", "url", "snippet"}
    url, headers, _ = opener.calls[0]
    assert url == web.SEARCH_URL + "?q=bayes+theorem" and headers["user-agent"].startswith("edgent/")
    assert len(search("bayes theorem", max_results=50, opener=opener, resolver=PUBLIC_RESOLVER)) == 3
    with pytest.raises(ToolFailure) as ei:
        search("   ", opener=opener, resolver=PUBLIC_RESOLVER)
    assert ei.value.error.category.value == "invalid_input"


def test_search_blocks_and_empty_pages() -> None:
    forbidden = urllib.error.HTTPError(web.SEARCH_URL, 403, "Forbidden", {}, None)  # type: ignore[arg-type]
    opener = FakeOpener({web.SEARCH_URL + "?q=blocked": forbidden, web.SEARCH_URL + "?q=captcha": lambda url: FakeResponse(DDG_BLOCKED_HTML, url),
                         web.SEARCH_URL + "?q=nothing": lambda url: FakeResponse(DDG_EMPTY_HTML, url), web.SEARCH_URL + "?q=weird": lambda url: FakeResponse("<html><body>???</body></html>", url)})
    for q in ("blocked", "captcha", "weird"):
        with pytest.raises(ToolFailure) as ei:
            search(q, opener=opener, resolver=PUBLIC_RESOLVER)
        err = ei.value.error
        assert err.category.value == "unavailable" and err.retryable is False and "try web_fetch on a known URL" in err.remediation, q
    assert search("nothing", opener=opener, resolver=PUBLIC_RESOLVER) == []
    with pytest.raises(ToolFailure) as ei:
        search("x", opener=FakeOpener({}), resolver=lambda host: ["127.0.0.1"])
    assert ei.value.error.category.value == "invalid_input"


# ----------------------------------------------------------------------------- review additions
def test_fetch_normalises_urls_http_client_cannot_send() -> None:
    """IDN hosts and non-ASCII or blank characters raised raw UnicodeEncodeError / ValueError from http.client."""
    opener = FakeOpener({"https://": lambda url: FakeResponse("ok", url, content_type="text/plain")})
    cases = {
        "https://例え.テスト/パス?q=値": "https://xn--r8jz45g.xn--zckzah/%E3%83%91%E3%82%B9?q=%E5%80%A4",
        "https://en.wikipedia.org/wiki/Bayes’_theorem": "https://en.wikipedia.org/wiki/Bayes%E2%80%99_theorem",
        "https://a.example/my doc.txt": "https://a.example/my%20doc.txt",
        "https://user:pw@a.example:8443/x%20y?a=1&b=2#frag": "https://user:pw@a.example:8443/x%20y?a=1&b=2",
        "https://[2606:4700::1111]/z": "https://[2606:4700::1111]/z",
        "https://example.com/page": "https://example.com/page",
    }
    for given, sent in cases.items():
        r = fetch(given, opener=opener, resolver=PUBLIC_RESOLVER)
        assert opener.calls[-1][0] == sent and r.url == given and r.text == "ok", given


def test_fetch_maps_url_value_errors_to_invalid_input() -> None:
    class Rejecting:
        def open(self, req: Any, timeout: float | None = None) -> Any:
            raise ValueError("Tunnel host can't contain control characters")

    with pytest.raises(ToolFailure) as ei:
        fetch("https://exa mple.com/x", opener=Rejecting(), resolver=PUBLIC_RESOLVER)
    assert ei.value.error.category.value == "invalid_input" and "cannot be requested" in ei.value.error.message
    with pytest.raises(ToolFailure) as ei:
        fetch("https://example.com:99999/", opener=Rejecting(), resolver=PUBLIC_RESOLVER)
    assert ei.value.error.category.value == "invalid_input" and "port" in ei.value.error.message


@pytest.mark.parametrize("url", ["http://[64:ff9b::7f00:1]/", "http://[2002:7f00:1::1]/", "http://224.0.0.1/", "http://[::ffff:10.0.0.1]/", "http://[ff02::1]/"])
def test_check_url_sees_through_embedded_and_multicast_addresses(url: str) -> None:
    with pytest.raises(ToolFailure) as ei:
        check_url(url, resolver=PUBLIC_RESOLVER)
    assert ei.value.error.category.value == "invalid_input"
    assert check_url("http://[2606:4700::1111]/", resolver=PUBLIC_RESOLVER).hostname == "2606:4700::1111"
