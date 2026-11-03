"""Headless Chromium through the sandbox with the page lab; the browser runs are skipped where none is installed."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from edgent.adapters.browser import chromium
from edgent.adapters.browser.chromium import (
    dom_excerpt,
    find_browser,
    parse_report,
    report_defects,
    run_page,
    wrap_page,
)
from edgent.adapters.sandbox import SandboxResult
from edgent.adapters.sandbox.runner import MAX_OUTPUT, _cap
from edgent.domain.errors import ToolFailure
from tests.conftest import skip_unless

chromium_available = skip_unless("chromium")

DEFECTIVE_PAGE = """<style>
body{font-family:sans-serif;margin:20px;background:#fff;color:#111}
#a{position:absolute;left:40px;top:40px;font-size:20px}
#b{position:absolute;left:60px;top:48px;font-size:20px}
#c{position:absolute;left:40px;top:200px}
.faint{color:#cfcfcf}
.tiny{font-size:9px}
#wide{position:absolute;left:1200px;top:300px;width:300px;height:30px;background:#eee}
#scroller{overflow:auto;width:200px;height:50px;margin-top:120px}
#inner{width:800px}
</style>
<h1>Widget</h1>
<p id="a">Overlapping text one</p>
<p id="b">Overlapping text two</p>
<p id="c"><span class="faint">faint grey text</span> and <span class="tiny">tiny text</span></p>
<div id="wide">way off to the right</div>
<div id="scroller"><div id="inner">scrolls inside, not an overflow</div></div>
<svg width="300" height="60" viewBox="0 0 300 60"><text x="5" y="30" font-size="14" fill="#333">label</text><text x="280" y="30" font-size="14" fill="#333">clipped</text></svg>
<script>
console.error("boom", {a: 1});
window.__edgentTest = function () { return ["expected 3 got 4", null, "", "second failure"]; };
notDefinedAnywhere();
</script>
"""


@chromium_available
def test_run_page_measures_a_defective_page(tmp_path: Path):
    res = run_page(DEFECTIVE_PAGE, tmp_path / "lab", width=1280, height=800, timeout_s=60, title="Defective")
    assert res.ok, res.errors
    assert res.screenshot_path and Path(res.screenshot_path).stat().st_size > 0
    report = res.report
    assert report is not None and report["viewport"] == {"width": 1280, "height": 800} and report["title"] == "Defective"
    assert any("boom" in e for e in report["console_errors"]) and any("notDefinedAnywhere" in e for e in report["console_errors"])
    assert report["tests_ran"] is True and report["test_failures"] == ["expected 3 got 4", "second failure"]
    m = report["measurements"]
    assert any({o["a"].split()[0], o["b"].split()[0]} == {"p#a", "p#b"} for o in m["overlaps"])
    assert [o["element"].split()[0] for o in m["overflow"]] == ["div#wide"] and m["overflow"][0]["axis"] == "x"
    assert [s["element"].split()[0] for s in m["small_text"]] == ["span.tiny"] and m["small_text"][0]["font_px"] == 9
    assert any(c["element"].startswith("span.faint") and c["ratio"] < 4.5 for c in m["low_contrast"])
    assert [c["element"] for c in m["clipped"]] == ['text "clipped"'] and m["clipped"][0]["beyond_px"][0] > 5
    assert "Overlapping text one" in res.dom_excerpt and "__edgentTest" not in res.dom_excerpt
    html = Path(res.html_path).read_text(encoding="utf-8")
    assert "Content-Security-Policy" in html and html.index('id="edgent-pagelab"') < html.index("Overlapping")
    defects = report_defects(report)
    kinds = {d["kind"] for d in defects}
    assert {"console_error", "test_failure", "overflow_x", "text_overlap", "small_text", "low_contrast", "clipped"} <= kinds
    assert all(d["severity"] in ("error", "warn") for d in defects)


@chromium_available
def test_run_page_clean_full_document_without_screenshot(tmp_path: Path):
    doc = (
        "<!doctype html><html><head><title>Clean</title><style>body{background:#fff;color:#222;font:16px sans-serif}</style></head>"
        "<body><p>All good here.</p><script>window.__edgentTest = () => Promise.resolve([]);</script></body></html>"
    )
    res = run_page(doc, tmp_path / "lab", width=800, height=600, screenshot=False)
    assert res.ok and res.screenshot_path is None
    r = res.report
    assert r["tests_ran"] and r["test_failures"] == [] and r["console_errors"] == []
    assert all(r["measurements"][k] == [] for k in ("overflow", "clipped", "overlaps", "small_text", "low_contrast"))
    assert report_defects(r) == []
    wrapped = Path(res.html_path).read_text(encoding="utf-8")
    assert wrapped.startswith("<!doctype html><html><head><meta charset") and "<title>Clean</title>" in wrapped


@chromium_available
def test_csp_blocks_scripts_the_page_did_not_inline(tmp_path: Path):
    lab = tmp_path / "lab"
    lab.mkdir()
    (lab / "external.js").write_text("window.externalRan = true;", encoding="utf-8")
    page = (
        "<script>document.addEventListener('securitypolicyviolation', function (e) { (window.blocked = window.blocked || []).push(e.blockedURI); });</script>"
        f'<script src="{(lab / "external.js").as_uri()}"></script>'
        "<p>guarded</p>"
        "<script>window.__edgentTest = function () { var f = []; if (window.externalRan) f.push('external script ran');"
        " if (!(window.blocked || []).length) f.push('no CSP violation was reported'); return f; };</script>"
    )
    res = run_page(page, lab, screenshot=False)
    assert res.ok, res.errors
    assert res.report["test_failures"] == []


def test_find_browser_prefers_a_headless_shell_sibling(tmp_path: Path):
    chrome = tmp_path / "chromium-1234" / "chrome-linux" / "chrome"
    chrome.parent.mkdir(parents=True)
    chrome.write_text("")
    assert find_browser(str(chrome)) == str(chrome)
    shell = tmp_path / "chromium_headless_shell-1234" / "chrome-linux" / "headless_shell"
    shell.parent.mkdir(parents=True)
    shell.write_text("")
    assert find_browser(str(chrome)) == str(shell)
    assert find_browser(str(shell)) == str(shell)
    local = tmp_path / "bin" / "chromium"
    local.parent.mkdir()
    local.write_text("")
    assert find_browser(str(local)) == str(local)
    (local.parent / "headless_shell").write_text("")
    assert find_browser(str(local)) == str(local.parent / "headless_shell")


def test_unavailable_when_no_browser(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(chromium, "find_chromium", lambda: None)
    assert find_browser() is None
    with pytest.raises(ToolFailure) as info:
        run_page("<p>x</p>", tmp_path / "lab")
    assert info.value.error.category.value == "unavailable" and "EDGENT_CHROMIUM" in info.value.error.remediation


def test_run_page_with_a_fake_runner_parses_the_report(tmp_path: Path):
    calls: list[tuple[list[str], str, dict]] = []
    report = {"version": 1, "console_errors": [], "tests_ran": False, "test_failures": [], "measurements": {"overflow": [], "clipped": [], "overlaps": [], "small_text": [], "low_contrast": []}}

    def runner(argv: list[str], cwd: str, **kw) -> SandboxResult:
        calls.append((argv, cwd, kw))
        png = next((a.split("=", 1)[1] for a in argv if a.startswith("--screenshot=")), None)
        if png:
            Path(png).write_bytes(b"\x89PNG fake")
        dom = '<html><body><p>hi &amp; bye</p><script id="edgent-report" type="application/json">' + json.dumps(report) + "</script></body></html>"
        return SandboxResult(0, dom, "", 0.01, argv=argv)

    res = run_page("<p>hi &amp; bye</p>", tmp_path / "lab", width=640, height=480, timeout_s=10, chromium="/fake/headless_shell", runner=runner)
    assert res.ok and res.report == report and res.dom_excerpt == "hi & bye"
    assert len(calls) == 1
    argv, cwd, kw = calls[0]
    assert argv[0] == "/fake/headless_shell" and "--dump-dom" in argv and "--window-size=640,480" in argv and "--headless=new" in argv
    assert argv[-1].startswith("file://") and argv[-1].endswith("page.html") and cwd == str(tmp_path / "lab")
    assert kw["allow_network"] is False and kw["timeout_s"] == 10 and kw["memory_bytes"] == chromium.CHROMIUM_ADDRESS_SPACE_BYTES
    assert res.screenshot_path and Path(res.screenshot_path).read_bytes().startswith(b"\x89PNG")


def test_run_page_reports_a_missing_report_and_a_crash(tmp_path: Path):
    def crashing(argv: list[str], cwd: str, **kw) -> SandboxResult:
        return SandboxResult(-5, "", "Trace/breakpoint trap", 0.01, argv=argv)

    res = run_page("<p>x</p>", tmp_path / "lab", chromium="/fake/headless_shell", runner=crashing)
    assert not res.ok and res.report is None and res.screenshot_path is None
    assert any("exited -5" in e for e in res.errors) and any("no page-lab report" in e for e in res.errors)

    def timing_out(argv: list[str], cwd: str, **kw) -> SandboxResult:
        return SandboxResult(-9, "", "", 30.0, timed_out=True, argv=argv)

    res = run_page("<p>x</p>", tmp_path / "lab", chromium="/fake/headless_shell", runner=timing_out, timeout_s=7)
    assert not res.ok and any("did not finish within 7 s" in e for e in res.errors)


def test_parse_report_excerpt_and_wrap():
    assert parse_report('<script id="edgent-report" type="application/json">{"a": 1}</script>') == {"a": 1}
    assert parse_report('<script id="edgent-report" type="application/json">[1]</script>') is None
    assert parse_report("no report") is None
    assert dom_excerpt("<style>p{}</style><p>Hello <b>world</b></p><script>x()</script>", chars=8) == "Hello w…"
    wrapped = wrap_page("<p>frag</p>", title="T <x>")
    assert wrapped.startswith("<!doctype html>") and "<title>T &lt;x&gt;</title>" in wrapped and "connect-src 'none'" in wrapped
    doc = wrap_page("<html><body><p>no head</p></body></html>")
    assert doc.index("<head>") < doc.index("<p>no head</p>") and "edgent-pagelab" in doc
    defects = report_defects({"console_errors": ["x"], "measurements": {"overflow": [{"element": "div", "axis": "y", "overflow_px": [0, 40]}]}}, {"overflow_y": "error"})
    assert [(d["kind"], d["severity"]) for d in defects] == [("console_error", "error"), ("overflow_y", "error")]
    assert report_defects(None) == []


def test_html_size_limit(tmp_path: Path):
    with pytest.raises(ToolFailure) as info:
        run_page("x" * (int(chromium.PAGE_HTML_LIMIT.value) + 1), tmp_path / "lab", chromium="/fake", runner=lambda *a, **k: None)
    assert info.value.error.code == "limit:page html"


@chromium_available
def test_run_page_keeps_the_report_of_a_page_larger_than_the_sandbox_output_cap(tmp_path: Path):
    """The DOM dump carries the page lab and the page itself; the report sits at its very end."""
    page = "".join(f"<p>paragraph {i} of a long page that is well past the sandbox's default output cap</p>\n" for i in range(600))
    assert len(page) > MAX_OUTPUT
    res = run_page(page + "<script>window.__edgentTest = () => [];</script>", tmp_path / "lab", screenshot=False)
    assert res.ok, res.errors
    assert res.report["tests_ran"] and res.report["counts"]["text_elements"] == 600


@chromium_available
def test_svg_label_contrast_is_measured_against_the_shape_beneath(tmp_path: Path):
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 200" width="400" height="200" font-family="system-ui, sans-serif">'
        '<rect x="10" y="10" width="200" height="80" fill="#1b1f24"/><text id="light" x="20" y="60" font-size="18" fill="#ffffff">light on dark</text>'
        '<rect x="220" y="10" width="170" height="80" fill="#f4f6f8"/><text id="faint" x="230" y="60" font-size="18" fill="#cfcfcf">faint on light</text>'
        '<text id="plain" x="20" y="150" font-size="18" fill="#1b1f24">dark on page</text><text id="grey" x="220" y="150" font-size="18" fill="#bbbbbb">grey on page</text>'
        "</svg>"
    )
    res = run_page(svg, tmp_path / "lab", width=500, height=300, screenshot=False)
    assert res.ok, res.errors
    low = {c["element"].split()[0]: c for c in res.report["measurements"]["low_contrast"]}
    assert set(low) == {"text#faint", "text#grey"}
    assert low["text#faint"]["bg"] == "rgb(244, 246, 248)" and low["text#grey"]["bg"] == "rgb(255, 255, 255)"


def test_dom_dump_is_not_cut_at_the_sandbox_default(tmp_path: Path):
    report = {"version": 1, "console_errors": [], "tests_ran": True, "test_failures": [], "measurements": {"overflow": [], "clipped": [], "overlaps": [], "small_text": [], "low_contrast": []}}
    dom = "<html><body>" + "<p>filler</p>" * 3000 + '<script id="edgent-report" type="application/json">' + json.dumps(report) + "</script></body></html>"
    assert len(dom) > MAX_OUTPUT
    seen: list[dict] = []

    def capped(argv: list[str], cwd: str, **kw) -> SandboxResult:  # honours max_output exactly as the sandbox does
        seen.append(kw)
        text, truncated = _cap(dom.encode("utf-8"), kw.get("max_output", MAX_OUTPUT))
        return SandboxResult(0, text, "", 0.01, truncated=truncated, argv=argv)

    res = run_page("<p>x</p>", tmp_path / "lab", chromium="/fake/headless_shell", runner=capped, screenshot=False)
    assert res.ok and res.report == report and seen[0]["max_output"] == chromium.DOM_DUMP_LIMIT.value

    def default_cap(argv: list[str], cwd: str, **kw) -> SandboxResult:  # a runner that ignores the ceiling
        text, truncated = _cap(dom.encode("utf-8"), MAX_OUTPUT)
        return SandboxResult(0, text, "", 0.01, truncated=truncated, argv=argv)

    res = run_page("<p>x</p>", tmp_path / "lab", chromium="/fake/headless_shell", runner=default_cap, screenshot=False)
    assert not res.ok and res.report is None and any("cut off" in e and "dom dump" in e for e in res.errors)


def test_relative_out_dir_is_made_absolute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    cwds: list[str] = []

    def runner(argv: list[str], cwd: str, **kw) -> SandboxResult:
        cwds.append(cwd)
        return SandboxResult(0, "", "", 0.01, argv=argv)

    run_page("<p>x</p>", "lab", chromium="/fake/headless_shell", runner=runner, screenshot=False)
    assert cwds == [str((tmp_path / "lab").resolve())]
