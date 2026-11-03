"""Headless Chromium as a verifier: open a local page, run its self-test, measure the layout, take a picture.

``run_page`` writes a wrapper document (the content plus ``pagelab.js`` and a Content-Security-Policy that
blocks every external script, style, fetch and frame so a widget cannot phone home), runs Chromium in the
sandbox with the network off and a named timeout, and returns the screenshot, the page lab's JSON report and
a DOM excerpt. The binary comes from ``edgent.adapters.toolchain.find_chromium``; Playwright's
``headless_shell`` is preferred over the full browser when it sits beside it, because it starts in a fraction
of the time and needs no display, GPU or D-Bus. When no browser exists the caller gets the typed
``unavailable`` error with the install hint. Tests inject a ``runner`` with ``run_subprocess``'s signature.
"""
from __future__ import annotations

import html as html_lib
import json
import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from ...domain.errors import ErrorCategory, Limit, ToolFailure
from ...domain.utils import ensure_dir
from ..sandbox import SandboxResult, run_subprocess
from ..toolchain import INSTALL_HINTS, find_chromium

CHROMIUM_TIMEOUT_LIMIT = Limit("chromium page timeout", 120, "s", floor=5)
# Chromium reserves well over 64 GiB of virtual address space at start-up (the V8 sandbox and PartitionAlloc
# regions) and dies with SIGTRAP under a smaller RLIMIT_AS. The ceiling that matters is the wall-clock timeout.
CHROMIUM_ADDRESS_SPACE_BYTES = 256 << 30
PAGE_HTML_LIMIT = Limit("page html", 2_000_000, "chars")
# The DOM dump carries the page, the page lab and the report at its end; the sandbox's default 20 000-char cap
# would cut the report off any page over a few kilobytes, so the dump gets its own, much larger ceiling.
DOM_DUMP_LIMIT = Limit("dom dump", 8_000_000, "chars")
VIRTUAL_TIME_BUDGET_MS = 4000  # timers and animation frames the page may consume before the DOM is dumped
VIEWPORT_MIN_PX = 200
VIEWPORT_MAX_PX = 4096
DOM_EXCERPT_CHARS = 1500
STDERR_TAIL_CHARS = 600
REPORT_SCRIPT_ID = "edgent-report"
PAGELAB_SCRIPT_ID = "edgent-pagelab"
CONTENT_SECURITY_POLICY = (
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob: file:; "
    "font-src data: file:; media-src data: blob: file:; connect-src 'none'; frame-src 'none'; object-src 'none'; "
    "form-action 'none'; base-uri 'none'"
)
HEADLESS_FLAGS: tuple[str, ...] = (
    "--headless=new",
    "--disable-gpu",
    "--no-sandbox",
    "--hide-scrollbars",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-extensions",
    "--disable-sync",
    "--disable-background-networking",
    "--disable-dev-shm-usage",
    "--run-all-compositor-stages-before-draw",
)
_HEADLESS_SHELL_NAMES = ("headless_shell", "headless_shell.exe", "chrome-headless-shell")
_REPORT_RE = re.compile(rf'<script id="{REPORT_SCRIPT_ID}" type="application/json">(.*?)</script>', re.DOTALL)
_SCRIPT_RE = re.compile(r"<script\b[^>]*>.*?</script>", re.DOTALL | re.IGNORECASE)
_STYLE_RE = re.compile(r"<style\b[^>]*>.*?</style>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")

Runner = Callable[..., SandboxResult]


@dataclass
class PageResult:
    ok: bool
    screenshot_path: str | None
    report: dict[str, Any] | None
    dom_excerpt: str
    errors: list[str] = field(default_factory=list)
    html_path: str = ""
    duration_s: float = 0.0
    returncode: int = 0
    stderr_tail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "screenshot_path": self.screenshot_path,
            "report": self.report,
            "dom_excerpt": self.dom_excerpt,
            "errors": list(self.errors),
            "html_path": self.html_path,
            "duration_s": round(self.duration_s, 3),
            "returncode": self.returncode,
        }


# ----------------------------------------------------------------------------- discovery
@lru_cache(maxsize=1)
def pagelab_source() -> str:
    return (Path(__file__).with_name("pagelab.js")).read_text(encoding="utf-8")


def find_browser(found: str | None = None) -> str | None:
    """The browser to run: ``found`` (or ``find_chromium()``), upgraded to a ``headless_shell`` sibling when one exists.

    Playwright installs ``chromium-NNNN/chrome-linux/chrome`` next to ``chromium_headless_shell-NNNN/chrome-linux/headless_shell``.
    """
    path = found if found is not None else find_chromium()
    if not path:
        return None
    p = Path(path)
    if p.name in _HEADLESS_SHELL_NAMES:
        return str(p)
    candidates: list[Path] = [p.with_name(name) for name in _HEADLESS_SHELL_NAMES]
    parts = list(p.parts)
    for i, part in enumerate(parts):
        if part.startswith("chromium-"):
            sibling = parts[:i] + ["chromium_headless_shell-" + part[len("chromium-") :]] + parts[i + 1 : -1]
            candidates += [Path(*sibling, name) for name in _HEADLESS_SHELL_NAMES]
    for c in candidates:
        if c.is_file() or c.is_symlink():
            return str(c)
    return str(p)


def unavailable() -> ToolFailure:
    return ToolFailure(
        "unavailable",
        "Chromium is required for browser checks (widget_run, diagram_render) and was not found on this machine.",
        ErrorCategory.unavailable,
        remediation=INSTALL_HINTS["chromium"] + " Then run `edgent doctor` to confirm. Until then validate the markup statically and record the visual check as an unknown.",
        toolchain="chromium",
    )


# ----------------------------------------------------------------------------- wrapping
def wrap_page(html: str, title: str = "", csp: str = CONTENT_SECURITY_POLICY) -> str:
    """The content with the CSP and the page lab in its head (added to a full document, or wrapped around a fragment)."""
    policy = html_lib.escape(csp, quote=False).replace('"', "&quot;")  # single quotes are CSP syntax; keep them readable
    head_extra = f'<meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="{policy}"><script id="{PAGELAB_SCRIPT_ID}">{pagelab_source()}</script>'
    if re.search(r"<html[\s>]", html, re.IGNORECASE):
        m = re.search(r"<head[^>]*>", html, re.IGNORECASE)
        if m:
            return html[: m.end()] + head_extra + html[m.end() :]
        m = re.search(r"<html[^>]*>", html, re.IGNORECASE)
        assert m is not None
        return html[: m.end()] + "<head>" + head_extra + "</head>" + html[m.end() :]
    return f"<!doctype html><html><head>{head_extra}<title>{html_lib.escape(title)}</title></head><body>{html}</body></html>"


def parse_report(dom: str) -> dict[str, Any] | None:
    m = _REPORT_RE.search(dom or "")
    if not m:
        return None
    raw = m.group(1)
    for candidate in (raw, html_lib.unescape(raw)):  # script text is dumped raw; unescape only as a fallback
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return data if isinstance(data, dict) else None
    return None


def dom_excerpt(dom: str, chars: int = DOM_EXCERPT_CHARS) -> str:
    """The page's visible text, collapsed, for a quick look; scripts and styles removed."""
    text = _SCRIPT_RE.sub(" ", dom or "")
    text = _STYLE_RE.sub(" ", text)
    text = html_lib.unescape(_TAG_RE.sub(" ", text))
    text = " ".join(text.split())
    return text if len(text) <= chars else text[: chars - 1] + "…"


DEFAULT_SEVERITIES: dict[str, str] = {
    "console_error": "error",
    "test_failure": "error",
    "overflow_x": "error",
    "overflow_y": "warn",
    "clipped": "error",
    "text_overlap": "error",
    "small_text": "error",
    "low_contrast": "warn",
    "pagelab_error": "warn",
}


def report_defects(report: dict[str, Any] | None, severities: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """The page lab's report as a flat list of ``{"kind", "severity", "detail", ...}`` (severities overridable per kind)."""
    sev = {**DEFAULT_SEVERITIES, **(severities or {})}
    out: list[dict[str, Any]] = []
    if not report:
        return out

    def add(kind: str, detail: str, **extra: Any) -> None:
        out.append({"kind": kind, "severity": sev.get(kind, "warn"), "detail": detail, **extra})

    for msg in report.get("console_errors") or []:
        add("console_error", str(msg))
    for msg in report.get("test_failures") or []:
        add("test_failure", str(msg))
    m = report.get("measurements") if isinstance(report.get("measurements"), dict) else {}
    for item in m.get("overflow") or []:
        px = item.get("overflow_px") or [0, 0]
        if item.get("axis") == "y":
            add("overflow_y", f"{item.get('element')} extends {px[1]} px below the viewport", element=item.get("element"))
        else:
            add("overflow_x", f"{item.get('element')} extends {px[0]} px past the side of the viewport", element=item.get("element"))
    for item in m.get("clipped") or []:
        px = item.get("beyond_px") or [0, 0]
        add("clipped", f"{item.get('element')} extends {px[0]} px sideways and {px[1]} px vertically beyond its drawing's viewBox and is cut off", element=item.get("element"))
    for item in m.get("overlaps") or []:
        px = item.get("overlap_px") or [0, 0]
        add("text_overlap", f"{item.get('a')} overlaps {item.get('b')} by {px[0]} x {px[1]} px", a=item.get("a"), b=item.get("b"))
    for item in m.get("small_text") or []:
        add("small_text", f"{item.get('element')} is {item.get('font_px')} px; the floor is {item.get('min_px', 12)} px", font_px=item.get("font_px"))
    for item in m.get("low_contrast") or []:
        add("low_contrast", f"{item.get('element')} has contrast {item.get('ratio')}:1 ({item.get('fg')} on {item.get('bg')}); {item.get('required')}:1 is required", ratio=item.get("ratio"))
    for msg in report.get("pagelab_errors") or []:
        add("pagelab_error", str(msg))
    return out


def _tail(text: str, chars: int = STDERR_TAIL_CHARS) -> str:
    text = (text or "").strip()
    return text if len(text) <= chars else "…" + text[-chars:]


def _clamp_viewport(value: Any, default: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = default
    return max(VIEWPORT_MIN_PX, min(n, VIEWPORT_MAX_PX))


# ----------------------------------------------------------------------------- running
def run_page(
    html: str,
    out_dir: str | os.PathLike[str],
    width: int = 1280,
    height: int = 800,
    timeout_s: float = 30.0,
    screenshot: bool = True,
    chromium: str | None = None,
    runner: Runner | None = None,
    name: str = "page",
    title: str = "",
) -> PageResult:
    """Open ``html`` (a fragment or a full document) in headless Chromium and bring back the picture and the report.

    One Chromium run produces both the screenshot and the DOM dump; when the report is missing from that dump
    a second, DOM-only run is tried before giving up. ``errors`` lists what went wrong with the run itself; the
    page's own console errors and test failures are inside ``report``.
    """
    if len(html) > PAGE_HTML_LIMIT.value:
        err = PAGE_HTML_LIMIT.exceeded(len(html), "Split the page, or move data out of the markup into a smaller structure.")
        raise ToolFailure(err.code, err.message, err.category, remediation=err.remediation)
    exe = chromium or find_browser()
    if not exe:
        raise unavailable()
    out = Path(ensure_dir(out_dir)).resolve()  # the sandbox pins TMPDIR and HOME to the cwd; a relative one breaks Chromium
    width, height = _clamp_viewport(width, 1280), _clamp_viewport(height, 800)
    timeout = float(min(max(timeout_s, CHROMIUM_TIMEOUT_LIMIT.floor or 0), CHROMIUM_TIMEOUT_LIMIT.value))
    html_path = out / f"{name}.html"
    html_path.write_text(wrap_page(html, title), encoding="utf-8")
    png = out / f"{name}.png"
    if png.exists():
        png.unlink()
    base = [exe, *HEADLESS_FLAGS, f"--window-size={width},{height}", f"--virtual-time-budget={VIRTUAL_TIME_BUDGET_MS}", "--dump-dom"]
    url = html_path.resolve().as_uri()
    run = runner or run_subprocess

    def launch(argv: list[str]) -> SandboxResult:
        return run(
            argv, str(out), timeout_s=timeout, cpu_seconds=int(math.ceil(timeout)), memory_bytes=CHROMIUM_ADDRESS_SPACE_BYTES,
            allow_network=False, max_output=int(DOM_DUMP_LIMIT.value),
        )

    r = launch([*base, *([f"--screenshot={png}"] if screenshot else []), url])
    errors: list[str] = []
    if r.timed_out:
        errors.append(f"chromium did not finish within {timeout:g} s; {CHROMIUM_TIMEOUT_LIMIT.describe()}")
    elif r.returncode != 0:
        errors.append(f"chromium exited {r.returncode}: {_tail(r.stderr) or 'no output'}")
    dom = r.stdout or ""
    report = parse_report(dom)
    duration = r.duration_s
    truncated = r.truncated
    if report is None and screenshot and not r.timed_out:
        r2 = launch([*base, url])
        duration += r2.duration_s
        if r2.stdout:
            dom = r2.stdout
            report = parse_report(dom)
            truncated = r2.truncated
    if report is None and not r.timed_out:
        if truncated:
            errors.append(f"the DOM dump was cut off before the page-lab report. {DOM_DUMP_LIMIT.describe()} Make the page smaller (less markup, no large inline data)")
        else:
            errors.append("no page-lab report in the DOM: the page threw before load, navigated away, or Chromium could not run the inline script")
    screenshot_path = str(png) if screenshot and png.is_file() and png.stat().st_size > 0 else None
    if screenshot and screenshot_path is None and not errors:
        errors.append("the screenshot was not written")
    return PageResult(
        ok=not errors,
        screenshot_path=screenshot_path,
        report=report,
        dom_excerpt=dom_excerpt(dom),
        errors=errors,
        html_path=str(html_path),
        duration_s=duration,
        returncode=r.returncode,
        stderr_tail=_tail(r.stderr),
    )


__all__ = [
    "CHROMIUM_ADDRESS_SPACE_BYTES",
    "CHROMIUM_TIMEOUT_LIMIT",
    "CONTENT_SECURITY_POLICY",
    "DEFAULT_SEVERITIES",
    "DOM_DUMP_LIMIT",
    "PAGE_HTML_LIMIT",
    "VIRTUAL_TIME_BUDGET_MS",
    "PageResult",
    "dom_excerpt",
    "find_browser",
    "pagelab_source",
    "parse_report",
    "report_defects",
    "run_page",
    "unavailable",
    "wrap_page",
]
