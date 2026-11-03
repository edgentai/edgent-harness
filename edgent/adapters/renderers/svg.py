"""SVG diagrams: validate by parsing, rasterise through Chromium, measure the labels with the page lab.

``validate_svg`` is pure Python (well-formed XML, an ``svg`` root with its namespace, a viewBox or a size,
no scripts, label font sizes against the 12 px floor). ``render_svg_png`` wraps the SVG in a white page the
size of its viewBox and screenshots it; ``measure_svg`` reads the page lab's report: labels overlapping,
anything drawn outside the viewBox, small text and low contrast. Diagrams render on white because that is
the book page; a dark-theme diagram should carry its own background rect.
"""
from __future__ import annotations

import html as html_lib
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...domain.errors import ErrorCategory, Limit, ToolFailure
from ...domain.utils import ensure_dir
from ..browser.chromium import report_defects, run_page

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
SVG_SOURCE_LIMIT = Limit("svg source", 2_000_000, "chars")
SVG_MIN_FONT_PX = 12.0
SVG_DEFAULT_FONT_PX = 16.0
RENDER_WIDTH_PX_DEFAULT = 1200
RENDER_SIZE_PX_MIN = 64
RENDER_SIZE_PX_MAX = 4096
RENDER_TIMEOUT_S = 45.0
MAX_SMALL_TEXT_REPORTED = 25
DIAGRAM_SEVERITIES: dict[str, str] = {"overflow_x": "error", "overflow_y": "error", "low_contrast": "warn"}
_LENGTH_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+(?:e[+-]?[0-9]+)?)\s*([a-z%]*)\s*$", re.IGNORECASE)
_UNIT_PX = {"": 1.0, "px": 1.0, "pt": 4 / 3, "pc": 16.0, "mm": 96 / 25.4, "cm": 96 / 2.54, "in": 96.0}
_FONT_KEYWORDS = {"xx-small": 9.0, "x-small": 10.0, "small": 13.0, "medium": 16.0, "large": 18.0, "x-large": 24.0, "xx-large": 32.0}
_STYLE_FONT_RE = re.compile(r"font-size\s*:\s*([^;]+)", re.IGNORECASE)


@dataclass
class SvgValidation:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    width: float | None = None
    height: float | None = None
    view_box: list[float] | None = None
    element_count: int = 0
    text_count: int = 0
    font_sizes: list[float] = field(default_factory=list)
    small_text: list[dict[str, Any]] = field(default_factory=list)
    has_script: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "width": self.width,
            "height": self.height,
            "view_box": self.view_box,
            "element_count": self.element_count,
            "text_count": self.text_count,
            "font_sizes": sorted(set(self.font_sizes)),
            "small_text": list(self.small_text),
            "has_script": self.has_script,
        }


# ----------------------------------------------------------------------------- parsing helpers
def _local(tag: str) -> tuple[str, str]:
    if tag.startswith("{"):
        ns, _, name = tag[1:].partition("}")
        return name, ns
    return tag, ""


def _length(value: str | None, em_px: float = SVG_DEFAULT_FONT_PX) -> float | None:
    if value is None:
        return None
    m = _LENGTH_RE.match(value)
    if not m:
        return None
    number, unit = float(m.group(1)), m.group(2).lower()
    if unit in ("%",):
        return None
    if unit in ("em", "rem"):
        return number * em_px
    factor = _UNIT_PX.get(unit)
    return number * factor if factor is not None else None


def _font_size(el: ET.Element) -> float | None:
    raw = el.get("font-size")
    if raw is None:
        style = el.get("style") or ""
        m = _STYLE_FONT_RE.search(style)
        raw = m.group(1).strip() if m else None
    if raw is None:
        return None
    key = raw.strip().lower()
    if key in _FONT_KEYWORDS:
        return _FONT_KEYWORDS[key]
    return _length(raw)


def _effective_font(el: ET.Element, parents: dict[ET.Element, ET.Element]) -> float:
    node: ET.Element | None = el
    while node is not None:
        size = _font_size(node)
        if size is not None:
            return size
        node = parents.get(node)
    return SVG_DEFAULT_FONT_PX


def _own_text(el: ET.Element) -> str:
    parts = [el.text or ""]
    for child in el:
        parts.append(child.tail or "")
    return " ".join(" ".join(parts).split())


def _view_box(raw: str | None) -> tuple[list[float] | None, str | None]:
    if raw is None:
        return None, None
    try:
        nums = [float(v) for v in raw.replace(",", " ").split()]
    except ValueError:
        return None, f"viewBox {raw!r} is not four numbers"
    if len(nums) != 4:
        return None, f"viewBox {raw!r} is not four numbers"
    if nums[2] <= 0 or nums[3] <= 0:
        return None, f"viewBox {raw!r} has a zero or negative size"
    return nums, None


# ----------------------------------------------------------------------------- validation
def validate_svg(text: str) -> SvgValidation:
    """Parse and check an SVG without a browser; ``ok`` is False when it would not render as a standalone file."""
    v = SvgValidation(ok=False)
    if not text or not text.strip():
        v.errors.append("the SVG is empty")
        return v
    if len(text) > SVG_SOURCE_LIMIT.value:
        v.errors.append(SVG_SOURCE_LIMIT.exceeded(len(text), "Simplify the drawing or split it.").message)
        return v
    try:
        root = ET.fromstring(text)
    except ET.ParseError as e:
        line, col = getattr(e, "position", (None, None))
        where = f" (line {line}, column {col})" if line else ""
        v.errors.append(f"not well-formed XML{where}: {e}")
        return v
    tag, ns = _local(root.tag)
    if tag != "svg":
        v.errors.append(f"the root element is <{tag}>, not <svg>")
    elif ns != SVG_NS:
        v.errors.append('the <svg> root needs xmlns="http://www.w3.org/2000/svg"; without it the file renders as nothing')
    v.view_box, vb_error = _view_box(root.get("viewBox"))
    if vb_error:
        v.errors.append(vb_error)
    v.width, v.height = _length(root.get("width")), _length(root.get("height"))
    if v.view_box is None and not vb_error and (v.width is None or v.height is None):
        v.errors.append("the <svg> needs a viewBox (preferred) or both width and height in absolute units")
    parents = {child: parent for parent in root.iter() for child in parent}
    for el in root.iter():
        v.element_count += 1
        name, _ = _local(el.tag)
        if name == "script":
            v.has_script = True
            v.errors.append("<script> is not allowed in a diagram; use a widget for anything interactive")
        elif name == "foreignObject":
            v.warnings.append("<foreignObject> content is not measured; prefer native SVG text")
        elif name == "text":
            v.text_count += 1
        if name in ("text", "tspan", "textPath"):
            content = _own_text(el)
            if content:
                size = _effective_font(el, parents)
                v.font_sizes.append(round(size, 2))
                if size < SVG_MIN_FONT_PX and len(v.small_text) < MAX_SMALL_TEXT_REPORTED:
                    v.small_text.append({"text": content[:60], "font_px": round(size, 2)})
        handlers = sorted(a for a in el.attrib if _local(a)[0].lower().startswith("on"))
        if handlers:
            v.has_script = True
            v.errors.append(f"<{name}> has an event handler attribute ({handlers[0]}); a diagram is static, use a widget for anything interactive")
        for attr in ("href", XLINK_HREF):
            ref = el.get(attr)
            if ref and ref.lower().startswith(("http://", "https://", "//")):
                v.warnings.append(f"<{name}> references {ref[:60]}; external resources never load in the sandbox and render blank")
    v.ok = not v.errors
    return v


def static_defects(v: SvgValidation, include_small_text: bool = True) -> list[dict[str, Any]]:
    """Defects known without a browser: validation errors, warnings and (optionally) small labels."""
    out: list[dict[str, Any]] = [{"kind": "invalid", "severity": "error", "detail": e} for e in v.errors]
    out += [{"kind": "warning", "severity": "warn", "detail": w} for w in v.warnings]
    if include_small_text:
        out += [{"kind": "small_text", "severity": "error", "detail": f"label {s['text']!r} is {s['font_px']:g} px; the floor is {SVG_MIN_FONT_PX:g} px", "font_px": s["font_px"]} for s in v.small_text]
    return out


# ----------------------------------------------------------------------------- rendering
def render_size(v: SvgValidation, width: int = RENDER_WIDTH_PX_DEFAULT) -> tuple[int, int]:
    """Pixel size for the picture: the viewBox aspect at ``width``, else the intrinsic size, else 4:3."""
    w = int(min(max(width, RENDER_SIZE_PX_MIN), RENDER_SIZE_PX_MAX))
    if v.view_box:
        h = int(round(w * v.view_box[3] / v.view_box[2]))
        if h > RENDER_SIZE_PX_MAX:
            w = int(round(w * RENDER_SIZE_PX_MAX / h))
            h = RENDER_SIZE_PX_MAX
        return max(w, RENDER_SIZE_PX_MIN), max(h, RENDER_SIZE_PX_MIN)
    if v.width and v.height:
        return int(min(max(round(v.width), RENDER_SIZE_PX_MIN), RENDER_SIZE_PX_MAX)), int(min(max(round(v.height), RENDER_SIZE_PX_MIN), RENDER_SIZE_PX_MAX))
    return w, int(round(w * 0.75))


def svg_page(svg_text: str, width: int, height: int, background: str = "#ffffff") -> str:
    """An HTML page exactly the size of the drawing, so viewport overflow means 'outside the viewBox'."""
    return (
        '<!doctype html><html><head><meta charset="utf-8"><title>diagram</title><style>'
        f"html,body{{margin:0;padding:0;background:{html_lib.escape(background)};width:{width}px;height:{height}px;overflow:hidden}}"
        f"body>svg{{display:block;width:{width}px;height:{height}px}}"
        f"</style></head><body>{svg_text}</body></html>"
    )


def _invalid(v: SvgValidation) -> ToolFailure:
    return ToolFailure(
        "invalid_input",
        "the SVG is not valid: " + "; ".join(v.errors),
        ErrorCategory.invalid_input,
        remediation="Fix the listed problems and resend the whole SVG. Do not repeat the unchanged call.",
        errors=list(v.errors),
    )


def render_svg_png(
    svg_text: str,
    out_png: str | Path,
    width: int = RENDER_WIDTH_PX_DEFAULT,
    chromium: str | None = None,
    runner: Any = None,
    out_dir: str | Path | None = None,
    timeout_s: float = RENDER_TIMEOUT_S,
    validation: SvgValidation | None = None,
) -> dict[str, Any]:
    """Rasterise through Chromium. Returns ``{"png_path", "width", "height", "report", "errors"}``; the report is the page lab's."""
    v = validation or validate_svg(svg_text)
    if not v.ok:
        raise _invalid(v)
    w, h = render_size(v, width)
    target = Path(out_png)
    ensure_dir(target.parent)
    work = Path(out_dir) if out_dir is not None else target.parent / ".pagelab" / target.stem
    page = run_page(svg_page(svg_text, w, h), work, width=w, height=h, timeout_s=timeout_s, screenshot=True, chromium=chromium, runner=runner, name="diagram")
    if page.screenshot_path is None:
        raise ToolFailure(
            "render_failed",
            "Chromium produced no picture of the diagram: " + ("; ".join(page.errors) or "unknown reason"),
            ErrorCategory.execution_failure,
            remediation="Check the SVG renders in a browser; if Chromium itself failed, run `edgent doctor` and read the error.",
            errors=list(page.errors),
        )
    shutil.copyfile(page.screenshot_path, target)
    return {"png_path": str(target), "width": w, "height": h, "report": page.report, "errors": list(page.errors)}


def measure_svg(
    svg_text: str,
    chromium: str | None = None,
    runner: Any = None,
    report: dict[str, Any] | None = None,
    out_dir: str | Path | None = None,
    validation: SvgValidation | None = None,
    timeout_s: float = RENDER_TIMEOUT_S,
) -> list[dict[str, Any]]:
    """Defects of a diagram: static ones, plus the page lab's when ``report`` is given or Chromium can produce one."""
    v = validation or validate_svg(svg_text)
    if not v.ok:
        return static_defects(v)
    if report is None:
        w, h = render_size(v, RENDER_WIDTH_PX_DEFAULT)
        work = Path(out_dir) if out_dir is not None else Path(tempfile.mkdtemp(prefix="edgent-pagelab-"))  # never the caller's cwd
        page = run_page(svg_page(svg_text, w, h), work, width=w, height=h, timeout_s=timeout_s, screenshot=False, chromium=chromium, runner=runner, name="diagram-measure")
        report = page.report
        if report is None:
            return static_defects(v) + [{"kind": "no_report", "severity": "error", "detail": "; ".join(page.errors) or "no page-lab report"}]
    return static_defects(v, include_small_text=False) + report_defects(report, DIAGRAM_SEVERITIES)


__all__ = [
    "DIAGRAM_SEVERITIES",
    "RENDER_WIDTH_PX_DEFAULT",
    "SVG_MIN_FONT_PX",
    "SVG_SOURCE_LIMIT",
    "SvgValidation",
    "measure_svg",
    "render_size",
    "render_svg_png",
    "static_defects",
    "svg_page",
    "validate_svg",
]
