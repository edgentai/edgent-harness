"""Office documents: docx, epub, odt and rtf through pandoc; pptx through the standard library.

Pandoc cannot read pptx, and a slide deck has real page boundaries, so slides are parsed from the package
XML directly: one ``[p.N]`` per slide in presentation order, speaker notes included. Flowing documents have
no fixed pages; they get one ``[p.N]`` per top-level heading section, and the record says that pages are
approximate. Without pandoc, docx falls back to a plain-paragraph reading of ``word/document.xml``; the
other formats report ``unavailable`` with the install hint.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

from ...domain.errors import ErrorCategory, ToolFailure
from ...domain.workspace import Workspace
from .base import IngestResult, Sniff, coverage_complete, missing_tool, page_marker, run_command, title_from_stem
from .text import first_heading

PANDOC_TIMEOUT_S = 300.0
_PANDOC_FORMATS = {"docx": "docx", "epub": "epub", "odt": "odt", "rtf": "rtf"}
_HEADING_RE = re.compile(r"^(#{1,6})\s+\S")
_NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
}
APPROXIMATE_PAGES = "page numbers are approximate: one [p.N] per {unit}; cite L<n> lines of the text view for precision"


def _bad_package(src: Path, what: str) -> ToolFailure:
    return ToolFailure("unsupported", f"{src.name} is not a readable {what} package.", ErrorCategory.unsupported, remediation="Re-save the file from its editor and add it again, or export it to PDF.", path=str(src))


# ----------------------------------------------------------------------------- pptx (stdlib)
def _slide_order(z: zipfile.ZipFile) -> list[str]:
    """Slide part names in presentation order, from presentation.xml and its relationships."""
    names = set(z.namelist())
    try:
        rels = ET.fromstring(z.read("ppt/_rels/presentation.xml.rels"))
        pres = ET.fromstring(z.read("ppt/presentation.xml"))
        targets = {r.get("Id"): r.get("Target", "") for r in rels.findall("rel:Relationship", _NS)}
        ordered = []
        for sld in pres.findall("p:sldIdLst/p:sldId", _NS):
            target = targets.get(sld.get(f"{{{_NS['r']}}}id"), "")
            part = "ppt/" + target.lstrip("/").removeprefix("ppt/")
            if part in names:
                ordered.append(part)
        if ordered:
            return ordered
    except (KeyError, ET.ParseError):
        pass
    slides = [n for n in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]
    return sorted(slides, key=lambda n: int(re.search(r"(\d+)\.xml$", n).group(1)))  # type: ignore[union-attr]


def _paragraphs(root: ET.Element) -> list[str]:
    out: list[str] = []
    for para in root.iter(f"{{{_NS['a']}}}p"):
        text = "".join(t.text or "" for t in para.iter(f"{{{_NS['a']}}}t"))
        text = " ".join(text.split())
        if text:
            out.append(text)
    return out


def _notes_for(z: zipfile.ZipFile, slide_part: str) -> list[str]:
    rels_name = slide_part.replace("ppt/slides/", "ppt/slides/_rels/") + ".rels"
    try:
        rels = ET.fromstring(z.read(rels_name))
    except (KeyError, ET.ParseError):
        return []
    for r in rels.findall("rel:Relationship", _NS):
        if r.get("Type", "").endswith("/notesSlide"):
            part = "ppt/slides/" + r.get("Target", "")
            part = re.sub(r"ppt/slides/\.\./", "ppt/", part)
            try:
                return _paragraphs(ET.fromstring(z.read(part)))
            except (KeyError, ET.ParseError):
                return []
    return []


def pptx_slides(src: Path) -> list[dict[str, Any]]:
    """[{'title': first paragraph, 'text': paragraphs, 'notes': paragraphs}] in presentation order."""
    try:
        with zipfile.ZipFile(src) as z:
            slides = []
            for part in _slide_order(z):
                try:
                    paras = _paragraphs(ET.fromstring(z.read(part)))
                except ET.ParseError:
                    paras = []
                notes = [n for n in _notes_for(z, part) if not re.fullmatch(r"\d+", n)]  # drop the slide-number placeholder
                slides.append({"title": paras[0] if paras else "", "text": paras, "notes": notes})
            return slides
    except zipfile.BadZipFile as e:
        raise _bad_package(src, "pptx") from e


def _ingest_pptx(src: Path, title: str | None) -> IngestResult:
    slides = pptx_slides(src)
    parts = []
    for n, slide in enumerate(slides, 1):
        lines = slide["text"] or ["(slide without text)"]
        if slide["notes"]:
            lines += ["", "speaker notes:"] + slide["notes"]
        parts.append(f"{page_marker(n)}\n" + "\n".join(lines) + "\n")
    derived = next((s["title"] for s in slides if s["title"]), None)
    extra = {"slides": len(slides), "engine": "stdlib-pptx", "notes_slides": sum(1 for s in slides if s["notes"])}
    return IngestResult("office", "\n".join(parts), title or derived or title_from_stem(src.stem), coverage_complete(pages_total=len(slides)), ["one [p.N] per slide, in presentation order"], extra)


# ----------------------------------------------------------------------------- docx and friends
def pandoc_to_markdown(pandoc: str, src: Path, fmt: str) -> str:
    r = run_command([pandoc, "-f", fmt, "-t", "gfm-raw_html", "--wrap=none", str(src)], PANDOC_TIMEOUT_S)
    if not r.ok:
        reason = "timed out" if r.timed_out else (r.stderr.strip().splitlines() or ["unknown error"])[-1][:300]
        raise ToolFailure("unsupported", f"pandoc could not convert {src.name} ({fmt}): {reason}", ErrorCategory.unsupported, remediation="Re-save the document from its editor, or export it to PDF and add that instead.", path=str(src))
    return r.stdout.decode("utf-8", "replace")


def docx_plain_text(src: Path) -> str:
    """Paragraph text from word/document.xml with headings as '#' lines and tables as pipe rows. No pandoc."""
    try:
        with zipfile.ZipFile(src) as z:
            root = ET.fromstring(z.read("word/document.xml"))
    except (zipfile.BadZipFile, KeyError, ET.ParseError) as e:
        raise _bad_package(src, "docx") from e
    w = _NS["w"]
    lines: list[str] = []

    def para_text(p: ET.Element) -> str:
        bits = []
        for el in p.iter():
            if el.tag == f"{{{w}}}t":
                bits.append(el.text or "")
            elif el.tag == f"{{{w}}}tab":
                bits.append("\t")
            elif el.tag in (f"{{{w}}}br", f"{{{w}}}cr"):
                bits.append("\n")
        return "".join(bits).strip()

    body = root.find(f"{{{w}}}body")
    for el in body if body is not None else []:
        if el.tag == f"{{{w}}}p":
            text = para_text(el)
            style = el.find(f"{{{w}}}pPr/{{{w}}}pStyle")
            val = (style.get(f"{{{w}}}val") or "") if style is not None else ""
            m = re.fullmatch(r"(?i)heading\s*(\d)", val)
            if text and m:
                lines += ["", "#" * int(m.group(1)) + " " + text, ""]
            elif text and val.lower() == "title":
                lines += ["", "# " + text, ""]
            elif text:
                lines += [text, ""]
        elif el.tag == f"{{{w}}}tbl":
            for row in el.iter(f"{{{w}}}tr"):
                cells = [" ".join(para_text(p) for p in cell.iter(f"{{{w}}}p")).strip() for cell in row.iter(f"{{{w}}}tc")]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip("\n") + "\n"


def split_sections(markdown: str) -> tuple[list[str], int]:
    """Split on the top-most heading level present; text before the first heading is its own section."""
    lines = markdown.replace("\r\n", "\n").split("\n")
    levels = [len(m.group(1)) for ln in lines if (m := _HEADING_RE.match(ln))]
    if not levels:
        return [markdown.strip("\n")], 0
    top = min(levels)
    sections: list[list[str]] = [[]]
    in_fence = False
    for ln in lines:
        if ln.startswith("```"):
            in_fence = not in_fence
        m = _HEADING_RE.match(ln)
        if m and not in_fence and len(m.group(1)) == top and any(s.strip() for s in sections[-1]):
            sections.append([])
        sections[-1].append(ln)
    return [s for s in ("\n".join(sec).strip("\n") for sec in sections) if s], top


def ingest_office(src: Path, source_id: str, workspace: Workspace, toolchain: Any, info: Sniff, title: str | None = None, **_: Any) -> IngestResult:
    fmt = info.format
    if fmt == "pptx":
        return _ingest_pptx(src, title)
    if fmt not in _PANDOC_FORMATS:
        raise ToolFailure("unsupported", f"{src.name}: office format {fmt!r} is not supported.", ErrorCategory.unsupported, remediation="Export it to docx, epub or PDF and add that instead.", path=str(src))
    pandoc = toolchain.path("pandoc") if toolchain is not None else None
    limitations: list[str] = []
    missing: list[dict[str, str]] = []
    if pandoc:
        markdown = pandoc_to_markdown(pandoc, src, _PANDOC_FORMATS[fmt])
        engine = "pandoc"
    elif fmt == "docx":
        markdown = docx_plain_text(src)
        engine = "stdlib-docx"
        missing.append(missing_tool("pandoc", "docx conversion with lists, tables and images"))
        limitations.append("pandoc is not installed: plain paragraphs only (lists, images and formatting were simplified)")
    else:
        toolchain.require("pandoc", f"{fmt} conversion")  # raises the typed unavailable error with the install hint
        raise AssertionError("unreachable")
    sections, level = split_sections(markdown)
    unit = f"heading-{level} section" if level else "document (no headings found)"
    parts = [f"{page_marker(n)}\n{text}\n" for n, text in enumerate(sections, 1)]
    limitations.append(APPROXIMATE_PAGES.format(unit=unit))
    extra = {"sections": len(sections), "heading_level": level, "engine": engine, "format": fmt}
    derived = first_heading(markdown)
    return IngestResult("office", "\n".join(parts), title or derived or title_from_stem(src.stem), coverage_complete(pages_total=len(sections)), limitations, extra, missing)
