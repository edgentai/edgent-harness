"""Text-like sources: plain text, Markdown, source code, subtitle files and saved HTML pages.

The content is copied under the header unchanged, so a grep hit's line number in the view is the citation
(``<id>:L<n>``). Code is wrapped in a fenced block with its language. SRT and VTT subtitle files become a
timeline with ``[t=HH:MM:SS]`` markers, so an existing caption track makes a lecture citable by time
without a transcriber. A local HTML file is converted to Markdown like a fetched page (kind ``web``), because
a page full of tags is not something anyone can grep or cite.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ...domain.errors import Limit, ToolFailure
from ...domain.workspace import Workspace
from ..html2md import html_to_markdown
from ..web import decode_body
from .base import IngestResult, Sniff, coverage_complete, hms, parse_hms, title_from_stem

TEXT_SOURCE_BYTES_LIMIT = Limit("text source size", 20_000_000, "bytes", does_not_raise=("the sources_add timeout",))
_HEADING_RE = re.compile(r"^#\s+(.+?)\s*#*\s*$")
_CUE_TIME_RE = re.compile(r"(\d{1,2}:\d{2}(?::\d{2})?[.,]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}(?::\d{2})?[.,]\d{1,3})")
_TAG_RE = re.compile(r"<[^>]{1,80}>")
_FENCE_RUN_RE = re.compile(r"`{3,}")


def decode_text(raw: bytes) -> tuple[str, str, str | None]:
    """(text, encoding, limitation). UTF-8 (with or without BOM), UTF-16 with a BOM, else Latin-1 with a note."""
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return raw.decode("utf-16"), "utf-16", None
        except UnicodeDecodeError:
            pass
    try:
        return raw.decode("utf-8-sig"), "utf-8", None
    except UnicodeDecodeError as e:
        text = raw.decode("latin-1")
        return text, "latin-1", f"file is not valid UTF-8 ({e.reason} at byte {e.start}); decoded as Latin-1, so some characters may be wrong"


def first_heading(text: str) -> str | None:
    for line in text.splitlines()[:200]:
        m = _HEADING_RE.match(line.strip())
        if m:
            return m.group(1).strip()
    return None


def fence_for(text: str) -> str:
    longest = max((len(m.group(0)) for m in _FENCE_RUN_RE.finditer(text)), default=0)
    return "`" * max(3, longest + 1)


def subtitles_to_timeline(text: str) -> tuple[str, dict[str, Any]]:
    """SRT or VTT cues to ``[t=HH:MM:SS]`` blocks; cues starting in the same second share a marker."""
    blocks: dict[int, list[str]] = {}
    last_end = 0
    cues = 0
    for chunk in re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n")):
        lines = [ln for ln in chunk.split("\n") if ln.strip()]
        for i, line in enumerate(lines):
            m = _CUE_TIME_RE.search(line)
            if not m:
                continue
            start, end = parse_hms(m.group(1)), parse_hms(m.group(2))
            if start is None:
                break
            body = " ".join(_TAG_RE.sub("", ln).strip() for ln in lines[i + 1 :] if ln.strip())
            if body:
                blocks.setdefault(start, []).append(body)
                cues += 1
            last_end = max(last_end, end or start)
            break
    out: list[str] = []
    for sec in sorted(blocks):
        out.append(f"[t={hms(sec)}]")
        out.extend(blocks[sec])
        out.append("")
    return "\n".join(out), {"cues": cues, "duration_s": last_end}


HTML_TO_MARKDOWN_NOTE = "HTML converted to Markdown: scripts, navigation, forms and media were dropped"


def _check_size(src: Path) -> int:
    size = src.stat().st_size
    if size > TEXT_SOURCE_BYTES_LIMIT.value:
        err = TEXT_SOURCE_BYTES_LIMIT.exceeded(size, "Split the file (for example with bash: split -l) and add the parts as separate sources.")
        raise ToolFailure(err.code, err.message, err.category, remediation=err.remediation, details=err.details)
    return size


def ingest_html(src: Path, source_id: str, workspace: Workspace, toolchain: Any, info: Sniff, title: str | None = None, **_: Any) -> IngestResult:
    """A saved web page or HTML export: Markdown under the header, the page title as the source title."""
    _check_size(src)
    text, encoding = decode_body(src.read_bytes())  # honours a BOM or a <meta charset>, like a fetched page
    markdown, page_title = html_to_markdown(text)
    extra: dict[str, Any] = {"chars": len(markdown), "encoding": encoding, "content_type": "text/html"}
    derived = page_title.strip() or first_heading(markdown) or title_from_stem(src.stem)
    return IngestResult("web", markdown, title or derived, coverage_complete(), [HTML_TO_MARKDOWN_NOTE], extra)


def ingest_text(src: Path, source_id: str, workspace: Workspace, toolchain: Any, info: Sniff, title: str | None = None, **_: Any) -> IngestResult:
    _check_size(src)
    text, encoding, note = decode_text(src.read_bytes())
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    limitations = [note] if note else []
    extra: dict[str, Any] = {"lines": text.count("\n") + (1 if text and not text.endswith("\n") else 0), "chars": len(text), "encoding": encoding}
    derived_title: str | None = None
    if info.format == "subtitles":
        body, cue_info = subtitles_to_timeline(text)
        extra.update(cue_info)
        if not cue_info["cues"]:
            limitations.append("no subtitle cues were recognised; the file was kept as plain text")
            body = text
        return IngestResult("text", body, title or title_from_stem(src.stem), coverage_complete(), limitations, extra)
    if info.kind == "code":
        fence = fence_for(text)
        body = f"{fence}{info.language or ''}\n{text.rstrip()}\n{fence}"
        extra["language"] = info.language
        return IngestResult("code", body, title or title_from_stem(src.stem), coverage_complete(), limitations, extra)
    if info.kind == "markdown":
        derived_title = first_heading(text)
    return IngestResult(info.kind, text, title or derived_title or title_from_stem(src.stem), coverage_complete(), limitations, extra)
