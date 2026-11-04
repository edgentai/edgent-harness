"""A small Markdown to HTML renderer for the reader and for builds without pandoc.

Covers what chapters use: front matter (stripped, returned as metadata),
ATX headings with ids, paragraphs, fenced code, blockquotes, nested lists,
pipe tables, emphasis, inline code, links, images, horizontal rules, and
the Edgent placeholders ``{{film: id}}``, ``{{widget: id}}`` and
``{{exercises: id}}``. Math (``$…$``, ``$$…$$``) is passed through
untouched for a client-side renderer. Citations ``[source:p.12]`` become
clickable spans carrying the locator.
"""
from __future__ import annotations

import html
import re
from typing import Any

from ..domain.utils import parse_front_matter, slugify

_CITE_RE = re.compile(r"\[([a-z0-9][a-z0-9\-_]*):(p\.\d+|t=\d{1,2}:\d{2}(?::\d{2})?|L\d+)\]")
_PLACEHOLDER_RE = re.compile(r"\{\{\s*(film|widget|exercises)\s*:\s*([a-zA-Z0-9\-_]+)\s*\}\}")
_MATH_SPLIT_RE = re.compile(r"(\$\$.+?\$\$|\$[^$\n]+?\$)", re.DOTALL)
_SCHEME_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.\-]*):")
LINK_SCHEMES = frozenset({"http", "https", "mailto"})  # what an <a href> may point at; anything else becomes "#"
IMAGE_SCHEMES = frozenset({"http", "https", "data"})  # what an <img src> may load; the reader's CSP allows data: images


def _attr(value: str) -> str:
    """Make text that is already escaped for element content safe inside a double-quoted attribute."""
    return value.replace('"', "&quot;").replace("'", "&#x27;")


def _url(value: str, schemes: frozenset[str]) -> str:
    """An attribute-safe URL: relative and fragment URLs pass; an absolute URL must use one of ``schemes``."""
    m = _SCHEME_RE.match(value)
    if m and m.group(1).lower() not in schemes:
        return "#"
    return _attr(value)


def _inline(text: str) -> str:
    """Inline markup on an already HTML-escaped-safe path: we escape, then re-insert tags."""
    out: list[str] = []
    for chunk in _MATH_SPLIT_RE.split(text):
        if not chunk:
            continue
        if chunk.startswith("$"):
            out.append(html.escape(chunk, quote=False))
            continue
        s = html.escape(chunk, quote=False)
        s = re.sub(r"`([^`]+)`", lambda m: f"<code>{m.group(1)}</code>", s)
        s = re.sub(r"!\[([^\]]*)\]\(([^)\s]+)\)", lambda m: f'<img alt="{_attr(m.group(1))}" src="{_url(m.group(2), IMAGE_SCHEMES)}">', s)
        s = _CITE_RE.sub(lambda m: f'<span class="cite" data-locator="{m.group(1)}:{m.group(2)}">[{m.group(1)}:{m.group(2)}]</span>', s)
        s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", lambda m: f'<a href="{_url(m.group(2), LINK_SCHEMES)}">{m.group(1)}</a>', s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"(?<![*\w])\*([^*\n]+)\*(?![*\w])", r"<em>\1</em>", s)
        s = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"<em>\1</em>", s)
        out.append(s)
    return "".join(out)


def markdown_to_html(text: str, with_ids: bool = True) -> tuple[str, dict[str, Any]]:
    """Return (html_body, metadata). Section ids come from headings (``slugify``)."""
    meta, body = parse_front_matter(text)
    lines = body.splitlines()
    out: list[str] = []
    i = 0
    para: list[str] = []
    list_stack: list[tuple[str, int]] = []  # (tag, indent)
    section_open = False
    used_ids: set[str] = set()

    def flush_para() -> None:
        nonlocal para
        if para:
            out.append(f"<p>{_inline(' '.join(s.strip() for s in para))}</p>")
            para = []

    def close_lists(to_indent: int = -1) -> None:
        while list_stack and list_stack[-1][1] > to_indent:
            tag, _ = list_stack.pop()
            out.append(f"</{tag}>")

    def close_section() -> None:
        nonlocal section_open
        if section_open:
            out.append("</section>")
            section_open = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        # fenced code
        if stripped.startswith("```"):
            flush_para()
            close_lists()
            lang = stripped[3:].strip()
            code: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            cls = f' class="language-{html.escape(lang)}"' if lang else ""
            out.append(f"<pre><code{cls}>{html.escape(chr(10).join(code))}</code></pre>")
            continue
        # display math block: pass through
        if stripped.startswith("$$") and (stripped == "$$" or not stripped.endswith("$$") or len(stripped) == 2):
            flush_para()
            close_lists()
            block = [line]
            i += 1
            while i < len(lines) and "$$" not in lines[i]:
                block.append(lines[i])
                i += 1
            if i < len(lines):
                block.append(lines[i])
                i += 1
            out.append('<div class="math-block">' + html.escape(chr(10).join(block), quote=False) + "</div>")
            continue
        # headings
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            flush_para()
            close_lists()
            level = len(m.group(1))
            title = m.group(2).strip()
            hid = slugify(title)
            while hid in used_ids:
                hid += "-x"
            used_ids.add(hid)
            if level <= 2:
                close_section()
                out.append(f'<section id="{hid}" data-section="{hid}">')
                section_open = True
            idattr = f' id="h-{hid}"' if with_ids else ""
            out.append(f"<h{level}{idattr}>{_inline(title)}</h{level}>")
            i += 1
            continue
        # horizontal rule
        if re.match(r"^(-{3,}|\*{3,}|_{3,})$", stripped):
            flush_para()
            close_lists()
            out.append("<hr>")
            i += 1
            continue
        # placeholders
        pm = _PLACEHOLDER_RE.fullmatch(stripped)
        if pm:
            flush_para()
            close_lists()
            kind, ident = pm.group(1), pm.group(2)
            out.append(f'<div class="placeholder placeholder-{kind}" data-kind="{kind}" data-id="{ident}">{kind}: {ident}</div>')
            i += 1
            continue
        # blockquote
        if stripped.startswith(">"):
            flush_para()
            close_lists()
            quote: list[str] = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip()[1:].strip())
                i += 1
            out.append(f"<blockquote><p>{_inline(' '.join(quote))}</p></blockquote>")
            continue
        # tables
        if "|" in stripped and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{2,}", lines[i + 1]):
            flush_para()
            close_lists()
            header = [c.strip() for c in stripped.strip("|").split("|")]
            i += 2
            rows: list[list[str]] = []
            while i < len(lines) and "|" in lines[i]:
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            out.append("<table><thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in header) + "</tr></thead><tbody>")
            for r in rows:
                out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>")
            out.append("</tbody></table>")
            continue
        # lists
        lm = re.match(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$", line)
        if lm:
            flush_para()
            indent = len(lm.group(1).expandtabs(4))
            tag = "ol" if lm.group(2)[0].isdigit() else "ul"
            if not list_stack or indent > list_stack[-1][1]:
                list_stack.append((tag, indent))
                out.append(f"<{tag}>")
            else:
                close_lists(indent)
                if not list_stack or list_stack[-1][1] != indent:
                    list_stack.append((tag, indent))
                    out.append(f"<{tag}>")
            out.append(f"<li>{_inline(lm.group(3))}</li>")
            i += 1
            continue
        # blank line
        if not stripped:
            flush_para()
            if list_stack and (i + 1 >= len(lines) or not re.match(r"^\s*([-*+]|\d+[.)])\s+", lines[i + 1])):
                close_lists()
            i += 1
            continue
        para.append(line)
        i += 1
    flush_para()
    close_lists()
    close_section()
    return "\n".join(out) + "\n", meta


def page(title: str, body_html: str, extra_head: str = "", math: bool = True) -> str:
    """A standalone HTML document around a rendered body; KaTeX auto-render when the client has it."""
    return (
        "<!doctype html>\n<html><head><meta charset=\"utf-8\">\n"
        f"<title>{html.escape(title)}</title>\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<style>body{max-width:46em;margin:2em auto;padding:0 1em;font:17px/1.6 Georgia,serif;color:#222;background:#fafaf7}"
        "pre{background:#f0f0ea;padding:.8em;overflow:auto}code{font-family:ui-monospace,monospace;font-size:.92em}"
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.3em .6em}.cite{color:#2a6;cursor:pointer}"
        ".placeholder{border:1px dashed #999;padding:.6em;margin:1em 0;color:#666}blockquote{border-left:3px solid #ccc;margin:0;padding-left:1em;color:#555}</style>\n"
        f"{extra_head}\n</head><body>\n{body_html}\n</body></html>\n"
    )
