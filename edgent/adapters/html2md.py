"""HTML to Markdown with the standard library, for web pages and HTML exports.

Good enough for reading: headings, paragraphs, lists, links, code, tables
(as pipe rows), images as alt text. Scripts, styles and navigation chrome
are dropped. Output is untrusted data wherever it is used.
"""
from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser

_BLOCK = {"p", "div", "section", "article", "main", "header", "footer", "br", "hr", "blockquote", "pre", "table", "tr", "ul", "ol", "li", "h1", "h2", "h3", "h4", "h5", "h6", "figure", "figcaption", "dl", "dt", "dd"}
_SKIP = {"script", "style", "noscript", "svg", "canvas", "iframe", "template", "head", "nav", "aside", "form", "button", "input", "select"}


class _MD(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.skip = 0
        self.pre = 0
        self.list_stack: list[tuple[str, int]] = []
        self.href: str | None = None
        self.cell = False
        self.title: str = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "title":  # the title lives inside <head>, which is otherwise skipped
            self._in_title = True
            return
        if tag in _SKIP:
            self.skip += 1
            return
        if self.skip:
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.out.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "p" or tag == "div" or tag == "section" or tag == "article":
            self.out.append("\n\n")
        elif tag == "br":
            self.out.append("\n")
        elif tag == "hr":
            self.out.append("\n\n---\n\n")
        elif tag == "pre":
            self.pre += 1
            self.out.append("\n\n```\n")
        elif tag == "code" and not self.pre:
            self.out.append("`")
        elif tag in ("strong", "b"):
            self.out.append("**")
        elif tag in ("em", "i"):
            self.out.append("*")
        elif tag == "blockquote":
            self.out.append("\n\n> ")
        elif tag in ("ul", "ol"):
            self.list_stack.append((tag, 0))
            self.out.append("\n")
        elif tag == "li":
            if self.list_stack:
                kind, n = self.list_stack[-1]
                self.list_stack[-1] = (kind, n + 1)
                bullet = f"{n + 1}." if kind == "ol" else "-"
            else:
                bullet = "-"
            self.out.append("\n" + "  " * (len(self.list_stack) - 1) + bullet + " ")
        elif tag == "a":
            self.href = a.get("href")
            self.out.append("[")
        elif tag == "img":
            alt = a.get("alt") or ""
            src = a.get("src") or ""
            self.out.append(f"![{alt}]({src})" if src else f"[image: {alt}]")
        elif tag == "tr":
            if self.out and not self.out[-1].endswith("\n"):
                self.out.append("\n")
            self.out.append("| ")
        elif tag in ("td", "th"):
            self.cell = True
        elif tag == "table":
            self.out.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
            return
        if tag in _SKIP:
            self.skip = max(0, self.skip - 1)
            return
        if self.skip:
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6", "p", "div", "section", "article", "blockquote", "figure", "figcaption"):
            self.out.append("\n")
        elif tag == "pre":
            self.pre = max(0, self.pre - 1)
            self.out.append("\n```\n\n")
        elif tag == "code" and not self.pre:
            self.out.append("`")
        elif tag in ("strong", "b"):
            self.out.append("**")
        elif tag in ("em", "i"):
            self.out.append("*")
        elif tag in ("ul", "ol"):
            if self.list_stack:
                self.list_stack.pop()
            self.out.append("\n")
        elif tag == "a":
            self.out.append(f"]({self.href})" if self.href else "]")
            self.href = None
        elif tag in ("td", "th"):
            self.out.append(" | ")
            self.cell = False
        elif tag == "tr":
            self.out.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip:
            return
        if self._in_title:
            self.title += data
            return
        if self.pre:
            self.out.append(data)
        else:
            text = re.sub(r"\s+", " ", data)
            if text.strip() or (self.out and not self.out[-1].endswith(("\n", " "))):
                self.out.append(text)


def html_to_markdown(html: str) -> tuple[str, str]:
    """Return (markdown, title)."""
    p = _MD()
    try:
        p.feed(html)
        p.close()
    except Exception:  # noqa: BLE001 - malformed HTML still yields what was parsed
        pass
    text = unescape("".join(p.out))
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n", p.title.strip()
