"""Text and document views: clean text, structure, page map, chunks with offsets."""
from __future__ import annotations

import re
from typing import Any

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")


def clean_text(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip("\n")


def structure(text: str) -> dict[str, Any]:
    headings = []
    tables = []
    offset = 0
    in_table: list[int] = []
    for line in text.split("\n"):
        m = HEADING_RE.match(line)
        if m:
            headings.append({"level": len(m.group(1)), "title": m.group(2).strip(), "offset": offset})
        if TABLE_ROW_RE.match(line):
            in_table.append(offset)
        elif in_table:
            tables.append({"offset": in_table[0], "rows": len(in_table)})
            in_table = []
        offset += len(line) + 1
    if in_table:
        tables.append({"offset": in_table[0], "rows": len(in_table)})
    return {"headings": headings, "tables": tables, "chars": len(text), "lines": text.count("\n") + 1}


def page_map(text: str) -> list[dict[str, Any]]:
    """Pages split on form feeds; a single page when there are none."""
    pages = []
    offset = 0
    for i, page in enumerate(text.split("\f"), 1):
        pages.append({"page": i, "offset": offset, "length": len(page)})
        offset += len(page) + 1
    return pages


def chunk(text: str, size: int = 1200, overlap: int = 150) -> list[dict[str, Any]]:
    """Chunks with character offsets, split on paragraph boundaries where possible."""
    chunks = []
    n = len(text)
    start = 0
    idx = 0
    while start < n:
        end = min(n, start + size)
        if end < n:
            cut = text.rfind("\n\n", start + size // 2, end)
            if cut == -1:
                cut = text.rfind(". ", start + size // 2, end)
                cut = cut + 1 if cut != -1 else -1
            if cut != -1:
                end = cut
        chunks.append({"index": idx, "offset": start, "length": end - start, "text": text[start:end]})
        idx += 1
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def locate(text: str, quote: str) -> int:
    """Find a quote in text, tolerant of whitespace differences. Returns offset or -1."""
    if not quote:
        return -1
    i = text.find(quote)
    if i != -1:
        return i
    norm = re.sub(r"\s+", " ", quote.strip()).lower()
    hay = re.sub(r"\s+", " ", text).lower()
    j = hay.find(norm)
    if j == -1:
        return -1
    # Map back approximately: count non-collapsed chars. Good enough for citation checks.
    count = 0
    for k, ch in enumerate(text):
        if count == j:
            return k
        if not (ch.isspace() and k > 0 and text[k - 1].isspace()):
            count += 1
    return -1
