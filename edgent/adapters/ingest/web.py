"""URL sources: fetch, keep the raw bytes, derive a text view, and route binaries to the file ingesters.

``ingest_url(url, workspace, fetcher=None)`` takes an injectable ``fetcher(url) -> (content, content_type,
final_url)`` so tests never touch the network. The default fetcher is ``edgent.adapters.web.fetch`` (urllib,
public hosts only, a 30 s timeout and the 5 MB ``FETCH_BYTES_LIMIT``). HTML becomes Markdown through
``html2md``; a PDF, image, media or office payload is saved under ``sources/raw`` and handed to the matching
ingester, so a link to a paper yields ``[p.N]`` markers like a local file would.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import urllib.parse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ...domain.errors import ErrorCategory, Limit, ToolFailure
from ...domain.utils import now_iso, sha256_bytes, slugify
from ...domain.workspace import Workspace
from .. import web as web_adapter
from ..html2md import html_to_markdown
from .base import (
    WORKSPACE_WRITE_LOCK,
    Sniff,
    SourceRecord,
    allocate_id,
    clean_title,
    coverage_complete,
    coverage_partial,
    find_by_origin,
    find_by_sha256,
    id_taken,
    in_flight_ids,
    place_raw,
    raw_extension,
    regenerate_index,
    release_id,
    remove_source_files,
    reserve_id,
    sniff_bytes,
    validate_id,
    write_meta,
    write_text_view,
)
from .text import HTML_TO_MARKDOWN_NOTE, first_heading

URL_FETCH_TIMEOUT_S = web_adapter.DEFAULT_TIMEOUT_S
URL_CONTENT_BYTES_LIMIT = Limit("url source size", web_adapter.FETCH_BYTES_LIMIT.value, "bytes", does_not_raise=("the sources_add_url timeout",))
Fetcher = Callable[[str], tuple[str | bytes, str, str]]
_HTML_TYPES = {"text/html", "application/xhtml+xml"}
_JSON_TYPES = {"application/json", "text/json", "application/ld+json"}
_MARKDOWN_TYPES = {"text/markdown", "text/x-markdown"}
_HTML_SNIFF_RE = re.compile(r"<(!doctype\s+html|html|head|body)\b", re.I)


def make_default_fetcher(opener: Any = None, resolver: Any = None, timeout_s: float = URL_FETCH_TIMEOUT_S, user_agent: str = web_adapter.DEFAULT_USER_AGENT) -> Fetcher:
    """urllib through the web adapter's protections; binary bodies come back as bytes, text as str."""

    def fetch(url: str) -> tuple[str | bytes, str, str]:
        r = web_adapter.fetch(url, timeout_s=timeout_s, opener=opener, resolver=resolver, user_agent=user_agent)
        kind = web_adapter.content_kind(r)
        return (r.data if kind in ("pdf", "binary") else r.text), r.content_type, r.final_url

    return fetch


def title_from_url(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    segments = [urllib.parse.unquote(s) for s in parts.path.split("/") if s]
    tail = segments[-1] if segments else ""
    tail = re.sub(r"\.(html?|php|aspx?|jsp|md|txt|pdf|docx|pptx|epub|png|jpe?g|gif|webp|mp4|m4a|mp3|wav|mkv|webm|json|xml|csv)$", "", tail, flags=re.I)
    base = tail or parts.netloc or url
    return clean_title(re.sub(r"[-_]+", " ", base), parts.netloc or "web page")


def _media_type(content_type: str) -> str:
    return (content_type or "").split(";")[0].strip().lower()


def _as_bytes(content: str | bytes) -> bytes:
    return content if isinstance(content, bytes) else content.encode("utf-8", "replace")


def _textual(content: str | bytes, media_type: str) -> tuple[str, str, str | None] | None:
    """(body markdown, raw extension, page title) for HTML, JSON, Markdown and text payloads; None for binaries."""
    text = content if isinstance(content, str) else None
    if text is None:
        if b"\x00" in content[:4096]:
            return None
        text = content.decode("utf-8", "replace")
    head = text[:2048]
    if media_type in _HTML_TYPES or (media_type in ("", "text/plain", "application/octet-stream") and _HTML_SNIFF_RE.search(head)):
        markdown, page_title = html_to_markdown(text)
        return markdown, ".html", page_title or None
    if media_type in _JSON_TYPES or media_type.endswith("+json"):
        try:
            return json.dumps(json.loads(text), indent=2, ensure_ascii=False) + "\n", ".json", None
        except ValueError:
            return text, ".json", None
    if media_type in _MARKDOWN_TYPES:
        return text, ".md", first_heading(text)
    if media_type.startswith("text/") or media_type.endswith("+xml") or media_type in ("", "application/octet-stream", "application/xml", "application/javascript", "application/x-yaml", "application/yaml"):
        return text, ".txt", None
    return None


def ingest_url(url: str, workspace: Workspace, fetcher: Fetcher | None = None, title: str | None = None, source_id: str | None = None, toolchain: Any = None, replace: bool = False) -> SourceRecord:
    """Fetch ``url`` and record it as a source. Text payloads get kind 'web'; binaries go to the file ingesters.
    ``replace`` re-ingests in place: the record with the same content or the same URL keeps its id."""
    url = str(url or "").strip()
    if not re.match(r"^https?://", url, re.I):
        raise ToolFailure("invalid_input", f"{url!r} is not an http(s) URL.", ErrorCategory.invalid_input, remediation="Give a full URL starting with http:// or https://; for a local file use sources_add.", url=url)
    if source_id is not None:
        source_id = validate_id(source_id)
    fetch = fetcher or make_default_fetcher()
    content, content_type, final_url = fetch(url)
    final_url = str(final_url or url)
    media_type = _media_type(content_type)
    data = _as_bytes(content)
    limitations: list[str] = []
    if len(data) > URL_CONTENT_BYTES_LIMIT.value:
        data = data[: int(URL_CONTENT_BYTES_LIMIT.value)]
        content = data if isinstance(content, bytes) else data.decode("utf-8", "replace")
        limitations.append(URL_CONTENT_BYTES_LIMIT.exceeded(len(data), "The rest of the page was not saved.").message)
    sha = sha256_bytes(data)
    binary = sniff_bytes(data[:8192], os.path.splitext(urllib.parse.urlsplit(final_url).path)[1].lower()) if media_type != "text/html" else None
    if media_type == "application/pdf" and (binary is None or binary.kind != "pdf"):
        binary = Sniff("pdf", "pdf", "application/pdf")
    if binary is not None and binary.kind != "binary":
        return _ingest_binary(data, binary, workspace, toolchain, title, source_id, url, final_url, replace)
    textual = _textual(content, media_type)
    if textual is None:
        raise ToolFailure(
            "unsupported",
            f"{final_url} returned {media_type or 'an unknown content type'}, which has no text view.",
            ErrorCategory.unsupported,
            remediation="Download the file with web_fetch (save=true) or bash and run sources_add on it if it is a document, image, audio or video; otherwise cite the URL without a locator.",
            url=final_url,
            content_type=media_type,
        )
    body, raw_ext, page_title = textual
    if raw_ext != ".json" and not page_title:
        page_title = first_heading(body)
    final_title = clean_title(title or page_title or title_from_url(final_url))
    fetched = now_iso()
    full_body = f"(origin: {final_url}, fetched {fetched[:19]}Z)\n\n{body.strip()}\n"
    if raw_ext == ".html":
        limitations.append(HTML_TO_MARKDOWN_NOTE)
    cov = coverage_partial() if any("exceeded" in lim for lim in limitations) else coverage_complete()
    with WORKSPACE_WRITE_LOCK:  # the commit phase: dedupe, id, files and index as one step against parallel adds
        existing = None if replace else find_by_sha256(workspace, sha)
        if existing is not None and (source_id is None or existing.id == source_id):
            existing.extra["already_present"] = True
            return existing
        sid = source_id or _replaced_id(workspace, sha, url, final_url, replace) or allocate_id(workspace, final_title, reserved=in_flight_ids(workspace))
        if id_taken(workspace, sid) and not replace:
            raise ToolFailure("invalid_input", f"source id {sid!r} is already taken.", ErrorCategory.invalid_input, remediation="Omit source_id, choose another one, or pass replace=true to overwrite that source.", source_id=sid)
        reserve_id(workspace, sid, sha)  # refuses an id a parallel call is still writing
        try:
            if id_taken(workspace, sid):
                remove_source_files(workspace, sid)
            raw_rel = place_raw(workspace, sid, raw_ext, data=data)
            record = SourceRecord(
                id=sid,
                kind="web",
                title=final_title,
                raw_path=raw_rel,
                text_path="",
                sha256=sha,
                size_bytes=len(data),
                coverage=cov,
                limitations=limitations,
                created_at=fetched,
                origin=url,
                extra={"url": final_url, "content_type": media_type, "fetched_at": fetched, "chars": len(body), "usable": True, "missing_toolchains": []},
            )
            write_text_view(workspace, record, full_body)
            write_meta(workspace, record)
            regenerate_index(workspace)
        finally:
            release_id(workspace, sid)
    return record


def _replaced_id(workspace: Workspace, sha: str, url: str, final_url: str, replace: bool) -> str | None:
    """On ``replace`` without an explicit id: the id of the record with the same content or the same URL."""
    if not replace:
        return None
    same = find_by_sha256(workspace, sha) or find_by_origin(workspace, url) or find_by_origin(workspace, final_url)
    return same.id if same is not None else None


def _ingest_binary(data: bytes, info: Sniff, workspace: Workspace, toolchain: Any, title: str | None, source_id: str | None, url: str, final_url: str, replace: bool) -> SourceRecord:
    from . import ingest_path  # noqa: PLC0415 - the package imports this module; the dispatcher is needed only here

    stem = slugify(title or title_from_url(final_url)) or "download"
    with tempfile.TemporaryDirectory(prefix="edgent-url-") as tmp:
        payload = Path(tmp) / f"{stem}{raw_extension(info, Path(urllib.parse.urlsplit(final_url).path))}"
        payload.write_bytes(data)
        return ingest_path(payload, workspace, toolchain, title=title or title_from_url(final_url), source_id=source_id, replace=replace, origin=url, extra={"url": final_url})


__all__ = ["Fetcher", "URL_CONTENT_BYTES_LIMIT", "URL_FETCH_TIMEOUT_S", "ingest_url", "make_default_fetcher", "title_from_url"]
