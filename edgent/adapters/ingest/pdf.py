"""PDF sources: text page by page under ``[p.N]`` markers, OCR for scanned pages when an engine exists.

PyMuPDF is used when importable; otherwise ``pdftotext -layout`` runs once and the output is split on form
feeds. A page with fewer than ``OCR_CANDIDATE_MIN_CHARS`` characters is treated as scanned: it is rasterised
(PyMuPDF or ``pdftoppm``) and read with tesseract when both exist, else recorded as failed. Coverage names
the failed pages, so a citation into one of them resolves to "unknown", never to "the source does not say".
"""
from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ...domain.errors import ErrorCategory, Limit, ToolFailure
from ...domain.workspace import Workspace
from .base import IngestResult, Sniff, coverage, missing_tool, page_marker, run_command, title_from_stem

OCR_CANDIDATE_MIN_CHARS = 20
PDF_PAGES_LIMIT = Limit("pdf pages", 5000, "pages", does_not_raise=("the sources_add timeout",))
PDF_OCR_PAGES_LIMIT = Limit("pdf OCR pages", 200, "pages", does_not_raise=("the sources_add timeout",))
PDF_TOOL_TIMEOUT_S = 900.0
OCR_PAGE_TIMEOUT_S = 180.0
OCR_RASTER_DPI = 200
_PAGES_RE = re.compile(r"^Pages:\s+(\d+)", re.M)


def _pymupdf() -> Any | None:
    try:
        import fitz  # noqa: PLC0415 - optional dependency, imported lazily

        return fitz
    except Exception:  # noqa: BLE001 - absent or broken install both mean "not available"
        return None


def _clean_page(text: str) -> str:
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return cleaned.strip("\n")


def extract_with_pymupdf(fitz: Any, src: Path, max_pages: int) -> tuple[list[str], str | None, int]:
    """(pages, metadata title, pages_in_file). A file PyMuPDF cannot open is a typed ``unsupported`` failure."""
    try:
        with fitz.open(str(src)) as doc:
            total = doc.page_count
            pages = [_clean_page(doc[i].get_text("text") or "") for i in range(min(total, max_pages))]
            title = (doc.metadata or {}).get("title") or None
    except Exception as e:  # noqa: BLE001 - PyMuPDF raises its own exception types; none may escape a tool
        raise ToolFailure(
            "unsupported",
            f"PyMuPDF could not read {src.name}: {type(e).__name__}: {str(e)[:300]}",
            ErrorCategory.unsupported,
            remediation="Check that the file is a complete, unencrypted PDF; re-export it or decrypt it (qpdf --decrypt) and add it again.",
            path=str(src),
        ) from e
    return pages, (str(title).strip() or None) if title else None, total


def extract_with_pdftotext(pdftotext: str, src: Path) -> list[str]:
    r = run_command([pdftotext, "-layout", "-enc", "UTF-8", str(src), "-"], PDF_TOOL_TIMEOUT_S)
    if not r.ok and not r.stdout:
        reason = "timed out" if r.timed_out else (r.stderr.strip().splitlines() or ["unknown error"])[-1][:300]
        raise ToolFailure(
            "unsupported",
            f"pdftotext could not read {src.name}: {reason}",
            ErrorCategory.unsupported,
            remediation="Check that the file is a complete, unencrypted PDF; re-export it or decrypt it (qpdf --decrypt) and add it again.",
            path=str(src),
        )
    text = r.stdout.decode("utf-8", "replace")
    if text.endswith("\f"):  # pdftotext ends every page, including the last, with a form feed
        text = text[:-1]
    return [_clean_page(p) for p in text.split("\f")]


def pdfinfo_pages(src: Path) -> int | None:
    exe = shutil.which("pdfinfo")
    if not exe:
        return None
    r = run_command([exe, str(src)], 60.0)
    m = _PAGES_RE.search(r.stdout.decode("utf-8", "replace")) if r.ok else None
    return int(m.group(1)) if m else None


def rasterise_page(src: Path, page: int, out_dir: Path, fitz: Any | None) -> Path | None:
    """One PNG of ``page`` (1-based) at OCR resolution, through PyMuPDF or pdftoppm; None when neither can."""
    if fitz is not None:
        try:
            with fitz.open(str(src)) as doc:
                pix = doc[page - 1].get_pixmap(dpi=OCR_RASTER_DPI)
                out = out_dir / f"page-{page}.png"
                pix.save(str(out))
                return out
        except Exception:  # noqa: BLE001 - fall through to pdftoppm
            pass
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        return None
    prefix = out_dir / f"pg{page}"
    r = run_command([pdftoppm, "-r", str(OCR_RASTER_DPI), "-f", str(page), "-l", str(page), "-png", str(src), str(prefix)], OCR_PAGE_TIMEOUT_S)
    if not r.ok:
        return None
    produced = sorted(out_dir.glob(f"pg{page}-*.png"))
    return produced[0] if produced else None


def ocr_pages(src: Path, candidates: list[int], tesseract: str, fitz: Any | None) -> tuple[dict[int, str], list[int], list[str]]:
    """OCR each candidate page; returns (text by page, pages still empty, notes)."""
    texts: dict[int, str] = {}
    failed: list[int] = []
    notes: list[str] = []
    with tempfile.TemporaryDirectory(prefix="edgent-ocr-") as tmp:
        for n in candidates:
            png = rasterise_page(src, n, Path(tmp), fitz)
            if png is None:
                failed.append(n)
                continue
            r = run_command([tesseract, str(png), "-"], OCR_PAGE_TIMEOUT_S)
            text = _clean_page(r.stdout.decode("utf-8", "replace")) if r.ok else ""
            png.unlink(missing_ok=True)
            if text.strip():
                texts[n] = text
            else:
                failed.append(n)
                if not r.ok:
                    notes.append(f"tesseract failed on page {n}: {'timed out' if r.timed_out else (r.stderr.strip().splitlines() or ['error'])[-1][:120]}")
    return texts, failed, notes


def _page_list(pages: list[int], shown: int = 12) -> str:
    head = ", ".join(str(p) for p in pages[:shown])
    return head + (f", ... ({len(pages)} in all)" if len(pages) > shown else "")


def ingest_pdf(src: Path, source_id: str, workspace: Workspace, toolchain: Any, info: Sniff, title: str | None = None, **_: Any) -> IngestResult:
    fitz = _pymupdf()
    limitations: list[str] = []
    missing: list[dict[str, str]] = []
    meta_title: str | None = None
    pdftotext = toolchain.path("pdftotext") if toolchain is not None else None
    if fitz is not None:
        pages, meta_title, total = extract_with_pymupdf(fitz, src, int(PDF_PAGES_LIMIT.value))
        engine = "pymupdf"
    elif pdftotext:
        pages = extract_with_pdftotext(pdftotext, src)
        total = pdfinfo_pages(src) or len(pages)
        engine = "pdftotext"
        if total > len(pages):
            pages += [""] * (total - len(pages))  # pdftotext stopped early: the rest is unread, not empty
    else:
        missing += [missing_tool("pdftotext", "PDF text extraction"), missing_tool("pymupdf", "PDF text extraction")]
        limitations.append("no PDF text engine: nothing could be extracted (install poppler-utils or pip install edgent[pdf], then sources_add again with replace=true)")
        body = "(no PDF text engine installed: the raw file was kept but no page text could be extracted)\n"
        return IngestResult("pdf", body, title or title_from_stem(src.stem), coverage("unknown", pages_total=None, ocr=False), limitations, {"engine": None}, missing, usable=False)
    if total > PDF_PAGES_LIMIT.value:
        pages = pages[: int(PDF_PAGES_LIMIT.value)]
        limitations.append(f"only the first {int(PDF_PAGES_LIMIT.value)} of {total} pages were extracted ({PDF_PAGES_LIMIT.describe()}) Split the PDF to cover the rest.")
    candidates = [n for n, text in enumerate(pages, 1) if len(text.strip()) < OCR_CANDIDATE_MIN_CHARS]
    tesseract = toolchain.path("tesseract") if toolchain is not None else None
    ocr_ran = False
    failed: list[int] = []
    if candidates and tesseract:
        to_ocr = candidates[: int(PDF_OCR_PAGES_LIMIT.value)]
        texts, _, notes = ocr_pages(src, to_ocr, tesseract, fitz)
        ocr_ran = True
        for n, text in texts.items():
            if len(text.strip()) > len(pages[n - 1].strip()):
                pages[n - 1] = text
        failed = [n for n in candidates if not pages[n - 1].strip()]
        limitations += notes
        if len(candidates) > len(to_ocr):
            limitations.append(f"OCR stopped after {len(to_ocr)} pages ({PDF_OCR_PAGES_LIMIT.describe()}); pages {_page_list(candidates[len(to_ocr):])} stay unread.")
        if failed:
            limitations.append(f"OCR found no text on pages {_page_list(failed)}: they may be blank or images without text.")
    elif candidates:
        failed = [n for n in candidates if not pages[n - 1].strip()]
        sparse = [n for n in candidates if n not in failed]
        missing.append(missing_tool("tesseract", "OCR of scanned PDF pages"))
        if failed:
            limitations.append(f"no OCR engine: {len(failed)} scanned pages unreadable (pages {_page_list(failed)}). Install tesseract-ocr and sources_add again with replace=true.")
        if sparse:
            limitations.append(f"pages {_page_list(sparse)} hold under {OCR_CANDIDATE_MIN_CHARS} characters of text and may be scans with stray marks; no OCR engine to check them.")
    if engine == "pdftotext":
        limitations.append("extracted with pdftotext -layout: column order and hyphenation may be approximate")
    failed_set = set(failed)
    reason = "no OCR engine installed" if not tesseract else "OCR produced nothing"
    parts: list[str] = []
    for n, text in enumerate(pages, 1):
        content = text if n not in failed_set else f"(no extractable text on this page: probably a scanned image; {reason})"
        parts.append(f"{page_marker(n)}\n{content}\n")
    extracted = len(pages) - len(failed_set)
    status = "complete" if not failed_set else ("partial" if extracted > 0 else "unknown")
    cov = coverage(status, pages_total=len(pages), pages_failed=sorted(failed_set), ocr=ocr_ran)
    extra = {"pages": len(pages), "engine": engine, "chars": sum(len(p) for p in pages), "pages_in_file": total}
    return IngestResult("pdf", "\n".join(parts), title or meta_title or title_from_stem(src.stem), cov, limitations, extra, missing, usable=extracted > 0 or ocr_ran)
