"""Ingestion adapters: sniffing, ids, text views, PDF pages, media timelines, images, URLs, office documents, locators."""
from __future__ import annotations

import shutil
import struct
import subprocess
import zipfile
import zlib
from pathlib import Path

import pytest

from edgent.adapters import ingest
from edgent.adapters.ingest import base, image, media, office, pdf, text, web
from edgent.adapters.toolchain import Toolchain
from edgent.domain.errors import ToolFailure
from edgent.domain.workspace import Workspace
from tests.conftest import skip_unless

PDF_ENGINE = pytest.mark.skipif(not (Toolchain.detect().has("pdftotext") or Toolchain.detect().has("pymupdf")), reason="neither pdftotext nor PyMuPDF is installed; PDF extraction is unverified here")


# ----------------------------------------------------------------------------- fixtures
def make_pdf(pages: list[str | None]) -> bytes:
    """A minimal valid PDF, one Helvetica text object per page; None gives a page without text (a 'scan')."""
    objs: list[bytes | None] = [None, None, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids: list[int] = []
    for content_text in pages:
        if content_text is None:
            content = b""
        else:
            esc = content_text.encode("latin-1").replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")
            content = b"BT /F1 24 Tf 72 700 Td (" + esc + b") Tj ET"
        objs.append(b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream")
        objs.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents " + str(len(objs)).encode() + b" 0 R /Resources << /Font << /F1 3 0 R >> >> >>")
        kids.append(len(objs))
    objs[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objs[1] = b"<< /Type /Pages /Kids [" + b" ".join(f"{k} 0 R".encode() for k in kids) + b"] /Count " + str(len(kids)).encode() + b" >>"
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + (body or b"") + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def make_png(width: int, height: int) -> bytes:
    raw = b"".join(b"\x00" + b"\x80\x10\x10" * width for _ in range(height))

    def chunk(tag: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)

    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def make_jpeg_header(width: int, height: int) -> bytes:
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x00" * 9
    sof0 = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, height, width, 1) + b"\x01\x11\x00"
    return b"\xff\xd8" + app0 + sof0 + b"\xff\xd9"


def make_webp_lossy(width: int, height: int) -> bytes:
    vp8 = b"\x00\x00\x00" + b"\x9d\x01\x2a" + struct.pack("<HH", width, height) + b"\x00" * 8
    return b"RIFF" + struct.pack("<I", 4 + 8 + len(vp8)) + b"WEBP" + b"VP8 " + struct.pack("<I", len(vp8)) + vp8


def make_docx_zip(path: Path, body_xml: str) -> None:
    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", f'<?xml version="1.0"?><w:document xmlns:w="{w}"><w:body>{body_xml}</w:body></w:document>')


def ffmpeg_video(path: Path) -> None:
    """Three seconds: 1.5 s red then 1.5 s blue (one hard scene cut) with a 440 Hz tone."""
    argv = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=red:s=320x240:r=10:d=1.5", "-f", "lavfi", "-i", "color=blue:s=320x240:r=10:d=1.5",
        "-f", "lavfi", "-i", "sine=f=440:d=3", "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-map", "2:a",
        "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(path),
    ]
    subprocess.run(argv, check=True, capture_output=True, timeout=120)


def ffmpeg_wav(path: Path, seconds: float = 2.0) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"sine=f=440:d={seconds}", str(path)], check=True, capture_output=True, timeout=60)


def pandoc_make(markdown: str, out: Path) -> None:
    src = out.with_suffix(".md")
    src.write_text(markdown, encoding="utf-8")
    subprocess.run(["pandoc", str(src), "-o", str(out)], check=True, capture_output=True, timeout=120)


DOC_MD = "# Title One\n\nFirst section text.\n\n# Title Two\n\nSecond section text with a list.\n\n- a\n- b\n\n## Sub\n\nMore.\n"


@pytest.fixture
def tc() -> Toolchain:
    return Toolchain.detect()


@pytest.fixture
def inbox(tmp_path: Path) -> Path:
    d = tmp_path / "inbox"
    d.mkdir()
    return d


def view(workspace: Workspace, rec: base.SourceRecord) -> str:
    return workspace.read_text(rec.text_path)


# ----------------------------------------------------------------------------- sniffing and ids
def test_sniff_by_magic_and_extension(inbox: Path) -> None:
    cases: dict[str, tuple[bytes, str, str]] = {
        "a.pdf": (b"%PDF-1.7 junk", "pdf", "pdf"),
        "b.png": (make_png(2, 2), "image", "png"),
        "c.jpg": (make_jpeg_header(4, 4), "image", "jpeg"),
        "d.gif": (b"GIF89a" + b"\x00" * 10, "image", "gif"),
        "e.webp": (make_webp_lossy(3, 3), "image", "webp"),
        "f.wav": (b"RIFF\x00\x00\x00\x00WAVEfmt ", "audio", "wav"),
        "g.mp3": (b"ID3\x04\x00" + b"\x00" * 10, "audio", "mp3"),
        "h.mp4": (b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00", "video", "mp4"),
        "i.m4a": (b"\x00\x00\x00\x18ftypM4A \x00\x00\x02\x00", "audio", "m4a"),
        "j.mkv": (b"\x1aE\xdf\xa3" + b"\x00" * 20, "video", "mkv"),
        "k.webm": (b"\x1aE\xdf\xa3\x93B\x82\x84webm" + b"\x00" * 20, "video", "webm"),
        "l.bin": (b"\x00\x01\x02binary", "binary", "bin"),
        "m.txt": (b"plain text\n", "text", "txt"),
        "n.md": (b"# heading\n", "markdown", "markdown"),
        "o.py": (b"print(1)\n", "code", "py"),
        "p": (b"no extension, still text\n", "text", "txt"),
        "q.srt": (b"1\n00:00:01,000 --> 00:00:02,000\nhi\n", "text", "subtitles"),
        "r.avi": (b"RIFF\x00\x00\x00\x00AVI LIST", "video", "avi"),
        "s.flac": (b"fLaC\x00\x00\x00\x22", "audio", "flac"),
        "t.rtf": (b"{\\rtf1\\ansi hello}", "office", "rtf"),
    }
    for name, (data, kind, fmt) in cases.items():
        (inbox / name).write_bytes(data)
        got = base.sniff(inbox / name)
        assert (got.kind, got.format) == (kind, fmt), name
        assert base.sniff_kind(inbox / name) == kind
    assert base.sniff(inbox / "o.py").language == "python"
    (inbox / "Dockerfile").write_bytes(b"FROM python:3.12\n")
    assert base.sniff(inbox / "Dockerfile").language == "dockerfile"
    make_docx_zip(inbox / "w.docx", "<w:p><w:r><w:t>x</w:t></w:r></w:p>")
    assert base.sniff(inbox / "w.docx").format == "docx"
    with zipfile.ZipFile(inbox / "deck.pptx", "w") as z:
        z.writestr("ppt/presentation.xml", "<p/>")
    assert base.sniff(inbox / "deck.pptx") == base.Sniff("office", "pptx", base.sniff(inbox / "deck.pptx").mime)
    with zipfile.ZipFile(inbox / "book.epub", "w") as z:
        z.writestr("mimetype", "application/epub+zip")
    assert base.sniff(inbox / "book.epub").format == "epub"
    with zipfile.ZipFile(inbox / "sheet.xlsx", "w") as z:
        z.writestr("xl/workbook.xml", "<x/>")
    assert base.sniff(inbox / "sheet.xlsx") == base.Sniff("binary", "xlsx", base.sniff(inbox / "sheet.xlsx").mime)
    with zipfile.ZipFile(inbox / "plain.zip", "w") as z:
        z.writestr("a.txt", "x")
    assert base.sniff(inbox / "plain.zip").kind == "binary"


def test_allocate_id_suffixes_on_collision(workspace: Workspace) -> None:
    assert base.allocate_id(workspace, "Bayes' Theorem: Lecture 3") == "bayes-theorem-lecture-3"
    workspace.write_text("sources/text/notes.md", "x")
    assert base.allocate_id(workspace, "notes") == "notes-2"
    workspace.write_text("sources/text/notes-2.meta.json", "{}")
    assert base.allocate_id(workspace, "notes") == "notes-3"
    (workspace.path("sources_raw") / "notes-3.pdf").write_bytes(b"%PDF")
    assert base.allocate_id(workspace, "notes") == "notes-4"
    assert base.allocate_id(workspace, "notes", reserved={"notes-4"}) == "notes-5"
    assert base.allocate_id(workspace, "   ") == "untitled"
    assert len(base.allocate_id(workspace, "x" * 200)) <= base.ID_MAX_CHARS
    with pytest.raises(ToolFailure):
        base.validate_id("Not A Slug")


def test_coverage_record_shape_and_round_trip() -> None:
    cov = base.coverage("partial", pages_total=10, pages_failed=[7, 3, 3])
    assert cov["status"] == "partial" and cov["detail"]["pages_failed"] == [3, 7]
    assert set(cov["detail"]) == {"pages_total", "pages_failed", "minutes_total", "minutes_transcribed", "ocr", "transcriber"}
    with pytest.raises(ValueError):
        base.coverage("full")
    rec = base.SourceRecord("a", "pdf", "A", "sources/raw/a.pdf", "sources/text/a.md", "ff", 3, cov, ["lim"], "2024-01-01T00:00:00+00:00", "/x/a.pdf", {"pages": 10})
    again = base.SourceRecord.from_dict(rec.to_dict())
    assert again == rec and again.meta_path == "sources/text/a.meta.json" and again.frames_dir == "sources/text/a.frames"
    legacy = base.SourceRecord.from_dict({"id": "b", "coverage": {"status": "bogus"}})
    assert legacy.coverage_status == "unknown" and legacy.text_path == "sources/text/b.md"


def test_timestamp_helpers() -> None:
    assert base.hms(0) == "00:00:00" and base.hms(3723.9) == "01:02:03" and base.hms(360000) == "100:00:00"
    assert base.parse_hms("00:14:32") == 872 and base.parse_hms("14:32") == 872 and base.parse_hms("92") == 92 and base.parse_hms("1:02:03.5") == 3723
    assert base.parse_hms("1:60") is None and base.parse_hms("a:b") is None and base.parse_hms("") is None
    assert base.TIME_MARKER_RE.match("[t=01:02:03]") and base.PAGE_MARKER_RE.match("[p.12]") and not base.PAGE_MARKER_RE.match("[p.12] text")


def test_parse_locator_forms() -> None:
    for raw, kind, value, ref in (
        ("lec:p.12", "page", 12, "12"), ("lec:p12", "page", 12, "12"), ("lec:t=00:14:32", "time", 872, "00:14:32"), ("LEC:t=14:32", "time", 872, "00:14:32"),
        ("lec:t=92", "time", 92, "00:01:32"), ("lec:L12", "line", 12, "12"), ("lec-2:l7", "line", 7, "7"), ("lec:L12-15", "line", 12, "12"),
    ):
        loc = base.parse_locator(raw)
        assert (loc.source_id, loc.kind, loc.value, loc.ref) == ("lec-2" if raw.startswith("lec-2") else "lec", kind, value, ref), raw
    assert base.parse_locator("lec:p.3").marker == "[p.3]" and base.parse_locator("lec:t=5").marker == "[t=00:00:05]" and base.parse_locator("lec:L2").marker == "L2"
    for bad in ("", "lec", "lec:", ":p.1", "lec:page 3", "lec:t=1:99", "Bad Id:p.1", "lec:x12"):
        with pytest.raises(base.LocatorError):
            base.parse_locator(bad)


# ----------------------------------------------------------------------------- text, markdown, code
def test_text_markdown_code_views(workspace: Workspace, tc: Toolchain, inbox: Path) -> None:
    (inbox / "notes.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    (inbox / "Bayes Theorem.md").write_text("# Bayes' theorem\n\nPosterior is prior times likelihood.\n", encoding="utf-8")
    (inbox / "calc.py").write_text("def f(x):\n    return x * 2\n", encoding="utf-8")
    txt = ingest.ingest_path(inbox / "notes.txt", workspace, tc)
    md = ingest.ingest_path(str(inbox / "Bayes Theorem.md"), workspace, tc)
    code = ingest.ingest_path(inbox / "calc.py", workspace, tc)
    assert (txt.id, txt.kind, txt.title) == ("notes", "text", "notes") and txt.coverage_status == "complete"
    assert (md.id, md.kind, md.title) == ("bayes-theorem", "markdown", "Bayes' theorem")
    assert (code.id, code.kind, code.extra["language"]) == ("calc", "code", "python")
    assert view(workspace, txt) == "# notes\nsource: notes | kind: text | coverage: complete\n\nalpha\nbeta\ngamma\n"
    assert view(workspace, md).startswith("# Bayes' theorem\nsource: bayes-theorem | kind: markdown | coverage: complete\n\n# Bayes' theorem\n")
    assert view(workspace, code) == "# calc\nsource: calc | kind: code | coverage: complete\n\n```python\ndef f(x):\n    return x * 2\n```\n"
    assert (workspace.root / "sources/raw/notes.txt").read_text() == "alpha\nbeta\ngamma\n" and (workspace.root / "sources/raw/calc.py").exists()
    assert txt.raw_path == "sources/raw/notes.txt" and txt.text_path == "sources/text/notes.md" and txt.size_bytes == 17 and len(txt.sha256) == 64
    assert txt.origin == str((inbox / "notes.txt").resolve()) and txt.created_at and txt.extra["usable"] is True and txt.extra["missing_toolchains"] == []
    meta = workspace.read_json("sources/text/notes.meta.json")
    assert meta["id"] == "notes" and meta["coverage"]["status"] == "complete" and "already_present" not in meta["extra"]
    index = workspace.read_text("sources/INDEX.md")
    assert index.startswith("# Sources\n") and "- notes | text | notes | coverage complete | sources/text/notes.md\n" in index
    assert index.index("- bayes-theorem") < index.index("- calc") < index.index("- notes")
    assert [r.id for r in ingest.list_sources(workspace)] == ["bayes-theorem", "calc", "notes"]
    assert ingest.get_source(workspace, "calc").kind == "code" and ingest.get_source(workspace, "nope") is None


def test_code_fence_grows_past_backticks_and_latin1_is_noted(workspace: Workspace, tc: Toolchain, inbox: Path) -> None:
    (inbox / "readme.sh").write_text("echo '```'\n", encoding="utf-8")
    rec = ingest.ingest_path(inbox / "readme.sh", workspace, tc)
    assert view(workspace, rec).count("````") == 2
    (inbox / "old.txt").write_bytes(b"caf\xe9\n")
    rec = ingest.ingest_path(inbox / "old.txt", workspace, tc)
    assert rec.extra["encoding"] == "latin-1" and any("Latin-1" in lim for lim in rec.limitations) and "café" in view(workspace, rec)


def test_subtitles_become_time_markers(workspace: Workspace, tc: Toolchain, inbox: Path) -> None:
    (inbox / "lecture.srt").write_text("1\n00:00:01,000 --> 00:00:03,500\nHello <i>there</i>\n\n2\n00:14:32,000 --> 00:14:35,000\nBayes again\nline two\n", encoding="utf-8")
    rec = ingest.ingest_path(inbox / "lecture.srt", workspace, tc)
    body = view(workspace, rec)
    assert rec.kind == "text" and rec.extra["cues"] == 2 and "[t=00:00:01]\nHello there\n" in body and "[t=00:14:32]\nBayes again line two\n" in body
    hit = ingest.resolve_locator(workspace, "lecture:t=14:33")
    assert hit["found"] and hit["marker"] == "[t=00:14:32]" and hit["text"].startswith("Bayes again") and hit["offset_s"] == 1


def test_id_collision_dedupe_and_replace(workspace: Workspace, tc: Toolchain, inbox: Path) -> None:
    (inbox / "notes.txt").write_text("one\n")
    (inbox / "sub").mkdir()
    (inbox / "sub" / "notes.txt").write_text("two\n")
    first = ingest.ingest_path(inbox / "notes.txt", workspace, tc)
    second = ingest.ingest_path(inbox / "sub" / "notes.txt", workspace, tc)
    assert (first.id, second.id) == ("notes", "notes-2")
    again = ingest.ingest_path(inbox / "notes.txt", workspace, tc)
    assert again.id == "notes" and again.extra["already_present"] is True and len(ingest.list_sources(workspace)) == 2
    assert "already_present" not in workspace.read_json("sources/text/notes.meta.json")["extra"]
    (inbox / "notes.txt").write_text("one, revised\n")
    replaced = ingest.ingest_path(inbox / "notes.txt", workspace, tc, replace=True, source_id="notes")
    assert replaced.id == "notes" and "revised" in view(workspace, replaced) and len(ingest.list_sources(workspace)) == 2
    with pytest.raises(ToolFailure) as e:
        ingest.ingest_path(inbox / "sub" / "notes.txt", workspace, tc, source_id="notes", replace=False)
    assert e.value.error.category.value == "invalid_input" and "replace" in e.value.error.remediation
    explicit = ingest.ingest_path(inbox / "sub" / "notes.txt", workspace, tc, title="Second Notes", source_id="second")
    assert explicit.id == "second" and explicit.title == "Second Notes" and explicit.extra.get("already_present") is None
    assert ingest.regenerate_index(workspace) == 3


def test_input_path_errors(workspace: Workspace, tc: Toolchain, inbox: Path) -> None:
    with pytest.raises(ToolFailure) as e:
        ingest.ingest_path(inbox / "missing.pdf", workspace, tc)
    assert e.value.error.category.value == "not_found" and "sources_add_url" in e.value.error.remediation
    with pytest.raises(ToolFailure) as e:
        ingest.ingest_path(inbox, workspace, tc)
    assert e.value.error.category.value == "invalid_input"
    (inbox / "blob.bin").write_bytes(b"\x00\x01\x02" * 50)
    with pytest.raises(ToolFailure) as e:
        ingest.ingest_path(inbox / "blob.bin", workspace, tc)
    assert e.value.error.category.value == "unsupported" and "supported inputs" in e.value.error.remediation
    with zipfile.ZipFile(inbox / "sheet.xlsx", "w") as z:
        z.writestr("xl/workbook.xml", "<x/>")
    with pytest.raises(ToolFailure) as e:
        ingest.ingest_path(inbox / "sheet.xlsx", workspace, tc)
    assert "CSV" in e.value.error.remediation
    workspace.write_text("notes/inside.md", "# Inside\n\nrelative paths resolve against the workspace root\n")
    rec = ingest.ingest_path("notes/inside.md", workspace, tc)
    assert rec.id == "inside" and rec.title == "Inside"
    raw = workspace.path("sources_raw") / "inside.md"
    assert raw.exists()
    same = ingest.ingest_path(str(raw), workspace, tc, replace=True)  # a file already in sources/raw is not copied onto itself
    assert same.id == "inside" and raw.exists()


# ----------------------------------------------------------------------------- pdf
@PDF_ENGINE
def test_pdf_pages_markers_and_coverage(workspace: Workspace, tc: Toolchain, inbox: Path) -> None:
    (inbox / "paper.pdf").write_bytes(make_pdf(["Hello Bayes theorem on page one", None, "Page three has (parens) and more"]))
    rec = ingest.ingest_path(inbox / "paper.pdf", workspace, tc, title="A Paper")
    body = view(workspace, rec)
    assert rec.kind == "pdf" and rec.id == "a-paper" and body.startswith("# A Paper\nsource: a-paper | kind: pdf | coverage: partial\n\n[p.1]\nHello Bayes theorem on page one\n")
    assert "\n[p.2]\n(no extractable text on this page" in body and "\n[p.3]\nPage three has (parens) and more\n" in body
    assert rec.coverage["status"] == "partial" and rec.coverage["detail"]["pages_total"] == 3 and rec.coverage["detail"]["pages_failed"] == [2]
    assert rec.extra["pages"] == 3 and rec.extra["engine"] in ("pdftotext", "pymupdf") and rec.extra["usable"] is True
    if not tc.has("tesseract"):
        assert any("no OCR engine: 1 scanned pages unreadable (pages 2)" in lim for lim in rec.limitations)
        assert rec.coverage["detail"]["ocr"] is False and any(m["name"] == "tesseract" for m in rec.extra["missing_toolchains"])
    (inbox / "clean.pdf").write_bytes(make_pdf(["Only a little text here, but enough", "Fig. 3"]))
    complete = ingest.ingest_path(inbox / "clean.pdf", workspace, tc)
    assert complete.coverage_status == "complete" and complete.coverage["detail"]["pages_failed"] == [] and "[p.2]\nFig. 3\n" in view(workspace, complete)
    if not tc.has("tesseract"):
        assert any("pages 2 hold under 20 characters" in lim for lim in complete.limitations)
    (inbox / "scan.pdf").write_bytes(make_pdf([None, None]))
    all_blank = ingest.ingest_path(inbox / "scan.pdf", workspace, tc)
    assert all_blank.coverage_status == "unknown" and all_blank.coverage["detail"]["pages_failed"] == [1, 2] and all_blank.extra["usable"] is False


@PDF_ENGINE
def test_pdf_locators_report_uncovered_pages(workspace: Workspace, tc: Toolchain, inbox: Path) -> None:
    (inbox / "paper.pdf").write_bytes(make_pdf(["Page one has enough text to count", None, "Page three has enough text as well"]))
    ingest.ingest_path(inbox / "paper.pdf", workspace, tc)
    p1 = ingest.resolve_locator(workspace, "paper:p.1")
    assert p1["found"] and p1["marker"] == "[p.1]" and p1["text"] == "Page one has enough text to count" and p1["line"] == 4 and "partial" in (p1["coverage_note"] or "")
    p2 = ingest.resolve_locator(workspace, "paper:p.2")
    assert not p2["found"] and p2["coverage_note"] == "page 2 had no extractable text: absence is unknown, not evidence" and p2["marker_present"] is True
    assert p2["nearest"] == ["p.1", "p.3"]
    p9 = ingest.resolve_locator(workspace, "paper:p.9")
    assert not p9["found"] and "beyond the last page (3)" in p9["coverage_note"] and p9["nearest"][0] == "p.3"
    t = ingest.resolve_locator(workspace, "paper:t=00:00:05")
    assert not t["found"] and "no time markers" in t["coverage_note"] and "p.N" in t["coverage_note"]


def test_pdf_without_engine_is_recorded_but_unusable(workspace: Workspace, inbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (inbox / "paper.pdf").write_bytes(make_pdf(["Some text"]))
    bare = Toolchain.detect()
    bare.binaries["pdftotext"] = None
    bare.binaries["tesseract"] = None
    monkeypatch.setattr(pdf, "_pymupdf", lambda: None)
    rec = ingest.ingest_path(inbox / "paper.pdf", workspace, bare)
    assert rec.kind == "pdf" and rec.coverage_status == "unknown" and rec.extra["usable"] is False
    assert {m["name"] for m in rec.extra["missing_toolchains"]} == {"pdftotext", "pymupdf"} and any("no PDF text engine" in lim for lim in rec.limitations)
    assert "(no PDF text engine installed" in view(workspace, rec) and (workspace.root / "sources/raw/paper.pdf").exists()


def test_pdftotext_page_splitting_helpers(tmp_path: Path) -> None:
    assert pdf._clean_page("a  \r\nb\n\n\n\nc\n") == "a\nb\n\nc"
    fake = tmp_path / "pdftotext"
    fake.write_text("#!/bin/sh\nprintf 'one\\n\\f\\fthree\\n\\f'\n")
    fake.chmod(0o755)
    assert pdf.extract_with_pdftotext(str(fake), tmp_path / "x.pdf") == ["one", "", "three"]
    failing = tmp_path / "pdftotext-fail"
    failing.write_text("#!/bin/sh\necho 'Syntax Error: broken' >&2\nexit 1\n")
    failing.chmod(0o755)
    with pytest.raises(ToolFailure) as e:
        pdf.extract_with_pdftotext(str(failing), tmp_path / "x.pdf")
    assert e.value.error.category.value == "unsupported" and "broken" in e.value.error.message


# ----------------------------------------------------------------------------- images
def test_image_dimensions_without_pil(inbox: Path) -> None:
    (inbox / "a.png").write_bytes(make_png(7, 3))
    (inbox / "b.jpg").write_bytes(make_jpeg_header(640, 480))
    (inbox / "c.gif").write_bytes(b"GIF89a" + struct.pack("<HH", 320, 200) + b"\x00" * 4)
    (inbox / "d.webp").write_bytes(make_webp_lossy(100, 50))
    bits = 99 | (49 << 14)
    vp8l = b"\x2f" + struct.pack("<I", bits) + b"\x00" * 8
    (inbox / "e.webp").write_bytes(b"RIFF" + struct.pack("<I", 4 + 8 + len(vp8l)) + b"WEBP" + b"VP8L" + struct.pack("<I", len(vp8l)) + vp8l)
    (inbox / "f.bmp").write_bytes(b"BM" + b"\x00" * 16 + struct.pack("<ii", 30, -20) + b"\x00" * 10)
    (inbox / "g.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 4)
    assert image.image_dimensions(inbox / "a.png") == (7, 3) and image.image_dimensions(inbox / "b.jpg") == (640, 480)
    assert image.image_dimensions(inbox / "c.gif") == (320, 200) and image.image_dimensions(inbox / "d.webp") == (100, 50)
    assert image.image_dimensions(inbox / "e.webp") == (100, 50) and image.image_dimensions(inbox / "f.bmp") == (30, 20)
    assert image.image_dimensions(inbox / "g.png") is None


def test_image_ingest_without_ocr_is_unknown(workspace: Workspace, tc: Toolchain, inbox: Path) -> None:
    (inbox / "Slide 7.png").write_bytes(make_png(16, 9))
    bare = Toolchain.detect()
    bare.binaries["tesseract"] = None
    rec = ingest.ingest_path(inbox / "Slide 7.png", workspace, bare)
    body = view(workspace, rec)
    assert rec.kind == "image" and rec.id == "slide-7" and rec.title == "Slide 7" and rec.extra["dimensions"] == [16, 9]
    assert body.startswith("# Slide 7\nsource: slide-7 | kind: image | coverage: unknown\n\n[p.1]\n(image: png, 16x9, raw copy at sources/raw/slide-7)\n(no OCR engine)\n")
    assert rec.coverage_status == "unknown" and rec.coverage["detail"]["ocr"] is False and rec.extra["usable"] is True
    assert any("no OCR engine" in lim for lim in rec.limitations) and rec.extra["missing_toolchains"][0]["name"] == "tesseract"
    hit = ingest.resolve_locator(workspace, "slide-7:p.1")
    assert hit["found"] and "(no OCR engine)" in hit["text"] and "unknown" in hit["coverage_note"]


@skip_unless("tesseract")
def test_image_ocr_runs_when_available(workspace: Workspace, tc: Toolchain, inbox: Path) -> None:
    (inbox / "blank.png").write_bytes(make_png(64, 64))
    rec = ingest.ingest_path(inbox / "blank.png", workspace, tc)
    assert rec.coverage_status == "complete" and rec.coverage["detail"]["ocr"] is True


# ----------------------------------------------------------------------------- media
@skip_unless("ffmpeg")
def test_video_keyframes_without_transcriber(workspace: Workspace, tc: Toolchain, inbox: Path) -> None:
    ffmpeg_video(inbox / "Lecture 01.mp4")
    rec = ingest.ingest_path(inbox / "Lecture 01.mp4", workspace, tc, transcriber=None if tc.has("faster_whisper") is False else None)
    body = view(workspace, rec)
    assert rec.kind == "video" and rec.id == "lecture-01" and rec.extra["duration_s"] == pytest.approx(3.0, abs=0.2) and rec.extra["dimensions"] == [320, 240]
    frames = sorted(p.name for p in (workspace.root / "sources/text/lecture-01.frames").iterdir())
    assert "00-00-00.jpg" in frames and "00-00-01.jpg" in frames, frames  # the first frame plus the red-to-blue scene cut at 1.5 s
    assert "[t=00:00:00]\n(keyframe: sources/text/lecture-01.frames/00-00-00.jpg)\n" in body and "[t=00:00:01]\n(keyframe: sources/text/lecture-01.frames/00-00-01.jpg)\n" in body
    assert body.startswith("# Lecture 01\nsource: lecture-01 | kind: video | coverage: ")
    if not tc.has("faster_whisper"):
        assert rec.coverage_status == "unknown" and rec.coverage["detail"]["minutes_transcribed"] == 0 and rec.coverage["detail"]["transcriber"] is None
        assert any("no transcriber installed (pip install edgent[stt])" in lim for lim in rec.limitations)
        assert any(m["name"] == "faster_whisper" for m in rec.extra["missing_toolchains"]) and rec.extra["usable"] is True
        hit = ingest.resolve_locator(workspace, "lecture-01:t=00:00:02")
        assert hit["found"] and hit["marker"] == "[t=00:00:01]" and "unknown, not absent" in hit["coverage_note"] and hit["offset_s"] == 1
    assert rec.coverage["detail"]["minutes_total"] == pytest.approx(0.05, abs=0.01)
    beyond = ingest.resolve_locator(workspace, "lecture-01:t=00:01:00")
    assert not beyond["found"] and "beyond the duration" in beyond["coverage_note"]
    page = ingest.resolve_locator(workspace, "lecture-01:p.1")
    assert not page["found"] and "t=HH:MM:SS" in page["coverage_note"]


@skip_unless("ffmpeg")
def test_video_with_injected_transcriber(workspace: Workspace, tc: Toolchain, inbox: Path) -> None:
    ffmpeg_video(inbox / "talk.mp4")
    seen: list[str] = []

    def fake(wav_path: str) -> media.Transcript:
        seen.append(wav_path)
        assert Path(wav_path).exists() and Path(wav_path).stat().st_size > 1000
        return media.Transcript([media.Segment(0.0, 1.4, "Welcome to the  lecture."), media.Segment(1.6, 2.9, "Today: Bayes.")], "en", "fake-stt")

    rec = ingest.ingest_path(inbox / "talk.mp4", workspace, tc, transcriber=fake)
    body = view(workspace, rec)
    assert seen and seen[0].endswith("audio.wav") and not Path(seen[0]).exists()  # the WAV lived in a temporary directory
    assert rec.coverage_status == "complete" and rec.coverage["detail"]["transcriber"] == "fake-stt" and rec.coverage["detail"]["minutes_transcribed"] == rec.coverage["detail"]["minutes_total"]
    assert "[t=00:00:00]\n(keyframe: sources/text/talk.frames/00-00-00.jpg)\nWelcome to the lecture.\n" in body and "[t=00:00:01]\n(keyframe: sources/text/talk.frames/00-00-01.jpg)\nToday: Bayes.\n" in body
    assert rec.extra["language"] == "en" and rec.extra["transcript_segments"] == 2 and "| transcript: fake-stt (en)" in body
    hit = ingest.resolve_locator(workspace, "talk:t=1")
    assert hit["found"] and hit["text"].endswith("Today: Bayes.") and hit["coverage_note"] is None
    crashing = ingest.ingest_path(inbox / "talk.mp4", workspace, tc, replace=True, transcriber=lambda p: (_ for _ in ()).throw(RuntimeError("model missing")))
    assert crashing.coverage_status == "unknown" and any("transcription failed" in lim for lim in crashing.limitations)


@skip_unless("ffmpeg")
def test_audio_without_transcriber_is_unusable(workspace: Workspace, tc: Toolchain, inbox: Path) -> None:
    ffmpeg_wav(inbox / "tone.wav")
    bare = Toolchain.detect()
    bare.modules["faster_whisper"] = False
    rec = ingest.ingest_path(inbox / "tone.wav", workspace, bare)
    assert rec.kind == "audio" and rec.coverage_status == "unknown" and rec.extra["usable"] is False and rec.extra["keyframes"] == 0
    assert "nothing in this file is greppable yet" in view(workspace, rec)
    miss = ingest.resolve_locator(workspace, "tone:t=1")
    assert not miss["found"] and "no time markers" in miss["coverage_note"] and "unknown, not absent" in miss["coverage_note"]
    with_text = ingest.ingest_path(inbox / "tone.wav", workspace, bare, replace=True, transcriber=lambda p: media.Transcript([media.Segment(0.2, 1.0, "beep")], None, "fake"))
    assert with_text.coverage_status == "complete" and "[t=00:00:00]\nbeep\n" in view(workspace, with_text) and with_text.extra["usable"] is True


def test_media_without_ffprobe_is_recorded_but_unusable(workspace: Workspace, inbox: Path) -> None:
    (inbox / "clip.mp4").write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64)
    bare = Toolchain.detect()
    bare.binaries["ffprobe"] = None
    rec = ingest.ingest_path(inbox / "clip.mp4", workspace, bare)
    assert rec.kind == "video" and rec.extra["usable"] is False and rec.extra["missing_toolchains"][0]["name"] == "ffprobe" and rec.coverage_status == "unknown"


@skip_unless("ffprobe")
def test_probe_rejects_garbage(tc: Toolchain, inbox: Path) -> None:
    (inbox / "bad.mp4").write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64)
    with pytest.raises(ToolFailure) as e:
        media.probe(tc.path("ffprobe"), inbox / "bad.mp4")
    assert e.value.error.category.value == "unsupported" and "re-encode" in e.value.error.remediation.lower()


def test_media_timeline_merges_frames_and_segments() -> None:
    frames = [media.Keyframe(0, "sources/text/x.frames/00-00-00.jpg", ["Title slide"]), media.Keyframe(65, "sources/text/x.frames/00-01-05.jpg")]
    transcript = media.Transcript([media.Segment(0.4, 3.0, "hello  world"), media.Segment(65.2, 70.0, "next"), media.Segment(80.0, 81.0, "   ")], "en", "t")
    body = media._timeline(frames, transcript)
    assert body == "[t=00:00:00]\n(keyframe: sources/text/x.frames/00-00-00.jpg)\nslide text: Title slide\nhello world\n\n[t=00:01:05]\n(keyframe: sources/text/x.frames/00-01-05.jpg)\nnext\n"
    assert media._sample(list(range(10)), 4) == [0, 2, 5, 7] and media._sample([1, 2], 5) == [1, 2]


# ----------------------------------------------------------------------------- web
HTML = "<!DOCTYPE html><html><head><title>Bayes' theorem - Wiki</title><script>evil()</script></head><body><nav>menu</nav><h1>Bayes' theorem</h1><p>Posterior is <b>prior</b> times likelihood.</p></body></html>"


def test_url_ingestion_through_injected_fetcher(workspace: Workspace, tc: Toolchain) -> None:
    calls: list[str] = []

    def fetcher(url: str) -> tuple[str, str, str]:
        calls.append(url)
        return HTML, "text/html; charset=utf-8", "https://example.org/wiki/Bayes_theorem"

    rec = ingest.ingest_url("https://example.org/wiki/Bayes", workspace, fetcher=fetcher, toolchain=tc)
    body = view(workspace, rec)
    assert calls == ["https://example.org/wiki/Bayes"] and rec.kind == "web" and rec.id == "bayes-theorem-wiki" and rec.title == "Bayes' theorem - Wiki"
    assert rec.origin == "https://example.org/wiki/Bayes" and rec.extra["url"] == "https://example.org/wiki/Bayes_theorem" and rec.extra["content_type"] == "text/html"
    assert body.startswith("# Bayes' theorem - Wiki\nsource: bayes-theorem-wiki | kind: web | coverage: complete\n\n(origin: https://example.org/wiki/Bayes_theorem, fetched ")
    assert "# Bayes' theorem" in body and "Posterior is **prior** times likelihood." in body and "evil()" not in body and "menu" not in body
    assert (workspace.root / rec.raw_path).read_text() == HTML and rec.raw_path == "sources/raw/bayes-theorem-wiki.html" and rec.coverage_status == "complete"
    assert "HTML converted to Markdown" in rec.limitations[0] and "- bayes-theorem-wiki | web | " in workspace.read_text("sources/INDEX.md")
    again = ingest.ingest_url("https://example.org/wiki/Bayes", workspace, fetcher=fetcher)
    assert again.id == rec.id and again.extra["already_present"] is True
    hit = ingest.resolve_locator(workspace, "bayes-theorem-wiki:L8")
    assert hit["found"] and hit["line_text"] == "Posterior is **prior** times likelihood."
    plain = ingest.ingest_url("https://example.org/notes.txt", workspace, fetcher=lambda u: ("just text\nsecond line\n", "text/plain", u), title="Plain Notes")
    assert plain.kind == "web" and plain.id == "plain-notes" and plain.raw_path.endswith(".txt") and "just text\nsecond line" in view(workspace, plain)
    data = ingest.ingest_url("https://api.example.org/v1/items", workspace, fetcher=lambda u: ('{"a": [1, 2]}', "application/json", u))
    assert data.id == "items" and '"a": [\n    1,' in view(workspace, data) and data.raw_path.endswith(".json")


def test_url_ingestion_errors_and_limits(workspace: Workspace) -> None:
    with pytest.raises(ToolFailure) as e:
        ingest.ingest_url("ftp://example.org/x", workspace, fetcher=lambda u: ("", "", u))
    assert e.value.error.category.value == "invalid_input"
    with pytest.raises(ToolFailure) as e:
        ingest.ingest_url("https://example.org/blob", workspace, fetcher=lambda u: (b"\x00\x01\x02" * 100, "application/octet-stream", u))
    assert e.value.error.category.value == "unsupported" and "sources_add" in e.value.error.remediation
    big = "x" * (int(web.URL_CONTENT_BYTES_LIMIT.value) + 10)
    rec = ingest.ingest_url("https://example.org/big.txt", workspace, fetcher=lambda u: (big, "text/plain", u))
    assert rec.coverage_status == "partial" and rec.size_bytes == web.URL_CONTENT_BYTES_LIMIT.value and any("url source size exceeded" in lim for lim in rec.limitations)
    assert web.title_from_url("https://example.org/docs/intro-to-bayes.html") == "intro to bayes" and web.title_from_url("https://example.org/") == "example.org" and web.title_from_url("https://x.org/a/paper.pdf") == "paper"


@PDF_ENGINE
def test_url_pdf_is_routed_to_the_pdf_ingester(workspace: Workspace, tc: Toolchain) -> None:
    payload = make_pdf(["Remote paper page one with enough words"])
    rec = ingest.ingest_url("https://example.org/papers/bayes.pdf", workspace, fetcher=lambda u: (payload, "application/pdf", u), toolchain=tc)
    assert rec.kind == "pdf" and rec.id == "bayes" and rec.title == "bayes" and rec.origin == "https://example.org/papers/bayes.pdf" and rec.raw_path == "sources/raw/bayes.pdf"
    assert "[p.1]\nRemote paper page one with enough words" in view(workspace, rec) and rec.coverage_status == "complete" and rec.extra["usable"] is True
    assert ingest.ingest_url("https://example.org/papers/bayes.pdf", workspace, fetcher=lambda u: (payload, "application/pdf", u), toolchain=tc).extra["already_present"] is True
    other = make_pdf(["A different paper served as octet-stream"])
    mislabelled = ingest.ingest_url("https://example.org/dl?id=7", workspace, fetcher=lambda u: (other, "application/octet-stream", u), toolchain=tc, title="Mislabelled PDF")
    assert mislabelled.kind == "pdf" and mislabelled.id == "mislabelled-pdf" and mislabelled.origin == "https://example.org/dl?id=7" and mislabelled.raw_path == "sources/raw/mislabelled-pdf.pdf"


def test_default_fetcher_uses_the_web_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_web import PUBLIC_RESOLVER, FakeOpener, FakeResponse

    opener = FakeOpener({"https://example.org/page": FakeResponse(HTML, "https://example.org/page")})
    fetch = web.make_default_fetcher(opener=opener, resolver=PUBLIC_RESOLVER)
    content, content_type, final_url = fetch("https://example.org/page")
    assert content == HTML and content_type == "text/html" and final_url == "https://example.org/page" and opener.calls[0][2] == web.URL_FETCH_TIMEOUT_S
    pdf_opener = FakeOpener({"https://example.org/x.pdf": FakeResponse(b"%PDF-1.4 bytes", "https://example.org/x.pdf", content_type="application/pdf")})
    content, content_type, _ = web.make_default_fetcher(opener=pdf_opener, resolver=PUBLIC_RESOLVER)("https://example.org/x.pdf")
    assert isinstance(content, bytes) and content_type == "application/pdf"
    with pytest.raises(ToolFailure):
        web.make_default_fetcher(opener=opener, resolver=PUBLIC_RESOLVER)("http://localhost:8000/secret")


# ----------------------------------------------------------------------------- office
@skip_unless("pandoc")
def test_docx_and_epub_through_pandoc(workspace: Workspace, tc: Toolchain, inbox: Path) -> None:
    pandoc_make(DOC_MD, inbox / "doc.docx")
    pandoc_make(DOC_MD, inbox / "doc.epub")
    docx = ingest.ingest_path(inbox / "doc.docx", workspace, tc)
    epub = ingest.ingest_path(inbox / "doc.epub", workspace, tc, title="Doc as EPUB")
    for rec in (docx, epub):
        body = view(workspace, rec)
        assert rec.kind == "office" and rec.coverage_status == "complete" and rec.coverage["detail"]["pages_total"] == 2 and rec.extra["engine"] == "pandoc"
        assert "[p.1]\n# Title One\n\nFirst section text.\n" in body and "\n[p.2]\n# Title Two\n" in body and "- a\n- b" in body and "<div" not in body
        assert any("page numbers are approximate" in lim for lim in rec.limitations)
    assert docx.title == "Title One" and epub.title == "Doc as EPUB" and (docx.id, epub.id) == ("doc", "doc-as-epub")
    hit = ingest.resolve_locator(workspace, "doc:p.2")
    assert hit["found"] and hit["text"].startswith("# Title Two") and hit["coverage_note"] is None


@skip_unless("pandoc")
def test_pptx_slides_without_pandoc(workspace: Workspace, inbox: Path) -> None:
    pandoc_make(DOC_MD, inbox / "deck.pptx")
    bare = Toolchain.detect()
    bare.binaries["pandoc"] = None
    rec = ingest.ingest_path(inbox / "deck.pptx", workspace, bare)
    body = view(workspace, rec)
    assert rec.kind == "office" and rec.extra["engine"] == "stdlib-pptx" and rec.extra["slides"] == 2 and rec.coverage["detail"]["pages_total"] == 2
    assert "[p.1]\nTitle One\nFirst section text.\n" in body and "\n[p.2]\nTitle Two\nSecond section text with a list.\na\nb\n" in body
    assert rec.limitations == ["one [p.N] per slide, in presentation order"] and rec.extra["missing_toolchains"] == []


@skip_unless("pandoc")
def test_docx_fallback_and_epub_unavailable_without_pandoc(workspace: Workspace, inbox: Path) -> None:
    pandoc_make(DOC_MD, inbox / "doc.docx")
    pandoc_make(DOC_MD, inbox / "doc.epub")
    bare = Toolchain.detect()
    bare.binaries["pandoc"] = None
    rec = ingest.ingest_path(inbox / "doc.docx", workspace, bare)
    body = view(workspace, rec)
    assert rec.extra["engine"] == "stdlib-docx" and "[p.1]\n# Title One\n\nFirst section text.\n" in body and "[p.2]\n# Title Two" in body
    assert any("pandoc is not installed" in lim for lim in rec.limitations) and rec.extra["missing_toolchains"][0]["name"] == "pandoc"
    with pytest.raises(ToolFailure) as e:
        ingest.ingest_path(inbox / "doc.epub", workspace, bare)
    assert e.value.error.category.value == "unavailable" and "pandoc" in e.value.error.remediation
    assert ingest.get_source(workspace, "doc-2") is None  # nothing half-written


def test_docx_plain_text_parser_and_section_split(inbox: Path) -> None:
    body = (
        '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Intro</w:t></w:r></w:p>'
        "<w:p><w:r><w:t>Hello</w:t><w:tab/><w:t>world</w:t></w:r></w:p>"
        "<w:tbl><w:tr><w:tc><w:p><w:r><w:t>a</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>b</w:t></w:r></w:p></w:tc></w:tr></w:tbl>"
        '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Next</w:t></w:r></w:p><w:p><w:r><w:t>Body</w:t></w:r></w:p>'
    )
    make_docx_zip(inbox / "d.docx", body)
    md = office.docx_plain_text(inbox / "d.docx")
    assert md == "# Intro\n\nHello\tworld\n\n| a | b |\n\n# Next\n\nBody\n"
    sections, level = office.split_sections(md)
    assert level == 1 and sections == ["# Intro\n\nHello\tworld\n\n| a | b |", "# Next\n\nBody"]
    assert office.split_sections("plain\ntext") == (["plain\ntext"], 0)
    assert office.split_sections("pre\n\n## A\n\nx\n\n## B\n\ny\n")[0] == ["pre", "## A\n\nx", "## B\n\ny"]
    (inbox / "bad.docx").write_bytes(b"PK\x03\x04 not really")
    with pytest.raises(ToolFailure) as e:
        office.docx_plain_text(inbox / "bad.docx")
    assert e.value.error.category.value == "unsupported"


# ----------------------------------------------------------------------------- locators on text and the index
def test_line_locators_and_windows(workspace: Workspace, tc: Toolchain, inbox: Path) -> None:
    (inbox / "long.txt").write_text("".join(f"line {i:03d} of the text\n" for i in range(1, 201)))
    ingest.ingest_path(inbox / "long.txt", workspace, tc)
    hit = ingest.resolve_locator(workspace, "long:L100", window_chars=120)
    assert hit["found"] and hit["line"] == 100 and hit["line_text"] == "line 097 of the text" and hit["marker"] == "L100"
    assert hit["text"].count("\n") <= 7 and "line 097 of the text" in hit["text"] and len(hit["text"]) <= 140
    assert ingest.resolve_locator(workspace, "long:L1")["line_text"] == "# long"
    miss = ingest.resolve_locator(workspace, "long:L5000")
    assert not miss["found"] and "beyond the end" in miss["coverage_note"] and miss["nearest"] == ["L203"]
    bad = ingest.resolve_locator(workspace, "long:page 3")
    assert not bad["found"] and "unknown form" in bad["error"] and bad["hint"]
    unknown = ingest.resolve_locator(workspace, "ghost:L1")
    assert not unknown["found"] and unknown["coverage_note"].startswith("no source with id 'ghost'") and unknown["known_ids"] == ["long"]
    huge = ingest.resolve_locator(workspace, "long:L10", window_chars=10 ** 9)
    assert len(huge["text"]) <= ingest.LOCATOR_WINDOW_LIMIT.value + 10


def test_index_regeneration_from_meta_files(workspace: Workspace, tc: Toolchain, inbox: Path) -> None:
    assert ingest.regenerate_index(workspace) == 0 and "no sources yet" in workspace.read_text("sources/INDEX.md")
    (inbox / "a.txt").write_text("a\n")
    (inbox / "b.md").write_text("# B | pipe in title\n")
    ingest.ingest_path(inbox / "a.txt", workspace, tc)
    ingest.ingest_path(inbox / "b.md", workspace, tc)
    workspace.write_text("sources/INDEX.md", "stale\n")
    (workspace.root / "sources/text/junk.meta.json").write_text("not json")
    assert ingest.regenerate_index(workspace) == 2
    index = workspace.read_text("sources/INDEX.md")
    assert index == "# Sources\n\n<!-- one line per source: id | kind | title | coverage | text view -->\n- a | text | a | coverage complete | sources/text/a.md\n- b | markdown | B / pipe in title | coverage complete | sources/text/b.md\n"
    (workspace.root / "sources/text/a.meta.json").unlink()
    assert ingest.regenerate_index(workspace) == 1 and "- a |" not in workspace.read_text("sources/INDEX.md")


def test_text_view_header_is_exact(workspace: Workspace) -> None:
    rec = base.SourceRecord("demo", "text", "  Demo | title  ", "sources/raw/demo.txt", "", "ff", 1, base.coverage_partial(pages_total=2, pages_failed=[2]))
    p = base.write_text_view(workspace, rec, "\n\nbody line\n\n")
    assert p.read_text() == "# Demo / title\nsource: demo | kind: text | coverage: partial\n\nbody line\n" and rec.text_path == "sources/text/demo.md"
    base.write_meta(workspace, rec)
    assert base.read_record(workspace, "demo").title == "Demo / title" and base.read_record(workspace, "../demo") is None


def test_run_command_handles_missing_binary_and_timeout(tmp_path: Path) -> None:
    r = base.run_command(["definitely-not-a-binary-xyz"], 5)
    assert r.returncode == 127 and not r.ok
    if shutil.which("sleep"):
        r = base.run_command(["sleep", "5"], 0.2)
        assert r.timed_out and not r.ok and "timed out" in r.stderr


def test_module_level_limits_have_units() -> None:
    for lim in (text.TEXT_SOURCE_BYTES_LIMIT, pdf.PDF_PAGES_LIMIT, pdf.PDF_OCR_PAGES_LIMIT, media.KEYFRAMES_LIMIT, media.MEDIA_DURATION_LIMIT, web.URL_CONTENT_BYTES_LIMIT, ingest.LOCATOR_WINDOW_LIMIT, ingest.MAX_INPUT_BYTES_LIMIT):
        assert lim.unit and lim.value > 0 and "exceeded" in lim.exceeded(lim.value + 1, "act").message
    assert media.SCENE_CHANGE_THRESHOLD == 0.3 and media.KEYFRAME_FLOOR_INTERVAL_S == 60 and media.AUDIO_SAMPLE_RATE_HZ == 16000



# ----------------------------------------------------------------------------- review additions: encodings, html, replace
def test_utf16_text_is_text_not_binary(workspace: Workspace, tc: Toolchain, inbox: Path) -> None:
    (inbox / "win.txt").write_bytes("hello\nwörld\n".encode("utf-16"))  # Notepad's "Unicode": a BOM and a NUL in every other byte
    assert base.sniff(inbox / "win.txt").kind == "text"
    rec = ingest.ingest_path(inbox / "win.txt", workspace, tc)
    assert rec.kind == "text" and rec.extra["encoding"] == "utf-16" and view(workspace, rec).endswith("\n\nhello\nwörld\n") and rec.limitations == []
    (inbox / "noext").write_bytes("x\n".encode("utf-16"))
    assert base.sniff(inbox / "noext").kind == "text"  # the BOM alone marks it as text
    (inbox / "crash.log").write_bytes(b"line one\nli\x00ne two\n")
    assert base.sniff(inbox / "crash.log").kind == "text"  # a stray NUL in a known text extension does not make it binary
    (inbox / "blob").write_bytes(b"\x00\x01\x02" * 20)
    assert base.sniff(inbox / "blob").kind == "binary"


def test_local_html_becomes_a_markdown_web_source(workspace: Workspace, tc: Toolchain, inbox: Path) -> None:
    page = '<html><head><meta charset="utf-8"><title>Bayes Lecture</title><script>x()</script></head><body><nav>menu</nav><h1>Bayes</h1><p>Posterior is <b>prior</b> times likelihood.</p></body></html>'
    (inbox / "Lecture Page.html").write_text(page, encoding="utf-8")
    rec = ingest.ingest_path(inbox / "Lecture Page.html", workspace, tc)
    body = view(workspace, rec)
    assert rec.kind == "web" and rec.id == "lecture-page" and rec.title == "Bayes Lecture" and rec.raw_path == "sources/raw/lecture-page.html"
    assert body.startswith("# Bayes Lecture\nsource: lecture-page | kind: web | coverage: complete\n\n# Bayes\n\nPosterior is **prior** times likelihood.")
    assert "x()" not in body and "menu" not in body and "<p>" not in body
    assert rec.limitations == [text.HTML_TO_MARKDOWN_NOTE] and rec.extra["content_type"] == "text/html" and rec.coverage_status == "complete"
    assert (workspace.root / rec.raw_path).read_text(encoding="utf-8") == page
    hit = ingest.resolve_locator(workspace, "lecture-page:L6")
    assert hit["found"] and hit["line_text"] == "Posterior is **prior** times likelihood."
    (inbox / "bare.htm").write_text("<p>no title here</p>")
    assert ingest.ingest_path(inbox / "bare.htm", workspace, tc).title == "bare"


def test_replace_reingests_the_same_path_in_place(workspace: Workspace, tc: Toolchain, inbox: Path) -> None:
    (inbox / "notes.txt").write_text("version one\n")
    first = ingest.ingest_path(inbox / "notes.txt", workspace, tc)
    (inbox / "notes.txt").write_text("version two, longer\n")
    second = ingest.ingest_path(inbox / "notes.txt", workspace, tc, replace=True)
    assert (first.id, second.id) == ("notes", "notes") and "version two" in view(workspace, second) and [r.id for r in ingest.list_sources(workspace)] == ["notes"]
    assert second.sha256 != first.sha256 and workspace.read_json("sources/text/notes.meta.json")["sha256"] == second.sha256
    (inbox / "other.txt").write_text("version three\n")
    third = ingest.ingest_path(inbox / "other.txt", workspace, tc, replace=True)  # a different path is a new source, never a silent overwrite
    assert third.id == "other" and len(ingest.list_sources(workspace)) == 2


def test_url_replace_without_source_id_keeps_the_id(workspace: Workspace) -> None:
    def page(version: str, final: str | None = None) -> web.Fetcher:
        return lambda u: (f"<html><head><title>Same Page</title></head><body><p>{version}</p></body></html>", "text/html", final or u)

    first = ingest.ingest_url("https://example.org/same", workspace, fetcher=page("version one"))
    second = ingest.ingest_url("https://example.org/same", workspace, fetcher=page("version two"), replace=True)
    assert (first.id, second.id) == ("same-page", "same-page") and second.sha256 != first.sha256
    assert "version two" in view(workspace, second) and "version one" not in view(workspace, second) and [r.id for r in ingest.list_sources(workspace)] == ["same-page"]
    moved = ingest.ingest_url("https://example.org/moved", workspace, fetcher=page("version three", final="https://example.org/same"), replace=True)
    assert moved.id == "same-page" and moved.origin == "https://example.org/moved"  # the URL a redirect lands on identifies the record too
    other = ingest.ingest_url("https://example.org/other", workspace, fetcher=page("version four"))
    assert other.id == "same-page-2" and len(ingest.list_sources(workspace)) == 2  # without replace a new page never overwrites another


def test_time_locator_before_the_first_marker_is_not_found(workspace: Workspace, tc: Toolchain, inbox: Path) -> None:
    (inbox / "cap.srt").write_text("1\n00:00:10,000 --> 00:00:12,000\nfirst words\n\n2\n00:00:20,000 --> 00:00:22,000\nsecond\n")
    ingest.ingest_path(inbox / "cap.srt", workspace, tc)
    early = ingest.resolve_locator(workspace, "cap:t=5")
    assert not early["found"] and early["text"] == "" and early["marker"] is None
    assert "before the first marker [t=00:00:10]" in early["coverage_note"] and early["nearest"] == ["t=00:00:10", "t=00:00:20"]
    exact = ingest.resolve_locator(workspace, "cap:t=10")
    assert exact["found"] and exact["offset_s"] == 0 and exact["marker"] == "[t=00:00:10]" and "nearest" not in exact
    later = ingest.resolve_locator(workspace, "cap:t=19")
    assert later["found"] and later["marker"] == "[t=00:00:10]" and later["offset_s"] == 9 and later["nearest"][0] == "t=00:00:20"


# ----------------------------------------------------------------------------- review additions: parallel ingests
def test_parallel_ingests_get_distinct_ids_and_a_complete_index(workspace: Workspace, tc: Toolchain, inbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading
    import time

    files = []
    for i in range(6):
        d = inbox / f"f{i}"
        d.mkdir()
        (d / "notes.txt").write_text(f"content {i}\n")
        files.append(d / "notes.txt")
    plain = ingest._INGESTERS["text"]

    def slow_text(*args: object, **kw: object) -> base.IngestResult:
        time.sleep(0.05)  # keeps every call in flight at once, so the ids must come from the reservations
        return plain(*args, **kw)

    monkeypatch.setitem(ingest._INGESTERS, "text", slow_text)
    results: dict[int, object] = {}

    def run(i: int) -> None:
        try:
            results[i] = ingest.ingest_path(files[i], workspace, tc)
        except Exception as e:  # noqa: BLE001 - the assertion below reports it
            results[i] = e

    threads = [threading.Thread(target=run, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(r.id for r in results.values() if isinstance(r, base.SourceRecord)) == ["notes", "notes-2", "notes-3", "notes-4", "notes-5", "notes-6"], results
    assert len(ingest.list_sources(workspace)) == 6 and workspace.read_text("sources/INDEX.md").count("- notes") == 6
    assert not base.in_flight_ids(workspace)
    (inbox / "dup.txt").write_text("the same bytes\n")
    outs: dict[str, base.SourceRecord] = {}

    def run_dup(k: str) -> None:
        outs[k] = ingest.ingest_path(inbox / "dup.txt", workspace, tc)

    pair = [threading.Thread(target=run_dup, args=(k,)) for k in ("a", "b")]
    for t in pair:
        t.start()
    for t in pair:
        t.join()
    assert {outs["a"].id, outs["b"].id} == {"dup"} and sum(1 for o in outs.values() if o.extra.get("already_present")) == 1  # the second waits for the first
    assert len(ingest.list_sources(workspace)) == 7 and not base.in_flight_ids(workspace)


def test_in_flight_reservation_guards_explicit_ids(workspace: Workspace, tc: Toolchain, inbox: Path) -> None:
    (inbox / "a.txt").write_text("a\n")
    with base.WORKSPACE_WRITE_LOCK:
        base.reserve_id(workspace, "busy", "deadbeef")
    try:
        assert base.in_flight_ids(workspace) == {"busy"}
        with pytest.raises(ToolFailure) as e:
            ingest.ingest_path(inbox / "a.txt", workspace, tc, source_id="busy")
        assert e.value.error.category.value == "invalid_input" and "already taken" in e.value.error.message
        with pytest.raises(ToolFailure) as e:
            ingest.ingest_path(inbox / "a.txt", workspace, tc, source_id="busy", replace=True)
        assert "another call" in e.value.error.message
        assert ingest.ingest_path(inbox / "a.txt", workspace, tc, title="busy").id == "busy-2"
    finally:
        base.release_id(workspace, "busy")
    assert not base.in_flight_ids(workspace) and ingest.get_source(workspace, "busy") is None


# ----------------------------------------------------------------------------- review additions: OCR and PyMuPDF paths through fakes
def fake_tesseract(bindir: Path, lines: list[str]) -> str:
    """A tesseract stand-in that prints ``lines`` for any image (the real one is absent here)."""
    bindir.mkdir(parents=True, exist_ok=True)
    exe = bindir / "tesseract"
    exe.write_text("#!/bin/sh\n" + "".join(f"echo '{ln}'\n" for ln in lines))
    exe.chmod(0o755)
    return str(exe)


@PDF_ENGINE
@pytest.mark.skipif(shutil.which("pdftoppm") is None, reason="pdftoppm not installed; PDF page rasterising for OCR is unverified here")
def test_pdf_ocr_path_with_a_fake_engine(workspace: Workspace, inbox: Path, tmp_path: Path) -> None:
    tc = Toolchain.detect()
    tc.binaries["tesseract"] = fake_tesseract(tmp_path / "ocr", ["OCR line one", "OCR line two"])
    (inbox / "scan.pdf").write_bytes(make_pdf(["Page one has real text here", None]))
    rec = ingest.ingest_path(inbox / "scan.pdf", workspace, tc)
    assert rec.coverage_status == "complete" and rec.coverage["detail"]["ocr"] is True and rec.coverage["detail"]["pages_failed"] == [] and rec.extra["missing_toolchains"] == []
    assert "\n[p.2]\nOCR line one\nOCR line two\n" in view(workspace, rec) and not any("OCR" in lim for lim in rec.limitations)
    hit = ingest.resolve_locator(workspace, "scan:p.2")
    assert hit["found"] and hit["text"] == "OCR line one\nOCR line two" and hit["coverage_note"] is None
    tc.binaries["tesseract"] = fake_tesseract(tmp_path / "silent", [])
    blank = ingest.ingest_path(inbox / "scan.pdf", workspace, tc, replace=True)
    assert blank.coverage_status == "partial" and blank.coverage["detail"]["pages_failed"] == [2] and blank.coverage["detail"]["ocr"] is True
    assert any("OCR found no text on pages 2" in lim for lim in blank.limitations) and "OCR produced nothing)" in view(workspace, blank)
    assert not ingest.resolve_locator(workspace, "scan:p.2")["found"]


def test_image_ocr_path_with_a_fake_engine(workspace: Workspace, inbox: Path, tmp_path: Path) -> None:
    tc = Toolchain.detect()
    tc.binaries["tesseract"] = fake_tesseract(tmp_path / "ocr", ["Slide title", "bullet one"])
    (inbox / "slide.png").write_bytes(make_png(4, 4))
    rec = ingest.ingest_path(inbox / "slide.png", workspace, tc)
    assert rec.coverage_status == "complete" and rec.coverage["detail"]["ocr"] is True and rec.extra["missing_toolchains"] == [] and rec.limitations == []
    assert view(workspace, rec).endswith("[p.1]\n(image: png, 4x4, raw copy at sources/raw/slide)\nSlide title\nbullet one\n")
    assert ingest.resolve_locator(workspace, "slide:p.1")["coverage_note"] is None
    exe = tmp_path / "failing" / "tesseract"
    exe.parent.mkdir()
    exe.write_text("#!/bin/sh\necho 'Error opening data file eng.traineddata' >&2\nexit 1\n")
    exe.chmod(0o755)
    tc.binaries["tesseract"] = str(exe)
    broken = ingest.ingest_path(inbox / "slide.png", workspace, tc, replace=True)
    assert broken.coverage_status == "unknown" and any("tesseract failed" in lim and "eng.traineddata" in lim for lim in broken.limitations)
    assert "(OCR failed; see limitations)" in view(workspace, broken)


@skip_unless("ffmpeg")
def test_keyframe_slide_text_with_a_fake_engine(workspace: Workspace, inbox: Path, tmp_path: Path) -> None:
    ffmpeg_video(inbox / "deck.mp4")
    tc = Toolchain.detect()
    tc.modules["faster_whisper"] = False
    tc.binaries["tesseract"] = fake_tesseract(tmp_path / "ocr", ["Agenda", "1. Bayes"])
    rec = ingest.ingest_path(inbox / "deck.mp4", workspace, tc)
    body = view(workspace, rec)
    assert rec.coverage["detail"]["ocr"] is True and rec.coverage_status == "unknown"  # slide text was read; what is said is still unknown
    assert "[t=00:00:00]\n(keyframe: sources/text/deck.frames/00-00-00.jpg)\nslide text: Agenda\nslide text: 1. Bayes\n" in body
    assert not any(m["name"] == "tesseract" for m in rec.extra["missing_toolchains"]) and any(m["name"] == "faster_whisper" for m in rec.extra["missing_toolchains"])
    hit = ingest.resolve_locator(workspace, "deck:t=0")
    assert hit["found"] and "slide text: Agenda" in hit["text"] and "no transcript" in hit["coverage_note"]


class _FakeFitz:
    """Enough of PyMuPDF's surface for the extraction path: ``open()`` gives a document with ``page_count``,
    ``metadata`` and pages whose ``get_text`` returns the text; ``fail`` makes ``open()`` raise like a damaged file."""

    def __init__(self, pages: list[str], title: str | None = "Meta Title", fail: Exception | None = None) -> None:
        self.pages, self.title, self.fail, self.opened = pages, title, fail, []

    def open(self, path: str) -> _FakeFitz:
        self.opened.append(path)
        if self.fail is not None:
            raise self.fail
        return self

    def __enter__(self) -> _FakeFitz:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def metadata(self) -> dict[str, str | None]:
        return {"title": self.title}

    def __getitem__(self, i: int) -> _FakeFitz:
        page = _FakeFitz([self.pages[i]])
        return page

    def get_text(self, kind: str) -> str:
        return self.pages[0]


def test_pymupdf_path_with_a_fake_module(workspace: Workspace, inbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeFitz(["Alpha page with enough text", "", "Gamma page with enough text"])
    monkeypatch.setattr(pdf, "_pymupdf", lambda: fake)
    tc = Toolchain.detect()
    tc.binaries["tesseract"] = None
    (inbox / "fz.pdf").write_bytes(make_pdf(["the fake module never reads this"]))
    rec = ingest.ingest_path(inbox / "fz.pdf", workspace, tc)
    assert rec.title == "Meta Title" and rec.extra["engine"] == "pymupdf" and rec.extra["pages_in_file"] == 3 and fake.opened == [str(inbox / "fz.pdf")]
    assert rec.coverage_status == "partial" and rec.coverage["detail"]["pages_failed"] == [2] and any("no OCR engine: 1 scanned pages unreadable (pages 2)" in lim for lim in rec.limitations)
    assert "[p.1]\nAlpha page with enough text\n" in view(workspace, rec) and not any("pdftotext" in lim for lim in rec.limitations)
    monkeypatch.setattr(pdf, "_pymupdf", lambda: _FakeFitz([], fail=RuntimeError("cannot open: encrypted")))
    with pytest.raises(ToolFailure) as e:
        ingest.ingest_path(inbox / "fz.pdf", workspace, tc, replace=True)
    assert e.value.error.category.value == "unsupported" and "encrypted" in e.value.error.message and "qpdf" in e.value.error.remediation
    assert ingest.get_source(workspace, "fz").extra["engine"] == "pymupdf" and not base.in_flight_ids(workspace)  # the failed replace left the record intact
