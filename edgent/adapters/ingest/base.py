"""The shared contract of source ingestion: records, text views, ids, sniffing, markers and coverage.

Every source becomes three things under ``sources/``: the raw file (``raw/<id>.<ext>``), a greppable text
view (``text/<id>.md``) and a ``SourceRecord`` (``text/<id>.meta.json``). The text view starts with an exact
two-line header and carries locator markers, each on its own line: ``[p.N]`` for a page and ``[t=HH:MM:SS]``
for a timestamp. Lines of plain text files are addressed by their line number in the view (``L<n>``), so no
line markers are written. ``INDEX.md`` is regenerated from the meta files and never edited by hand.

Coverage is a producer fact: an ingester declares what it covered and what it could not, so a grep miss on
an uncovered region is reported as unknown, never as absence.
"""
from __future__ import annotations

import os
import re
import shutil
import threading
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...domain.errors import ErrorCategory, Limit, ToolFailure
from ...domain.utils import ensure_dir, read_json, slugify, write_json
from ...domain.workspace import FILES, LAYOUT, Workspace

SOURCE_KINDS: tuple[str, ...] = ("pdf", "text", "markdown", "code", "video", "audio", "image", "web", "office")
COVERAGE_STATUSES: tuple[str, ...] = ("complete", "partial", "unknown")
TEXT_VIEW_EXT = ".md"
META_EXT = ".meta.json"
FRAMES_DIR_EXT = ".frames"
SNIFF_BYTES = 8192
ID_MAX_CHARS = 60
ID_SUFFIXES_LIMIT = Limit("source id collision suffixes", 1000, "ids")
COMMAND_OUTPUT_BYTES_LIMIT = Limit("ingest command output", 200_000_000, "bytes")
LOCATOR_FORMS = "<source-id>:p.<page>, <source-id>:t=HH:MM:SS (or MM:SS) or <source-id>:L<line>"
INDEX_HEADER = "# Sources\n\n<!-- one line per source: id | kind | title | coverage | text view -->\n"
INDEX_EMPTY = "_(no sources yet: add one with sources_add or sources_add_url)_\n"
SUPPORTED_INPUTS = (
    "PDF; text, Markdown, HTML and source code; docx, pptx, epub, odt and rtf; PNG, JPEG, GIF, WEBP, BMP and TIFF images; "
    "MP4, MOV, MKV, WEBM and AVI video; MP3, WAV, M4A, OGG and FLAC audio; SRT and VTT subtitles; http(s) URLs via sources_add_url"
)

# One process-wide lock serialises the commit phase of every ingest (id allocation, raw copy, view, meta, index), so
# tools running in parallel never share an id or write a stale INDEX.md. An ingest reserves its id before the slow
# work starts; the reservation lives in ``_IN_FLIGHT`` until its files exist on disk.
WORKSPACE_WRITE_LOCK = threading.RLock()
_INGEST_DONE = threading.Condition(WORKSPACE_WRITE_LOCK)
_IN_FLIGHT: dict[tuple[Path, str], str] = {}  # (workspace root, source id) -> sha256 of the input being ingested

PAGE_MARKER_RE = re.compile(r"^\[p\.(\d+)\]\s*$")
TIME_MARKER_RE = re.compile(r"^\[t=(\d{2,}):(\d{2}):(\d{2})\]\s*$")
ID_RE = re.compile(rf"^[a-z0-9][a-z0-9-]{{0,{ID_MAX_CHARS - 1}}}$")
_LOCATOR_PAGE_RE = re.compile(r"^p\.?\s*(\d+)$", re.I)
_LOCATOR_LINE_RE = re.compile(r"^L\s*(\d+)(?:\s*-\s*L?\s*(\d+))?$", re.I)
_LOCATOR_TIME_RE = re.compile(r"^t\s*=\s*(.+)$", re.I)

_MARKDOWN_EXTENSIONS = {".md", ".markdown", ".mdx", ".mkd", ".mdown"}
_HTML_EXTENSIONS = {".html", ".htm", ".xhtml"}
_UTF16_BOMS = (b"\xff\xfe", b"\xfe\xff")
_SUBTITLE_EXTENSIONS = {".srt", ".vtt"}
_TEXT_EXTENSIONS = {".txt", ".text", ".rst", ".csv", ".tsv", ".log", ".org", ".adoc", ".asciidoc", ".nfo", ".diff", ".patch"}
_VIDEO_EXTENSIONS = {".mp4": "mp4", ".m4v": "mp4", ".mov": "mov", ".mkv": "mkv", ".webm": "webm", ".avi": "avi", ".mpg": "mpeg", ".mpeg": "mpeg", ".wmv": "wmv", ".flv": "flv"}
_AUDIO_EXTENSIONS = {".mp3": "mp3", ".wav": "wav", ".m4a": "m4a", ".aac": "aac", ".ogg": "ogg", ".oga": "ogg", ".opus": "opus", ".flac": "flac", ".wma": "wma", ".aiff": "aiff", ".aif": "aiff"}
_IMAGE_EXTENSIONS = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".gif": "gif", ".webp": "webp", ".bmp": "bmp", ".tif": "tiff", ".tiff": "tiff", ".heic": "heic"}
_OFFICE_EXTENSIONS = {".docx": "docx", ".pptx": "pptx", ".epub": "epub", ".odt": "odt", ".rtf": "rtf"}
_FORMAT_EXT = {
    "pdf": ".pdf", "png": ".png", "jpeg": ".jpg", "gif": ".gif", "webp": ".webp", "bmp": ".bmp", "tiff": ".tif", "heic": ".heic",
    "wav": ".wav", "mp3": ".mp3", "flac": ".flac", "ogg": ".ogg", "m4a": ".m4a", "aac": ".aac", "opus": ".opus", "wma": ".wma", "aiff": ".aiff",
    "mp4": ".mp4", "mov": ".mov", "mkv": ".mkv", "webm": ".webm", "avi": ".avi", "mpeg": ".mpg", "wmv": ".wmv", "flv": ".flv",
    "docx": ".docx", "pptx": ".pptx", "epub": ".epub", "odt": ".odt", "rtf": ".rtf", "markdown": ".md", "html": ".html", "json": ".json",
}
_CODE_LANGUAGES: dict[str, str] = {
    ".py": "python", ".pyi": "python", ".pyx": "cython", ".ipynb": "json", ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "tsx", ".jsx": "jsx", ".json": "json", ".jsonl": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".ini": "ini", ".cfg": "ini", ".conf": "ini", ".env": "bash", ".sh": "bash", ".bash": "bash", ".zsh": "bash", ".fish": "fish",
    ".ps1": "powershell", ".bat": "batch", ".cmd": "batch", ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    ".hh": "cpp", ".cs": "csharp", ".java": "java", ".kt": "kotlin", ".kts": "kotlin", ".scala": "scala", ".go": "go", ".rs": "rust",
    ".rb": "ruby", ".php": "php", ".swift": "swift", ".m": "objectivec", ".mm": "objectivec", ".dart": "dart", ".html": "html", ".htm": "html",
    ".xhtml": "html", ".xml": "xml", ".svg": "xml", ".xsl": "xml", ".css": "css", ".scss": "scss", ".sass": "sass", ".less": "less",
    ".sql": "sql", ".r": "r", ".jl": "julia", ".lua": "lua", ".pl": "perl", ".pm": "perl", ".hs": "haskell", ".ml": "ocaml", ".mli": "ocaml",
    ".ex": "elixir", ".exs": "elixir", ".erl": "erlang", ".clj": "clojure", ".cljs": "clojure", ".lisp": "lisp", ".el": "lisp", ".scm": "scheme",
    ".rkt": "racket", ".tex": "latex", ".sty": "latex", ".cls": "latex", ".bib": "bibtex", ".proto": "protobuf", ".graphql": "graphql",
    ".gql": "graphql", ".cmake": "cmake", ".vue": "vue", ".svelte": "svelte", ".zig": "zig", ".nim": "nim", ".v": "verilog", ".sv": "verilog",
    ".vhd": "vhdl", ".vhdl": "vhdl", ".asm": "asm", ".s": "asm", ".f90": "fortran", ".f95": "fortran", ".f": "fortran", ".for": "fortran",
    ".mat": "matlab", ".sol": "solidity", ".tf": "hcl", ".hcl": "hcl", ".nix": "nix", ".dockerfile": "dockerfile", ".groovy": "groovy",
    ".gradle": "groovy", ".awk": "awk", ".sed": "sed", ".vim": "vim", ".ex2": "elixir",
}
_CODE_FILENAMES = {"dockerfile": "dockerfile", "makefile": "makefile", "gnumakefile": "makefile", "cmakelists.txt": "cmake", "justfile": "makefile", "rakefile": "ruby", "gemfile": "ruby"}


# ----------------------------------------------------------------------------- records
def coverage(status: str, *, pages_total: int | None = None, pages_failed: list[int] | None = None, minutes_total: float | None = None,
             minutes_transcribed: float | None = None, ocr: bool = False, transcriber: str | None = None) -> dict[str, Any]:
    """A coverage record with every detail key present, so consumers never guess which keys exist."""
    if status not in COVERAGE_STATUSES:
        raise ValueError(f"coverage status must be one of {COVERAGE_STATUSES}; got {status!r}")
    return {
        "status": status,
        "detail": {
            "pages_total": pages_total,
            "pages_failed": sorted(set(pages_failed or [])),
            "minutes_total": minutes_total,
            "minutes_transcribed": minutes_transcribed,
            "ocr": bool(ocr),
            "transcriber": transcriber,
        },
    }


def coverage_complete(**detail: Any) -> dict[str, Any]:
    return coverage("complete", **detail)


def coverage_partial(**detail: Any) -> dict[str, Any]:
    return coverage("partial", **detail)


def coverage_unknown(**detail: Any) -> dict[str, Any]:
    return coverage("unknown", **detail)


def coverage_status(cov: dict[str, Any] | None) -> str:
    status = (cov or {}).get("status")
    return status if status in COVERAGE_STATUSES else "unknown"


@dataclass
class SourceRecord:
    """What ``sources/text/<id>.meta.json`` holds. ``coverage`` follows ``coverage()``; ``origin`` is the input path or URL."""

    id: str
    kind: str
    title: str
    raw_path: str
    text_path: str
    sha256: str
    size_bytes: int
    coverage: dict[str, Any] = field(default_factory=coverage_unknown)
    limitations: list[str] = field(default_factory=list)
    created_at: str = ""
    origin: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def meta_path(self) -> str:
        return f"{LAYOUT['sources_text']}/{self.id}{META_EXT}"

    @property
    def frames_dir(self) -> str:
        return f"{LAYOUT['sources_text']}/{self.id}{FRAMES_DIR_EXT}"

    @property
    def coverage_status(self) -> str:
        return coverage_status(self.coverage)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "raw_path": self.raw_path,
            "text_path": self.text_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "coverage": self.coverage,
            "limitations": list(self.limitations),
            "created_at": self.created_at,
            "origin": self.origin,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SourceRecord:
        cov = d.get("coverage")
        if not isinstance(cov, dict) or "status" not in cov:
            cov = coverage_unknown()
        else:
            cov = coverage(coverage_status(cov), **{k: v for k, v in (cov.get("detail") or {}).items() if k in coverage_unknown()["detail"]})
        return cls(
            id=str(d["id"]),
            kind=str(d.get("kind", "text")),
            title=str(d.get("title") or d["id"]),
            raw_path=str(d.get("raw_path", "")),
            text_path=str(d.get("text_path", f"{LAYOUT['sources_text']}/{d['id']}{TEXT_VIEW_EXT}")),
            sha256=str(d.get("sha256", "")),
            size_bytes=int(d.get("size_bytes", 0) or 0),
            coverage=cov,
            limitations=[str(x) for x in d.get("limitations") or []],
            created_at=str(d.get("created_at", "")),
            origin=str(d.get("origin", "")),
            extra=dict(d.get("extra") or {}),
        )


@dataclass
class IngestResult:
    """What one ingester hands back; ``ingest_path`` turns it into the record, the view and the index line."""

    kind: str
    body: str
    title: str | None = None
    coverage: dict[str, Any] = field(default_factory=coverage_complete)
    limitations: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    missing: list[dict[str, str]] = field(default_factory=list)  # toolchains whose absence limited this view: {name, purpose, hint}
    usable: bool = True  # False when nothing text-bearing or viewable was derived (the tool then reports 'unavailable')


def missing_tool(name: str, purpose: str) -> dict[str, str]:
    from ..toolchain import INSTALL_HINTS

    return {"name": name, "purpose": purpose, "hint": INSTALL_HINTS.get(name, f"Install {name}.")}


# ----------------------------------------------------------------------------- sniffing
@dataclass(frozen=True)
class Sniff:
    kind: str  # a SOURCE_KINDS entry, or 'binary' when nothing here can read it
    format: str  # pdf, png, mp4, docx, python, markdown, subtitles, ...
    mime: str = "application/octet-stream"
    language: str | None = None  # fenced-block language for code


def sniff_bytes(head: bytes, ext: str = "") -> Sniff | None:
    """Classify by magic bytes alone; None when the bytes carry no signature."""
    if head.startswith(b"%PDF"):
        return Sniff("pdf", "pdf", "application/pdf")
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return Sniff("image", "png", "image/png")
    if head.startswith(b"\xff\xd8\xff"):
        return Sniff("image", "jpeg", "image/jpeg")
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return Sniff("image", "gif", "image/gif")
    if head[:4] == b"RIFF" and len(head) >= 12:
        tag = head[8:12]
        if tag == b"WEBP":
            return Sniff("image", "webp", "image/webp")
        if tag == b"WAVE":
            return Sniff("audio", "wav", "audio/wav")
        if tag == b"AVI ":
            return Sniff("video", "avi", "video/x-msvideo")
    if head.startswith(b"ID3"):
        return Sniff("audio", "mp3", "audio/mpeg")
    if head.startswith(b"fLaC"):
        return Sniff("audio", "flac", "audio/flac")
    if head.startswith(b"OggS"):
        return Sniff("audio", "ogg", "audio/ogg")
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in (b"M4A ", b"M4B ", b"M4P "):
            return Sniff("audio", "m4a", "audio/mp4")
        if brand == b"qt  ":
            return Sniff("video", "mov", "video/quicktime")
        if brand in (b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1"):
            return Sniff("image", "heic", "image/heic")
        return Sniff("video", "mp4", "video/mp4")
    if head.startswith(b"\x1aE\xdf\xa3"):
        return Sniff("video", "webm", "video/webm") if b"webm" in head[:64] else Sniff("video", "mkv", "video/x-matroska")
    if head.startswith(b"{\\rtf"):
        return Sniff("office", "rtf", "application/rtf")
    if head.startswith(b"BM") and ext == ".bmp":
        return Sniff("image", "bmp", "image/bmp")
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return Sniff("image", "tiff", "image/tiff")
    if head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2") and ext in ("", ".mp3"):
        return Sniff("audio", "mp3", "audio/mpeg")
    return None


def _sniff_zip(path: Path) -> Sniff:
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            mimetype = z.read("mimetype").strip() if "mimetype" in names else b""
    except (zipfile.BadZipFile, OSError, KeyError):
        return Sniff("binary", "zip", "application/zip")
    if mimetype == b"application/epub+zip" or "META-INF/container.xml" in names:
        return Sniff("office", "epub", "application/epub+zip")
    if mimetype.startswith(b"application/vnd.oasis.opendocument.text"):
        return Sniff("office", "odt", "application/vnd.oasis.opendocument.text")
    if "word/document.xml" in names:
        return Sniff("office", "docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    if "ppt/presentation.xml" in names:
        return Sniff("office", "pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation")
    if "xl/workbook.xml" in names:
        return Sniff("binary", "xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    return Sniff("binary", "zip", "application/zip")


def sniff(path: str | os.PathLike[str]) -> Sniff:
    """Classify a file by magic bytes first, then by extension, then by whether it looks like text."""
    p = Path(path)
    with open(p, "rb") as f:
        head = f.read(SNIFF_BYTES)
    ext = p.suffix.lower()
    name = p.name.lower()
    if head.startswith(b"PK\x03\x04"):
        return _sniff_zip(p)
    by_magic = sniff_bytes(head, ext)
    if by_magic is not None:
        return by_magic
    if ext in _MARKDOWN_EXTENSIONS:
        return Sniff("markdown", "markdown", "text/markdown")
    if ext in _SUBTITLE_EXTENSIONS:
        return Sniff("text", "subtitles", "text/plain")
    if ext in _HTML_EXTENSIONS:
        return Sniff("web", "html", "text/html")
    if ext in _TEXT_EXTENSIONS or head.startswith(_UTF16_BOMS):  # UTF-16 text is full of NUL bytes; it is still text
        return Sniff("text", ext.lstrip(".") or "txt", "text/plain")
    language = _CODE_LANGUAGES.get(ext) or _CODE_FILENAMES.get(name)
    if language:
        return Sniff("code", ext.lstrip(".") or name, "text/plain", language)
    if ext == ".pdf":
        return Sniff("pdf", "pdf", "application/pdf")
    if ext in _VIDEO_EXTENSIONS:
        return Sniff("video", _VIDEO_EXTENSIONS[ext], "video/*")
    if ext in _AUDIO_EXTENSIONS:
        return Sniff("audio", _AUDIO_EXTENSIONS[ext], "audio/*")
    if ext in _IMAGE_EXTENSIONS:
        return Sniff("image", _IMAGE_EXTENSIONS[ext], "image/*")
    if ext in _OFFICE_EXTENSIONS:
        return Sniff("office", _OFFICE_EXTENSIONS[ext], "application/octet-stream")
    if b"\x00" in head:
        return Sniff("binary", ext.lstrip(".") or "unknown", "application/octet-stream")
    return Sniff("text", ext.lstrip(".") or "txt", "text/plain")


def sniff_kind(path: str | os.PathLike[str]) -> str:
    return sniff(path).kind


def raw_extension(info: Sniff, path: Path) -> str:
    """The extension the raw copy gets: canonical for sniffed formats, the original one for text."""
    ext = _FORMAT_EXT.get(info.format)
    if ext:
        return ext
    suffix = path.suffix.lower()
    if suffix and re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
        return suffix
    return ".txt" if info.kind in ("text", "markdown", "code") else ".bin"


# ----------------------------------------------------------------------------- ids, titles, paths
def clean_title(title: str | None, fallback: str = "Untitled") -> str:
    t = " ".join(str(title or "").replace("|", "/").split())
    return t[:200] if t else fallback


def title_from_stem(stem: str) -> str:
    return clean_title(re.sub(r"[-_]+", " ", stem).strip(), "Untitled")


def text_view_rel(source_id: str) -> str:
    return f"{LAYOUT['sources_text']}/{source_id}{TEXT_VIEW_EXT}"


def meta_rel(source_id: str) -> str:
    return f"{LAYOUT['sources_text']}/{source_id}{META_EXT}"


def frames_rel(source_id: str) -> str:
    return f"{LAYOUT['sources_text']}/{source_id}{FRAMES_DIR_EXT}"


def id_taken(workspace: Workspace, source_id: str) -> bool:
    text_dir = workspace.path("sources_text")
    if (text_dir / f"{source_id}{TEXT_VIEW_EXT}").exists() or (text_dir / f"{source_id}{META_EXT}").exists():
        return True
    return any(workspace.path("sources_raw").glob(f"{source_id}.*"))


def allocate_id(workspace: Workspace, title_or_stem: str, reserved: set[str] | None = None) -> str:
    """Slug of the title or file stem, unique within the workspace: ``-2``, ``-3`` ... on collision."""
    base = slugify(title_or_stem, ID_MAX_CHARS) or "source"
    if base[0] == "-" or not ID_RE.match(base):
        base = "source"
    candidate = base
    n = 1
    while id_taken(workspace, candidate) or (reserved and candidate in reserved):
        n += 1
        if n > ID_SUFFIXES_LIMIT.value:
            err = ID_SUFFIXES_LIMIT.exceeded(n, f"Give sources_add a distinct title; {base!r} already has {int(ID_SUFFIXES_LIMIT.value)} variants.")
            raise ToolFailure(err.code, err.message, err.category, remediation=err.remediation)
        suffix = f"-{n}"
        candidate = f"{base[: ID_MAX_CHARS - len(suffix)].rstrip('-')}{suffix}"
    return candidate


def validate_id(source_id: str) -> str:
    sid = str(source_id).strip()
    if not ID_RE.match(sid):
        raise ToolFailure(
            "invalid_input",
            f"source id {source_id!r} must be lower-case letters, digits and hyphens, at most {ID_MAX_CHARS} characters, starting with a letter or digit.",
            ErrorCategory.invalid_input,
            remediation="Omit source_id to derive one from the title or file name, or give a slug such as 'lecture-03'.",
            source_id=source_id,
        )
    return sid


def in_flight_ids(workspace: Workspace) -> set[str]:
    """Ids reserved by ingests of this workspace that are still running."""
    return {sid for root, sid in _IN_FLIGHT if root == workspace.root}


def reserve_id(workspace: Workspace, source_id: str, sha256: str) -> None:
    """Hold ``source_id`` for an ingest in progress. Call with ``WORKSPACE_WRITE_LOCK`` held."""
    key = (workspace.root, source_id)
    if key in _IN_FLIGHT:
        raise ToolFailure(
            "invalid_input",
            f"source id {source_id!r} is being added by another call right now.",
            ErrorCategory.invalid_input,
            remediation="Wait for that call to finish, then check sources_list before adding it again.",
            source_id=source_id,
        )
    _IN_FLIGHT[key] = sha256


def release_id(workspace: Workspace, source_id: str) -> None:
    with _INGEST_DONE:
        _IN_FLIGHT.pop((workspace.root, source_id), None)
        _INGEST_DONE.notify_all()


def wait_for_in_flight_sha(workspace: Workspace, sha256: str) -> None:
    """Block while a parallel ingest of the same content is running, so the dedupe check can see its record.
    Call with ``WORKSPACE_WRITE_LOCK`` held; the lock is released while waiting."""
    while any(root == workspace.root and sha == sha256 for (root, _), sha in _IN_FLIGHT.items()):
        _INGEST_DONE.wait()


def view_header(record: SourceRecord) -> str:
    return f"# {clean_title(record.title)}\nsource: {record.id} | kind: {record.kind} | coverage: {record.coverage_status}\n\n"


def write_text_view(workspace: Workspace, record: SourceRecord, body: str) -> Path:
    """Header, blank line, body. Sets ``record.text_path``. Coverage must be final before this is called."""
    record.title = clean_title(record.title)
    text = view_header(record) + body.strip("\n") + "\n"
    rel = text_view_rel(record.id)
    p = workspace.write_text(rel, text)
    record.text_path = rel
    return p


def write_meta(workspace: Workspace, record: SourceRecord) -> Path:
    p = workspace.resolve(meta_rel(record.id))
    write_json(p, record.to_dict())
    return p


def read_record(workspace: Workspace, source_id: str) -> SourceRecord | None:
    if not ID_RE.match(str(source_id)):
        return None
    try:
        data = read_json(workspace.resolve(meta_rel(source_id)))
    except Exception:  # noqa: BLE001 - a malformed meta file is reported as absent, not as a crash
        return None
    if not isinstance(data, dict) or "id" not in data:
        return None
    try:
        return SourceRecord.from_dict(data)
    except (KeyError, TypeError, ValueError):
        return None


def all_records(workspace: Workspace) -> list[SourceRecord]:
    out: list[SourceRecord] = []
    for p in sorted(workspace.path("sources_text").glob(f"*{META_EXT}")):
        rec = read_record(workspace, p.name[: -len(META_EXT)])
        if rec is not None:
            out.append(rec)
    return out


def find_by_sha256(workspace: Workspace, sha256: str) -> SourceRecord | None:
    for rec in all_records(workspace):
        if rec.sha256 == sha256:
            return rec
    return None


def find_by_origin(workspace: Workspace, origin: str | None) -> SourceRecord | None:
    """The record ingested from this path or URL, so ``replace`` re-ingests it in place even when the content changed."""
    if not origin:
        return None
    for rec in all_records(workspace):
        if rec.origin == origin or rec.extra.get("url") == origin:
            return rec
    return None


def regenerate_index(workspace: Workspace) -> int:
    """Rewrite ``sources/INDEX.md`` from the meta files: one line per source, sorted by id."""
    with WORKSPACE_WRITE_LOCK:  # re-entrant: an ingest's commit phase already holds it
        records = all_records(workspace)
        lines = [INDEX_HEADER]
        if not records:
            lines.append(INDEX_EMPTY)
        for r in records:
            lines.append(f"- {r.id} | {r.kind} | {clean_title(r.title)} | coverage {r.coverage_status} | {r.text_path}\n")
        workspace.write_text(FILES["index"], "".join(lines))
    return len(records)


def place_raw(workspace: Workspace, source_id: str, ext: str, *, src: Path | None = None, data: bytes | None = None) -> str:
    """Copy a file (or write bytes) to ``sources/raw/<id><ext>``; returns the workspace-relative path."""
    rel = f"{LAYOUT['sources_raw']}/{source_id}{ext}"
    dest = workspace.resolve(rel)
    ensure_dir(dest.parent)
    for stale in dest.parent.glob(f"{source_id}.*"):
        if stale != dest and stale.is_file() and (src is None or stale.resolve() != src.resolve()):
            stale.unlink()
    if data is not None:
        tmp = dest.with_name(f"{dest.name}.tmp.{os.getpid()}")
        tmp.write_bytes(data)
        os.replace(tmp, dest)
    elif src is not None and src.resolve() != dest.resolve():
        shutil.copyfile(src, dest)
    return rel


def remove_source_files(workspace: Workspace, source_id: str) -> None:
    """Delete a source's view, meta, frames and raw copies (used by replace)."""
    text_dir = workspace.path("sources_text")
    for p in (text_dir / f"{source_id}{TEXT_VIEW_EXT}", text_dir / f"{source_id}{META_EXT}"):
        if p.exists():
            p.unlink()
    frames = text_dir / f"{source_id}{FRAMES_DIR_EXT}"
    if frames.is_dir():
        shutil.rmtree(frames, ignore_errors=True)
    for raw in workspace.path("sources_raw").glob(f"{source_id}.*"):
        if raw.is_file():
            raw.unlink()


# ----------------------------------------------------------------------------- markers and locators
def hms(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def parse_hms(text: str) -> int | None:
    """``HH:MM:SS``, ``MM:SS`` or ``SS`` (fractions allowed) to whole seconds; None when malformed."""
    parts = text.strip().split(":")
    if not 1 <= len(parts) <= 3 or not all(parts):
        return None
    try:
        last = float(parts[-1].replace(",", "."))
        fields = [int(x) for x in parts[:-1]]
    except ValueError:
        return None
    if last < 0 or any(f < 0 for f in fields) or (len(parts) > 1 and last >= 60) or (len(parts) == 3 and fields[1] >= 60):
        return None
    total = int(last)
    for i, f in enumerate(reversed(fields)):
        total += f * (60 if i == 0 else 3600)
    return total


def page_marker(n: int) -> str:
    return f"[p.{int(n)}]"


def time_marker(seconds: float) -> str:
    return f"[t={hms(seconds)}]"


@dataclass(frozen=True)
class Locator:
    source_id: str
    kind: str  # page | time | line
    value: int  # page number, seconds, or line number
    ref: str  # what the citation shows: '12', '00:14:32', '12'
    raw: str

    @property
    def marker(self) -> str:
        if self.kind == "page":
            return page_marker(self.value)
        if self.kind == "time":
            return time_marker(self.value)
        return f"L{self.value}"


class LocatorError(ValueError):
    pass


def parse_locator(locator: str) -> Locator:
    """``id:p.12``, ``id:t=00:14:32`` (also ``t=14:32``), ``id:L12``. Raises LocatorError with the accepted forms."""
    raw = str(locator or "").strip()
    sid, sep, rest = raw.partition(":")
    sid = sid.strip().lower()
    rest = rest.strip()
    if not sep or not sid or not rest:
        raise LocatorError(f"locator {raw!r} must look like {LOCATOR_FORMS}")
    if not ID_RE.match(sid):
        raise LocatorError(f"locator {raw!r} has an invalid source id {sid!r} (lower-case letters, digits and hyphens)")
    m = _LOCATOR_PAGE_RE.match(rest)
    if m:
        return Locator(sid, "page", int(m.group(1)), m.group(1), raw)
    m = _LOCATOR_LINE_RE.match(rest)
    if m:
        return Locator(sid, "line", int(m.group(1)), m.group(1), raw)
    m = _LOCATOR_TIME_RE.match(rest)
    if m:
        secs = parse_hms(m.group(1))
        if secs is None:
            raise LocatorError(f"locator {raw!r} has a malformed timestamp {m.group(1)!r}; use t=HH:MM:SS or t=MM:SS")
        return Locator(sid, "time", secs, hms(secs), raw)
    raise LocatorError(f"locator {raw!r} has an unknown form {rest!r}; use {LOCATOR_FORMS}")


# ----------------------------------------------------------------------------- subprocesses
@dataclass
class CommandResult:
    returncode: int
    stdout: bytes
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


COMMAND_MEMORY_BYTES = 2 << 30  # memory ceiling for a toolchain run on an untrusted file (bytes): ffmpeg and pandoc on large inputs
COMMAND_PASSTHROUGH_ENV = ("TESSDATA_PREFIX", "FONTCONFIG_PATH", "LANG", "LC_ALL")  # what the toolchains need from the environment


def run_command(argv: list[str], timeout_s: float, cwd: str | os.PathLike[str] | None = None) -> CommandResult:
    """Run a toolchain binary on untrusted bytes inside the sandbox: wall-clock and CPU limits, a memory ceiling, no
    network, a scrubbed environment. A missing binary is exit 127, never an exception."""
    from ..sandbox.runner import run_subprocess

    env_extra = {k: os.environ[k] for k in COMMAND_PASSTHROUGH_ENV if k in os.environ}
    try:
        r = run_subprocess(
            argv,
            cwd=str(cwd) if cwd is not None else os.getcwd(),
            timeout_s=timeout_s,
            cpu_seconds=max(1, int(timeout_s)),
            memory_bytes=COMMAND_MEMORY_BYTES,
            allow_network=False,
            env_extra=env_extra,
            max_output=int(COMMAND_OUTPUT_BYTES_LIMIT.value),
        )
    except OSError as e:  # the sandbox reports a missing or refused executable as exit 127/126; this is the runner itself failing
        return CommandResult(126, b"", str(e))
    stderr = r.stderr + (f"\n[timed out after {timeout_s:g} s]" if r.timed_out else "") + ("\n[stdout truncated]" if r.truncated else "")
    return CommandResult(-9 if r.timed_out else r.returncode, r.stdout.encode("utf-8", "replace"), stderr, timed_out=r.timed_out)
