"""The sources tools through the registry: contracts, add, add_url, list, cite check with evidence, coverage, reindex."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from edgent.adapters.toolchain import Toolchain
from edgent.domain.evidence import Evidence
from edgent.tools.edu import sources as sources_mod
from edgent.tools.edu.sources import SOURCES_TOOL_NAMES, build_sources_registry
from edgent.tools.registry import ReadOnly, ToolContext, ToolRegistry
from tests.conftest import skip_unless
from tests.test_ingest import HTML, PDF_ENGINE, ffmpeg_video, ffmpeg_wav, make_pdf, make_png


@pytest.fixture
def tools(registry: ToolRegistry) -> ToolRegistry:
    return build_sources_registry(registry)


@pytest.fixture
def inbox(tmp_path: Path) -> Path:
    d = tmp_path / "inbox"
    d.mkdir()
    return d


def call(tools: ToolRegistry, ctx: ToolContext, tool: str, **args: Any) -> Any:
    return tools.call(tool, args, ctx)


def ok(tools: ToolRegistry, ctx: ToolContext, tool: str, **args: Any) -> dict[str, Any]:
    r = call(tools, ctx, tool, **args)
    assert r.ok, r.error
    return r.output


def err(tools: ToolRegistry, ctx: ToolContext, tool: str, **args: Any) -> Any:
    r = call(tools, ctx, tool, **args)
    assert not r.ok, r.output
    assert r.error.remediation
    return r


# ----------------------------------------------------------------------------- contracts
def test_registry_contents_and_contracts(tools: ToolRegistry) -> None:
    assert tools.names() == sorted(SOURCES_TOOL_NAMES) and len(SOURCES_TOOL_NAMES) == 6
    expected = {"sources_add": "write", "sources_add_url": "network", "sources_list": "read", "sources_cite_check": "read", "sources_coverage": "read", "sources_reindex": "write"}
    for name, permission in expected.items():
        spec = tools.get(name)
        assert spec is not None and spec.permission == permission and spec.group == "sources", name
        assert spec.parameters.get("additionalProperties") is False and spec.when_not_to_use and len(spec.description) > 80, name
        assert spec.full_description().count("Do not use when") == 1
        for prop, schema in spec.parameters["properties"].items():
            assert schema.get("description"), f"{name}.{prop} has no description"
    assert tools.get("sources_add").side_effects and tools.get("sources_add_url").side_effects
    assert not tools.get("sources_cite_check").side_effects and not tools.get("sources_list").side_effects
    assert not tools.get("sources_reindex").side_effects and tools.get("sources_reindex").idempotent  # never replayed from the idempotency store
    assert tools.get("sources_add").timeout_s >= 3600 and tools.get("sources_add").cost_class == "expensive"
    assert tools.get("sources_cite_check").max_inline_chars > sources_mod.CITE_WINDOW_LIMIT.value
    assert tools.schema_bytes(compact=True) < 6_000
    again = build_sources_registry(tools)
    assert again is tools and len(tools.names()) == 6
    fresh = build_sources_registry()
    assert fresh.names() == sorted(SOURCES_TOOL_NAMES)
    manifest = fresh.mcp_manifest()["tools"]
    assert {t["name"]: t["annotations"]["x-permission"] for t in manifest}["sources_add_url"] == "network"


def test_read_only_policy_denies_writes(tools: ToolRegistry, ctx: ToolContext, inbox: Path) -> None:
    (inbox / "a.txt").write_text("a\n")
    ctx.permissions = ReadOnly()
    r = err(tools, ctx, "sources_add", path=str(inbox / "a.txt"))
    assert r.error.category.value == "permission_denied"
    assert ok(tools, ctx, "sources_list")["count"] == 0


# ----------------------------------------------------------------------------- add
def test_sources_add_markdown(tools: ToolRegistry, ctx: ToolContext, workspace: Any, inbox: Path) -> None:
    (inbox / "Bayes Notes.md").write_text("# Bayes' theorem\n\n" + "Posterior is prior times likelihood over evidence. " * 20 + "\n", encoding="utf-8")
    out = ok(tools, ctx, "sources_add", path=str(inbox / "Bayes Notes.md"))
    assert out["id"] == "bayes-notes" and out["kind"] == "markdown" and out["title"] == "Bayes' theorem" and out["coverage"]["status"] == "complete"
    assert out["text_path"] == "sources/text/bayes-notes.md" and out["raw_path"] == "sources/raw/bayes-notes.md" and out["untrusted"] is True
    assert len(out["preview"]) == sources_mod.PREVIEW_CHARS and out["preview"].startswith("# Bayes' theorem\nsource: bayes-notes | kind: markdown | coverage: complete\n\n")
    assert out["already_present"] is False and "missing_toolchains" not in out and "_attach" not in out
    assert "- bayes-notes | markdown | Bayes' theorem | coverage complete | sources/text/bayes-notes.md" in workspace.read_text("sources/INDEX.md")
    again = ok(tools, ctx, "sources_add", path=str(inbox / "Bayes Notes.md"), title="ignored for duplicates")
    assert again["already_present"] is True and again["id"] == "bayes-notes"
    titled = ok(tools, ctx, "sources_add", path="sources/raw/bayes-notes.md", title="Second Copy", replace=False, source_id="copy-2")
    assert titled["id"] == "copy-2" and titled["title"] == "Second Copy"
    listing = ok(tools, ctx, "sources_list")
    assert listing["count"] == 2 and listing["sources"][0] == {"id": "bayes-notes", "kind": "markdown", "title": "Bayes' theorem", "coverage": "complete", "size_bytes": (inbox / "Bayes Notes.md").stat().st_size, "text_path": "sources/text/bayes-notes.md", "limitations": 0}
    assert ok(tools, ctx, "sources_list", kind="pdf")["count"] == 0 and "no sources of kind" in ok(tools, ctx, "sources_list", kind="pdf")["hint"]


def test_sources_add_errors(tools: ToolRegistry, ctx: ToolContext, inbox: Path) -> None:
    r = err(tools, ctx, "sources_add", path=str(inbox / "missing.pdf"))
    assert r.error.category.value == "not_found" and "sources_add_url" in r.error.remediation
    assert err(tools, ctx, "sources_add", path=str(inbox)).error.category.value == "invalid_input"
    (inbox / "blob.bin").write_bytes(b"\x00\x01" * 100)
    r = err(tools, ctx, "sources_add", path=str(inbox / "blob.bin"))
    assert r.error.category.value == "unsupported" and "supported inputs" in r.error.remediation
    assert err(tools, ctx, "sources_add", path=str(inbox / "blob.bin"), source_id="Bad Id").error.category.value == "invalid_input"
    assert err(tools, ctx, "sources_add", path=str(inbox / "blob.bin"), extra=1).error.category.value == "invalid_input"
    assert err(tools, ctx, "sources_add").error.category.value == "invalid_input"


def test_sources_add_image_attaches_and_reports_missing_ocr(tools: ToolRegistry, ctx: ToolContext, workspace: Any, inbox: Path) -> None:
    (inbox / "diagram.png").write_bytes(make_png(12, 8))
    bare = Toolchain.detect()
    bare.binaries["tesseract"] = None
    ctx.toolchain = bare
    r = call(tools, ctx, "sources_add", path=str(inbox / "diagram.png"))
    assert r.ok and r.output["kind"] == "image" and r.output["coverage"]["status"] == "unknown" and r.output["extra"]["dimensions"] == [12, 8]
    assert r.output["missing_toolchains"][0]["name"] == "tesseract" and "replace=true" in r.output["hint"]
    assert len(r.attachments) == 1 and r.attachments[0].type == "image" and r.attachments[0].path == str(workspace.root / "sources/raw/diagram.png")
    assert "_attach" not in r.output


@PDF_ENGINE
def test_sources_add_pdf_without_engine_is_unavailable_with_partial(tools: ToolRegistry, ctx: ToolContext, inbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from edgent.adapters.ingest import pdf as pdf_mod

    (inbox / "paper.pdf").write_bytes(make_pdf(["Text that nobody can read without an engine"]))
    bare = Toolchain.detect()
    bare.binaries["pdftotext"] = None
    monkeypatch.setattr(pdf_mod, "_pymupdf", lambda: None)
    ctx.toolchain = bare
    r = err(tools, ctx, "sources_add", path=str(inbox / "paper.pdf"))
    assert r.error.category.value == "unavailable" and "pdftotext" in r.error.message and "replace=true" in r.error.remediation and "poppler" in r.error.remediation
    assert r.output["id"] == "paper" and r.output["coverage"]["status"] == "unknown" and r.error.partial["id"] == "paper"
    listing = ok(tools, ctx, "sources_list")
    assert listing["count"] == 1 and listing["sources"][0]["coverage"] == "unknown"
    ctx.toolchain = Toolchain.detect()
    fixed = ok(tools, ctx, "sources_add", path=str(inbox / "paper.pdf"), replace=True)
    assert fixed["id"] == "paper" and fixed["coverage"]["status"] == "complete" and "[p.1]\nText that nobody" in fixed["preview"]


@skip_unless("ffmpeg")
def test_sources_add_video_is_added_with_unknown_coverage(tools: ToolRegistry, ctx: ToolContext, workspace: Any, inbox: Path) -> None:
    ffmpeg_video(inbox / "lecture.mp4")
    bare = Toolchain.detect()
    bare.modules["faster_whisper"] = False
    bare.binaries["tesseract"] = None
    ctx.toolchain = bare
    out = ok(tools, ctx, "sources_add", path=str(inbox / "lecture.mp4"), title="Lecture 1")
    assert out["kind"] == "video" and out["coverage"]["status"] == "unknown" and out["coverage"]["detail"]["minutes_transcribed"] == 0
    assert {m["name"] for m in out["missing_toolchains"]} >= {"faster_whisper"} and any("edgent[stt]" in m["hint"] for m in out["missing_toolchains"])
    assert "(keyframe: sources/text/lecture-1.frames/00-00-00.jpg)" in out["preview"]
    cov = ok(tools, ctx, "sources_coverage", source_id="lecture-1")
    assert cov["coverage"]["status"] == "unknown" and any("no transcriber installed" in lim for lim in cov["limitations"]) and cov["extra"]["keyframes"] >= 2
    cite = ok(tools, ctx, "sources_cite_check", locator="lecture-1:t=00:00:02")
    assert cite["found"] and cite["marker"] == "[t=00:00:01]" and "unknown, not absent" in cite["coverage_note"] and cite["evidence_id"]
    ev = ctx.evidence.get(cite["evidence_id"])
    assert ev.locations == [{"kind": "time", "ref": "00:00:01"}] and ev.limitations == [cite["coverage_note"]] and ev.locator() == "lecture-1:t=00:00:01"


@skip_unless("ffmpeg")
def test_sources_add_audio_without_transcriber_is_unavailable_but_recorded(tools: ToolRegistry, ctx: ToolContext, inbox: Path) -> None:
    ffmpeg_wav(inbox / "interview.wav")
    bare = Toolchain.detect()
    bare.modules["faster_whisper"] = False
    ctx.toolchain = bare
    r = err(tools, ctx, "sources_add", path=str(inbox / "interview.wav"))
    assert r.error.category.value == "unavailable" and "faster_whisper" in r.error.message and "edgent[stt]" in r.error.remediation
    assert r.output["id"] == "interview" and r.output["kind"] == "audio"
    assert ok(tools, ctx, "sources_list")["sources"][0]["id"] == "interview"
    miss = ok(tools, ctx, "sources_cite_check", locator="interview:t=00:00:01")
    assert miss["found"] is False and "unknown, not absent" in miss["coverage_note"] and miss["evidence_id"] is None


# ----------------------------------------------------------------------------- add_url
def test_sources_add_url_with_injected_fetcher(tools: ToolRegistry, ctx: ToolContext, workspace: Any) -> None:
    seen: list[str] = []

    def fetcher(url: str) -> tuple[str, str, str]:
        seen.append(url)
        return HTML, "text/html", url

    ctx.extra["url_fetcher"] = fetcher
    out = ok(tools, ctx, "sources_add_url", url="https://example.org/bayes")
    assert seen == ["https://example.org/bayes"] and out["kind"] == "web" and out["id"] == "bayes-theorem-wiki" and out["origin"] == "https://example.org/bayes"
    assert out["preview"].startswith("# Bayes' theorem - Wiki\nsource: bayes-theorem-wiki | kind: web | coverage: complete\n\n(origin: https://example.org/bayes") and out["untrusted"] is True
    assert (workspace.root / "sources/raw/bayes-theorem-wiki.html").exists()
    assert err(tools, ctx, "sources_add_url", url="not a url").error.category.value == "invalid_input"
    assert err(tools, ctx, "sources_add_url", url="   ").error.category.value == "invalid_input"
    ctx.extra["url_fetcher"] = lambda u: (b"\x00\x01" * 10, "application/octet-stream", u)
    assert err(tools, ctx, "sources_add_url", url="https://example.org/blob").error.category.value == "unsupported"


def test_sources_add_url_default_fetcher_uses_web_opener(tools: ToolRegistry, ctx: ToolContext) -> None:
    from tests.test_web import PUBLIC_RESOLVER, FakeOpener, FakeResponse

    opener = FakeOpener({"https://example.org/page": FakeResponse(HTML, "https://example.org/page")})
    ctx.extra["web_opener"] = opener
    ctx.extra["web_resolver"] = PUBLIC_RESOLVER
    ctx.settings = {"web": {"timeout_s": 7, "user_agent": "test-agent"}}
    out = ok(tools, ctx, "sources_add_url", url="https://example.org/page", title="Fetched Page")
    assert out["id"] == "fetched-page" and opener.calls[0][1]["user-agent"] == "test-agent" and opener.calls[0][2] == 7.0
    r = err(tools, ctx, "sources_add_url", url="http://localhost:9/x")
    assert r.error.category.value == "invalid_input"


# ----------------------------------------------------------------------------- cite check
@PDF_ENGINE
def test_cite_check_creates_evidence_and_reports_uncovered_pages(tools: ToolRegistry, ctx: ToolContext, workspace: Any, inbox: Path) -> None:
    (inbox / "paper.pdf").write_bytes(make_pdf(["Bayes is introduced on page one", None, "Page three says likelihood matters"]))
    ok(tools, ctx, "sources_add", path=str(inbox / "paper.pdf"))
    hit = ok(tools, ctx, "sources_cite_check", locator="paper:p.3", window_chars=200)
    assert hit == {
        "locator": "paper:p.3", "source_id": "paper", "kind": "pdf", "found": True, "marker": "[p.3]", "text": "Page three says likelihood matters",
        "evidence_id": hit["evidence_id"], "coverage_note": hit["coverage_note"], "untrusted": True, "line": 10, "truncated": False, "text_path": "sources/text/paper.md",
    }
    assert hit["evidence_id"].startswith("ev_") and "partial" in hit["coverage_note"]
    ev = ctx.evidence.get(hit["evidence_id"])
    assert isinstance(ev, Evidence) and ev.operation == "source_passage" and ev.parameters == {"locator": "paper:p.3"} and ev.confidence == "observed" and ev.authority == "source"
    assert ev.result == {"text": "Page three says likelihood matters", "marker": "[p.3]"} and ev.locations == [{"kind": "page", "ref": "3"}] and ev.locator() == "paper:p.3"
    rec = workspace.read_json("sources/text/paper.meta.json")
    assert ev.subject == {"source_id": "paper", "sha256": rec["sha256"]} and ev.limitations == [hit["coverage_note"]]
    assert ok(tools, ctx, "sources_cite_check", locator="paper:p.3", window_chars=200)["evidence_id"] == hit["evidence_id"] and ctx.evidence.count() == 1
    scanned = ok(tools, ctx, "sources_cite_check", locator="paper:p.2")
    assert scanned["found"] is False and scanned["coverage_note"] == "page 2 had no extractable text: absence is unknown, not evidence"
    assert scanned["nearest"] == ["p.1", "p.3"] and scanned["evidence_id"] is None and scanned["marker_present"] is True and "do not invent" in scanned["hint"]
    beyond = ok(tools, ctx, "sources_cite_check", locator="paper:p.12")
    assert beyond["found"] is False and "beyond the last page (3)" in beyond["coverage_note"] and beyond["nearest"] == ["p.3", "p.2", "p.1"]
    assert ctx.evidence.count() == 1


def test_cite_check_lines_unknown_sources_and_bad_locators(tools: ToolRegistry, ctx: ToolContext, inbox: Path) -> None:
    (inbox / "notes.md").write_text("# Notes\n\nFirst point.\nSecond point about priors.\n")
    ok(tools, ctx, "sources_add", path=str(inbox / "notes.md"))
    hit = ok(tools, ctx, "sources_cite_check", locator="notes:L7")
    assert hit["found"] and hit["line_text"] == "Second point about priors." and hit["marker"] == "L7" and hit["coverage_note"] is None
    ev = ctx.evidence.get(hit["evidence_id"])
    assert ev.locations == [{"kind": "line", "ref": "7"}] and ev.locator() == "notes:L7" and ev.limitations == []
    missing = ok(tools, ctx, "sources_cite_check", locator="ghost:L1")
    assert missing["found"] is False and missing["coverage_note"].startswith("no source with id 'ghost'") and missing["known_ids"] == ["notes"] and missing["evidence_id"] is None
    r = err(tools, ctx, "sources_cite_check", locator="notes p.1")
    assert r.error.category.value == "invalid_input" and "p.<page>" in r.error.remediation
    assert err(tools, ctx, "sources_cite_check", locator="notes:L1", window_chars=1).error.category.value == "invalid_input"
    capped = ok(tools, ctx, "sources_cite_check", locator="notes:L4", window_chars=10 ** 6)
    assert capped["found"] and len(capped["text"]) < sources_mod.CITE_WINDOW_LIMIT.value
    ctx.evidence = None
    no_store = ok(tools, ctx, "sources_cite_check", locator="notes:L6")
    assert no_store["found"] and no_store["evidence_id"] is None and "no evidence store" in no_store["note"]


# ----------------------------------------------------------------------------- coverage and reindex
def test_coverage_and_reindex(tools: ToolRegistry, ctx: ToolContext, workspace: Any, inbox: Path) -> None:
    (inbox / "a.txt").write_text("alpha\n")
    ok(tools, ctx, "sources_add", path=str(inbox / "a.txt"), title="Alpha")
    cov = ok(tools, ctx, "sources_coverage", source_id="ALPHA")
    assert cov["source_id"] == "alpha" and cov["coverage"]["status"] == "complete" and cov["limitations"] == [] and cov["missing_toolchains"] == []
    assert cov["text_path"] == "sources/text/alpha.md" and cov["raw_path"] == "sources/raw/alpha.txt" and cov["origin"].endswith("a.txt") and "usable" not in cov["extra"]
    r = err(tools, ctx, "sources_coverage", source_id="nope")
    assert r.error.category.value == "not_found" and "alpha" in r.error.message and r.error.details["known_ids"] == ["alpha"]
    workspace.write_text("sources/INDEX.md", "stale\n")
    assert ok(tools, ctx, "sources_reindex") == {"count": 1, "index_path": "sources/INDEX.md"}
    assert "- alpha | text | Alpha | coverage complete | sources/text/alpha.md" in workspace.read_text("sources/INDEX.md")
    (workspace.root / "sources/text/alpha.meta.json").unlink()
    assert ok(tools, ctx, "sources_reindex")["count"] == 0 and ok(tools, ctx, "sources_list")["count"] == 0


def test_tool_outputs_are_json_serialisable_and_traced(tools: ToolRegistry, ctx: ToolContext, inbox: Path) -> None:
    import json

    from edgent.observability import Tracer

    ctx.tracer = Tracer(ctx.workspace.path("traces"), "s1")
    (inbox / "a.md").write_text("# A\n\ntext\n")
    for name, args in (("sources_add", {"path": str(inbox / "a.md")}), ("sources_list", {}), ("sources_cite_check", {"locator": "a:L4"}), ("sources_coverage", {"source_id": "a"}), ("sources_reindex", {})):
        r = call(tools, ctx, name, **args)
        assert r.ok and json.dumps(r.output)
    events = [json.loads(line) for line in Path(ctx.tracer.path).read_text().splitlines()]
    assert [e["data"]["name"] for e in events if e["kind"] == "tool_call"] == ["sources_add", "sources_list", "sources_cite_check", "sources_coverage", "sources_reindex"]



# ----------------------------------------------------------------------------- review additions
def test_parallel_sources_add_through_the_registry(tools: ToolRegistry, ctx: ToolContext, workspace: Any, inbox: Path) -> None:
    for i in range(3):
        d = inbox / f"d{i}"
        d.mkdir()
        (d / "notes.md").write_text(f"# Notes\n\nbody {i}\n")
    results = tools.call_parallel([("sources_add", {"path": str(inbox / f"d{i}" / "notes.md")}, None) for i in range(3)], ctx)
    assert all(r.ok for r in results), [r.error for r in results if not r.ok]
    assert sorted(r.output["id"] for r in results) == ["notes", "notes-2", "notes-3"] and not any(r.replayed for r in results)
    listing = ok(tools, ctx, "sources_list")
    assert listing["count"] == 3 and listing["untrusted"] is True and workspace.read_text("sources/INDEX.md").count("- notes") == 3
    assert ok(tools, ctx, "sources_coverage", source_id="notes-2")["untrusted"] is True


def test_sources_add_html_file_and_replace_after_a_change(tools: ToolRegistry, ctx: ToolContext, inbox: Path) -> None:
    (inbox / "page.html").write_text("<html><head><title>Page One</title></head><body><p>alpha text</p></body></html>")
    out = ok(tools, ctx, "sources_add", path=str(inbox / "page.html"))
    assert out["kind"] == "web" and out["id"] == "page" and out["title"] == "Page One" and out["raw_path"] == "sources/raw/page.html"  # id from the stem, title from <title>
    assert "alpha text" in out["preview"] and "<p>" not in out["preview"] and out["untrusted"] is True
    (inbox / "page.html").write_text("<html><head><title>Page One</title></head><body><p>beta text</p></body></html>")
    again = ok(tools, ctx, "sources_add", path=str(inbox / "page.html"), replace=True)
    assert again["id"] == "page" and "beta text" in again["preview"] and again["already_present"] is False and ok(tools, ctx, "sources_list")["count"] == 1
    early = ok(tools, ctx, "sources_cite_check", locator="page:L99")
    assert early["found"] is False and "beyond the end" in early["coverage_note"] and early["evidence_id"] is None
