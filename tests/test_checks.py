"""The checks against fixture chapters: a good chapter passes; a bad one fails with the right defects."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from edgent.adapters.providers.mock import MockProvider
from edgent.domain.evidence import Evidence
from edgent.domain.obligations import Obligation, derive_obligations
from edgent.domain.outline import Outline, Unit
from edgent.tools.checks import cite as cite_mod
from edgent.tools.checks import render as render_mod
from edgent.tools.checks.cite import check_cite, local_resolve, parse_locator, resolve_citation
from edgent.tools.checks.execute import check_execute, normalise_output
from edgent.tools.checks.judge import DEFAULT_RUBRIC, check_judge
from edgent.tools.checks.prereqs import check_prereqs
from edgent.tools.checks.readability import (
    GRADE_BANDS,
    check_readability,
    level_from_text,
    strip_markup,
    syllables,
    text_stats,
)
from edgent.tools.checks.registry import CHECK_TOOL_NAMES, build_checks_registry, close_with_checks, run_checks
from edgent.tools.checks.render import check_render
from edgent.tools.registry import ToolContext, ToolRegistry, ToolSpec, obj
from tests.conftest import skip_unless

GOOD_CHAPTER = """---
title: Vectors
unit_id: chapter-vectors
persona: second-year undergraduate
level: undergraduate
introduces: [vector, vector addition]
requires: []
sources: [textbook]
---
# Vectors

Suppose you want to describe the position of a drone relative to its launch pad with enough precision to bring it home. You need three numbers, one for each direction, and you need them together as a single mathematical object. That object is a vector, and this chapter introduces the operations that make vectors useful.

## Definition

A vector is an ordered list of numbers, written $v = (v_1, v_2, v_3)$, whose entries are called components. The textbook introduces the same idea through displacement [textbook:p.12], which is exactly the motivation we started from.

$$
v + w = (v_1 + w_1, v_2 + w_2, v_3 + w_3)
$$

## A worked example

Adding two vectors means adding their corresponding components, position by position, which the following program demonstrates.

```python
v = [1, 2, 3]
w = [4, 5, 6]
print([a + b for a, b in zip(v, w)])
```

```output
[5, 7, 9]
```

Because the components are added separately, vector addition inherits the commutative and associative properties of ordinary addition. You can now add vectors and explain why the order of the operands never changes the result.
"""

BAD_CHAPTER = """---
title: Matrices
unit_id: chapter-matrices
level: undergraduate
introduces: [matrix]
requires: [determinant]
---
# Matrices

A matrix is a rectangular array of numbers, and because this chapter keeps every one of its sentences deliberately long so that a reader at any level of preparation would struggle to hold the subject and the verb together in working memory, it runs on and on without ever reaching the point. The eigenvalue decomposition appears later in the book. The claim about rank comes from [textbook:p.99] and this one from [ghost:p.1].

```python
x = 1 / 0
print(x)
```

```python
print("hello")
```

```output
goodbye
```
"""

TEXTBOOK = """# Textbook
source: textbook | kind: pdf | coverage: partial

[p.11]
Chapter one opens with arrows.

[p.12]
A displacement is an arrow from one point to another; it is the first example of a vector.

[p.13]
Scaling an arrow stretches it.
"""

LECTURE = """# Lecture
[t=00:00:00]
Welcome to the course.

[t=00:14:00]
Now we define a vector as a displacement.

[t=00:20:00]
Next, matrices.
"""


@pytest.fixture
def chapters(workspace) -> dict[str, str]:
    workspace.write_text("book/chapters/01-vectors.md", GOOD_CHAPTER)
    workspace.write_text("book/chapters/02-matrices.md", BAD_CHAPTER)
    workspace.write_text("sources/text/textbook.md", TEXTBOOK)
    workspace.write_json("sources/text/textbook.meta.json", {"id": "textbook", "kind": "pdf", "coverage": {"status": "partial", "detail": {"pages_total": 120, "pages_failed": [99]}}})
    workspace.write_text("sources/text/lecture.md", LECTURE)
    o = Outline(workspace.path("outline"))
    o.title, o.persona = "Linear Algebra", "second-year undergraduate"
    o.add(Unit(id="chapter-vectors", kind="chapter", title="Vectors", path="book/chapters/01-vectors.md", meta={"level": "undergraduate"}))
    o.add(Unit(id="chapter-matrices", kind="chapter", title="Matrices", path="book/chapters/02-matrices.md", depends_on=["chapter-vectors"]))
    o.add(Unit(id="chapter-eigen", kind="chapter", title="Eigenvalues", path="book/chapters/03-eigen.md", depends_on=["chapter-matrices"], meta={"introduces": ["eigenvalue decomposition", "determinant"]}))
    return {"good": "book/chapters/01-vectors.md", "bad": "book/chapters/02-matrices.md"}


@pytest.fixture
def local_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Citations resolve with the local marker rules regardless of whether the sources module is present."""
    monkeypatch.setattr(cite_mod, "external_resolver", lambda: None)


def _msgs(result: dict[str, Any], severity: str | None = None) -> list[str]:
    return [d["message"] for d in result["defects"] if severity is None or d["severity"] == severity]


def _evidence(ctx: ToolContext, eid: str | None) -> Evidence:
    assert eid is not None
    ev = ctx.evidence.get(eid)
    assert ev is not None
    return ev


# ----------------------------------------------------------------------------- render
def test_render_good_chapter_passes_and_records_evidence(ctx: ToolContext, chapters: dict[str, str]) -> None:
    r = check_render(ctx, {"path": chapters["good"]})
    assert r["passed"] and r["authority"] == "controlled_replay" and _msgs(r, "medium") == [] and _msgs(r, "high") == []
    ev = _evidence(ctx, r["evidence_id"])
    assert ev.operation == "check_render" and ev.authority == "controlled_replay" and ev.subject["path"] == chapters["good"]
    if ctx.toolchain.has("pandoc"):
        assert r["confidence"] == "observed" and r["detail"]["pandoc"]["returncode"] == 0
        assert (ctx.workspace.root / r["detail"]["pandoc"]["output"]).exists() and r["detail"]["pandoc"]["output"].endswith("01-vectors.html")
    else:
        assert r["limitations"] == ["no pandoc: structure only"] and r["confidence"] == "derived"


@skip_unless("pandoc")
def test_render_with_pandoc_produces_katex_html(ctx: ToolContext, chapters: dict[str, str]) -> None:
    r = check_render(ctx, {"path": chapters["good"]})
    html = (ctx.workspace.root / "book/build/01-vectors.html").read_text(encoding="utf-8")
    assert "katex" in html and "<h1" in html and r["detail"]["pandoc"]["warnings"] == []


def test_render_structural_fallback_without_pandoc(ctx: ToolContext, chapters: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render_mod, "_pandoc_path", lambda _ctx: None)
    r = check_render(ctx, {"path": chapters["good"]})
    assert r["passed"] and r["limitations"] == ["no pandoc: structure only"] and "pandoc" not in r["detail"]
    assert _evidence(ctx, r["evidence_id"]).limitations == ["no pandoc: structure only"]


def test_render_reports_structural_defects(ctx: ToolContext, workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render_mod, "_pandoc_path", lambda _ctx: None)
    text = "# No front matter\n\nAn unbalanced $x here.\n\n$$\nx = 1\n\n### jump\n\n![missing](../../media/diagrams/nope.svg) and [out](../../../etc/passwd) and [ok](https://example.org)\n\n{{film: ghost}}\n\n```\nplain\n```\n\n```\nprint(1)\n"
    workspace.write_text("book/chapters/x.md", text)
    r = check_render(ctx, {"path": "book/chapters/x.md"})
    msgs = " | ".join(_msgs(r))
    assert not r["passed"]
    assert "no front matter" in msgs and "unbalanced $ on line 3" in msgs and "unbalanced $$" in msgs
    assert "never closed" in msgs and "broken link ../../media/diagrams/nope.svg" in msgs and "escapes the workspace" in msgs
    assert "heading level jumps from h1 to h3" in msgs and "{{film: ghost}} has no built target" in msgs and "has no language" in msgs
    assert r["detail"]["links_checked"] == 2


def test_render_rejects_escape_missing_and_binary_paths(ctx: ToolContext, workspace) -> None:
    from edgent.domain.errors import ToolFailure

    with pytest.raises(ToolFailure) as e:
        check_render(ctx, {"path": "../outside.md"})
    assert e.value.error.category.value == "invalid_input" and e.value.error.remediation
    with pytest.raises(ToolFailure) as e:
        check_render(ctx, {"path": "book/chapters/none.md"})
    assert e.value.error.category.value == "not_found"
    workspace.write_text("book/x.html", "<p>")
    with pytest.raises(ToolFailure) as e:
        check_render(ctx, {"path": "book/x.html"})
    assert e.value.error.category.value == "unsupported"


# ----------------------------------------------------------------------------- execute
def test_execute_good_chapter_runs_blocks_and_matches_output(ctx: ToolContext, chapters: dict[str, str]) -> None:
    r = check_execute(ctx, {"path": chapters["good"]})
    assert r["passed"] and r["confidence"] == "observed" and r["authority"] == "controlled_replay"
    (entry,) = r["detail"]["blocks"]
    assert entry["ok"] and entry["output_matches"] and entry["stdout_preview"].strip() == "[5, 7, 9]"
    ev = _evidence(ctx, r["evidence_id"])
    assert ev.operation == "check_execute" and ev.confidence == "observed" and ev.result["failed"] == []


def test_execute_bad_chapter_reports_traceback_and_output_mismatch(ctx: ToolContext, chapters: dict[str, str]) -> None:
    r = check_execute(ctx, {"path": chapters["bad"]})
    assert not r["passed"]
    high = _msgs(r, "high")
    assert any("block 1 failed: ZeroDivisionError" in m for m in high)
    assert any("block 2 printed 'hello', the chapter shows 'goodbye'" in m for m in high)
    assert r["detail"]["blocks"][0]["error"].startswith("ZeroDivisionError")
    assert r["defects"][0]["location"] == f"{chapters['bad']}:L12"


def test_execute_continue_marker_shares_definitions_and_scratch_dir(ctx: ToolContext, workspace) -> None:
    text = (
        "---\ntitle: T\n---\n# T\n\n```python\nx = 21\nopen('note.txt', 'w').write('kept')\nprint('first')\n```\n\n```output\nfirst\n```\n\n"
        "```python\n# edgent: continue\nprint(x * 2)\n```\n\n```output\n42\n```\n\n```python\nprint(open('note.txt').read())\n```\n\n```output\nkept\n```\n\n```js\nconsole.log(1)\n```\n"
    )
    workspace.write_text("book/chapters/c.md", text)
    r = check_execute(ctx, {"path": "book/chapters/c.md"})
    assert r["passed"], r["defects"]
    blocks = r["detail"]["blocks"]
    assert [b["block"] for b in blocks] == [1, 3, 5] and blocks[1]["continued"] and blocks[1]["stdout_preview"].strip() == "42"
    assert (workspace.root / r["detail"]["scratch_dir"] / "block_003.py").read_text().startswith("x = 21")


def test_execute_timeout_and_limits(ctx: ToolContext, workspace) -> None:
    from edgent.domain.errors import ToolFailure

    workspace.write_text("book/chapters/slow.md", "---\ntitle: S\n---\n# S\n\n```python\nimport time\ntime.sleep(5)\n```\n")
    r = check_execute(ctx, {"path": "book/chapters/slow.md", "timeout_s": 1})
    assert not r["passed"] and "timed out after 1 s" in r["defects"][0]["message"] and r["detail"]["blocks"][0]["timed_out"]
    with pytest.raises(ToolFailure) as e:
        check_execute(ctx, {"path": "book/chapters/slow.md", "timeout_s": 10_000})
    assert e.value.error.code == "limit:check_execute block timeout" and "300 s" in e.value.error.message
    workspace.write_text("book/chapters/none.md", "---\ntitle: N\n---\n# N\n\nprose only\n")
    r = check_execute(ctx, {"path": "book/chapters/none.md"})
    assert r["passed"] and r["limitations"] == ["no python code blocks in the file"]


def test_normalise_output_ignores_whitespace_and_blank_lines() -> None:
    assert normalise_output("a  b\n\n  c \n") == normalise_output("a b\nc") == ["a b", "c"]


# ----------------------------------------------------------------------------- cite
def test_parse_locator_forms() -> None:
    assert parse_locator("textbook:p.12") == ("textbook", "page", "12")
    assert parse_locator("lecture:t=14:32") == ("lecture", "time", "00:14:32")
    assert parse_locator("notes:L7") == ("notes", "line", "7")
    assert parse_locator("Bad Locator") is None


def test_local_resolver_pages_times_lines_and_coverage(workspace, chapters) -> None:
    r = local_resolve(workspace, "textbook:p.12")
    assert r["resolved"] and r["text"].startswith("A displacement is an arrow") and r["line"] == 7
    r = local_resolve(workspace, "lecture:t=00:16:30")
    assert r["resolved"] and "define a vector" in r["text"]
    r = local_resolve(workspace, "lecture:t=00:00:00")
    assert r["resolved"] and "Welcome" in r["text"]
    r = local_resolve(workspace, "textbook:L8")
    assert r["resolved"] and "first example of a vector" in r["text"]
    r = local_resolve(workspace, "textbook:p.99")
    assert not r["resolved"] and r["uncovered"] and "page 99 failed extraction" in r["coverage"]
    r = local_resolve(workspace, "textbook:p.50")
    assert not r["resolved"] and r["uncovered"] and "coverage partial" in r["coverage"]
    r = local_resolve(workspace, "lecture:p.3")
    assert not r["resolved"] and not r["uncovered"] and "no [p.3] marker" in r["reason"]
    r = local_resolve(workspace, "ghost:p.1")
    assert not r["resolved"] and "no text view for source 'ghost'" in r["reason"]
    assert not local_resolve(workspace, "???")["resolved"]


def test_cite_good_chapter_resolves_and_links_passages(ctx: ToolContext, chapters: dict[str, str], local_only: None) -> None:
    r = check_cite(ctx, {"path": chapters["good"]})
    assert r["passed"] and r["authority"] == "source" and r["detail"]["untrusted"] is True
    (entry,) = r["detail"]["citations"]
    assert entry["resolved"] and entry["locator"] == "textbook:p.12" and entry["excerpt"].startswith("A displacement")
    passage = _evidence(ctx, entry["evidence_id"])
    assert passage.operation == "source_passage" and passage.authority == "source" and passage.locator() == "textbook:p.12"
    assert _evidence(ctx, r["evidence_id"]).links == [entry["evidence_id"]]


def test_cite_bad_chapter_reports_unresolved_and_uncovered(ctx: ToolContext, chapters: dict[str, str], local_only: None) -> None:
    r = check_cite(ctx, {"path": chapters["bad"]})
    assert not r["passed"]
    by_sev = {d["severity"]: d["message"] for d in r["defects"]}
    assert "[textbook:p.99] absence is unknown" in by_sev["medium"] and "failed extraction" in by_sev["medium"]
    assert "[ghost:p.1] does not resolve" in by_sev["high"]


def test_cite_uses_the_ingest_resolver_when_present_and_falls_back_when_it_raises(workspace, chapters, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake(ws: Any, locator: str) -> dict[str, Any]:
        calls.append(locator)
        if locator == "textbook:p.12":
            return {"found": True, "text": "from ingest", "coverage_note": "coverage partial", "line": 7, "source_id": "textbook"}
        if locator == "textbook:p.99":
            return {"found": False, "text": "", "coverage_note": "page 99 could not be read", "marker_present": True}
        if locator == "textbook:p.11":
            return {"found": False, "coverage_note": "no source with id 'textbook'"}
        raise RuntimeError("boom")

    monkeypatch.setattr(cite_mod, "external_resolver", lambda: fake)
    r = resolve_citation(workspace, "textbook:p.12")
    assert r["resolved"] and r["text"] == "from ingest" and r["via"] == "ingest"
    r = resolve_citation(workspace, "textbook:p.99")
    assert not r["resolved"] and r["uncovered"] and r["via"] == "ingest"
    r = resolve_citation(workspace, "textbook:p.11")
    assert r["resolved"] and r["via"] == "local" and r["note"] == "no source record; resolved from the text view"
    r = resolve_citation(workspace, "lecture:t=00:14:00")
    assert r["resolved"] and r["via"] == "local" and "ingest resolver raised RuntimeError" in r["note"]
    assert calls == ["textbook:p.12", "textbook:p.99", "textbook:p.11", "lecture:t=00:14:00"]


def test_cite_without_citations_passes_with_a_limitation(ctx: ToolContext, workspace, local_only: None) -> None:
    workspace.write_text("book/chapters/n.md", "---\ntitle: N\n---\n# N\n\nNo citations.\n")
    r = check_cite(ctx, {"path": "book/chapters/n.md"})
    assert r["passed"] and r["limitations"] == ["no citations in the file"]


# ----------------------------------------------------------------------------- readability
def test_level_keywords_and_bands() -> None:
    assert level_from_text("second-year undergraduate") == "undergraduate"
    assert level_from_text("a high school student") == "secondary"
    assert level_from_text("PhD researcher") == "graduate"
    assert level_from_text("kindergarten class") == "kg"
    assert level_from_text("grade 3 pupils") == "primary"
    assert level_from_text("middle school") == "middle"
    assert level_from_text("working engineers") == "professional"
    assert level_from_text("no idea") is None and level_from_text("") is None
    assert set(GRADE_BANDS) == {"kg", "primary", "middle", "secondary", "undergraduate", "graduate", "professional"}


def test_syllables_and_stats_strip_markup() -> None:
    assert syllables("vector") == 2 and syllables("the") == 1 and syllables("mathematical") == 5 and syllables("make") == 1
    prose = strip_markup("# Heading\n\nSee $x^2$ and `code` and [a link](u) and [src:p.1] here. **Bold** text ends.\n\n```python\nprint(1)\n```\n")
    assert "Heading" not in prose and "x^2" not in prose and "code" not in prose and "a link" in prose and "src:p.1" not in prose
    stats = text_stats(prose)
    assert stats["sentences"] == 2 and stats["words"] == 10 and stats["longest_sentence_words"] == 7


def test_readability_good_chapter_in_band_and_bad_chapter_flags_long_sentence(ctx: ToolContext, chapters: dict[str, str]) -> None:
    r = check_readability(ctx, {"path": chapters["good"]})
    assert r["passed"], r["defects"]
    assert r["detail"]["level"] == "undergraduate" and r["detail"]["level_source"] == "front matter level" and 10 <= r["detail"]["fk_grade"] <= 15
    ev = _evidence(ctx, r["evidence_id"])
    assert ev.operation == "check_readability" and ev.confidence == "derived" and ev.parameters["level"] == "undergraduate"
    r = check_readability(ctx, {"path": chapters["bad"]})
    assert not r["passed"]
    msgs = _msgs(r, "medium")
    assert any("exceeds the undergraduate ceiling of 35" in m and "sentence of" in m for m in msgs)
    assert r["detail"]["sentences_over_ceiling"] == 1 and r["detail"]["longest_sentence_words"] > 40


def test_readability_level_sources_and_short_text(ctx: ToolContext, workspace, chapters) -> None:
    from edgent.domain.errors import ToolFailure

    workspace.write_text("book/chapters/short.md", "---\ntitle: S\nunit_id: chapter-vectors\n---\n# S\n\nA vector is short. So is this.\n")
    r = check_readability(ctx, {"path": "book/chapters/short.md"})
    assert r["passed"] and r["limitations"][0].startswith("too short to grade") and r["detail"]["level_source"] == "outline unit level"
    workspace.write_text("book/chapters/nolevel.md", "---\ntitle: S\n---\n# S\n\nA vector is short. So is this.\n")
    r = check_readability(ctx, {"path": "book/chapters/nolevel.md"})
    assert r["detail"]["level"] == "undergraduate" and r["detail"]["level_source"] == "outline persona"
    r = check_readability(ctx, {"path": "book/chapters/nolevel.md", "level": "kindergarten"})
    assert r["detail"]["level"] == "kg" and r["detail"]["level_source"] == "argument"
    with pytest.raises(ToolFailure) as e:
        check_readability(ctx, {"path": "book/chapters/nolevel.md", "level": "martian"})
    assert e.value.error.category.value == "invalid_input"


def test_readability_grade_outside_band_is_a_defect(ctx: ToolContext, workspace) -> None:
    simple = "The cat sat. The dog ran. We had fun. It was hot. " * 6
    workspace.write_text("book/chapters/simple.md", f"---\ntitle: S\nlevel: graduate\n---\n# S\n\n{simple}\n")
    r = check_readability(ctx, {"path": "book/chapters/simple.md"})
    assert not r["passed"] and "below the graduate band 12-18" in r["defects"][0]["message"]
    workspace.write_text("book/chapters/simple-kg.md", f"---\ntitle: S\nlevel: kg\n---\n# S\n\n{simple}\n")
    assert check_readability(ctx, {"path": "book/chapters/simple-kg.md"})["passed"]


# ----------------------------------------------------------------------------- prereqs
def test_prereqs_good_chapter_passes_and_bad_chapter_has_forward_reference(ctx: ToolContext, chapters: dict[str, str]) -> None:
    r = check_prereqs(ctx, {"path": chapters["good"]})
    assert r["passed"] and r["detail"]["unit_id"] == "chapter-vectors" and r["detail"]["earlier_units"] == []
    r = check_prereqs(ctx, {"path": chapters["bad"], "unit_id": "chapter-matrices"})
    assert not r["passed"]
    high = _msgs(r, "high")
    assert high == ["forward reference: 'determinant' is required here but introduced later by chapter-eigen"]
    low = _msgs(r, "low")
    assert low and "uses 'eigenvalue decomposition', which chapter-eigen introduces later" in low[0]
    assert r["detail"]["earlier_units"] == ["chapter-vectors"] and r["detail"]["later_terms_used"] == 1
    ev = _evidence(ctx, r["evidence_id"])
    assert ev.operation == "check_prereqs" and ev.result["forward_references"] == 1


def test_prereqs_dependency_chain_counts_as_earlier_and_missing_concepts_fail(ctx: ToolContext, workspace, chapters) -> None:
    o = Outline(workspace.path("outline"))
    o.add(Unit(id="chapter-late", kind="chapter", title="Late", path="book/chapters/09-late.md", depends_on=["chapter-eigen"]))
    workspace.write_text("book/chapters/09-late.md", "---\ntitle: Late\nunit_id: chapter-late\nrequires: [determinant, vector, topology]\n---\n# Late\n\nUses determinants.\n")
    r = check_prereqs(ctx, {"path": "book/chapters/09-late.md"})
    assert _msgs(r, "high") == ["'topology' is required but no unit introduces it"]
    workspace.write_text("book/chapters/orphan.md", "---\ntitle: O\nrequires: [vector]\n---\n# O\n\nText.\n")
    r = check_prereqs(ctx, {"path": "book/chapters/orphan.md"})
    assert not r["passed"] and "not an outline unit" in r["defects"][0]["message"]
    workspace.write_text("book/chapters/orphan2.md", "---\ntitle: O\n---\n# O\n\nText.\n")
    r = check_prereqs(ctx, {"path": "book/chapters/orphan2.md"})
    assert r["passed"] and r["limitations"] == ["not an outline unit; nothing to check against"]


def test_prereqs_reads_introduces_from_other_chapter_files(ctx: ToolContext, workspace, chapters) -> None:
    workspace.write_text("book/chapters/03-eigen.md", "---\ntitle: Eigen\nunit_id: chapter-eigen\nintroduces: [spectrum]\nrequires: [matrix]\n---\n# Eigen\n\nThe spectrum.\n")
    o = Outline(workspace.path("outline"))
    o.update("chapter-eigen", meta={})
    workspace.write_text("book/chapters/01-vectors.md", GOOD_CHAPTER.replace("# Vectors\n", "# Vectors\n\nThe spectrum of a matrix is a later idea.\n"))
    r = check_prereqs(ctx, {"path": "book/chapters/01-vectors.md"})
    assert r["passed"] and [d["message"] for d in r["defects"]] == ["uses 'matrix', which chapter-matrices introduces later", "uses 'spectrum', which chapter-eigen introduces later"]
    r = check_prereqs(ctx, {"path": "book/chapters/03-eigen.md"})
    assert r["passed"]


# ----------------------------------------------------------------------------- judge
def test_judge_parses_verdict_and_records_model_inference_evidence(ctx: ToolContext, provider: MockProvider, chapters: dict[str, str]) -> None:
    provider.push("Here is my verdict:\n```json\n" + json.dumps({"score": 8, "defects": [{"location": "Definition", "severity": "low", "message": "define component first"}], "summary": "Solid chapter."}) + "\n```")
    r = check_judge(ctx, {"path": chapters["good"], "unit_id": "chapter-vectors"})
    assert r["passed"] and r["authority"] == "model_inference" and r["confidence"] == "inferred" and r["detail"]["score"] == 8
    assert r["defects"] == [{"location": "Definition", "severity": "low", "message": "define component first"}]
    req = provider.calls[-1]
    assert req.system_text().startswith("[edgent-role: critic]") and DEFAULT_RUBRIC.splitlines()[0] in req.system_text()
    assert "untrusted data" in req.last_user_text() and "Suppose you want to describe" in req.last_user_text()
    assert req.metadata["role"] == "critic" and req.json_schema is not None
    ev = _evidence(ctx, r["evidence_id"])
    assert ev.operation == "check_judge" and ev.authority == "model_inference" and ev.confidence == "inferred" and ev.provider == "mock:mock"


def test_judge_fails_on_low_score_or_high_defect_and_handles_garbage(ctx: ToolContext, provider: MockProvider, chapters: dict[str, str]) -> None:
    provider.push(json.dumps({"score": 9, "defects": [{"location": "x", "severity": "high", "message": "wrong theorem"}], "summary": ""}))
    r = check_judge(ctx, {"path": chapters["good"], "rubric": "custom rubric", "role": "examiner"})
    assert not r["passed"] and r["detail"]["role"] == "examiner" and "custom rubric" in provider.calls[-1].system_text()
    provider.push(json.dumps({"score": 3, "defects": [], "summary": "weak"}))
    assert not check_judge(ctx, {"path": chapters["good"]})["passed"]
    provider.push("I cannot judge this.")
    r = check_judge(ctx, {"path": chapters["good"]})
    assert not r["passed"] and r["evidence_id"] is None and "no verdict" in r["limitations"] and r["defects"][0]["severity"] == "low"


def test_judge_unavailable_without_provider(ctx: ToolContext, chapters: dict[str, str]) -> None:
    from edgent.domain.errors import ToolFailure

    ctx.provider = None
    with pytest.raises(ToolFailure) as e:
        check_judge(ctx, {"path": chapters["good"]})
    assert e.value.error.category.value == "unavailable" and "critic round" in e.value.error.remediation
    ctx.provider = MockProvider(fail_on_roles={"critic"})
    with pytest.raises(ToolFailure) as e:
        check_judge(ctx, {"path": chapters["good"]})
    assert e.value.error.code == "provider_error" and e.value.error.category.value == "unavailable"


# ----------------------------------------------------------------------------- registry and the engine
def test_registry_contracts(registry: ToolRegistry) -> None:
    reg = build_checks_registry(registry)
    assert reg is registry and set(CHECK_TOOL_NAMES) <= set(reg.names())
    for name in CHECK_TOOL_NAMES:
        spec = reg.get(name)
        assert spec is not None and spec.group == "checks" and spec.parameters["additionalProperties"] is False
        assert "path" in spec.parameters["required"] and len(spec.description) > 80
    assert reg.get("check_judge").cost_class == "expensive" and reg.get("check_cite").permission == "read"
    build_checks_registry(registry)  # idempotent


def test_run_checks_through_registry_good_and_bad(registry: ToolRegistry, ctx: ToolContext, chapters: dict[str, str], local_only: None) -> None:
    reg = build_checks_registry(registry)
    r = reg.call("run_checks", {"path": chapters["good"]}, ctx)
    assert r.ok, r.error
    out = r.output
    assert out["passed"] and out["in_outline"] and out["summary"]["blocking_open"] == [] and out["summary"]["advisory_open"] == ["chapter-vectors:judge"]
    judge_row = next(row for row in out["ledger"] if row["kind"] == "judge")
    assert judge_row["status"] == "open" and judge_row["advisory"] and judge_row["reason"].startswith("advisory: not run")
    assert all(row["evidence_ids"] for row in out["ledger"] if row["status"] == "closed")
    r = reg.call("run_checks", {"path": chapters["bad"], "only": ["execute", "cite", "prereqs"]}, ctx)
    assert r.ok
    out = r.output
    assert not out["passed"]
    status = {row["id"]: row["status"] for row in out["ledger"]}
    assert status["chapter-matrices:execute:1"] == "open" and status["chapter-matrices:execute:2"] == "open"
    assert status["chapter-matrices:cite:ghost:p.1"] == "open" and status["chapter-matrices:prereqs"] == "open"
    assert status["chapter-matrices:render"] == "open" and status["chapter-matrices:readability"] == "open"  # not run because of `only`
    reasons = {row["id"]: row["reason"] for row in out["ledger"]}
    assert reasons["chapter-matrices:execute:1"].startswith("ZeroDivisionError")
    assert "no text view for source 'ghost'" in reasons["chapter-matrices:cite:ghost:p.1"] and "failed extraction" in reasons["chapter-matrices:cite:textbook:p.99"]


def test_run_checks_generic_chapter_and_unknown_unit(registry: ToolRegistry, ctx: ToolContext, workspace, local_only: None) -> None:
    reg = build_checks_registry(registry)
    workspace.write_text("book/chapters/free.md", GOOD_CHAPTER.replace("unit_id: chapter-vectors\n", ""))
    r = reg.call("run_checks", {"path": "book/chapters/free.md", "only": ["structure", "execute"]}, ctx)
    assert r.ok and not r.output["in_outline"] and r.output["unit_id"] == "free" and r.output["passed"]
    r = reg.call("run_checks", {"path": "book/chapters/free.md", "unit_id": "nope"}, ctx)
    assert not r.ok and r.error.category.value == "not_found"
    r = reg.call("run_checks", {"path": "book/chapters/free.md", "only": ["bogus"]}, ctx)
    assert not r.ok and r.error.category.value == "invalid_input"


def test_run_checks_with_judge_consumes_one_model_call(ctx: ToolContext, provider: MockProvider, chapters: dict[str, str], local_only: None) -> None:
    provider.push(json.dumps({"score": 9, "defects": [], "summary": "fine"}))
    out = run_checks(ctx, {"path": chapters["good"], "judge": True})
    judge_row = next(row for row in out["ledger"] if row["kind"] == "judge")
    assert judge_row["status"] == "closed" and judge_row["evidence_ids"] and len(provider.calls) == 1


def test_close_with_checks_honours_authority_and_external_closers(ctx: ToolContext, chapters: dict[str, str]) -> None:
    reg = ToolRegistry()
    calls: list[dict[str, Any]] = []

    def weak(_ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        calls.append(args)
        return {"passed": True, "authority": "model_inference", "evidence_id": "ev_weak"}

    def strong(_ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        calls.append(args)
        return {"passed": True, "authority": "controlled_replay", "evidence_ids": ["ev_strong"], "unresolved_references": [{"id": "f1"}]}

    reg.register(ToolSpec(name="film_render", description="d", parameters=obj({"film_id": {"type": "string"}}, ["film_id"]), fn=weak, permission="compute"))
    reg.register(ToolSpec(name="book_build", description="d", parameters=obj({"format": {"type": "string"}}), fn=strong, permission="compute"))
    reg.register(ToolSpec(name="widget_run", description="d", parameters=obj({"secret": {"type": "string"}}, ["secret"]), fn=weak, permission="compute"))
    film = derive_obligations(Unit(id="film-1", kind="film", title="F"), None)
    integ = derive_obligations(Unit(id="integ", kind="integration", title="I"), None)
    widget = derive_obligations(Unit(id="w1", kind="widget", title="W"), None)
    obs, results = close_with_checks(ctx, film + integ + widget, registry=reg)
    by_id = {o.id: o for o in obs}
    assert by_id["film-1:render"].status == "open" and "model_inference < controlled_replay" in by_id["film-1:render"].reason
    assert by_id["film-1:measure"].status == "open" and "check_measure is not available" in by_id["film-1:measure"].reason
    assert by_id["integ:render"].status == "closed" and by_id["integ:render"].evidence_ids == ["ev_strong"]
    assert by_id["integ:structure"].status == "open" and "unresolved references: f1" in by_id["integ:structure"].reason
    assert by_id["w1:render"].status == "open" and "needs arguments this gate cannot derive" in by_id["w1:render"].reason
    assert calls[0] == {"film_id": "film-1"} and calls[1] == {}
    assert results["film_render media/films/film-1"]["passed"] is True and "skipped" in results["widget_run media/widgets/w1.html"]
    assert results["book_build book/build"]["evidence_id"] is None


def test_close_with_checks_leaves_obligations_open_when_a_closer_fails(ctx: ToolContext, workspace) -> None:
    unit = Unit(id="c", kind="chapter", title="C", path="book/chapters/missing.md")
    obs, results = close_with_checks(ctx, derive_obligations(unit, "---\ntitle: T\n---\n# T\n\n```python\nprint(1)\n```\n"))
    assert all(o.status == "open" for o in obs if not o.advisory)
    assert obs[0].reason.startswith("check_render: not_found") and results["check_render book/chapters/missing.md"]["error"]["category"] == "not_found"


def test_close_with_checks_skips_closed_and_blocked(ctx: ToolContext) -> None:
    done = Obligation(id="a", kind="render", description="d", target="t", required_authority="controlled_replay", closer="check_render", status="closed", reason="earlier")
    blocked = Obligation(id="b", kind="execute", description="d", target="t", required_authority="controlled_replay", closer="check_execute", status="blocked", reason="no executor for js")
    obs, results = close_with_checks(ctx, [done, blocked])
    assert results == {} and done.reason == "earlier" and blocked.status == "blocked"


def test_check_output_shape_is_json_serialisable(ctx: ToolContext, chapters: dict[str, str], local_only: None) -> None:
    for fn in (check_render, check_execute, check_cite, check_readability, check_prereqs):
        out = fn(ctx, {"path": chapters["bad"]})
        assert set(out) >= {"passed", "authority", "confidence", "defects", "evidence_id", "detail"}
        assert all(set(d) == {"location", "severity", "message"} for d in out["defects"])
        json.dumps(out)
    assert (Path(ctx.workspace.path("evidence"))).exists() and ctx.evidence.count() >= 5


# ----------------------------------------------------------------------------- engine edge cases found in review
def test_external_closer_verdicts_honour_ok_false_and_compacted_outputs(ctx: ToolContext) -> None:
    from edgent.tools.checks.registry import closer_registry

    reg = ToolRegistry()
    reg.register(ToolSpec(name="film_render", description="d", parameters=obj({"film_id": {"type": "string"}}, ["film_id"]), fn=lambda c, a: {"ok": False, "defects": [{"beat": 2, "kind": "overlap", "severity": "error", "detail": "labels overlap in beat 2"}]}, permission="compute"))
    reg.register(ToolSpec(name="assess_items_check", description="d", parameters=obj({"set_id": {"type": "string"}}, ["set_id"]), fn=lambda c, a: {"passed": a["set_id"] == "good", "evidence_ids": ["ev_keys"], "items": [{"id": str(i), "issues": ["x" * 60]} for i in range(200)]}, permission="compute"))
    film = derive_obligations(Unit(id="film-1", kind="film", title="F"), None)
    good = derive_obligations(Unit(id="good", kind="exercise_set", title="G"), None)
    bad = derive_obligations(Unit(id="bad", kind="exercise_set", title="B"), None)
    obs, results = close_with_checks(ctx, film + good + bad, registry=reg)
    by_id = {o.id: o for o in obs}
    assert by_id["film-1:render"].status == "open" and by_id["film-1:render"].reason == "labels overlap in beat 2"
    assert by_id["good:keys"].status == "closed" and by_id["good:keys"].evidence_ids == ["ev_keys"]  # read back from the compacted file
    assert by_id["bad:keys"].status == "open" and by_id["bad:keys"].reason == "assess_items_check reported defects"
    assert results["assess_items_check assessments/good/items.json"]["passed"] is True
    ctx.extra["registry"] = reg
    assert closer_registry(ctx) is reg
    del ctx.extra["registry"]
    fallback = closer_registry(ctx)
    assert fallback is not None and fallback is not reg and {"assess_items_check", "book_build"} <= set(fallback.names()) and fallback.get("film_render") is None
    assert closer_registry(ctx) is fallback


def test_local_args_drop_a_level_the_check_cannot_parse() -> None:
    from edgent.tools.checks.registry import _local_args

    def ob(level: str) -> Obligation:
        return Obligation(id="u:readability", kind="readability", description="d", target="c.md", required_authority="controlled_replay", closer="check_readability", args={"path": "c.md", "level": level})

    assert _local_args("check_readability", [ob("a curious adult returning to maths")]) == {"path": "c.md"}
    assert _local_args("check_readability", [ob("second-year undergraduate")]) == {"path": "c.md", "level": "second-year undergraduate"}


def test_execute_scratch_directories_are_per_file_not_per_stem(ctx: ToolContext, workspace) -> None:
    text = "---\ntitle: T\n---\n# T\n\n```python\nprint('a')\n```\n"
    workspace.write_text("book/chapters/same.md", text)
    workspace.write_text("book/sections/same.md", text.replace("'a'", "'b'"))
    a = check_execute(ctx, {"path": "book/chapters/same.md"})
    b = check_execute(ctx, {"path": "book/sections/same.md"})
    assert a["passed"] and b["passed"] and a["detail"]["scratch_dir"] != b["detail"]["scratch_dir"] and a["detail"]["untrusted"] is True
    assert (workspace.root / a["detail"]["scratch_dir"] / "block_001.py").read_text().strip() == "print('a')"
    assert (workspace.root / b["detail"]["scratch_dir"] / "block_001.py").read_text().strip() == "print('b')"
