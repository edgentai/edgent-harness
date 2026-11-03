"""Obligation derivation per unit kind, the content parsers, and the authority rule for closing."""
from __future__ import annotations

import pytest

from edgent.domain.obligations import (
    CLOSERS,
    Obligation,
    blocking_obligations,
    citations,
    close_obligation,
    code_blocks,
    concept_list,
    derive_obligations,
    ledger_summary,
    placeholders,
    strip_code,
)
from edgent.domain.outline import Outline, Unit

CHAPTER = """---
title: Vectors
unit_id: chapter-vectors
level: undergraduate
introduces: [vector, vector addition]
requires: [real number]
sources: [textbook]
---
# Vectors

A vector is a list [textbook:p.12] of numbers; see also [lecture:t=00:14:32] and [notes:L7], and again [textbook:p.12].

```python
v = [1, 2, 3]
print(sum(v))
```

```output
6
```

```python
# edgent: continue
print(v[0])
```

```bash
echo hi
```

```
no language
```

{{film: f1}}
{{widget: w1}}
![cap](../../media/diagrams/d1.svg)
{{exercises: set-1}}
"""


# ----------------------------------------------------------------------------- parsers
def test_code_blocks_languages_output_and_continue_marker() -> None:
    blocks = code_blocks(CHAPTER)
    assert [b.language for b in blocks] == ["python", "output", "python", "bash", ""]
    assert blocks[0].expected_output == "6\n" and blocks[0].line == 13 and blocks[0].closed
    assert blocks[2].continues_previous and blocks[2].expected_output is None
    assert blocks[0].executable and not blocks[3].executable
    d = blocks[0].to_dict()
    assert d["has_expected_output"] and d["index"] == 1


def test_code_blocks_unterminated_fence_and_pandoc_attribute_form() -> None:
    text = "# T\n\n```{.python #one}\nprint(1)\n\nno close"
    (b,) = code_blocks(text)
    assert b.language == "python" and not b.closed
    assert b.code == "print(1)\n\nno close\n"


def test_output_block_only_counts_when_adjacent() -> None:
    text = "```python\nprint(1)\n```\n\nsome prose\n\n```output\n1\n```\n"
    blocks = code_blocks(text)
    assert blocks[0].expected_output is None


def test_citations_are_distinct_outside_code_with_lines() -> None:
    found = citations(CHAPTER)
    assert [c["locator"] for c in found] == ["textbook:p.12", "lecture:t=00:14:32", "notes:L7"]
    assert found[0]["count"] == 2 and found[0]["line"] == 11 and found[0]["source_id"] == "textbook"
    assert citations("```python\nx = '[src:p.1]'\n```\n") == []


def test_placeholders_and_strip_code_keep_line_numbers() -> None:
    refs = placeholders(CHAPTER)
    assert [(r["kind"], r["id"]) for r in refs] == [("film", "f1"), ("widget", "w1"), ("exercises", "set-1")]
    stripped = strip_code(CHAPTER)
    assert len(stripped.splitlines()) == len(CHAPTER.splitlines())
    assert "print(sum" not in stripped and "{{film: f1}}" in stripped


def test_concept_list_normalises() -> None:
    assert concept_list(["Vector", " vector_addition ", "vector"]) == ["vector", "vector addition"]
    assert concept_list("a, b; c") == ["a", "b", "c"]
    assert concept_list(None) == [] and concept_list(42) == []


# ----------------------------------------------------------------------------- derivation
def _unit(kind: str, uid: str = "u1", **kw) -> Unit:
    return Unit(id=uid, kind=kind, title=uid, **kw)  # type: ignore[arg-type]


def test_chapter_obligations_cover_structure_render_blocks_citations_readability_prereqs_judge() -> None:
    unit = _unit("chapter", "chapter-vectors", path="book/chapters/01-vectors.md", meta={"level": "undergraduate"})
    obs = derive_obligations(unit, CHAPTER)
    kinds = [o.kind for o in obs]
    assert kinds[:2] == ["structure", "render"]
    executes = [o for o in obs if o.kind == "execute"]
    assert [o.args["block"] for o in executes] == [1, 3, 4, 5]
    assert executes[0].status == "open" and executes[0].closer == "check_execute"
    assert executes[2].status == "blocked" and executes[2].reason == "no executor for bash"
    assert executes[3].status == "blocked" and "(no language)" in executes[3].reason
    cites = [o for o in obs if o.kind == "cite"]
    assert [o.args["locator"] for o in cites] == ["textbook:p.12", "lecture:t=00:14:32", "notes:L7"]
    assert all(o.required_authority == "source" for o in cites)
    readability = next(o for o in obs if o.kind == "readability")
    assert readability.args["level"] == "undergraduate"
    assert any(o.kind == "prereqs" for o in obs)
    judge = next(o for o in obs if o.kind == "judge")
    assert judge.advisory and judge.required_authority == "model_inference" and judge.closer == "check_judge"
    assert all(o.target == "book/chapters/01-vectors.md" and o.args["unit_id"] == "chapter-vectors" for o in obs)
    assert len({o.id for o in obs}) == len(obs)


def test_chapter_without_content_still_owes_structure_and_render() -> None:
    obs = derive_obligations(_unit("section", "s1", path="book/chapters/02.md"), None)
    assert [o.kind for o in obs] == ["structure", "render", "readability", "judge"]


def test_prereqs_obligation_comes_from_unit_meta_when_front_matter_is_silent(tmp_path) -> None:
    unit = _unit("lesson", "l1", path="x.md", meta={"requires": ["sets"]})
    obs = derive_obligations(unit, "---\ntitle: T\n---\n# T\n")
    assert any(o.kind == "prereqs" for o in obs)
    obs2 = derive_obligations(_unit("lesson", "l2", path="y.md"), "---\ntitle: T\n---\n# T\n")
    assert not any(o.kind == "prereqs" for o in obs2)


def test_readability_level_falls_back_to_outline_persona(workspace) -> None:
    outline = Outline(workspace.path("outline"))
    outline.persona = "high school student"
    obs = derive_obligations(_unit("chapter", "c1", path="c.md"), "---\ntitle: T\n---\n# T\n", outline)
    assert next(o for o in obs if o.kind == "readability").args["level"] == "high school student"


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("film", [("render", "film_render"), ("measure", "check_measure"), ("look", "film_look"), ("judge", "check_judge")]),
        ("widget", [("render", "widget_run"), ("measure", "check_measure"), ("judge", "check_judge")]),
        ("figure", [("render", "diagram_render"), ("measure", "check_measure")]),
        ("exercise_set", [("keys", "assess_items_check"), ("judge", "check_judge")]),
        ("assessment", [("keys", "assess_items_check"), ("judge", "check_judge")]),
        ("integration", [("structure", "book_build"), ("render", "book_build")]),
    ],
)
def test_media_assessment_and_integration_kinds(kind: str, expected: list[tuple[str, str]]) -> None:
    obs = derive_obligations(_unit(kind, "x1"), None)
    assert [(o.kind, o.closer) for o in obs] == expected
    assert all(o.required_authority == ("model_inference" if o.kind == "judge" else "controlled_replay") for o in obs)
    assert all(o.advisory == (o.kind == "judge") for o in obs)


def test_default_targets_follow_the_workspace_layout() -> None:
    assert derive_obligations(_unit("film", "intro"), None)[0].target == "media/films/intro"
    assert derive_obligations(_unit("widget", "w"), None)[0].target == "media/widgets/w.html"
    assert derive_obligations(_unit("figure", "f"), None)[0].target == "media/diagrams/f.svg"
    assert derive_obligations(_unit("exercise_set", "s"), None)[0].target == "assessments/s/items.json"
    assert derive_obligations(_unit("film", "p", path="media/films/custom"), None)[0].target == "media/films/custom"
    assert derive_obligations(_unit("film", "m", meta={"path": "media/films/meta"}), None)[0].target == "media/films/meta"


def test_research_owes_a_citation_per_locator_and_an_advisory_judge() -> None:
    obs = derive_obligations(_unit("research", "r1", path="knowledge/brief.md"), "Claim [a:p.1] and [b:L3].\n")
    assert [(o.kind, o.args.get("locator")) for o in obs] == [("cite", "a:p.1"), ("cite", "b:L3"), ("judge", None)]


def test_extra_checks_add_obligations_without_duplicating_closers() -> None:
    unit = _unit("chapter", "c", path="c.md", checks=["check_render", "widget_run", "mystery_tool"])
    obs = derive_obligations(unit, "---\ntitle: T\n---\n# T\n")
    closers = [o.closer for o in obs]
    assert closers.count("check_render") == 2  # structure and render share it; `checks` adds nothing for it
    extra = [o for o in obs if o.closer in ("widget_run", "mystery_tool")]
    assert [(o.kind, o.closer) for o in extra] == [("render", "widget_run"), ("render", "mystery_tool")]


def test_custom_kind_has_no_default_obligations() -> None:
    assert derive_obligations(_unit("custom", "c"), "text") == []
    assert CLOSERS["section"] is CLOSERS["chapter"]


# ----------------------------------------------------------------------------- closing
def _ob(kind: str = "execute", authority: str = "controlled_replay", **kw) -> Obligation:
    return Obligation(id=f"u:{kind}", kind=kind, description="d", target="t", required_authority=authority, closer="check_" + kind, **kw)


def test_close_obligation_refuses_a_weaker_verifier() -> None:
    ob = close_obligation(_ob(), "model_inference", ["ev_1"], True)
    assert ob.status == "open" and ob.reason == "verifier authority too weak: model_inference < controlled_replay"
    assert ob.evidence_ids == []
    ob = close_obligation(_ob("cite", "source"), "external_service", ["ev_1"], True)
    assert ob.status == "open" and "external_service < source" in ob.reason


def test_close_obligation_accepts_equal_or_stronger_authority_and_records_evidence() -> None:
    ob = close_obligation(_ob(), "controlled_replay", ["ev_1", "ev_1", ""], True, "ran")
    assert ob.status == "closed" and ob.evidence_ids == ["ev_1"] and ob.reason == "ran"
    ob = close_obligation(_ob("judge", "model_inference"), "source", ["ev_2"], True)
    assert ob.status == "closed" and ob.reason.startswith("closed by check_judge")
    ob = close_obligation(_ob(), "controlled_replay", None, False)
    assert ob.status == "open" and ob.reason == "check_execute reported defects"
    ob = close_obligation(_ob(), "nonsense", [], True)
    assert ob.status == "open" and "unknown verifier authority" in ob.reason


def test_blocking_and_summary_treat_advisory_and_blocked_apart() -> None:
    obs = [_ob(), _ob("judge", "model_inference", advisory=True), _ob("cite", "source", status="blocked", reason="no executor")]
    assert [o.id for o in blocking_obligations(obs)] == ["u:execute"]
    s = ledger_summary(obs)
    assert s == {"total": 3, "by_status": {"open": 2, "blocked": 1}, "blocking_open": ["u:execute"], "advisory_open": ["u:judge"], "blocked": ["u:cite"]}
    d = obs[1].to_dict()
    assert d["advisory"] is True and Obligation.from_dict(d) == obs[1]


def test_inline_code_spans_carry_no_citations_or_placeholders() -> None:
    text = "Write citations as `[src:p.1]` and films as `{{film: demo}}`; a real one: [src:p.2].\n"
    assert [c["locator"] for c in citations(text)] == ["src:p.2"]
    assert placeholders(text) == []
