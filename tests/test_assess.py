"""Assessment tools: creation through the mock examiner, mechanical key checks in the sandbox, rendering, grading."""
from __future__ import annotations

import datetime as dt
import json
from typing import Any

import pytest

from edgent.adapters.providers.base import ProviderError
from edgent.domain.learner import LearnerStore
from edgent.domain.utils import FakeClock
from edgent.tools.edu import assess as assess_mod
from edgent.tools.edu.assess import (
    ASSESS_TOOL_NAMES,
    build_assess_registry,
    learner_view,
    normalise_item,
    validate_item,
)
from edgent.tools.registry import ToolContext, ToolRegistry

T0 = dt.datetime(2026, 11, 3, 9, 0, tzinfo=dt.UTC)


def item_set(**over: Any) -> dict[str, Any]:
    base = {
        "set_id": "q1",
        "title": "Vectors quiz",
        "objectives": ["add two vectors"],
        "level": "undergraduate",
        "items": [
            {"id": "i1", "kind": "numeric", "stem": "First entry of (1,2)+(3,4)?", "answer": 4, "tolerance": 0, "solution": "1+3=4", "concept": "vector addition", "difficulty": 1, "verify": {"type": "numeric", "expression": "1+3"}},
            {"id": "i2", "kind": "mcq", "stem": "Which of these is a vector? Think of $\\vec v$.", "options": ["(1, 2)", "the number 3", "x^2"], "answer": 0, "solution": "a list of numbers is a vector", "concept": "vector", "difficulty": 1, "verify": {"type": "none"}},
            {"id": "i3", "kind": "short", "stem": "Name the operation written a·b.", "answer": "dot product", "solution": "the dot (scalar, inner) product", "concept": "dot product", "difficulty": 2, "verify": {"type": "regex", "pattern": "(the )?(dot|scalar|inner) product"}},
            {"id": "i4", "kind": "code", "stem": "Write add(a, b) for two lists.", "answer": "def add(a, b):\n    return [x + y for x, y in zip(a, b)]", "solution": "zip the lists", "concept": "vector addition", "difficulty": 2, "verify": {"type": "code", "code": "ns = {}\nexec(answer, ns)\nassert ns['add']([1, 2], [3, 4]) == [4, 6]"}},
            {"id": "i5", "kind": "proof", "stem": "Show that u+v = v+u.", "answer": "componentwise commutativity of addition", "solution": "each entry commutes", "concept": "vector addition", "difficulty": 3, "verify": {"type": "none"}},
        ],
    }
    base.update(over)
    return base


@pytest.fixture
def tools(registry: ToolRegistry, ctx: ToolContext) -> ToolRegistry:
    ctx.extra["clock"] = FakeClock(T0.timestamp())
    return build_assess_registry(registry)


def ok(tools: ToolRegistry, ctx: ToolContext, name: str, **args: Any) -> dict[str, Any]:
    r = tools.call(name, args, ctx)
    assert r.ok, r.error
    return r.output


def err(tools: ToolRegistry, ctx: ToolContext, name: str, **args: Any) -> Any:
    r = tools.call(name, args, ctx)
    assert not r.ok, r.output
    assert r.error.remediation
    return r.error


def write_key(ctx: ToolContext, data: dict[str, Any]) -> None:
    ctx.workspace.write_json(f"assessments/{data['set_id']}/key.json", data)
    ctx.workspace.write_json(f"assessments/{data['set_id']}/items.json", learner_view(data))


# ----------------------------------------------------------------------------- contracts
def test_registry_contracts(tools: ToolRegistry) -> None:
    assert tools.names() == sorted(ASSESS_TOOL_NAMES)
    expected = {"assess_create": "write", "assess_items_check": "compute", "assess_render": "write", "assess_grade": "write"}
    for name, permission in expected.items():
        spec = tools.get(name)
        assert spec is not None and spec.permission == permission and spec.group == "assess", name
        assert spec.parameters["additionalProperties"] is False and spec.when_not_to_use and len(spec.description) > 80
        for prop, schema in spec.parameters["properties"].items():
            assert schema.get("description"), f"{name}.{prop}"
    assert tools.get("assess_create").side_effects and tools.get("assess_create").cost_class == "expensive"
    assert not tools.get("assess_items_check").side_effects and tools.get("assess_items_check").timeout_s > assess_mod.KEY_CHECK_TIMEOUT_LIMIT.value
    assert not tools.get("assess_render").side_effects and tools.get("assess_grade").side_effects  # a re-render must see the current key; a re-grade is a retry
    assert tools.schema_bytes(compact=True) < 6000
    again = build_assess_registry(tools)
    assert again is tools and len(tools.names()) == 4


def test_item_validation_rules() -> None:
    good = [normalise_item(it, i) for i, it in enumerate(item_set()["items"], 1)]
    assert all(validate_item(it) == [] for it in good)
    assert validate_item(normalise_item({"kind": "mcq", "stem": "s", "options": ["a", "b"], "answer": 0, "concept": "c"}, 1)) == ["mcq needs 3-5 non-empty text options"]
    assert "mcq answer must be an integer index" in validate_item(normalise_item({"kind": "mcq", "stem": "s", "options": ["a", "b", "c"], "answer": 3, "concept": "c"}, 1))[0]
    assert validate_item(normalise_item({"kind": "numeric", "stem": "s", "answer": "four", "concept": "c"}, 1)) == ["numeric answer must be a number"]
    assert validate_item(normalise_item({"kind": "essay", "stem": "s", "answer": "x", "concept": "c"}, 1))[0].startswith("kind must be one of")
    issues = validate_item(normalise_item({"kind": "short", "stem": "", "answer": "", "concept": "", "difficulty": 9, "verify": {"type": "regex"}}, 1))
    assert len(issues) == 5 and "verify.pattern is required for verify.type regex" in issues
    assert validate_item(normalise_item("not an object", 3)) == ["item is not an object"]
    assert validate_item(normalise_item({"kind": "short", "stem": "s", "answer": "x", "concept": "c" * 121, "verify": {"type": "none"}}, 1)) == ["concept longer than 120 chars"]
    norm = normalise_item({"kind": "mcq", "stem": " s ", "options": ["a", "b", "c"], "answer": 1.0, "difficulty": 2.0, "concept": "c"}, 7)
    assert norm["id"] == "i7" and norm["answer"] == 1 and norm["difficulty"] == 2 and norm["stem"] == "s" and norm["verify"] == {"type": "none"} and norm["solution"] == ""


# ----------------------------------------------------------------------------- assess_create
def test_create_writes_key_and_learner_items_and_drops_invalid(tools: ToolRegistry, ctx: ToolContext, provider: Any) -> None:
    data = item_set()
    data["items"].append({"id": "bad", "kind": "mcq", "stem": "broken", "options": ["only"], "answer": 7, "concept": "x", "difficulty": 1})
    data["items"].append({"id": "i1", "kind": "numeric", "stem": "duplicate id", "answer": 1, "concept": "x", "difficulty": 1, "verify": {"type": "none"}})
    provider.push("Here is the set:\n```json\n" + json.dumps(data) + "\n```")
    out = ok(tools, ctx, "assess_create", set_id="q1", objectives=["add two vectors", " compute a dot product "], n_items=6, level="undergraduate", kinds=["numeric", "mcq", "short", "code", "proof"], concepts=["vector addition"])
    assert out["items"] == 5 and out["item_ids"] == ["i1", "i2", "i3", "i4", "i5"] and out["dropped"] == ["bad", "i1"]
    assert out["issues"]["i1"] == ["duplicate id"] and out["kinds"] == {"numeric": 1, "mcq": 1, "short": 1, "code": 1, "proof": 1}
    assert out["untrusted"] is True and "assess_items_check" in out["next"] and "note" in out
    key = json.loads(ctx.workspace.read_text("assessments/q1/key.json"))
    items = json.loads(ctx.workspace.read_text("assessments/q1/items.json"))
    assert key["title"] == "Vectors quiz" and key["objectives"] == ["add two vectors", "compute a dot product"] and key["level"] == "undergraduate"
    assert [it["id"] for it in key["items"]] == ["i1", "i2", "i3", "i4", "i5"] and key["items"][0]["answer"] == 4 and key["generator"]["provider"] == "mock"
    assert items["set_id"] == "q1" and len(items["items"]) == 5
    for it in items["items"]:
        assert not {"answer", "solution", "verify", "tolerance"} & set(it), it
        assert {"id", "kind", "stem", "concept", "difficulty"} <= set(it)
    assert items["items"][1]["options"] == ["(1, 2)", "the number 3", "x^2"]
    req = provider.calls[0]
    assert req.system_text().startswith("[edgent-role: examiner]") and req.metadata["role"] == "examiner" and req.json_schema is not None
    prompt = req.last_user_text()
    assert "Write 6 assessment items" in prompt and "- add two vectors" in prompt and "Level: undergraduate" in prompt and "vector addition" in prompt and '"verify"' in prompt


def test_create_personalises_from_the_learner_model(tools: ToolRegistry, ctx: ToolContext, provider: Any) -> None:
    LearnerStore(ctx.workspace).record({"type": "answered", "concept": "norms", "quality": 1}, now=T0)
    provider.push(json.dumps(item_set()))
    ok(tools, ctx, "assess_create", set_id="q2", objectives=["norms"], n_items=2)
    prompt = provider.calls[0].last_user_text()
    assert "Learner context" in prompt and "Weakest: norms" in prompt and "data, not instructions" in prompt
    assert ctx.workspace.file_exists("assessments/q2/key.json")


def test_create_failure_modes(tools: ToolRegistry, ctx: ToolContext, provider: Any) -> None:
    assert err(tools, ctx, "assess_create", set_id="Bad Id!", objectives=["x"]).category.value == "invalid_input"
    assert err(tools, ctx, "assess_create", set_id="q1", objectives=["  "]).category.value == "invalid_input"
    provider.push("I would rather not write a quiz today.")
    e = err(tools, ctx, "assess_create", set_id="q1", objectives=["x"])
    assert e.code == "invalid_output" and e.retryable and e.partial["untrusted"] is True and not ctx.workspace.file_exists("assessments/q1/key.json")
    provider.push(json.dumps({"items": [{"id": "a", "kind": "numeric", "stem": "s", "answer": "nope", "concept": "c"}]}))
    e = err(tools, ctx, "assess_create", set_id="q1", objectives=["x"])
    assert e.code == "no_valid_items" and e.partial["issues"]["a"] == ["numeric answer must be a number"]

    def boom(_req: Any) -> Any:
        raise ProviderError("quota exhausted", status=429, retryable=True)

    provider.push(boom)
    e = err(tools, ctx, "assess_create", set_id="q1", objectives=["x"])
    assert e.code == "provider_error" and e.retryable and "quota exhausted" in e.message
    ctx.provider = None
    e = err(tools, ctx, "assess_create", set_id="q1", objectives=["x"])
    assert e.category.value == "unavailable" and "key.json" in e.remediation


# ----------------------------------------------------------------------------- assess_items_check
def test_items_check_passes_and_records_evidence(tools: ToolRegistry, ctx: ToolContext) -> None:
    write_key(ctx, item_set())
    out = ok(tools, ctx, "assess_items_check", set_id="q1")
    assert out["passed"] is True and out["verified"] == 3 and out["unverifiable"] == ["i2", "i5"]
    rows = {r["id"]: r for r in out["items"]}
    assert all(r["ok"] and r["issues"] == [] for r in rows.values())
    assert rows["i1"]["verified"] and rows["i3"]["verified"] and rows["i4"]["verified"] and not rows["i2"]["verified"]
    assert len(out["evidence_ids"]) == 1
    ev = ctx.evidence.get(out["evidence_ids"][0])
    assert ev is not None and ev.operation == "assessment_key_check" and ev.confidence == "derived" and ev.authority == "controlled_replay"
    assert ev.result["verified_items"] == ["i1", "i3", "i4"] and ev.subject["source_id"] == "assessment:q1" and ev.limitations
    report = json.loads(ctx.workspace.read_text("assessments/q1/check.json"))
    assert report["passed"] and report["evidence_ids"] == out["evidence_ids"]
    assert ok(tools, ctx, "assess_items_check", set_id="q1")["evidence_ids"] == out["evidence_ids"]  # same key, same record


def test_items_check_detects_a_wrong_numeric_key(tools: ToolRegistry, ctx: ToolContext) -> None:
    data = item_set()
    data["items"][0]["answer"] = 5  # 1+3 is not 5
    write_key(ctx, data)
    out = ok(tools, ctx, "assess_items_check", set_id="q1")
    assert out["passed"] is False and out["evidence_ids"] == [] and "hint" in out
    row = next(r for r in out["items"] if r["id"] == "i1")
    assert not row["ok"] and "differs from the recomputed value 4" in row["issues"][0]
    assert ctx.evidence.count() == 0


def test_items_check_tolerance_regex_code_and_distractors(tools: ToolRegistry, ctx: ToolContext) -> None:
    data = item_set()
    data["items"] = [
        {"id": "pi", "kind": "numeric", "stem": "pi to 2 places", "answer": 3.14, "tolerance": 0.01, "concept": "c", "difficulty": 1, "verify": {"type": "numeric", "expression": "math.pi"}},
        {"id": "tight", "kind": "numeric", "stem": "pi exactly", "answer": 3.14, "tolerance": 0, "concept": "c", "difficulty": 1, "verify": {"type": "numeric", "expression": "math.pi"}},
        {"id": "nan", "kind": "numeric", "stem": "oops", "answer": 1, "concept": "c", "difficulty": 1, "verify": {"type": "numeric", "expression": "1/0"}},
        {"id": "rx", "kind": "short", "stem": "s", "answer": "gradient", "concept": "c", "difficulty": 1, "verify": {"type": "regex", "pattern": "slope"}},
        {"id": "rxbad", "kind": "short", "stem": "s", "answer": "gradient", "concept": "c", "difficulty": 1, "verify": {"type": "regex", "pattern": "("}},
        {"id": "code", "kind": "code", "stem": "s", "answer": "x = 1", "concept": "c", "difficulty": 1, "verify": {"type": "code", "code": "assert answer == 'x = 2'"}},
        {"id": "dup", "kind": "mcq", "stem": "s", "options": ["Alpha", "alpha ", "beta"], "answer": 0, "concept": "c", "difficulty": 1},
        {"id": "dup2", "kind": "mcq", "stem": "s", "options": ["alpha", "Beta", "beta.", "gamma"], "answer": 0, "concept": "c", "difficulty": 1},
    ]
    write_key(ctx, data)
    out = ok(tools, ctx, "assess_items_check", set_id="q1")
    rows = {r["id"]: r for r in out["items"]}
    assert rows["pi"]["ok"] and rows["pi"]["verified"]
    assert not rows["tight"]["ok"] and "tolerance" in rows["tight"]["issues"][0]
    assert not rows["nan"]["ok"] and "ZeroDivisionError" in rows["nan"]["issues"][0]
    assert rows["rx"]["issues"] == ["verify.pattern does not match the key answer"]
    assert rows["rxbad"]["issues"][0].startswith("verify.pattern does not compile")
    assert not rows["code"]["ok"] and "AssertionError" in rows["code"]["issues"][0]
    assert rows["dup"]["issues"] == ["more than one option reads as the correct answer 'Alpha'"]
    assert rows["dup2"]["issues"] == ["distractors are not distinct: 'beta'"]
    assert out["passed"] is False and out["verified"] == 1


def test_items_check_sandbox_is_isolated_and_reports_a_dead_run(tools: ToolRegistry, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:
    data = item_set()
    data["items"] = [{"id": "env", "kind": "numeric", "stem": "s", "answer": 0, "concept": "c", "difficulty": 1, "verify": {"type": "numeric", "expression": "len([k for k in __import__('os').environ if k == 'EDGENT_HOME'])"}}]
    write_key(ctx, data)
    assert ok(tools, ctx, "assess_items_check", set_id="q1")["passed"] is True  # the scrubbed sandbox environment has no EDGENT_HOME
    monkeypatch.setattr(assess_mod, "_VERIFY_HARNESS", "import sys\nsys.exit(3)\n")
    out = ok(tools, ctx, "assess_items_check", set_id="q1")
    assert out["passed"] is False and "printed no result" in out["items"][0]["issues"][0]


def test_items_check_not_found_and_broken_key(tools: ToolRegistry, ctx: ToolContext) -> None:
    e = err(tools, ctx, "assess_items_check", set_id="nope")
    assert e.category.value == "not_found" and "assess_create" in e.remediation
    ctx.workspace.write_text("assessments/q1/key.json", "{not json")
    assert err(tools, ctx, "assess_items_check", set_id="q1").category.value == "integrity"
    ctx.workspace.write_json("assessments/q1/key.json", {"set_id": "q1", "items": []})
    assert err(tools, ctx, "assess_items_check", set_id="q1").category.value == "integrity"
    assert err(tools, ctx, "assess_items_check", set_id="../etc").category.value == "invalid_input"


# ----------------------------------------------------------------------------- assess_render
def test_render_omits_answers_unless_with_key(tools: ToolRegistry, ctx: ToolContext) -> None:
    write_key(ctx, item_set())
    out = ok(tools, ctx, "assess_render", set_id="q1")
    assert out["path"] == "assessments/q1/rendered.html" and out["artifact_path"] == out["path"] and out["items"] == 5 and out["with_key"] is False
    page = ctx.workspace.read_text("assessments/q1/rendered.html")
    assert page.startswith("<!doctype html>") and "<script" not in page and "<style>" in page and "Vectors quiz" in page
    assert "First entry of (1,2)+(3,4)?" in page and "the number 3" in page and "$\\vec v$" in page and "not rendered" in page
    for secret in ("1+3=4", "a list of numbers is a vector", "dot product", "zip the lists", "zip(a, b)", "componentwise", "verify"):
        assert secret not in page, secret
    assert page.count('class="item"') == 5 and 'id="item-i3"' in page
    out = ok(tools, ctx, "assess_render", set_id="q1", with_key=True)
    key_page = ctx.workspace.read_text(out["path"])
    assert out["path"] == "assessments/q1/rendered_key.html" and "answer key" in key_page
    assert "A. (1, 2)" in key_page and "1+3=4" in key_page and "dot product" in key_page and "zip(a, b)" in key_page and "verified by numeric" in key_page
    data = item_set()
    data["items"][0]["answer"] = "four"
    write_key(ctx, data)
    assert err(tools, ctx, "assess_render", set_id="q1").category.value == "invalid_input"
    data = item_set()
    data["items"][1]["stem"] = "<script>alert(1)</script> & co"
    write_key(ctx, data)
    page = ctx.workspace.read_text(ok(tools, ctx, "assess_render", set_id="q1")["path"])
    assert "<script" not in page and "&lt;script&gt;" in page and "&amp; co" in page


# ----------------------------------------------------------------------------- assess_grade
def test_grade_scores_and_updates_mastery(tools: ToolRegistry, ctx: ToolContext) -> None:
    write_key(ctx, item_set())
    out = ok(tools, ctx, "assess_grade", set_id="q1", responses={"i1": "4.0", "i2": "b", "i3": "The Scalar Product", "i5": "they commute"})
    rows = {r["id"]: r for r in out["items"]}
    assert rows["i1"]["status"] == "correct" and rows["i3"]["status"] == "correct"
    assert rows["i2"]["status"] == "wrong" and rows["i2"]["given"] == 1 and rows["i2"]["expected"] == "A. (1, 2)" and rows["i2"]["solution"] == "a list of numbers is a vector"
    assert rows["i4"]["status"] == "blank" and rows["i4"]["expected"].startswith("def add")
    assert rows["i5"]["status"] == "needs_review" and "expected" not in rows["i5"]
    assert out["score"] == {"correct": 2, "wrong": 1, "blank": 1, "needs_review": 1, "total": 5, "gradable": 4, "score": 0.5, "percent": 50}
    assert out["untrusted"] is True and out["progress"]["points"] == 20 and out["progress"]["events_total"] == 5 and out["progress"]["streak_days"] == 1
    store = LearnerStore(ctx.workspace)
    concepts = store.mastery()["concepts"]
    assert concepts["dot product"]["repetitions"] == 1 and concepts["dot product"]["level"] == 3.0
    assert concepts["vector"]["lapses"] == 1 and concepts["vector"]["repetitions"] == 0
    assert concepts["vector addition"]["lapses"] == 1  # i1 correct then i4 blank on the same concept
    events = store.ledger()["events"]
    assert [(e["type"], e["quality"]) for e in events] == [("answered", 5.0), ("answered", 1.0), ("answered", 5.0), ("answered", 0.0), ("assessment_done", None)]
    assert events[0]["detail"] == "q1/i1: correct" and events[0]["session_id"] == "s1" and events[0]["at"] == T0.isoformat()
    assert events[-1]["detail"] == "q1: 2/4 correct, 1 need review"
    grades = json.loads(ctx.workspace.read_text("assessments/q1/grades.json"))
    assert grades["score"]["percent"] == 50 and grades["responses"]["i2"] == "b"
    e = err(tools, ctx, "assess_grade", set_id="q1", responses={"i9": 1})
    assert e.category.value == "invalid_input" and "i9" in e.message and "i1, i2" in e.remediation


def test_grade_mcq_response_shapes_and_numeric_tolerance(tools: ToolRegistry, ctx: ToolContext) -> None:
    data = item_set()
    data["items"][0]["tolerance"] = 0.5
    write_key(ctx, data)
    for resp, status in ((0, "correct"), ("A", "correct"), ("0", "correct"), ("(1, 2)", "correct"), ("1", "wrong"), ("Z", "wrong"), ("nonsense", "wrong"), (2.0, "wrong")):
        out = ok(tools, ctx, "assess_grade", set_id="q1", responses={"i2": resp, "i1": "4.4"})
        rows = {r["id"]: r for r in out["items"]}
        assert rows["i2"]["status"] == status, resp
        assert rows["i1"]["status"] == "correct"
    assert {r["id"]: r["status"] for r in ok(tools, ctx, "assess_grade", set_id="q1", responses={"i1": "4.6"})["items"]}["i1"] == "wrong"
    assert {r["id"]: r["status"] for r in ok(tools, ctx, "assess_grade", set_id="q1", responses={"i1": "four"})["items"]}["i1"] == "wrong"


def test_grade_runs_code_items_in_the_sandbox(tools: ToolRegistry, ctx: ToolContext) -> None:
    write_key(ctx, item_set())
    good = "def add(a, b):\n    return [x + y for x, y in zip(a, b)]"
    out = ok(tools, ctx, "assess_grade", set_id="q1", responses={"i4": good})
    assert {r["id"]: r["status"] for r in out["items"]}["i4"] == "correct"
    out = ok(tools, ctx, "assess_grade", set_id="q1", responses={"i4": "def add(a, b):\n    return a + b"})
    row = {r["id"]: r for r in out["items"]}["i4"]
    assert row["status"] == "wrong" and "AssertionError" in row["note"] and row["expected"] == good
    out = ok(tools, ctx, "assess_grade", set_id="q1", responses={"i4": "import sys; sys.exit(0)"})
    assert {r["id"]: r["status"] for r in out["items"]}["i4"] == "wrong"


# ----------------------------------------------------------------------------- review additions
def test_verify_scripts_are_unique_per_call_and_parallel_grades_do_not_cross(tools: ToolRegistry, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:
    write_key(ctx, item_set())
    names: list[Any] = []
    real = assess_mod.run_python

    def spy(code: str, cwd: str, **kw: Any) -> Any:
        names.append(kw.get("filename"))
        return real(code, cwd, **kw)

    monkeypatch.setattr(assess_mod, "run_python", spy)
    ok(tools, ctx, "assess_items_check", set_id="q1")
    ok(tools, ctx, "assess_items_check", set_id="q1")
    assert len(names) == 2 and names[0] != names[1] and all(n.startswith("assess_verify_q1_") for n in names)
    good = "def add(a, b):\n    return [x + y for x, y in zip(a, b)]"
    bad = "def add(a, b):\n    return a + b"
    calls = [("assess_grade", {"set_id": "q1", "responses": {"i4": good}}, None), ("assess_grade", {"set_id": "q1", "responses": {"i4": bad}}, None)]
    results = tools.call_parallel(calls, ctx)
    assert all(r.ok for r in results), [r.error for r in results]
    assert [{row["id"]: row["status"] for row in r.output["items"]}["i4"] for r in results] == ["correct", "wrong"]
    assert len(names) == 4 and len(set(names)) == 4


def test_items_check_duplicate_ids_keep_results_on_their_own_rows(tools: ToolRegistry, ctx: ToolContext) -> None:
    data = item_set()
    data["items"] = [data["items"][0], dict(data["items"][0], stem="again", answer=5)]
    write_key(ctx, data)
    out = ok(tools, ctx, "assess_items_check", set_id="q1")
    first, second = out["items"]
    assert first == {"id": "i1", "ok": True, "issues": [], "verified": True}
    assert second["issues"] == ["duplicate id"] and not second["ok"] and not second["verified"]
    assert out["passed"] is False and out["verified"] == 1 and out["untrusted"] is True
