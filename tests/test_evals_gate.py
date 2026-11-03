import json

from edgent.domain.evidence import Evidence, EvidenceStore
from edgent.evals.gate import FixtureClaim, Scenario, evaluate, load_scenarios, record_results
from edgent.observability.tracing import TraceEvent


def ev(seq, kind, data):
    return TraceEvent(seq, 0.0, "s", kind, data)


def test_gate_passes_and_fails(workspace, tmp_path):
    store = EvidenceStore(workspace.path("evidence"))
    store.put(Evidence("source_passage", {"locator": "t:p.1"}, {"quote": "d < 1"}, "observed", "source", subject={"source_id": "t"}))
    sc = Scenario(
        id="cite", task="why", expected_first_tool="grep", required_subsequence=["grep", "read_file", "sources_cite_check"],
        forbidden_tools=["bash"], answer_must_contain=["d < 1"], epistemic_cue_required=True,
        claims=[FixtureClaim("c1", "d < 1", "source_passage", "source", "observed", "result.quote")],
    )
    events = [
        ev(1, "model_call", {"usage": {"prompt_tokens": 100, "completion_tokens": 10}}),
        ev(2, "tool_call", {"name": "grep", "arguments": {"pattern": "damping"}, "ok": True}),
        ev(3, "tool_call", {"name": "read_file", "arguments": {"path": "a"}, "ok": True}),
        ev(4, "tool_call", {"name": "sources_cite_check", "arguments": {"locator": "t:p.1"}, "ok": True}),
        ev(5, "model_call", {"usage": {"prompt_tokens": 120, "completion_tokens": 20}}),
    ]
    r = evaluate(sc, events, "The source says d < 1 (unverified beyond page 1).", store)
    assert r.passed, r.failed_checks()
    assert r.prompt_tokens == 220 and r.tool_calls == 3
    bad_events = events + [ev(6, "tool_call", {"name": "grep", "arguments": {"pattern": "damping"}, "ok": False, "error": {"code": "invalid_input"}}), ev(7, "tool_call", {"name": "bash", "arguments": {"command": "ls"}, "ok": True})]
    r2 = evaluate(sc, bad_events, "d > 1", store, stop_reason="max_iterations")
    failed = set(r2.failed_checks())
    assert {"exit_ok", "no_repeated_calls", "no_validation_failures", "no_forbidden_tools", "answer_contains:d < 1", "epistemic_cue"} <= failed
    weak = EvidenceStore(tmp_path / "ev2")
    weak.put(Evidence("source_passage", {"locator": "t:p.1"}, {"quote": "d < 1"}, "inferred", "model_inference"))
    r3 = evaluate(sc, events, "d < 1 unverified", weak)
    assert "claim:c1" in r3.failed_checks() and "too weak" in [c["detail"] for c in r3.checks if c["name"] == "claim:c1"][0]


def test_load_and_record(tmp_path):
    (tmp_path / "s.json").write_text(json.dumps([{"id": "a", "task": "t", "claims": [{"id": "x", "expected": 1, "evidence_operation": "op"}]}, {"id": "b", "task": "u"}]))
    scs = load_scenarios(tmp_path)
    assert [s.id for s in scs] == ["a", "b"] and scs[0].claims[0].required_authority == "source"
    r = evaluate(scs[1], [], "ok", None)
    row = record_results([r], tmp_path / "results.md", "unit")
    assert "1/1" in row and (tmp_path / "results.md").read_text().count("| when |") == 1
    record_results([r], tmp_path / "results.md", "again")
    assert (tmp_path / "results.md").read_text().count("| again |") == 1
