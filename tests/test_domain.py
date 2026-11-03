from __future__ import annotations

import json

import pytest

from edgent.domain.errors import ConfigurationError, ErrorCategory, HarnessError, Limit, ToolFailure, err, ok
from edgent.domain.evidence import Evidence, EvidenceIntegrityError, EvidenceStore, authority_at_least
from edgent.domain.outline import Outline, OutlineError, Unit
from edgent.domain.types import CompletionRequest, ContentPart, Message, Reply, ToolCall, Usage
from edgent.domain.unknowns import Unknown, UnknownsLedger, UnknownTransitionError
from edgent.domain.utils import extract_json, parse_front_matter, slugify
from edgent.domain.workspace import Workspace, WorkspaceError


def test_message_roundtrip():
    m = Message.assistant("hi", [ToolCall("c1", "read_file", {"path": "a"})], reasoning="because")
    d = json.loads(json.dumps(m.to_dict()))
    back = Message.from_dict(d)
    assert back.text() == "hi" and back.tool_calls[0].name == "read_file" and back.reasoning == "because"
    t = Message.tool("c1", "read_file", "content", persisted_path="x")
    assert Message.from_dict(t.to_dict()).meta["persisted_path"] == "x"
    img = ContentPart.of_media("image", "/tmp/x.png")
    assert img.mime == "image/png" and Message.user("look", [img]).media()[0].type == "image"


def test_usage_and_reply():
    u = Usage(10, 5, cached_tokens=4) + Usage(1, 1)
    assert u.total_tokens == 17 and u.cached_tokens == 4
    r = Reply(text="x", tool_calls=[ToolCall.new("t")])
    assert r.as_message().tool_calls[0].name == "t"
    req = CompletionRequest([Message.system("sys"), Message.user("q")])
    assert req.system_text() == "sys" and req.last_user_text() == "q" and req.prompt_bytes_estimate() > 0


def test_error_contract():
    e = HarnessError("x", "bad", ErrorCategory.invalid_input, remediation="fix")
    d = e.to_dict()
    assert d["category"] == "invalid_input" and "fix" in d["remediation"]
    assert HarnessError.from_dict(d).code == "x"
    tf = ToolFailure("unavailable", "no ffmpeg", "unavailable", remediation="install", toolchain="ffmpeg")
    assert tf.error.details["toolchain"] == "ffmpeg" and tf.error.category is ErrorCategory.unavailable
    assert ok(1).ok and not err("e").ok
    with pytest.raises(RuntimeError):
        err("e").unwrap()


def test_limit_rule(monkeypatch):
    lim = Limit.from_env("prompt bytes", 1000, "bytes", env="EDGENT_TEST_LIMIT", floor=10, maximum=5000, environ={})
    assert lim.source == "default"
    e = lim.exceeded(1200, "Send less.")
    assert "1200" in e.message and "no runtime override" not in e.remediation and "EDGENT_TEST_LIMIT" in e.remediation
    lim2 = Limit.from_env("prompt bytes", 1000, "bytes", env="EDGENT_TEST_LIMIT", floor=10, maximum=5000, environ={"EDGENT_TEST_LIMIT": "2000"})
    assert lim2.value == 2000 and lim2.source == "setting"
    with pytest.raises(ConfigurationError):
        Limit.from_env("x", 1, "u", env="E", floor=10, environ={"E": "5"})
    with pytest.raises(ConfigurationError):
        Limit.from_env("x", 1, "u", env="E", environ={"E": "abc"})
    fixed = Limit("depth", 10, "levels", does_not_raise=("NODE_OPTIONS",))
    assert "no runtime override" in fixed.exceeded(11, "Flatten.").remediation


def test_utils():
    assert extract_json('text ```json\n{"a": 1}\n``` more') == {"a": 1}
    assert extract_json("prefix {\"a\": [1,2]} suffix") == {"a": [1, 2]}
    assert slugify("Hello, World! 2") == "hello-world-2"
    meta, body = parse_front_matter("---\nname: x\ndescription: y z\ntools: [a, b]\nskills:\n  - s1\n  - s2\nbackground: true\n---\nBody")
    assert meta == {"name": "x", "description": "y z", "tools": ["a", "b"], "skills": ["s1", "s2"], "background": True} and body == "Body"


def test_workspace_init_and_paths(workspace: Workspace):
    assert workspace.exists() and workspace.path("sources_text").is_dir()
    assert (workspace.root / "STUDY.md").read_text().startswith("# Test Workspace")
    assert workspace.is_git_repo() and "edgent init" in workspace.git_log()
    with pytest.raises(WorkspaceError):
        workspace.resolve("../outside")
    with pytest.raises(WorkspaceError):
        workspace.resolve("/etc/passwd")
    workspace.write_text("book/chapters/01-intro.md", "# Intro\n\nHello damping factor\n")
    hits = workspace.search("damping", "book")
    assert hits and hits[0].path == "book/chapters/01-intro.md" and hits[0].line == 3
    assert "book/chapters/01-intro.md" in workspace.glob("book/**/*.md")
    assert workspace.commit("add intro") is not None and workspace.commit("nothing") is None
    assert any(e["path"] == "book/chapters" for e in workspace.list("book"))


def test_workspace_instruction_chain_and_settings(workspace: Workspace, home):
    (home / "STUDY.md").write_text("user-wide")
    (home / "settings.json").write_text(json.dumps({"permissions": {"allow": ["Bash(ls *)"]}, "mode": "ask"}))
    workspace.write_text(".edgent/settings.json", json.dumps({"permissions": {"allow": ["Bash(manim *)"]}, "mode": "auto"}))
    workspace.write_text("book/STUDY.md", "book rules")
    chain = workspace.instruction_chain("book/chapters")
    assert [c[1] for c in chain] == ["user-wide", open(workspace.root / "STUDY.md").read(), "book rules"]
    s = workspace.settings()
    assert s["mode"] == "auto" and s["permissions"]["allow"] == ["Bash(ls *)", "Bash(manim *)"]
    workspace.append_progress("did a thing", "s1")
    assert "did a thing" in workspace.progress_tail()
    assert Workspace.find(workspace.root / "book" / "chapters").root == workspace.root


def test_evidence_store(workspace: Workspace):
    store = EvidenceStore(workspace.path("evidence"))
    ev = Evidence("source_passage", {"query": "damping"}, {"quote": "d < 1"}, "observed", "source", locations=[{"kind": "page", "ref": 12}], subject={"source_id": "textbook"})
    assert ev.evidence_id.startswith("ev_") and ev.verify_id() and ev.locator() == "textbook:p.12"
    assert store.put(ev) == "added" and store.put(ev) == "duplicate"
    same = Evidence.from_dict(ev.to_dict())
    assert same.evidence_id == ev.evidence_id
    tampered = Evidence.from_dict({**ev.to_dict(), "result": {"quote": "d > 1"}})
    with pytest.raises(EvidenceIntegrityError):
        store.put(tampered)
    assert store.get(ev.evidence_id).result == {"quote": "d < 1"} and store.find(operation="source_passage")[0].subject["source_id"] == "textbook"
    assert authority_at_least("source", "model_inference") and not authority_at_least("model_inference", "controlled_replay")


def test_unknowns_ledger(workspace: Workspace):
    ledger = UnknownsLedger(workspace.path("unknowns"))
    u = Unknown("Does the source define the damping factor?", "sources", scope="chapter-1", severity="high", recommended_probes=[{"operation": "deep_research", "rationale": "no source covers it"}])
    assert ledger.record(u) == "added" and ledger.record(Unknown(u.question, u.domain, scope="chapter-1")) == "duplicate"
    assert [x.unknown_id for x in ledger.open()] == [u.unknown_id]
    with pytest.raises(UnknownTransitionError):
        ledger.set_status(u.unknown_id, "resolved")
    with pytest.raises(UnknownTransitionError):
        ledger.resolve(u.unknown_id, "verified", "found it", [])
    ledger.resolve(u.unknown_id, "verified", "found on page 12", ["ev_abc"])
    reloaded = UnknownsLedger(workspace.path("unknowns"))
    got = reloaded.get(u.unknown_id)
    assert got.status == "resolved" and got.revision == 2 and got.history[-1]["to"] == "resolved"
    assert "Closed" in reloaded.render_markdown()
    ledger.set_status(u.unknown_id, "open", "new contradicting source")
    ledger.add_evidence(u.unknown_id, "ev_xyz", contradicting=True)
    assert ledger.get(u.unknown_id).status == "contradicted"


def test_outline_transitions(workspace: Workspace):
    o = Outline(workspace.path("outline"))
    o.title = "Linear Algebra"
    a = o.new_unit("chapter", "Vectors", "what a vector is")
    b = o.new_unit("chapter", "Matrices", depends_on=[a.id])
    o.new_unit("chapter", "Vectors")  # same title -> different id
    assert o.has("chapter-vectors-2")
    assert [u.id for u in o.ready()] == [a.id, "chapter-vectors-2"]
    with pytest.raises(OutlineError):
        o.set_status(b.id, "in_progress")
    with pytest.raises(OutlineError):
        o.set_status(a.id, "done")
    o.set_status(a.id, "in_progress")
    o.set_status(a.id, "partial", open_obligations=[{"kind": "execute", "reason": "code block failed"}])
    assert o.get(a.id).open_obligations and o.get(a.id).attempts == 1
    o.set_status(a.id, "done")
    assert o.get(a.id).open_obligations == [] and b.id in [u.id for u in o.ready()]
    with pytest.raises(OutlineError):
        o.update(a.id, status="pending")
    with pytest.raises(OutlineError):
        o.add(Unit(id=a.id, kind="chapter", title="dup"))
    with pytest.raises(OutlineError):
        o.add(Unit(id="x", kind="chapter", title="x", depends_on=["nope"]))
    reloaded = Outline(workspace.path("outline"))
    assert reloaded.get(a.id).status == "done" and reloaded.summary()["by_status"]["done"] == 1
    assert "`chapter-vectors`" in reloaded.render_markdown()
    # cycle detection
    c = o.new_unit("section", "Cyc", depends_on=[b.id])
    o.get(b.id).depends_on.append(c.id)
    assert any("cycle" in p for p in o.validate())
