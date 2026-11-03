from __future__ import annotations

import json

import pytest

from edgent.application.sessions import STANCE_MAX_ENTRIES, Session, SessionStore
from edgent.domain.types import Message, ToolCall, Usage


def test_create_append_save_load_roundtrip(workspace):
    s = Session.create(workspace, mode="chat", title="Linear algebra", session_id="chat-1")
    assert s.dir == workspace.path("sessions") / "chat-1" and s.state_path.is_file() and s.cleared_dir.is_dir()
    assert s.append(Message.user("What is a basis?")) == 1
    assert s.append(Message.assistant("", [ToolCall("c1", "read_file", {"path": "sources/text/book.md"})])) == 2
    assert s.append(Message.tool("c1", "read_file", "A basis is a linearly independent spanning set.", untrusted=True)) == 3
    assert s.append(Message.assistant("A basis is a minimal spanning set.")) == 4
    assert s.messages[-1].meta["seq"] == 4 and s.seq == 4
    s.add_stance("Examples before definitions")
    s.todo.append({"id": "t1", "content": "write chapter 1", "status": "pending"})
    s.add_usage(Usage(100, 20, cost_usd=0.01))
    s.meta["unit_id"] = "chapter-1"
    s.save()

    lines = [json.loads(ln) for ln in s.transcript_path.read_text().splitlines()]
    assert [ln["seq"] for ln in lines] == [1, 2, 3, 4] and all("ts" in ln for ln in lines) and lines[1]["tool_calls"][0]["name"] == "read_file"

    back = Session.load(workspace, "chat-1")
    assert back.title == "Linear algebra" and back.mode == "chat" and back.seq == 4 and len(back.messages) == 4
    assert back.messages[2].role == "tool" and back.messages[2].meta["untrusted"] is True and back.messages[1].tool_calls[0].arguments == {"path": "sources/text/book.md"}
    assert back.stance == ["Examples before definitions"] and back.todo[0]["content"] == "write chapter 1"
    assert back.usage_total.prompt_tokens == 100 and back.usage_total.cost_usd == pytest.approx(0.01) and back.meta["unit_id"] == "chapter-1"
    assert "Examples before definitions" in back.notes_path.read_text() and "write chapter 1" in back.notes_path.read_text()
    assert Session.exists(workspace, "chat-1") and not Session.exists(workspace, "nope") and not Session.exists(workspace, "../chat-1")


def test_load_replays_transcript_tail_after_crash(workspace):
    s = Session.create(workspace, session_id="crashy")
    s.append(Message.user("one"))
    s.save()
    s.append(Message.assistant("two"))  # persisted to the transcript, but the process dies before save()
    s.append(Message.user("three"))
    back = Session.load(workspace, "crashy")
    assert [m.text() for m in back.messages] == ["one", "two", "three"] and back.seq == 3
    assert back.append(Message.assistant("four")) == 4
    assert len(back.transcript_path.read_text().splitlines()) == 4


def test_missing_and_invalid_ids(workspace):
    with pytest.raises(FileNotFoundError):
        Session.load(workspace, "absent")
    with pytest.raises(ValueError):
        Session.create(workspace, session_id="../escape")
    with pytest.raises(ValueError):
        Session.load(workspace, "a/b")
    Session.create(workspace, session_id="dup")
    with pytest.raises(FileExistsError):
        Session.create(workspace, session_id="dup")


def test_helpers(workspace):
    s = Session.create(workspace, session_id="h")
    for i in range(5):
        s.append(Message.user(f"message number {i}"))
    assert len(s.recent(2)) == 2 and s.recent(2)[-1].text() == "message number 4" and s.recent(0) == []
    assert s.tokens() > 0 and s.last_text("user") == "message number 4" and s.last_text("assistant") == ""
    assert s.add_stance("  keep   it short ") and not s.add_stance("keep it short") and not s.add_stance("   ")
    for i in range(STANCE_MAX_ENTRIES + 5):
        s.add_stance(f"stance {i}")
    assert len(s.stance) == STANCE_MAX_ENTRIES and s.stance[-1] == f"stance {STANCE_MAX_ENTRIES + 4}" and "keep it short" not in s.stance
    total = s.add_usage(Usage(10, 5)) and s.add_usage(Usage(1, 1))
    assert total.prompt_tokens == 11 and s.usage_total.completion_tokens == 6


def test_recall_keywords_regex_and_context(workspace):
    s = Session.create(workspace, session_id="r")
    s.append(Message.user("Explain the coin flip example with two coins"))
    s.append(Message.assistant("With two coins there are four outcomes."))
    s.append(Message.user("Now the dice example"))
    s.append(Message.assistant("A die has six faces; two dice give 36 outcomes."))
    s.append(Message.user("Back to the COINS: what about three coins?"))
    s.append(Message.assistant("Three coins give eight outcomes."))
    hits = s.recall("coins outcomes", k=3)
    assert hits and hits[0]["score"] == 2 and hits[0]["seq"] == 6  # most terms, then most recent
    assert all(h["score"] >= 1 for h in hits) and {h["seq"] for h in hits} <= {1, 2, 5, 6}
    first = hits[0]
    assert first["role"] == "assistant" and "eight outcomes" in first["text"]
    assert [c["seq"] for c in first["context"]] == [5] and "three coins" in first["context"][0]["text"]
    wide = s.recall("dice", window=2)[0]
    assert wide["seq"] == 4 and [c["seq"] for c in wide["context"]] == [2, 3, 5, 6]
    rx = s.recall("re:\\b(coin|die)s?\\b", k=10)
    assert rx and rx[0]["seq"] == 5 and rx[0]["score"] == 2 and len(rx) == 5
    assert s.recall("   ") == [] and s.recall("zebra") == []
    with pytest.raises(ValueError):
        s.recall("re:(unclosed")
    # tool calls are searchable by name and arguments; cleared results stay verbatim in the transcript
    s.append(Message.assistant("", [ToolCall("c9", "grep", {"pattern": "damping"})]))
    s.messages[-1].content = []  # the in-memory copy may change; the transcript does not
    assert s.recall("damping")[0]["seq"] == 7


def test_recall_skips_torn_lines(workspace):
    s = Session.create(workspace, session_id="torn")
    s.append(Message.user("hello world"))
    with open(s.transcript_path, "a", encoding="utf-8") as f:
        f.write('{"seq": 2, "role": "assistant", "content": [{"type": "text", "te')
    assert s.recall("hello")[0]["seq"] == 1
    back = Session.load(workspace, "torn")
    assert back.seq == 1 and len(back.messages) == 1


def test_dangling_tool_calls_and_children(workspace):
    s = Session.create(workspace, session_id="d")
    assert s.dangling_tool_calls() == []
    s.append(Message.user("go"))
    s.append(Message.assistant("", [ToolCall("a", "bash", {}), ToolCall("b", "bash", {})]))
    assert [tc.id for tc in s.dangling_tool_calls()] == ["a", "b"]
    s.append(Message.tool("a", "bash", "ok"))
    assert [tc.id for tc in s.dangling_tool_calls()] == ["b"]
    s.append(Message.tool("b", "bash", "ok"))
    s.append(Message.assistant("done"))
    assert s.dangling_tool_calls() == []
    s.record_child("child-1")
    s.record_child("child-1")
    s.record_child("child-2")
    assert s.meta["children"] == ["child-1", "child-2"]


def test_corrupt_state_is_skipped_by_list_and_named_by_load(workspace):
    store = SessionStore(workspace)
    store.create(session_id="good")
    bad = store.create(session_id="bad")
    bad.state_path.write_text("{not json", encoding="utf-8")
    assert [r["session_id"] for r in store.list()] == ["good"] and store.latest().session_id == "good"
    with pytest.raises(ValueError) as ei:
        Session.load(workspace, "bad")
    assert "not valid JSON" in str(ei.value) and "bad" in str(ei.value)


def test_session_store(workspace):
    store = SessionStore(workspace)
    assert store.list() == [] and store.latest() is None and not store.exists("x")
    a = store.create(mode="chat", title="A", session_id="a")
    a.created_at = "2025-01-01T00:00:00+00:00"
    a.append(Message.user("hi"))
    a.save()
    b = store.create(mode="build", title="B", session_id="b", parent_id=None)
    b.created_at = "2025-02-01T00:00:00+00:00"
    b.save()
    c = store.create(mode="subagent", title="C", session_id="c", parent_id="b")
    c.created_at = "2025-03-01T00:00:00+00:00"
    c.save()
    (workspace.path("sessions") / "junk").mkdir()
    rows = store.list()
    assert [r["session_id"] for r in rows] == ["c", "b", "a"]
    assert rows[2] == {"session_id": "a", "mode": "chat", "title": "A", "created_at": "2025-01-01T00:00:00+00:00", "messages": 1, "parent_id": None}
    assert rows[0]["parent_id"] == "b" and [r["session_id"] for r in store.list(mode="build")] == ["b"]
    assert store.latest().session_id == "c" and store.latest(mode="chat").session_id == "a" and store.latest(mode="nope") is None
    assert store.get("b").title == "B" and store.exists("a")
    with pytest.raises(FileNotFoundError):
        store.get("zzz")
