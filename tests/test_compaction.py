from __future__ import annotations

from edgent.adapters.providers.mock import MockProvider
from edgent.application.compaction import (
    COMPACTOR_INSTRUCTION,
    ROLLUP_INSTRUCTION,
    CompactionLimits,
    Compactor,
    extractive_summary,
    find_locators,
    render_block,
)
from edgent.application.sessions import Session
from edgent.domain.types import ContentPart, Message, Reply, ToolCall


def exchange(session: Session, i: int, result_chars: int = 600) -> None:
    session.append(Message.user(f"Question {i}: explain topic {i} from sources/text/book.md please. Keep it short."))
    session.append(Message.assistant("", [ToolCall(f"c{i}", "read_file", {"path": "sources/text/book.md", "offset": i})]))
    session.append(Message.tool(f"c{i}", "read_file", f"[p.{i}] " + ("content " * (result_chars // 8)), untrusted=True))
    session.append(Message.assistant(f"Topic {i} is explained at book:p.{i}. The key idea is number {i}."))


def test_stage_one_clears_old_tool_results(workspace):
    s = Session.create(workspace, session_id="c1")
    for i in range(4):
        exchange(s, i)
    s.messages[2].meta["persisted_path"] = ".edgent/tool-results/call_0.json"  # already persisted by the registry
    c = Compactor(None, CompactionLimits(keep_recent_messages=4, clear_tool_result_over_chars=400))
    before = s.tokens()
    assert c.pressure(s, 1000) == (s.tokens() + s.notes_tokens()) / 1000 and c.pressure(s, 0) == 0.0
    report = c.maybe_compact(s, context_tokens=10_000)
    assert report.stage_reached == 0 and report.tool_results_cleared == 0  # under the target: nothing happens
    report = c.maybe_compact(s, context_tokens=int(before / 0.6) - 5)  # just over the target
    assert report.stage_reached == 1 and report.tool_results_cleared == 3 and report.after_tokens < before and report.notes_added == 0
    first, second = s.messages[2], s.messages[6]
    assert first.meta["cleared"] and first.text() == "[result of read_file persisted at .edgent/tool-results/call_0.json; re-read with read_file if needed]"
    assert second.meta["cleared"] and second.meta["persisted_path"] == "sessions/c1/cleared/7.txt"
    assert (s.cleared_dir / "7.txt").read_text().startswith("[p.1] content") and not (s.cleared_dir / "3.txt").exists()
    assert "[p.3]" in s.messages[-2].text() and "cleared" not in s.messages[-2].meta  # the recent window is untouched
    assert Session.load(workspace, "c1").messages[6].meta["cleared"] is True  # saved
    assert "[p.1] content" in s.recall("re:\\[p\\.1\\]")[0]["text"]  # the transcript keeps the original
    assert c.maybe_compact(s, context_tokens=int(s.tokens() / 0.6) + 50).tool_results_cleared == 0  # idempotent


def test_stage_two_extractive_notes_and_block_boundaries(workspace):
    s = Session.create(workspace, session_id="c2")
    for i in range(6):
        exchange(s, i, result_chars=100)
    total = len(s.messages)
    c = Compactor(None, CompactionLimits(keep_recent_messages=4, block_size=6, roll_at=50))
    report = c.maybe_compact(s, context_tokens=10, force=False)
    assert report.stage_reached == 2 and report.notes_added >= 1 and report.summariser == "extractive"
    assert len(s.messages) < total and s.messages[0].role == "user" and all(m.role != "tool" or s.messages[0] is not m for m in s.messages[:1])
    note = s.notes[0]
    assert note["level"] == 0 and note["covers"] == [1, 8] and "Asked: Question 0" in note["text"] and "read_file(path=sources/text/book.md" in note["text"]
    assert "book:p.0" in note["text"] and "sources/text/book.md" in note["text"] and "Concluded:" in note["text"] and note["at"]
    assert len(s.transcript_path.read_text().splitlines()) == total  # the transcript still holds everything
    assert "Asked: Question 0" in s.notes_path.read_text()
    back = Session.load(workspace, "c2")
    assert len(back.notes) == len(s.notes) and len(back.messages) == len(s.messages)
    # cutting never leaves a tool result without its call: the first kept message is a user message
    s2 = Session.create(workspace, session_id="c2b")
    s2.append(Message.user("start"))
    for i in range(12):
        s2.append(Message.assistant("", [ToolCall(f"k{i}", "bash", {"command": f"ls {i}"})]))
        s2.append(Message.tool(f"k{i}", "bash", "ok"))
    c2 = Compactor(None, CompactionLimits(keep_recent_messages=3, block_size=4))
    assert c2.cut_index(s2.messages) == 5 and s2.messages[5].role == "assistant"
    c2.maybe_compact(s2, context_tokens=1, force=True)
    assert s2.messages[0].role != "tool" and s2.notes[0]["covers"][0] == 1


def test_provider_summariser_uses_compactor_role(workspace):
    provider = MockProvider()
    seen = []

    def compact(req):
        seen.append(req)
        return Reply(text="NOTE: the user wants short answers; read sources/text/book.md p.1-3.")

    provider.on_role("compactor", compact)
    s = Session.create(workspace, session_id="c3")
    for i in range(5):
        exchange(s, i, result_chars=100)
    c = Compactor(provider, CompactionLimits(keep_recent_messages=4, block_size=4, roll_at=50))
    report = c.maybe_compact(s, context_tokens=200_000, force=True)
    assert report.notes_added == 1 and report.summariser == "provider" and s.notes[0]["text"].startswith("NOTE: the user wants")
    req = seen[0]
    assert req.metadata["role"] == "compactor" and req.effort == "low" and COMPACTOR_INSTRUCTION in req.system_text() and "[edgent-role: compactor]" in req.system_text()
    assert "[1] USER:" in req.last_user_text() and "Question 0" in req.last_user_text() and "TOOL read_file" in req.last_user_text()
    # a failing or empty compactor falls back to the extractive summary instead of breaking the loop
    provider.on_role("compactor", lambda req: Reply(text="   "))
    c.maybe_compact(s, context_tokens=200_000, force=True)
    assert c.last_summariser == "extractive" and "empty summary" in c.last_fallback_reason and len(s.notes) == 2
    custom = Compactor(provider, CompactionLimits(keep_recent_messages=2, block_size=2), summariser=lambda msgs, instr: f"custom {len(msgs)} {instr is COMPACTOR_INSTRUCTION}")
    custom.maybe_compact(s, context_tokens=200_000, force=True)
    assert s.notes[-1]["text"].startswith("custom ") and s.notes[-1]["text"].endswith("True") and custom.last_summariser == "custom"


def test_roll_up_and_note_caps(workspace):
    s = Session.create(workspace, session_id="c4")
    for i in range(1, 15):
        s.notes.append({"level": 0, "covers": [i * 10, i * 10 + 9], "text": f"note {i} says decision {i}", "at": "t"})
    c = Compactor(None, CompactionLimits(roll_at=12, roll_group=8, note_max_chars_by_level=(50, 30)))
    assert c.roll_up(s) == 1
    assert [n["level"] for n in s.notes] == [1, 0, 0, 0, 0, 0, 0] and s.notes[0]["covers"] == [10, 89] and s.notes[1]["covers"] == [90, 99]
    assert len(s.notes[0]["text"]) <= 30 and s.notes[0]["text"].endswith("…[note truncated]")
    # higher levels roll up too, with the last cap applying to every level beyond the tuple
    s.notes = [{"level": 1, "covers": [i, i], "text": "x" * 100, "at": "t"} for i in range(13)]
    assert c.roll_up(s) == 1 and s.notes[0]["level"] == 2 and len(s.notes[0]["text"]) <= 30 and len(s.notes) == 6
    assert c.limits.note_max_chars(0) == 50 and c.limits.note_max_chars(7) == 30
    # the summariser sees note wrappers for roll-ups
    seen = {}
    rc = Compactor(None, CompactionLimits(roll_at=1, roll_group=2), summariser=lambda msgs, instr: seen.update(n=len(msgs), instr=instr) or "merged")
    s.notes = [{"level": 0, "covers": [1, 2], "text": "a", "at": "t"}, {"level": 0, "covers": [3, 4], "text": "b", "at": "t"}]
    assert rc.roll_up(s) == 1 and seen == {"n": 2, "instr": ROLLUP_INSTRUCTION} and s.notes == [{"level": 1, "covers": [1, 4], "text": "merged", "at": s.notes[0]["at"]}]


def test_force_runs_all_stages_and_report_shape(workspace):
    s = Session.create(workspace, session_id="c5")
    for i in range(8):
        exchange(s, i)
    c = Compactor(None, CompactionLimits(keep_recent_messages=4, block_size=4, roll_at=2, roll_group=2))
    r = c.maybe_compact(s, context_tokens=200_000, force=True)
    assert r.stage_reached >= 2 and r.tool_results_cleared == 7 and r.notes_added >= 1 and r.before_tokens > r.after_tokens
    d = r.to_dict()
    assert set(d) == {"stage_reached", "before_tokens", "after_tokens", "notes_added", "tool_results_cleared", "rolled_up", "summariser"}
    empty = Session.create(workspace, session_id="c6")
    assert c.maybe_compact(empty, 100, force=True).stage_reached == 1 and empty.notes == []


def test_target_uses_the_same_measure_as_pressure(workspace):
    s = Session.create(workspace, session_id="c7")
    for i in range(6):
        exchange(s, i, result_chars=100)
    c = Compactor(None, CompactionLimits(keep_recent_messages=4, block_size=4, roll_at=50))
    window = int((s.tokens() + 300) / 0.6)  # the messages alone sit under the target; with a 600-token overhead they do not
    assert c.pressure(s, window, overhead_tokens=600) == c.occupied_tokens(s, 600) / window
    assert c.maybe_compact(s, window).stage_reached == 0 and len(s.notes) == 0
    report = c.maybe_compact(s, window, overhead_tokens=600)
    assert report.stage_reached >= 2 and report.notes_added >= 1 and len(s.notes) == report.notes_added
    # no window to measure against: nothing happens unless forced
    fresh = Session.create(workspace, session_id="c8")
    for i in range(3):
        exchange(fresh, i, result_chars=100)
    assert Compactor(None).maybe_compact(fresh, 0).stage_reached == 0 and fresh.notes == []
    forced = Compactor(None, CompactionLimits(keep_recent_messages=2, block_size=2)).maybe_compact(fresh, 0, force=True)
    assert forced.stage_reached >= 2 and forced.notes_added >= 1 and len(fresh.messages) < 12 and fresh.messages[0].role == "user"


def test_extractive_summary_and_helpers():
    msgs = [
        Message.user("Explain damping. Then show me the formula in lecture:t=00:14:32."),
        Message.assistant("", [ToolCall("a", "grep", {"pattern": "damping", "path": "sources/text"})]),
        Message.tool("a", "grep", "sources/text/lecture.md:12: damping factor"),
        Message.assistant("Damping is at book:p.12. It reduces oscillation amplitude."),
    ]
    text = extractive_summary(msgs)
    assert text.startswith("Asked: Explain damping.") and "Called: grep(pattern=damping, path=sources/text)" in text
    assert "Concluded: Damping is at book:p.12. It reduces oscillation amplitude." in text
    assert "lecture:t=00:14:32" in text and "book:p.12" in text and "sources/text/lecture.md" in text
    assert find_locators("see notes:L12 and media/films/a/out.mp4") == ["notes:L12", "media/films/a/out.mp4"]
    assert extractive_summary([]) == "(no content)"
    notes = [Message(role="user", content=[ContentPart.of_text("n" * 2000)], meta={"kind": "note", "level": 0, "covers": [1, 2]})] * 2
    merged = extractive_summary(notes)
    assert merged.count("\n") == 1 and len(merged) < 2000
    block = render_block(msgs + notes[:1])
    assert "USER:" in block and "ASSISTANT (calls grep(pattern=damping, path=sources/text)):" in block and "TOOL grep:" in block and "[note level 0, messages 1-2]" in block
