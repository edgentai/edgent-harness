from __future__ import annotations

from edgent.adapters.providers.mock import MockProvider
from edgent.application.compaction import (
    BYTES_PER_TOKEN_ESTIMATE,
    COMPACTOR_INSTRUCTION,
    ROLLUP_INSTRUCTION,
    CompactionLimits,
    Compactor,
    extractive_summary,
    find_locators,
    render_block,
)
from edgent.application.loop import AgentLoop
from edgent.application.sessions import Session
from edgent.domain.capabilities import Capabilities
from edgent.domain.types import ContentPart, Message, Reply, ToolCall
from edgent.tools.registry import S, ToolRegistry, obj, opt


def exchange(session: Session, i: int, result_chars: int = 600) -> None:
    session.append(Message.user(f"Question {i}: explain topic {i} from sources/text/book.md please. Keep it short."))
    session.append(Message.assistant("", [ToolCall(f"c{i}", "read_file", {"path": "sources/text/book.md", "offset": i})]))
    session.append(Message.tool(f"c{i}", "read_file", f"[p.{i}] " + ("content " * (result_chars // 8)), untrusted=True))
    session.append(Message.assistant(f"Topic {i} is explained at book:p.{i}. The key idea is number {i}."))


def assert_no_orphan_tool_results(messages: list[Message]) -> None:
    """Every kept tool result must follow the assistant message that called it: a cut never splits a call from its results."""
    live: set[str] = set()
    for m in messages:
        if m.role == "assistant":
            live = {tc.id for tc in m.tool_calls}
        elif m.role == "tool":
            assert m.tool_call_id in live, f"tool result {m.tool_call_id!r} (seq {m.meta.get('seq')}) kept without its call"
        else:
            live = set()


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
    assert len(s.messages) < total and s.messages[0].role == "user"
    assert_no_orphan_tool_results(s.messages)
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
    assert_no_orphan_tool_results(s2.messages)


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
    assert set(d) == {"stage_reached", "before_tokens", "after_tokens", "notes_added", "tool_results_cleared", "rolled_up", "summariser", "reached_recent_window"}
    assert d["reached_recent_window"] is False  # the ordinary stages freed something: the recent window was not touched
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


def test_forced_compaction_clears_inside_the_recent_window_but_never_the_current_turn(workspace):
    s = Session.create(workspace, session_id="c9")
    exchange(s, 0, result_chars=5000)  # a previous turn whose result is far larger than anything the window needs
    s.append(Message.user("Question 1: and now?"))
    s.append(Message.assistant("", [ToolCall("cur", "read_file", {"path": "sources/text/book.md"})]))
    s.append(Message.tool("cur", "read_file", "[p.1] " + "fresh " * 1000, untrusted=True))
    c = Compactor(None, CompactionLimits(keep_recent_messages=8))
    # proactive compaction keeps protecting the window: the whole session sits inside it
    assert c.maybe_compact(s, context_tokens=10).stage_reached == 1 and not any(m.meta.get("cleared") for m in s.messages)
    report = c.maybe_compact(s, context_tokens=200_000, force=True)
    assert report.reached_recent_window and report.tool_results_cleared == 1 and report.notes_added == 0 and report.to_dict()["reached_recent_window"] is True
    assert s.messages[2].meta["cleared"] and s.messages[2].text().startswith("[result of read_file persisted at ")
    assert s.messages[4].role == "user" and s.messages[4].text() == "Question 1: and now?"  # the last user message is intact
    assert "fresh" in s.messages[6].text() and "cleared" not in s.messages[6].meta  # so is the current turn's result
    assert len(s.messages) == 7 and s.notes == []
    # forcing again, with nothing left to clear, summarises the previous turn into a note; the current turn stays whole
    again = c.maybe_compact(s, context_tokens=200_000, force=True)
    assert again.tool_results_cleared == 0 and again.notes_added == 1 and again.reached_recent_window and len(s.notes) == 1
    assert [m.role for m in s.messages] == ["user", "assistant", "tool"] and s.messages[0].text() == "Question 1: and now?" and "fresh" in s.messages[2].text()
    assert s.notes[0]["covers"] == [1, 4] and "Asked: Question 0" in s.notes[0]["text"]
    assert_no_orphan_tool_results(s.messages)
    # and once only the current turn is left, forcing is a no-op
    third = c.maybe_compact(s, context_tokens=200_000, force=True)
    assert third.tool_results_cleared == 0 and third.notes_added == 0 and not third.reached_recent_window and len(s.messages) == 3
    # a session that is a single turn cannot be shrunk at all, even forced: the model has not seen its results yet
    one = Session.create(workspace, session_id="c9b")
    one.append(Message.user("read it"))
    one.append(Message.assistant("", [ToolCall("x", "read_file", {"path": "big.txt"})]))
    one.append(Message.tool("x", "read_file", "y" * 20_000, untrusted=True))
    r = c.maybe_compact(one, context_tokens=200_000, force=True)
    assert r.tool_results_cleared == 0 and r.notes_added == 0 and not r.reached_recent_window and len(one.messages[2].text()) == 20_000


def test_forced_compaction_summarises_into_the_recent_window_when_nothing_else_frees(workspace):
    s = Session.create(workspace, session_id="c10")
    s.append(Message.user("Question 0: here is the whole brief. " + "Constraint: keep it short. " * 300))
    s.append(Message.assistant("Noted: the brief is at sources/text/brief.md."))
    s.append(Message.user("Question 1: continue."))
    s.append(Message.assistant("Sure, next I will check sources/text/brief.md."))
    s.append(Message.user("Question 2: go on."))
    c = Compactor(None, CompactionLimits(keep_recent_messages=8, block_size=10))
    report = c.maybe_compact(s, context_tokens=200_000, force=True)
    assert report.reached_recent_window and report.notes_added >= 1 and report.tool_results_cleared == 0 and report.stage_reached >= 2
    assert s.messages[-1].role == "user" and s.messages[-1].text() == "Question 2: go on."
    assert s.messages[0].role == "user"  # the kept conversation still starts at a user message
    assert_no_orphan_tool_results(s.messages)
    assert s.notes[0]["covers"][0] == 1 and "Asked: Question 0" in s.notes[0]["text"] and "sources/text/brief.md" in s.notes[0]["text"]
    # with a tight window the loop keeps cutting up to, and never past, the current turn
    s2 = Session.create(workspace, session_id="c10b")
    for i in range(3):
        exchange(s2, i, result_chars=100)
    s2.append(Message.user("Question 3: current."))
    s2.append(Message.assistant("", [ToolCall("now", "bash", {"command": "ls"})]))
    s2.append(Message.tool("now", "bash", "a.md b.md"))
    Compactor(None, CompactionLimits(keep_recent_messages=8, block_size=10)).maybe_compact(s2, context_tokens=1, force=True)
    assert [m.role for m in s2.messages] == ["user", "assistant", "tool"] and s2.messages[0].text() == "Question 3: current."
    assert s2.notes and s2.notes[-1]["covers"][1] == 12


def test_byte_ceiling_narrows_the_window(workspace):
    s = Session.create(workspace, session_id="c11")
    for i in range(3):
        exchange(s, i, result_chars=600)
    plain = Compactor(MockProvider())  # no byte ceiling: the token window is the window
    assert plain.window_tokens(1_000_000) == 1_000_000 and plain.window_tokens(0) == 0 and Compactor(None).window_tokens(5) == 5
    ceiling = Capabilities(max_prompt_bytes=4000, context_tokens=1_000_000)
    narrow = Compactor(MockProvider(caps=ceiling), CompactionLimits(keep_recent_messages=2, block_size=4, clear_tool_result_over_chars=400))
    assert narrow.window_tokens(1_000_000) == 4000 // BYTES_PER_TOKEN_ESTIMATE == 1000 and narrow.window_tokens(0) == 1000
    assert narrow.pressure(s, 1_000_000) == narrow.occupied_tokens(s) / 1000 > plain.pressure(s, 1_000_000)
    # proactive compaction fires against the narrowed window although the token window alone would never trigger it
    assert plain.maybe_compact(s, context_tokens=1_000_000).stage_reached == 0
    report = narrow.maybe_compact(s, context_tokens=1_000_000)
    assert report.stage_reached >= 1 and report.tool_results_cleared == 2 and narrow.pressure(s, 1_000_000) < 1.0


def test_byte_ceiling_provider_recovers_on_the_turn_after_an_oversized_result(workspace):
    """One allowed tool result that alone exceeds the byte ceiling must not wedge the following turns."""
    workspace.write_text("big.txt", "lorem ipsum " * 1600)  # about 19,000 chars, within a tool's inline ceiling
    reg = ToolRegistry()

    @reg.tool("read_file", "Read a workspace file.", obj({"path": opt(S, "Workspace-relative path.")}, ["path"]), permission="read", max_inline_chars=30_000)
    def read_file(ctx, a):
        return {"path": a["path"], "text": ctx.workspace.read_text(a["path"]), "untrusted": True}

    provider = MockProvider(caps=Capabilities(max_prompt_bytes=20_000, context_tokens=1_000_000))
    provider.push(MockProvider.tool_reply("read_file", {"path": "big.txt"}), "turn 1 done", "turn 2 done", "turn 3 done")
    session = Session.create(workspace, mode="chat", session_id="c12")
    loop = AgentLoop(provider, reg, workspace, session, system_prompt="You are Edgent.")
    first = loop.run_turn("hello 0")
    assert first.stop_reason == "error" and "prompt size exceeded" in first.text  # the result of the current turn is kept whole
    assert any(m.role == "tool" and not m.meta.get("cleared") for m in session.messages)
    results = [loop.run_turn(f"hello {i}") for i in range(1, 4)]
    assert [r.stop_reason for r in results] == ["done"] * 3 and [r.text for r in results] == ["turn 1 done", "turn 2 done", "turn 3 done"]
    cleared = [m for m in session.messages if m.meta.get("cleared")]
    assert len(cleared) == 1 and cleared[0].role == "tool" and "lorem ipsum lorem ipsum" in (workspace.root / cleared[0].meta["persisted_path"]).read_text()
    assert all(c.prompt_bytes_estimate() <= 20_000 for c in provider.calls)
