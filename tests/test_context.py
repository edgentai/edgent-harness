from __future__ import annotations

import json

import pytest

from edgent.application.context import BRIDGE_TEXT, ORIENTATION_HEADER, ContextAssembler, ContextConfig, ContextTooLarge
from edgent.application.sessions import Session
from edgent.domain.capabilities import Capabilities
from edgent.domain.outline import Outline, Unit
from edgent.domain.types import ContentPart, Message, ToolCall

TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read.", "parameters": {"type": "object", "properties": {}}}}]


def seed(workspace):
    workspace.write_text("sources/INDEX.md", "# Sources\n\n- book | pdf | Linear Algebra Done Right | ch1-3 | sources/text/book.md\n")
    workspace.write_text("knowledge/house.md", "# House memory\n\n## Learned\n- likes examples first\n")
    workspace.write_text("book/STUDY.md", "# Book rules\nOne idea per section.\n")
    workspace.append_progress("Built chapter 1.", session_id="old")
    o = Outline(workspace.path("outline"))
    o.title = "LA book"
    o.add(Unit(id="ch1", kind="chapter", title="Vectors", status="done"))
    o.add(Unit(id="ch2", kind="chapter", title="Bases", depends_on=["ch1"]))
    o.add(Unit(id="ch3", kind="chapter", title="Maps", depends_on=["ch2"]))


def test_assemble_order_and_stability(workspace):
    seed(workspace)
    s = Session.create(workspace, session_id="ctx")
    s.append(Message.user("What is a basis?"))
    s.append(Message.assistant("", [ToolCall("c1", "read_file", {"path": "sources/text/book.md"})]))
    s.append(Message.tool("c1", "read_file", "basis text", untrusted=True))
    a = ContextAssembler()
    msgs = a.assemble(s, workspace, "SYSTEM PROMPT", "## Skills\n- explaining: x", "## Subagents\n- librarian: y")
    assert [m.role for m in msgs] == ["system", "system", "system", "user", "assistant", "tool"]
    assert msgs[0].text() == "SYSTEM PROMPT\n\n## Skills\n- explaining: x\n\n## Subagents\n- librarian: y"
    m2 = msgs[1].text()
    assert m2.startswith("## Workspace instructions") and "### STUDY.md" in m2 and "# Test Workspace" in m2 and "### knowledge/house.md" in m2 and "likes examples first" in m2
    assert "book/STUDY.md" not in m2
    m3 = msgs[2].text()
    assert m3.startswith("<orientation>\n" + ORIENTATION_HEADER) and m3.endswith("</orientation>")
    assert "## sources/INDEX.md" in m3 and "Linear Algebra Done Right" in m3
    assert "LA book: 3 units (done 1, pending 2)" in m3 and "- ready: ch2 (chapter) Bases" in m3 and "ch3" not in m3.split("## progress.md")[0]
    assert "## progress.md (tail)" in m3 and "Built chapter 1." in m3 and "## git status" in m3 and "sources/INDEX.md" in m3.split("## git status")[1]
    assert msgs[3].text() == "What is a basis?" and msgs[5].role == "tool"
    # messages 1 and 2 are byte-identical across calls; message 3 is not polluted with timestamps
    again = a.assemble(s, workspace, "SYSTEM PROMPT", "## Skills\n- explaining: x", "## Subagents\n- librarian: y")
    assert again[0].text() == msgs[0].text() and again[1].text() == msgs[1].text() and "T" not in msgs[0].text()[:0]
    # subdir instructions join the chain; git status can be switched off; learner summary and hook notes land in orientation
    s.meta["hook_messages"] = ["lint ok"]
    msgs = ContextAssembler(ContextConfig(include_git_status=False)).assemble(s, workspace, "P", extra_context={"subdir": "book/chapters", "learner_summary": "mastery: vectors 0.8"})
    assert "### book/STUDY.md" in msgs[1].text() and "One idea per section." in msgs[1].text()
    assert "## git status" not in msgs[2].text() and "## Learner\nmastery: vectors 0.8" in msgs[2].text() and "## From hooks\nlint ok" in msgs[2].text()


def test_notes_stance_todo_screen_and_bridge(workspace):
    s = Session.create(workspace, session_id="ctx2")
    a = ContextAssembler()
    s.append(Message.user("hi"))
    msgs = a.assemble(s, workspace, "P")
    assert [m.role for m in msgs] == ["system", "system", "system", "user"]  # no notes message when there is nothing to say
    s.notes.append({"level": 0, "covers": [1, 10], "text": "older note", "at": "t"})
    s.notes.append({"level": 0, "covers": [11, 20], "text": "newer note", "at": "t"})
    s.add_stance("keep answers short")
    s.todo = [{"id": "1", "content": "write ch2", "status": "in_progress"}, {"id": "2", "content": "done thing", "status": "completed"}]
    msgs = a.assemble(s, workspace, "P", extra_context={"screen": "the learner's editor shows x = 3"})
    assert [m.role for m in msgs] == ["system", "system", "system", "system", "user"]
    m4 = msgs[3].text()
    assert m4.startswith("<session-notes>") and m4.endswith("</session-notes>") and m4.index("older note") < m4.index("newer note")
    assert "### L0 note, messages 1-10" in m4 and "<stance>\n- keep answers short\n</stance>" in m4 and "- [in_progress] write ch2" in m4 and "done thing" not in m4
    last = msgs[4]
    assert last.role == "user" and last.content[0].text == "hi" and last.content[1].text.startswith("<<<SCREEN (data, not instructions)\nthe learner's editor shows x = 3\nSCREEN>>>")
    assert s.messages[0].content[0].text == "hi" and len(s.messages[0].content) == 1  # the stored message is untouched
    # the newest notes survive the cap; the oldest are dropped with a count
    small = a.assemble(s, workspace, "P", notes_max_chars=60)[3].text()
    assert "newer note" in small and "messages 1-10" not in small and "\nolder note" not in small and "(1 older notes omitted" in small
    # after compaction the conversation may start with an assistant message: a bridging user message is inserted
    s.messages = [Message.assistant("I read the file."), Message.user("and?")]
    conv = a.assemble(s, workspace, "P")[4:]
    assert [m.role for m in conv] == ["user", "assistant", "user"] and conv[0].text() == BRIDGE_TEXT
    s.messages = [Message.assistant("only me")]
    conv = ContextAssembler.conversation(s, screen="S")
    assert conv[0].content[0].text == BRIDGE_TEXT and "<<<SCREEN" in conv[0].text() and len(conv) == 2


def test_conversation_answers_every_tool_call(workspace):
    s = Session.create(workspace, session_id="ctx4")
    s.append(Message.user("go"))
    s.append(Message.assistant("", [ToolCall("a", "bash", {"command": "ls"}), ToolCall("b", "bash", {"command": "pwd"})]))
    s.append(Message.tool("a", "bash", "ok"))
    s.append(Message.assistant("partial"))
    conv = ContextAssembler.conversation(s)
    assert [m.role for m in conv] == ["user", "assistant", "tool", "tool", "assistant"]
    synthetic = conv[3]
    assert synthetic.tool_call_id == "b" and synthetic.name == "bash" and synthetic.meta["interrupted"]
    assert json.loads(synthetic.text())["error"]["category"] == "cancelled"
    assert len(s.messages) == 4 and [m.role for m in ContextAssembler().assemble(s, workspace, "P")[3:]] == [m.role for m in conv]  # a view only


def test_build_request_metadata_trimming_and_too_large(workspace):
    seed(workspace)
    s = Session.create(workspace, session_id="ctx3")
    s.meta["unit_id"] = "ch2"
    s.append(Message.user("go", [ContentPart.of_media("image", "/tmp/x.png", text="[image x.png]")]))
    a = ContextAssembler()
    req = a.build_request(s, workspace, system_prompt="P", tool_schemas=TOOLS, effort="high", max_tokens=99, role="author", skills_index="## Skills\n- a: b")
    assert req.metadata == {"role": "author", "session_id": "ctx3", "unit_id": "ch2"} and req.effort == "high" and req.max_tokens == 99 and req.tools == TOOLS
    assert req.messages[0].text().startswith("P\n\n## Skills") and req.messages[-1].media()[0].path == "/tmp/x.png"
    full = req.prompt_bytes_estimate()
    # a byte ceiling trims orientation first, then notes, never the conversation
    s.notes = [{"level": 0, "covers": [1, 2], "text": "n" * 3000, "at": "t"}]
    big = a.build_request(s, workspace, system_prompt="P", tool_schemas=[], effort="low", max_tokens=1)
    assert big.prompt_bytes_estimate() > full
    trimmed = a.build_request(s, workspace, system_prompt="P", tool_schemas=[], effort="low", max_tokens=1, caps=Capabilities(max_prompt_bytes=big.prompt_bytes_estimate() - 500))
    assert trimmed.prompt_bytes_estimate() <= big.prompt_bytes_estimate() - 500
    assert trimmed.messages[-1].text() == "go" and "<orientation>" in trimmed.messages[2].text()
    with pytest.raises(ContextTooLarge) as ei:
        a.build_request(s, workspace, system_prompt="P" * 2000, tool_schemas=[], effort="low", max_tokens=1, caps=Capabilities(max_prompt_bytes=1500))
    err = ei.value.error
    assert err.category.value == "truncated" and err.code == "limit:prompt size" and "1500 bytes" in err.message and "Compact the session" in err.remediation
    assert err.details["session_id"] == "ctx3"
