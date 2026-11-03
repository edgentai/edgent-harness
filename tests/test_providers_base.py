from __future__ import annotations

import pytest

from edgent.adapters.providers.base import ProviderError, find_role, role_marker
from edgent.adapters.providers.mock import MockProvider
from edgent.domain.capabilities import Capabilities
from edgent.domain.types import CompletionRequest, ContentPart, Message, Reply, ToolCall
from edgent.tools.text_protocol import parse_tool_calls, tool_instructions

TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]


def test_mock_script_and_handlers():
    p = MockProvider()
    p.push("first", MockProvider.tool_reply("read_file", {"path": "x"}), lambda req: Reply(text="third:" + req.last_user_text()))
    p.on_role("critic", lambda req: Reply(text="critique"))
    req = CompletionRequest([Message.system(role_marker("main")), Message.user("hello")], tools=TOOLS)
    assert p.complete(req).text == "first"
    r = p.complete(req)
    assert r.tool_calls[0].name == "read_file" and r.finish_reason == "tool_calls" and r.usage.prompt_tokens > 0
    assert p.complete(req).text == "third:hello"
    assert p.complete(CompletionRequest([Message.system(role_marker("critic")), Message.user("x")])).text == "critique"
    assert p.complete(CompletionRequest([Message.user("plain")])).text.startswith("[mock] plain")
    assert find_role([Message.system("x " + role_marker("author"))]) == "author"
    assert len(p.calls) == 5 and p.health()["ok"]


def test_text_protocol_parse():
    text = 'Thinking.\n```tool\n{"name": "read_file", "arguments": {"path": "a.md"}}\n```\nand\n```tool\n[{"name": "grep", "arguments": {"pattern": "x"}}, {"name": "bad"}]\n```'
    clean, calls, errors = parse_tool_calls(text)
    assert [c.name for c in calls] == ["read_file", "grep", "bad"] and clean.startswith("Thinking.") and "```" not in clean
    _, calls, errors = parse_tool_calls("```tool\nnot json\n```")
    assert not calls and errors
    assert "read_file" in tool_instructions(TOOLS)


def test_prepare_text_protocol_and_system_folding():
    caps = Capabilities(native_tools=False, system_channel=False, images=False, multi_turn=False, max_prompt_bytes=None)
    p = MockProvider(caps=caps)
    msgs = [
        Message.system("SYS"),
        Message.user("look", [ContentPart.of_media("image", "/tmp/a.png", text="[image a.png]")]),
        Message.assistant("", [ToolCall("c1", "read_file", {"path": "a"})]),
        Message.tool("c1", "read_file", "CONTENT"),
        Message.user("now answer"),
    ]
    req = p.prepare(CompletionRequest(msgs, tools=TOOLS))
    assert req.tools == [] and len(req.messages) == 1 and req.messages[0].role == "user"
    text = req.messages[0].plain_text()
    assert "<<<SYSTEM" in text and "read_file" in text and "TOOL RESULT read_file" in text and "[image a.png]" in text and "now answer" in text
    # finalize parses the text protocol
    p.push('I will read.\n```tool\n{"name": "read_file", "arguments": {"path": "a"}}\n```')
    r = p.complete(CompletionRequest([Message.user("q")], tools=TOOLS))
    assert r.tool_calls and r.tool_calls[0].arguments == {"path": "a"} and r.text == "I will read."


def test_prepare_refuses_oversize_prompt():
    p = MockProvider(caps=Capabilities(max_prompt_bytes=100))
    with pytest.raises(ProviderError) as ei:
        p.complete(CompletionRequest([Message.user("x" * 500)]))
    assert ei.value.error.category.value == "truncated" and "Send less" in ei.value.error.remediation


def test_prepare_media_stand_in():
    p = MockProvider(caps=Capabilities(images=False))
    req = p.prepare(CompletionRequest([Message.user("see", [ContentPart.of_media("image", "/tmp/a.png", text="[image: a.png]")])]))
    assert all(part.type == "text" for part in req.messages[0].content) and "[image: a.png]" in req.messages[0].plain_text()
