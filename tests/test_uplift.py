"""Comprehensive tests for the Claude-Code / Antigravity / OpenClaw-class edgent-harness.

Covers:
1. Sentinel/CDATA text tool protocol, Anthropic prompt caching, single-turn flattening, 128 KB ceiling,
   real OpenAI & Anthropic SSE streaming, and ModelRouter.
2. General tools: line-range & multimodal read_file, surgical edit_file + revert safety, sandboxed bash
   (persistent cwd, output offloading, background task_status/output/stop), web_fetch (SSRF guard, HTML-to-text,
   cache), web_search, permission modes (plan/ask/accept_edits/auto) & pattern deny rules (`bash(rm -rf*)`).
3. AgentLoop reliability: hierarchical AGENTS.md/CLAUDE.md + skills/agents discovery, multimodal image
   attachment, LLM-backed compaction + grep session_recall, doom-loop detection, truncation auto-continue,
   cancel_event, per-turn checkpointing & undo(), HITL request_review cross-instance resume, and fresh-context
   `reviewer` subagent.
4. Real stdio subprocess MCPClient <-> MCPServer round-trip & dynamic tool registration.
5. OpenClaw-style HeartbeatScheduler, streaming TerminalREPL, local HTTP + SSE HarnessServer, and CLI.
"""
from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

from harness.__main__ import main as cli_main
from harness.core.client import MockModelClient, ModelRequest, ModelResponse, OpenAICompatibleClient, StreamEvent
from harness.core.config import HarnessConfig
from harness.core.providers import AnthropicMessagesClient, ModelRouter, PromptTooLargeError
from harness.core.tool_protocol import parse_tool_calls, strip_tool_calls
from harness.core.types import Message, ToolCall
from harness.modality.artifacts import ArtifactStore
from harness.modality.sidecars import Sidecars
from harness.sandbox.workspace import Workspace
from harness.session import (
    AgentLoop,
    HeartbeatJob,
    HeartbeatScheduler,
    HookDecision,
    HookManager,
    PermissionMode,
    PermissionPolicy,
    SkillRegistry,
    run_doctor,
)
from harness.tools import (
    IdempotencyStore,
    MCPClient,
    MCPServer,
    ToolContext,
    build_default_registry,
    register_general_tools,
)
from harness.ui import HarnessServer, TerminalREPL

# Minimal valid 1x1 PNG (67 bytes)
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


class FakeHTTPResponse:
    def __init__(self, payload: dict | bytes | str, headers: dict[str, str] | None = None, status: int = 200) -> None:
        if isinstance(payload, dict):
            self._body = json.dumps(payload).encode("utf-8")
        elif isinstance(payload, str):
            self._body = payload.encode("utf-8")
        else:
            self._body = payload
        self.headers = headers or {"Content-Type": "application/json"}
        self.status = status

    def read(self, n: int = -1) -> bytes:
        return self._body if n < 0 else self._body[:n]

    def __iter__(self):
        return iter(self._body.splitlines(keepends=True))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class ProviderAndStreamingTests(unittest.TestCase):
    def test_text_protocol_preserves_latex_html_svg_and_cdata(self):
        raw = (
            "Applying surgical update.\n"
            '<edgent_call tool="edit_file" id="c1">\n'
            '<edgent_arg name="path">docs/ch1.md</edgent_arg>\n'
            '<edgent_arg name="old"><![CDATA[The slope is $\\frac{dy}{dx}$ inside <code>foo()</code>.]]></edgent_arg>\n'
            '<edgent_arg name="new" sentinel="__END__">\n'
            'SVG <text x="0">Label</text> and $\\frac{1}{1+e^{-x}}$ with ]]> intact.\n'
            "__END__</edgent_arg>\n"
            "</edgent_call>\n"
            '<edgent_call tool="load_skill" name="debugging"/>'
        )
        calls = parse_tool_calls(raw)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].name, "edit_file")
        self.assertEqual(calls[0].arguments["old"], r"The slope is $\frac{dy}{dx}$ inside <code>foo()</code>.")
        self.assertIn(r"$\frac{1}{1+e^{-x}}$", calls[0].arguments["new"])
        self.assertIn('<text x="0">Label</text>', calls[0].arguments["new"])
        self.assertIn("]]>", calls[0].arguments["new"])
        self.assertEqual(calls[1].name, "load_skill")
        self.assertEqual(calls[1].arguments["name"], "debugging")
        self.assertEqual(strip_tool_calls(raw), "Applying surgical update.")

    def test_anthropic_prompt_caching_ceiling_and_single_turn_flatten(self):
        captured: list[dict] = []

        def fake_opener(http_req, timeout=None):
            captured.append(json.loads(http_req.data.decode("utf-8")))
            return FakeHTTPResponse({
                "model": "claude-test",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 120, "output_tokens": 45, "cache_read_input_tokens": 80},
                "_bridge": {"thinking_tokens": 150},
                "content": [{"type": "text", "text": "ok"}],
            })

        tools = [{"type": "function", "function": {"name": "read_file", "description": "read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
        # Native multi-turn mode enables Anthropic prompt cache_control breakpoints on system + last tool
        c_native = AnthropicMessagesClient("http://127.0.0.1:8766", "claude-sonnet", effort="high", max_prompt_bytes=4096, opener=fake_opener)
        resp = c_native.complete(ModelRequest(messages=[Message.text("system", "sys"), Message.text("user", "hi")], tools=tools))
        self.assertEqual(resp.usage.cached_tokens, 80)
        self.assertEqual(resp.usage.reasoning_tokens, 150)
        self.assertIsInstance(captured[0]["system"], list)
        self.assertEqual(captured[0]["system"][0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(captured[0]["tools"][-1]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(captured[0]["messages"][-1]["content"][-1]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(captured[0]["output_config"]["effort"], "high")

        # Hard byte cap fails fast before transport
        with self.assertRaises(PromptTooLargeError):
            c_native.complete(ModelRequest(messages=[Message.text("user", "x" * 5000)]))

        # Single-turn bridge flattening collapses multi-turn history into one user message with string system
        c_flat = AnthropicMessagesClient(
            "http://127.0.0.1:8766", "bridge-model", native_tool_calls=False, single_turn_flatten=True, opener=fake_opener
        )
        c_flat.complete(ModelRequest(messages=[
            Message.text("system", "sys"),
            Message.text("user", "edit file"),
            Message(role="assistant", content=[], tool_calls=[ToolCall("c1", "read_file", {"path": "a.py"})]),
            Message(role="tool", content=[], tool_call_id="c1", name="read_file"),
        ]))
        self.assertIsInstance(captured[1]["system"], str)
        self.assertEqual(len(captured[1]["messages"]), 1)
        combined = "\n".join(b["text"] for b in captured[1]["messages"][0]["content"])
        self.assertIn("[ASSISTANT]", combined)
        self.assertIn("read_file", combined)

    def test_openai_and_anthropic_sse_streaming(self):
        # 1) OpenAI SSE stream with reasoning, text deltas, incremental tool call & usage
        openai_sse = "\n".join([
            'data: {"model":"qwen-test","choices":[{"delta":{"reasoning_content":"step 1"}}]}',
            'data: {"choices":[{"delta":{"content":"Hel"}}]}',
            'data: {"choices":[{"delta":{"content":"lo!"}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"tc1","function":{"name":"read_file","arguments":"{\\"path\\":"}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"a.txt\\"}"}}]},"finish_reason":"tool_calls"}]}',
            'data: {"choices":[],"usage":{"prompt_tokens":30,"completion_tokens":12}}',
            "data: [DONE]",
            "",
        ])
        o_client = OpenAICompatibleClient("http://127.0.0.1:8000/v1", "qwen-test", opener=lambda req, timeout=None: FakeHTTPResponse(openai_sse))
        o_events: list[StreamEvent] = []
        o_resp = o_client.complete_stream(ModelRequest(messages=[Message.text("user", "hi")]), on_event=o_events.append)
        self.assertEqual(o_resp.text, "Hello!")
        self.assertEqual(o_resp.reasoning, "step 1")
        self.assertEqual(len(o_resp.tool_calls), 1)
        self.assertEqual(o_resp.tool_calls[0].arguments, {"path": "a.txt"})
        self.assertEqual(o_resp.usage.prompt_tokens, 30)
        self.assertIn("text_delta", [e.kind for e in o_events])
        self.assertIn("thinking_delta", [e.kind for e in o_events])

        # 2) Anthropic SSE stream with thinking_delta, text_delta, input_json_delta & message_delta
        anth_sse = "\n".join([
            'event: message_start',
            'data: {"type":"message_start","message":{"model":"claude-stream","usage":{"input_tokens":40,"cache_read_input_tokens":20}}}',
            'event: content_block_start',
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}',
            'event: content_block_delta',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"plan it"}}',
            'event: content_block_start',
            'data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}',
            'event: content_block_delta',
            'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Working "}}',
            'event: content_block_delta',
            'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"now."}}',
            'event: content_block_start',
            'data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"tu_1","name":"bash","input":{}}}',
            'event: content_block_delta',
            'data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\\"command\\":\\"echo 1\\"}"}}',
            'event: message_delta',
            'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":18}}',
            'event: message_stop',
            'data: {"type":"message_stop"}',
            "",
        ])
        a_client = AnthropicMessagesClient("http://127.0.0.1:8766", "claude-stream", opener=lambda req, timeout=None: FakeHTTPResponse(anth_sse))
        a_events: list[StreamEvent] = []
        a_resp = a_client.complete_stream(ModelRequest(messages=[Message.text("user", "hi")]), on_event=a_events.append)
        self.assertEqual(a_resp.text, "Working now.")
        self.assertEqual(a_resp.reasoning, "plan it")
        self.assertEqual(a_resp.finish_reason, "tool_use")
        self.assertEqual(a_resp.tool_calls[0].name, "bash")
        self.assertEqual(a_resp.tool_calls[0].arguments, {"command": "echo 1"})
        self.assertEqual(a_resp.usage.cached_tokens, 20)
        self.assertEqual(a_resp.usage.completion_tokens, 18)

    def test_model_router_routes_complete_and_stream(self):
        c_def, c_rev = MockModelClient(), MockModelClient()
        c_def.push("default-reply")
        c_rev.push("reviewer-reply")
        router = ModelRouter(c_def)
        router.set_route("reviewer", c_rev, effort="high", temperature=0.0)
        evs: list[StreamEvent] = []
        r1 = router.complete_stream(ModelRequest(messages=[Message.text("user", "x")], metadata={"role": "reviewer"}), on_event=evs.append)
        r2 = router.complete(ModelRequest(messages=[Message.text("user", "x")], metadata={"role": "other"}))
        self.assertEqual((r1.text, r2.text), ("reviewer-reply", "default-reply"))
        self.assertEqual(c_rev.calls[0].temperature, 0.0)
        self.assertTrue(any(e.kind == "text_delta" and e.text == "reviewer-reply" for e in evs))


class GeneralToolsTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.ws = Workspace.open(self.d, create_subdirs=False)
        self.cfg = HarnessConfig(state_dir=os.path.join(self.d, ".edgent"))
        os.makedirs(self.cfg.state_dir, exist_ok=True)
        self.store = ArtifactStore(self.cfg.artifacts_dir)
        self.addCleanup(self.store.close)
        self.reg = register_general_tools(build_default_registry(IdempotencyStore(self.cfg.db_path)))
        self.addCleanup(lambda: self.reg._pool.shutdown(wait=False))

    def _ctx(self, policy=None, hooks=None, extra=None):
        ex = {"policy": policy, "hooks": hooks, **(extra or {})}
        return ToolContext("r1", "n1", self.store, self.ws, Sidecars(), self.cfg, extra=ex)

    def test_read_file_line_ranges_and_image_detection(self):
        self.ws.write_text("code.py", "line1\nline2\nline3\nline4\n")
        with open(self.ws.resolve("pixel.png"), "wb") as f:
            f.write(_TINY_PNG)
        ctx = self._ctx(policy=PermissionPolicy(mode=PermissionMode.AUTO))
        r_lines = self.reg.call("read_file", {"path": "code.py", "start_line": 2, "end_line": 3}, ctx)
        self.assertTrue(r_lines.ok)
        self.assertEqual(r_lines.output["content"], "2: line2\n3: line3")
        r_img = self.reg.call("read_file", {"path": "pixel.png"}, ctx)
        self.assertTrue(r_img.ok)
        self.assertEqual(r_img.output["modality"], "image")
        self.assertEqual((r_img.output["width"], r_img.output["height"]), (1, 1))
        self.assertEqual(r_img.output["attach_image_path"], "pixel.png")

    def test_surgical_edit_and_write_revert_not_swallowed(self):
        ctx = self._ctx(policy=PermissionPolicy(mode=PermissionMode.ACCEPT_EDITS))
        self.assertTrue(self.reg.call("write_file", {"path": "doc.md", "content": "# T\n\nPara one.\n\nPara two.\n"}, ctx).ok)
        r = self.reg.call("edit_file", {"path": "doc.md", "operation": "insert", "after": "¶2", "new": "Bridge."}, ctx)
        self.assertTrue(r.ok, r.error)
        self.assertIn("Para one.\n\nBridge.\n\nPara two.", self.ws.read_text("doc.md"))
        # Reverting back to exact original text via write_file must actually write (idempotent_replay=False)
        self.assertTrue(self.reg.call("write_file", {"path": "doc.md", "content": "# T\n\nPara one.\n\nPara two.\n"}, ctx).ok)
        self.assertNotIn("Bridge.", self.ws.read_text("doc.md"))
        miss = self.reg.call("edit_file", {"path": "doc.md", "operation": "replace", "old": "Para twoo.", "new": "X"}, ctx)
        self.assertFalse(miss.ok)
        self.assertIn("Closest paragraph", miss.error.message)

    def test_bash_persistent_cwd_output_offloading_background_and_deny(self):
        policy = PermissionPolicy(mode=PermissionMode.AUTO, deny=["bash(rm -rf*)"])
        ctx = self._ctx(policy=policy)
        self.addCleanup(lambda: ctx.extra["task_manager"].stop_all() if "task_manager" in ctx.extra else None)

        # Deny pattern blocks dangerous shell command before execution
        denied = self.reg.call("bash", {"command": "rm -rf /"}, ctx)
        self.assertFalse(denied.ok)
        self.assertEqual(denied.error.code, "permission_denied")

        # Persistent cwd across calls
        os.makedirs(self.ws.resolve("subdir"), exist_ok=True)
        r1 = self.reg.call("bash", {"command": "cd subdir && echo hi | tr a-z A-Z"}, ctx)
        self.assertTrue(r1.ok, r1.error)
        self.assertEqual(r1.output["stdout"].strip(), "HI")
        self.assertEqual(r1.output["cwd"], "subdir")
        r2 = self.reg.call("bash", {"command": "pwd"}, ctx)
        self.assertTrue(r2.output["stdout"].strip().endswith("/subdir"))

        # Large output (> 16 KB) is offloaded to .edgent/tool_outputs/
        big = self.reg.call("bash", {"command": f"{sys.executable} -c \"print('A' * 20000)\""}, ctx)
        self.assertTrue(big.ok)
        self.assertIsNotNone(big.output["stdout_log"])
        self.assertTrue(os.path.isfile(self.ws.resolve(big.output["stdout_log"])))

        # Background task start -> output -> stop
        bg = self.reg.call("bash", {"command": f"{sys.executable} -c \"import time; print('READY', flush=True); time.sleep(20)\"", "background": True}, ctx)
        self.assertTrue(bg.ok, bg.error)
        tid = bg.output["task_id"]
        time.sleep(0.25)
        out = self.reg.call("task_output", {"task_id": tid}, ctx)
        self.assertTrue(out.ok)
        self.assertIn("READY", out.output["output"])
        st_stop = self.reg.call("task_stop", {"task_id": tid}, ctx)
        self.assertTrue(st_stop.ok)
        self.assertFalse(st_stop.output["running"])

    def test_web_fetch_ssrf_guard_html_cleaning_and_web_search(self):
        fake_html = "<html><head><title>Test Page</title><style>p{color:red}</style></head><body><h1>Header</h1><script>alert(1)</script><p>Hello <b>world</b>.</p></body></html>"
        fake_ddg = '<a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fdoc">Example Doc</a><a class="result__snippet">Useful snippet.</a>'

        def opener(req, timeout=None):
            if "duckduckgo.com" in req.full_url:
                return FakeHTTPResponse(fake_ddg, headers={"Content-Type": "text/html"})
            return FakeHTTPResponse(fake_html, headers={"Content-Type": "text/html"})

        # In ACCEPT_EDITS mode, network tool requires approval (blocked without approve cb)
        ctx_no_net = self._ctx(policy=PermissionPolicy(mode=PermissionMode.ACCEPT_EDITS), extra={"web_opener": opener})
        self.assertFalse(self.reg.call("web_fetch", {"url": "https://example.org"}, ctx_no_net).ok)

        # In AUTO mode, SSRF loopback is still blocked in code!
        ctx = self._ctx(policy=PermissionPolicy(mode=PermissionMode.AUTO), extra={"web_opener": opener})
        ssrf = self.reg.call("web_fetch", {"url": "http://127.0.0.1:8080/secret"}, ctx)
        self.assertFalse(ssrf.ok)
        self.assertEqual(ssrf.error.code, "permission_denied")

        # Valid external URL (mocked opener; bypass DNS only if offline by using allow_private_urls for fake test)
        ctx_ok = self._ctx(policy=PermissionPolicy(mode=PermissionMode.AUTO), extra={"web_opener": opener, "allow_private_urls": True})
        fetched = self.reg.call("web_fetch", {"url": "https://example.org/guide"}, ctx_ok)
        self.assertTrue(fetched.ok, fetched.error)
        self.assertEqual(fetched.output["title"], "Test Page")
        self.assertIn("Hello world.", fetched.output["content"])
        self.assertNotIn("alert(1)", fetched.output["content"])
        self.assertTrue(os.path.isfile(self.ws.resolve(fetched.output["saved_path"])))

        searched = self.reg.call("web_search", {"query": "edgent harness"}, ctx_ok)
        self.assertTrue(searched.ok)
        self.assertEqual(searched.output["results"][0]["url"], "https://example.org/doc")
        self.assertEqual(searched.output["results"][0]["title"], "Example Doc")

    def test_plan_mode_and_hooks(self):
        ctx_plan = self._ctx(policy=PermissionPolicy(mode=PermissionMode.PLAN))
        self.assertFalse(self.reg.call("write_file", {"path": "a.txt", "content": "x"}, ctx_plan).ok)
        self.assertFalse(self.reg.call("bash", {"command": "echo x"}, ctx_plan).ok)

        hooks = HookManager()
        seen = []
        hooks.on_pre_tool_use(lambda n, a, c: HookDecision(allow=True, updated_arguments={**a, "content": a["content"] + "\n"}) if n == "write_file" else None)
        hooks.on_post_tool_use(lambda n, a, r, c: seen.append((n, r.ok)))
        ctx = self._ctx(policy=PermissionPolicy(mode=PermissionMode.AUTO, deny=["write_file(secrets/*)"]), hooks=hooks)
        self.assertFalse(self.reg.call("write_file", {"path": "secrets/k.txt", "content": "1"}, ctx).ok)
        self.assertTrue(self.reg.call("write_file", {"path": "ok.txt", "content": "hi"}, ctx).ok)
        self.assertEqual(self.ws.read_text("ok.txt"), "hi\n")
        self.assertEqual(seen, [("write_file", True)])


class AgentLoopReliabilityTests(unittest.TestCase):
    def test_hierarchical_context_discovery_and_multimodal_image_attachment(self):
        d = tempfile.mkdtemp()
        fake_home = tempfile.mkdtemp()
        os.makedirs(os.path.join(fake_home, ".edgent"), exist_ok=True)
        with open(os.path.join(fake_home, ".edgent", "AGENTS.md"), "w", encoding="utf-8") as f:
            f.write("Global rule: always be concise.")

        ws = Workspace.open(d, create_subdirs=False)
        ws.write_text("AGENTS.md", "Repo rule: run pytest before committing.")
        ws.write_text(".edgent/skills/refactor/SKILL.md", "---\nname: refactor\ndescription: Clean refactors\n---\nKeep diffs minimal.\n")
        with open(ws.resolve("chart.png"), "wb") as f:
            # Pad PNG with trailing comment/data > 10 KB to verify ToolRegistry never compacts image metadata
            f.write(_TINY_PNG + (b"\x00" * 10_000))

        client = MockModelClient()
        client.push(
            ModelResponse(text="", tool_calls=[ToolCall("c1", "read_file", {"path": "chart.png"})]),
            ModelResponse(text="I see a 1x1 PNG image."),
        )
        loop = AgentLoop(workspace=ws, client=client, policy=PermissionPolicy(mode=PermissionMode.AUTO), home_dir=fake_home)
        try:
            res = loop.send("vis-1", "Inspect chart.png")
            self.assertEqual(res.status, "completed")
            # System prompt includes both global & repo AGENTS.md + auto-discovered workspace skill!
            sys_prompt = client.calls[0].messages[0].plain_text()
            self.assertIn("Global rule: always be concise.", sys_prompt)
            self.assertIn("Repo rule: run pytest before committing.", sys_prompt)
            self.assertIn("refactor", loop.skills.names())
            # Second call includes the native image ContentPart attached from read_file!
            img_parts = [p for m in client.calls[1].messages for p in m.content if p.type == "image"]
            self.assertEqual(len(img_parts), 1)
            self.assertTrue(img_parts[0].data_url.startswith("data:image/png;base64,"))
            # Tool output text stays compact metadata without dumping raw base64
            tool_msgs = [m.plain_text() for m in client.calls[1].messages if m.role == "tool"]
            self.assertIn('"attached_as_image": true', tool_msgs[0])
            self.assertNotIn("data:image/png;base64,", tool_msgs[0])
        finally:
            loop.close()

    def test_llm_backed_compaction_and_session_recall(self):
        d = tempfile.mkdtemp()
        ws = Workspace.open(d, create_subdirs=False)
        client = MockModelClient()
        client.push(
            ModelResponse(text="Remember magic token ALPHA-9981."),
            ModelResponse(text="Understood, moving to step 2."),
            # Compactor LLM call response:
            ModelResponse(text="- **Goal & Constraints**: Track magic token ALPHA-9981.\n- **Open Items**: Continue."),
            # Next user turn uses session_recall to pull verbatim history:
            ModelResponse(text="", tool_calls=[ToolCall("r1", "session_recall", {"query": "ALPHA-9981"})]),
            ModelResponse(text="Recalled ALPHA-9981 from transcript."),
        )
        loop = AgentLoop(workspace=ws, client=client, use_llm_compaction=True, policy=PermissionPolicy(mode=PermissionMode.AUTO))
        try:
            loop.send("s-comp", "Store ALPHA-9981")
            loop.send("s-comp", "Step 2")
            summary = loop.compact("s-comp", keep_recent=2)
            self.assertIn("Goal & Constraints", summary)
            r3 = loop.send("s-comp", "What was the exact token?")
            self.assertIn("ALPHA-9981", r3.reply)
        finally:
            loop.close()

    def test_doom_loop_truncation_continuation_cancel_and_undo(self):
        d = tempfile.mkdtemp()
        ws = Workspace.open(d, create_subdirs=False)
        ws.write_text("existing.txt", "original\n")
        ws.write_text("to_delete.txt", "bye\n")

        # 1) Checkpoint & Undo (modify existing, create new, delete another -> undo restores all 3!)
        c_undo = MockModelClient()
        c_undo.push(
            ModelResponse(text="", tool_calls=[
                ToolCall("u1", "edit_file", {"path": "existing.txt", "operation": "replace", "old": "original", "new": "mutated"}),
                ToolCall("u2", "write_file", {"path": "created.txt", "content": "brand new"}),
                ToolCall("u3", "delete_file", {"path": "to_delete.txt"}),
            ]),
            ModelResponse(text="Applied 3 file mutations."),
        )
        loop = AgentLoop(workspace=ws, client=c_undo, policy=PermissionPolicy(mode=PermissionMode.AUTO))
        try:
            res = loop.send("undo-1", "Mutate workspace")
            self.assertIsNotNone(res.checkpoint_id)
            self.assertEqual(ws.read_text("existing.txt"), "mutated\n")
            self.assertTrue(ws.exists("created.txt"))
            self.assertFalse(ws.exists("to_delete.txt"))

            reverted = loop.undo("undo-1", steps=1)
            self.assertTrue(reverted["ok"])
            self.assertEqual(ws.read_text("existing.txt"), "original\n")
            self.assertFalse(ws.exists("created.txt"))
            self.assertEqual(ws.read_text("to_delete.txt"), "bye\n")
        finally:
            loop.close()

        # 2) Truncation auto-continuation stitches multi-part response
        c_cont = MockModelClient()
        c_cont.push(
            ModelResponse(text="Part one of answer.", finish_reason="max_tokens"),
            ModelResponse(text="Part two finishes it.", finish_reason="stop"),
        )
        loop2 = AgentLoop(workspace=ws, client=c_cont, policy=PermissionPolicy(mode=PermissionMode.AUTO))
        try:
            r = loop2.send("cont-1", "Write long answer")
            self.assertEqual(r.status, "completed")
            self.assertEqual(r.turns_used, 2)
            self.assertIn("Part one of answer.", r.reply)
            self.assertIn("Part two finishes it.", r.reply)
        finally:
            loop2.close()

        # 3) Doom-loop detection stops identical repeating calls cleanly
        c_doom = MockModelClient()
        same_call = ToolCall("d1", "read_file", {"path": "missing.txt"})
        for _ in range(5):
            c_doom.push(ModelResponse(text="", tool_calls=[same_call]))
        loop3 = AgentLoop(workspace=ws, client=c_doom, doom_loop_threshold=3, policy=PermissionPolicy(mode=PermissionMode.AUTO))
        try:
            rd = loop3.send("doom-1", "Read missing file")
            self.assertEqual(rd.status, "doom_loop")
            self.assertLessEqual(rd.turns_used, 4)
        finally:
            loop3.close()

        # 4) Cancel event stops turn cleanly before model call
        cancel = threading.Event()
        cancel.set()
        loop4 = AgentLoop(workspace=ws, client=MockModelClient(), policy=PermissionPolicy(mode=PermissionMode.AUTO))
        try:
            rc = loop4.send("cancel-1", "Do work", cancel_event=cancel)
            self.assertEqual(rc.status, "cancelled")
            self.assertEqual(rc.turns_used, 0)
        finally:
            loop4.close()

    def test_general_purpose_plan_review_edit_and_independent_reviewer_subagent(self):
        """Proves a purely general-purpose loop handles plan -> HITL review -> surgical edit -> independent reviewer."""
        d = tempfile.mkdtemp()
        ws = Workspace.open(d, create_subdirs=False)
        ws.write_text("source.md", "# Chapter 1\n\nIntro paragraph.\n\nKeep this untouched with $\\frac{1}{2}$.\n")
        ws.write_text("AGENTS.md", "When editing source.md for a persona, propose plan via request_review, edit surgically, and spawn `reviewer` before finishing.")

        main_client = MockModelClient()
        rev_client = MockModelClient()
        main_client.push(
            # Turn 1: inspect & pause for human review of plan
            ModelResponse(text="", tool_calls=[
                ToolCall("p1", "write_file", {"path": "plan.md", "content": "1. Insert concrete Python example in intro.\n2. Keep final paragraph verbatim."}),
                ToolCall("p2", "request_review", {"kind": "plan", "title": "Edit Plan", "summary": "Surgical edit plan ready", "artifact_paths": ["plan.md"]}),
            ]),
            # After human approves: surgical edit + spawn isolated reviewer subagent
            ModelResponse(text="", tool_calls=[
                ToolCall("e1", "edit_file", {"path": "source.md", "operation": "insert", "after": "Intro paragraph.", "new": "```python\nprint(6 * 7)\n```"}),
            ]),
            ModelResponse(text="", tool_calls=[
                ToolCall("r1", "spawn_subagent", {"role": "reviewer", "task": "Review source.md against plan.md."}),
            ]),
            ModelResponse(text="Updated source.md and verified by independent reviewer."),
        )
        rev_client.push(
            ModelResponse(text="", tool_calls=[ToolCall("rv1", "read_file", {"path": "source.md"})]),
            ModelResponse(text="PASS: Intro example added and untouched paragraph preserved verbatim."),
        )
        router = ModelRouter(main_client)
        router.set_route("reviewer", rev_client, effort="high", temperature=0.0)

        loop1 = AgentLoop(workspace=ws, client=router, policy=PermissionPolicy(mode=PermissionMode.AUTO))
        try:
            r1 = loop1.send("doc-task", "Adapt source.md using AGENTS.md instructions.")
            self.assertEqual(r1.status, "awaiting_review")
            self.assertEqual(r1.review_request["title"], "Edit Plan")
        finally:
            loop1.close()

        # Resume on a fresh AgentLoop instance after approval
        loop2 = AgentLoop(workspace=ws, client=router, policy=PermissionPolicy(mode=PermissionMode.AUTO))
        try:
            r2 = loop2.resume_review("doc-task", approved=True, feedback="Approved, proceed.")
            self.assertEqual(r2.status, "completed")
            updated = ws.read_text("source.md")
            self.assertIn("print(6 * 7)", updated)
            self.assertIn(r"Keep this untouched with $\frac{1}{2}$.", updated)
            # Reviewer ran on its dedicated routed client with isolated fresh context
            self.assertEqual(len(rev_client.calls), 2)
        finally:
            loop2.close()

    def test_cache_stable_orientation_micro_compact_and_grounding_guard(self):
        d = tempfile.mkdtemp()
        ws = Workspace.open(d, create_subdirs=False)
        ws.write_text("big.txt", "SECRET_LINE_42\n" + ("lorem ipsum dolor sit amet " * 200))

        # 1) Creating new files must NOT mutate msgs[1] orientation (preserves prompt cache prefix)
        c1 = MockModelClient()
        c1.push(
            ModelResponse(text="", tool_calls=[ToolCall("w1", "write_file", {"path": "newly_created.md", "content": "hi"})]),
            ModelResponse(text="Created file."),
            ModelResponse(text="Next turn still has stable prefix."),
        )
        loop = AgentLoop(workspace=ws, client=c1, policy=PermissionPolicy(mode=PermissionMode.AUTO))
        try:
            loop.send("cache-1", "Create a file.")
            loop.send("cache-1", "Follow up.")
            orient_t1 = c1.calls[0].messages[1].plain_text()
            orient_t2 = c1.calls[1].messages[1].plain_text()
            orient_t3 = c1.calls[2].messages[1].plain_text()
            self.assertEqual(orient_t1, orient_t2)
            self.assertEqual(orient_t1, orient_t3)
            self.assertNotIn("newly_created.md", orient_t3)
        finally:
            loop.close()

        # 2) Micro-compaction shrinks older bulky read_file outputs in memory while keeping transcript verbatim
        c2 = MockModelClient()
        c2.push(
            ModelResponse(text="", tool_calls=[ToolCall("r1", "read_file", {"path": "big.txt"})]),
            ModelResponse(text="Read big.txt."),
            ModelResponse(text="Turn 2."),
            ModelResponse(text="Turn 3."),
            ModelResponse(text="Turn 4."),
        )
        loop2 = AgentLoop(
            workspace=ws,
            client=c2,
            micro_compact_tool_results=True,
            micro_compact_keep_turns=2,
            micro_compact_trigger_chars=4_000,
            policy=PermissionPolicy(mode=PermissionMode.AUTO),
        )
        try:
            loop2.send("mc-1", "Read big.txt")
            loop2.send("mc-1", "Step 2")
            loop2.send("mc-1", "Step 3")
            loop2.send("mc-1", "Step 4")
            last_tool_texts = [m.plain_text() for m in c2.calls[-1].messages if m.role == "tool"]
            self.assertTrue(any("earlier read_file output cleared" in t for t in last_tool_texts))
            # Verbatim transcript still supports session_recall!
            self.assertTrue(len(loop2.sessions.recall("mc-1", "SECRET_LINE_42")) >= 1)
        finally:
            loop2.close()

        # 3) Reviewer grounding guard rejects Turn-1 ungrounded reply with 0 tool calls and nudges tool use
        c3 = MockModelClient()
        c3.push(
            ModelResponse(text="", tool_calls=[ToolCall("sp1", "spawn_subagent", {"role": "reviewer", "task": "Audit big.txt"})]),
            # Reviewer Turn 1: tries to hallucinate without calling any tool!
            ModelResponse(text="Everything looks broken without reading."),
            # Reviewer Turn 2 (after automatic grounding nudge): actually calls read_file!
            ModelResponse(text="", tool_calls=[ToolCall("rv1", "read_file", {"path": "big.txt", "start_line": 1, "end_line": 3})]),
            ModelResponse(text="Verified SECRET_LINE_42 in big.txt."),
            # Parent final reply:
            ModelResponse(text="Reviewer verified big.txt."),
        )
        loop3 = AgentLoop(workspace=ws, client=c3, policy=PermissionPolicy(mode=PermissionMode.AUTO))
        try:
            res = loop3.send("guard-1", "Run reviewer on big.txt")
            self.assertEqual(res.status, "completed")
            # Confirm the reviewer's returned tool result includes tools_used=["read_file"]
            parent_tool_msg = [m.plain_text() for m in c3.calls[-1].messages if m.role == "tool"][0]
            self.assertIn('"tools_used": ["read_file"]', parent_tool_msg)
            self.assertIn("Verified SECRET_LINE_42", parent_tool_msg)
        finally:
            loop3.close()


class MCPClientServerTests(unittest.TestCase):
    def test_real_subprocess_mcp_client_connects_to_harness_mcp_server(self):
        d = tempfile.mkdtemp()
        ws = Workspace.open(d, create_subdirs=False)
        ws.write_text("hello.txt", "Hello world.\n")
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        env = dict(os.environ)
        env["PYTHONPATH"] = repo_root

        client = MCPClient(
            [sys.executable, "-m", "harness", "mcp", "--workspace", d, "--mode", "auto", "--compact"],
            server_name="local",
            cwd=repo_root,
            env=env,
            timeout_s=20.0,
        )
        self.addCleanup(client.close)
        info = client.connect()
        self.assertEqual(info["serverInfo"]["name"], "edgent-harness")

        reg = build_default_registry()
        self.addCleanup(lambda: reg._pool.shutdown(wait=False))
        registered = client.register_into(reg)
        self.assertIn("mcp__local__edit_file", registered)
        self.assertIn("mcp__local__bash", registered)

        cfg = HarnessConfig(state_dir=os.path.join(d, ".edgent_client"))
        os.makedirs(cfg.state_dir, exist_ok=True)
        store = ArtifactStore(cfg.artifacts_dir)
        self.addCleanup(store.close)
        ctx = ToolContext("r", "n", store, ws, Sidecars(), cfg, extra={"policy": PermissionPolicy(mode=PermissionMode.AUTO)})
        res = reg.call("mcp__local__edit_file", {"path": "hello.txt", "operation": "replace", "old": "world", "new": "MCP"}, ctx)
        self.assertTrue(res.ok, res.error)
        self.assertEqual(ws.read_text("hello.txt"), "Hello MCP.\n")

    def test_mcp_server_general_prompts(self):
        d = tempfile.mkdtemp()
        ws = Workspace.open(d, create_subdirs=False)
        cfg = HarnessConfig(state_dir=os.path.join(d, ".edgent"))
        os.makedirs(cfg.state_dir, exist_ok=True)
        store = ArtifactStore(cfg.artifacts_dir)
        self.addCleanup(store.close)
        reg = register_general_tools(build_default_registry(IdempotencyStore(cfg.db_path)))
        self.addCleanup(lambda: reg._pool.shutdown(wait=False))
        srv = MCPServer(reg, lambda: ToolContext("mcp", "mcp", store, ws, Sidecars(), cfg))
        resp = srv.handle_message({"jsonrpc": "2.0", "id": 1, "method": "prompts/get", "params": {"name": "review_work", "arguments": {"target": "app.py"}}})
        self.assertIn("app.py", resp["result"]["messages"][0]["content"]["text"])


class RuntimeAndUITests(unittest.TestCase):
    def test_heartbeat_scheduler_periodic_runs_and_log(self):
        d = tempfile.mkdtemp()
        ws = Workspace.open(d, create_subdirs=False)
        client = MockModelClient()
        client.push("hb-1", "hb-2")
        loop = AgentLoop(workspace=ws, client=client, policy=PermissionPolicy(mode=PermissionMode.AUTO))
        self.addCleanup(loop.close)
        log_path = os.path.join(d, ".edgent", "heartbeat.jsonl")
        sched = HeartbeatScheduler(lambda sid, p, stg: loop.send(sid, p, stage=stg), log_path=log_path, poll_s=0.02)
        sched.add_job(HeartbeatJob("check-status", "hb-sess", "Check workspace status", interval_s=0.05, max_runs=2))
        sched.start()
        self.addCleanup(sched.stop)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if sched.list_jobs()[0]["runs"] >= 2:
                break
            time.sleep(0.03)
        sched.stop()
        self.assertEqual(sched.list_jobs()[0]["runs"], 2)
        with open(log_path, "r", encoding="utf-8") as f:
            recs = [json.loads(ln) for ln in f if ln.strip()]
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs[0]["status"], "completed")

    def test_terminal_repl_commands_streaming_and_interactive_approval(self):
        d = tempfile.mkdtemp()
        ws = Workspace.open(d, create_subdirs=False)
        client = MockModelClient()
        client.push(
            ModelResponse(text="", tool_calls=[ToolCall("w1", "write_file", {"path": "repl.txt", "content": "ok"})]),
            ModelResponse(text="Wrote repl.txt."),
        )
        loop = AgentLoop(workspace=ws, client=client, policy=PermissionPolicy(mode=PermissionMode.ASK))
        self.addCleanup(loop.close)
        out, err = io.StringIO(), io.StringIO()
        # User answers 'a' (always allow write_file)
        repl = TerminalREPL(loop, session_id="r1", stdin=io.StringIO("a\n"), stdout=out, stderr=err)
        self.assertIn("/undo", repl.handle_command("/help"))
        self.assertIn("mode switched", repl.handle_command("/mode ask"))
        res = repl.handle_line("Create repl.txt")
        self.assertEqual(res.status, "completed")
        self.assertEqual(ws.read_text("repl.txt"), "ok")
        self.assertIn("write_file", loop.policy.allow)
        self.assertIn("Wrote repl.txt.", out.getvalue())
        undo_out = repl.handle_command("/undo")
        self.assertTrue(json.loads(undo_out)["ok"])
        self.assertFalse(ws.exists("repl.txt"))

    def test_http_sse_server_endpoints_and_streaming(self):
        d = tempfile.mkdtemp()
        ws = Workspace.open(d, create_subdirs=False)
        client = MockModelClient()
        client.push(
            ModelResponse(text="", tool_calls=[ToolCall("s1", "write_file", {"path": "srv.txt", "content": "from-server"})]),
            ModelResponse(text="Server wrote srv.txt."),
            ModelResponse(text="Streamed reply works!"),
        )
        loop = AgentLoop(workspace=ws, client=client, policy=PermissionPolicy(mode=PermissionMode.AUTO))
        self.addCleanup(loop.close)
        srv = HarnessServer(loop, host="127.0.0.1", port=0, auth_token="secret-tok")
        srv.start_background()
        self.addCleanup(srv.stop)
        base = srv.base_url
        hdrs = {"Authorization": "Bearer secret-tok", "Content-Type": "application/json"}

        # Unauthorized request rejected
        with self.assertRaises(urllib.error.HTTPError) as ctx_err:
            urllib.request.urlopen(base + "/healthz", timeout=5)
        self.assertEqual(ctx_err.exception.code, 401)
        ctx_err.exception.close()

        # Healthz & tools
        with urllib.request.urlopen(urllib.request.Request(base + "/healthz", headers=hdrs), timeout=5) as r:
            hz = json.loads(r.read().decode("utf-8"))
        self.assertTrue(hz["ok"])

        # JSON POST message + undo
        req_msg = urllib.request.Request(
            base + "/v1/sessions/api-1/messages",
            data=json.dumps({"message": "Write srv.txt"}).encode("utf-8"),
            headers=hdrs,
            method="POST",
        )
        with urllib.request.urlopen(req_msg, timeout=10) as r:
            body = json.loads(r.read().decode("utf-8"))
        self.assertEqual(body["status"], "completed")
        self.assertEqual(ws.read_text("srv.txt"), "from-server")

        # SSE streaming POST message
        req_sse = urllib.request.Request(
            base + "/v1/sessions/api-1/messages",
            data=json.dumps({"message": "Stream back", "stream": True}).encode("utf-8"),
            headers={**hdrs, "Accept": "text/event-stream"},
            method="POST",
        )
        with urllib.request.urlopen(req_sse, timeout=10) as r:
            sse_raw = r.read().decode("utf-8")
        self.assertIn("event: text_delta", sse_raw)
        self.assertIn("Streamed reply works!", sse_raw)
        self.assertIn("event: result", sse_raw)

        # Undo via API
        req_undo = urllib.request.Request(base + "/v1/sessions/api-1/undo", data=b"{}", headers=hdrs, method="POST")
        with urllib.request.urlopen(req_undo, timeout=5) as r:
            u = json.loads(r.read().decode("utf-8"))
        self.assertTrue(u["ok"])
        self.assertFalse(ws.exists("srv.txt"))

    def test_doctor_and_cli_chat(self):
        info = run_doctor()
        self.assertTrue(info["ok"])
        self.assertIn("python3", info["binaries"])
        d = tempfile.mkdtemp()
        buf1, buf2 = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf1):
            self.assertEqual(cli_main(["--json", "doctor"]), 0)
        self.assertTrue(json.loads(buf1.getvalue())["ok"])
        with contextlib.redirect_stdout(buf2):
            self.assertEqual(cli_main(["--json", "chat", "Hello workspace", "--workspace", d, "--session", "s1"]), 0)
        self.assertEqual(json.loads(buf2.getvalue())["status"], "completed")


if __name__ == "__main__":
    unittest.main()
