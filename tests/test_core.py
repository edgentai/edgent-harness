import io
import json
import unittest
import urllib.error
from contextlib import contextmanager

from harness.core.client import MockModelClient, ModelError, ModelRequest, OpenAICompatibleClient, find_role, role_marker, split_thinking
from harness.core.config import EFFORT_PROFILES, EffortLevel, HarnessConfig
from harness.core.types import Budget, ContentPart, Message, SubAgentResult, Usage, Verdict, VerdictStatus
from harness.core.utils import SequentialIdGenerator, extract_json, redact, truncate


class ExtractJsonTests(unittest.TestCase):
    def test_bare_fenced_and_embedded(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})
        self.assertEqual(extract_json('Sure:\n```json\n{"a": [1,2]}\n```'), {"a": [1, 2]})
        self.assertEqual(extract_json('text before {"a": "x}y"} after'), {"a": "x}y"})
        self.assertIsNone(extract_json("no json here"))
        self.assertEqual(extract_json("[1, 2]"), [1, 2])


class ThinkingSplitTests(unittest.TestCase):
    def test_split(self):
        r, a = split_thinking("<think>hmm</think>42")
        self.assertEqual((r, a), ("hmm", "42"))
        r, a = split_thinking("plain")
        self.assertEqual((r, a), (None, "plain"))
        r, a = split_thinking("partial<think>unterminated")
        self.assertEqual((r, a), ("unterminated", "partial"))


class TypesTests(unittest.TestCase):
    def test_roundtrip(self):
        m = Message(role="user", content=[ContentPart.of_text("hi", untrusted=True)])
        self.assertEqual(Message.from_dict(m.to_dict()).content[0].untrusted, True)
        r = SubAgentResult(answer="x", usage=Usage(1, 2))
        self.assertEqual(SubAgentResult.from_dict(r.to_dict()).usage.completion_tokens, 2)
        v = Verdict(VerdictStatus.PARTIAL, [], ["gap"])
        self.assertEqual(Verdict.from_dict(v.to_dict()).status, VerdictStatus.PARTIAL)

    def test_budget(self):
        b = Budget(max_tokens=100, max_model_calls=2)
        b.charge(Usage(60, 50))
        self.assertIn("tokens", b.exhausted())
        self.assertTrue(b.near_exhaustion())
        self.assertEqual(b.used_model_calls, 1)


class ConfigTests(unittest.TestCase):
    def test_profiles_monotonic(self):
        levels = list(EffortLevel)
        for a, b in zip(levels, levels[1:]):
            self.assertLessEqual(EFFORT_PROFILES[a].thinking_budget, EFFORT_PROFILES[b].thinking_budget)
            self.assertLessEqual(EFFORT_PROFILES[a].max_tool_iterations, EFFORT_PROFILES[b].max_tool_iterations)
        self.assertEqual(EffortLevel.MAX.next(), None)
        self.assertEqual(EffortLevel.LOW.next(), EffortLevel.MEDIUM)

    def test_save_load(self):
        import os, tempfile

        d = tempfile.mkdtemp()
        cfg = HarnessConfig(state_dir=d, default_effort=EffortLevel.HIGH)
        cfg.serving.model = "m"
        cfg.save(os.path.join(d, "c.json"))
        back = HarnessConfig.load(os.path.join(d, "c.json"))
        self.assertEqual(back.default_effort, EffortLevel.HIGH)
        self.assertEqual(back.serving.model, "m")
        self.assertEqual(back.sandbox.allowed_executables, cfg.sandbox.allowed_executables)


class FakeResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class OpenAIClientTests(unittest.TestCase):
    def _client(self, responses):
        calls = []

        def opener(req, timeout=None):
            calls.append(json.loads(req.data.decode()) if req.data else None)
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return FakeResponse(json.dumps(item).encode())

        c = OpenAICompatibleClient("http://x/v1", "qwen", api_key="k", max_retries=2, ids=SequentialIdGenerator(), opener=opener)
        c._sleep = lambda *_: None
        return c, calls

    def test_payload_and_parse(self):
        body = {"choices": [{"message": {"content": "<think>r</think>answer", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1}"}}]}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 4}}}
        c, calls = self._client([body])
        msg = Message(role="user", content=[ContentPart.of_text("hi"), ContentPart(type="image", data_url="data:image/png;base64,AA==", mime="image/png")])
        resp = c.complete(ModelRequest([Message.text("system", "s"), msg], tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}], thinking_budget=0))
        self.assertEqual(resp.text, "answer")
        self.assertEqual(resp.reasoning, "r")
        self.assertEqual(resp.tool_calls[0].arguments, {"a": 1})
        self.assertEqual(resp.usage.cached_tokens, 4)
        payload = calls[0]
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(payload["messages"][1]["content"][1]["type"], "image_url")
        self.assertEqual(payload["messages"][0]["content"], "s")

    def test_retry_then_fail(self):
        import time as _t

        orig = _t.sleep
        _t.sleep = lambda *_: None
        try:
            c, _ = self._client([urllib.error.URLError("down"), urllib.error.URLError("down"), urllib.error.URLError("down")])
            with self.assertRaises(ModelError):
                c.complete(ModelRequest([Message.text("user", "x")]))
        finally:
            _t.sleep = orig

    def test_non_retryable_http_error(self):
        err = urllib.error.HTTPError("http://x", 400, "bad", {}, io.BytesIO(b"nope"))
        c, _ = self._client([err])
        with self.assertRaises(ModelError) as cm:
            c.complete(ModelRequest([Message.text("user", "x")]))
        self.assertEqual(cm.exception.status, 400)


class MockClientTests(unittest.TestCase):
    def test_role_dispatch_and_script(self):
        m = MockModelClient(ids=SequentialIdGenerator())
        self.assertEqual(find_role([Message.text("system", role_marker("planner"))]), "planner")
        r = m.complete(ModelRequest([Message.text("system", role_marker("classifier")), Message.text("user", "Suggest ten titles")]))
        self.assertEqual(extract_json(r.text)["problem_type"], "divergent")
        m.push("scripted")
        self.assertEqual(m.complete(ModelRequest([Message.text("user", "x")])).text, "scripted")
        self.assertGreater(r.usage.prompt_tokens, 0)


class UtilTests(unittest.TestCase):
    def test_redact_truncate(self):
        self.assertEqual(redact({"api_key": "x", "n": [{"token": 1}]}), {"api_key": "[REDACTED]", "n": [{"token": "[REDACTED]"}]})
        self.assertTrue(truncate("a" * 100, 20).endswith("[truncated]"))


if __name__ == "__main__":
    unittest.main()
