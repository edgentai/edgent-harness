from __future__ import annotations

from edgent.adapters.sandbox import InterpreterSession, run_python, run_subprocess
from edgent.adapters.toolchain import Toolchain
from edgent.domain.errors import ToolFailure
from edgent.observability import Metrics, Tracer, TraceReader


def test_run_python_isolated(workspace):
    r = run_python("import os, sys\nprint(sorted(k for k in os.environ if k in ('AWS_SECRET','HOME')))\nprint(2+2)", str(workspace.root), timeout_s=20)
    assert r.ok and "4" in r.stdout and "AWS_SECRET" not in r.stdout
    r = run_python("import time\ntime.sleep(5)", str(workspace.root), timeout_s=0.5)
    assert r.timed_out and not r.ok
    r = run_subprocess(["definitely-not-a-binary"], str(workspace.root))
    assert r.returncode == 127
    r = run_subprocess(["python3", "-c", "print('x'*50000)"], str(workspace.root), max_output=1000)
    assert r.truncated


def test_interpreter_session(workspace):
    s = InterpreterSession(str(workspace.root), timeout_s=20)
    try:
        assert s.exec("a = 21").get("ok")
        assert "42" in s.exec("print(a*2)")["stdout"]
        assert not s.exec("1/0")["ok"]
    finally:
        s.close()


def test_toolchain_report_and_require():
    tc = Toolchain.detect()
    rep = tc.report()
    assert any(i["name"] == "ffmpeg" for i in rep["items"]) and "manim_films" in rep["capabilities"]
    tc.binaries["manim"] = None
    tc.modules["manim"] = False
    try:
        tc.require("manim", "films")
    except ToolFailure as e:
        assert e.error.category.value == "unavailable" and "edgent doctor" in e.error.remediation
    else:
        raise AssertionError("expected ToolFailure")


def test_tracer_and_metrics(workspace):
    t = Tracer(workspace.path("traces"), "s1")
    t.record("model_call", {"usage": {"prompt_tokens": 100, "completion_tokens": 10, "cached_tokens": 50}, "api_key": "secret"})
    t.record("tool_call", {"name": "grep", "arguments": {"p": 1}, "ok": True})
    t.record("tool_call", {"name": "grep", "arguments": {"p": 1}, "ok": False})
    t2 = Tracer(workspace.path("traces"), "s1")
    assert t2._seq == 3
    events = list(TraceReader(t.path))
    assert events[0].data["api_key"] == "[REDACTED]"
    s = TraceReader(t.path).summary()
    assert s["cache_hit_rate"] == 0.5 and s["tool_errors"] == 1 and s["repeated_identical_calls"] == 1
    m = Metrics()
    m.record_tool_call("grep", False, 0.1)
    assert m.snapshot()["tool_error_rate"] == 1.0
