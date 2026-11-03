"""Run golden scenarios offline: a fresh workspace, the scripted mock provider as the fixture
environment, one turn through the real loop with the real tools, then the gate."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from ..adapters.providers.mock import MockProvider
from ..domain.types import Reply, ToolCall
from ..domain.workspace import Workspace
from ..observability.tracing import TraceReader
from .gate import GateResult, Scenario, evaluate


def scripted_reply(d: dict[str, Any]) -> Reply:
    calls = [ToolCall.new(c["name"], c.get("arguments") or {}) for c in d.get("tool_calls") or []]
    return Reply(text=d.get("text", ""), tool_calls=calls, finish_reason="tool_calls" if calls else "stop")


def run_scenario(scenario: Scenario, root: Path | None = None) -> tuple[list[Any], str, Any, str]:
    from ..application.app import App, AppConfig

    base = Path(root) if root else Path(tempfile.mkdtemp(prefix="edgent-eval-"))
    ws = Workspace.init(base / scenario.id, name=scenario.title or scenario.id, git=False)
    for rel, content in scenario.setup_files.items():
        ws.write_text(rel, content)
    provider = MockProvider()
    provider.push(*[scripted_reply(r) for r in scenario.scripted_replies])
    app = App(ws, AppConfig(provider="mock", effort="low"), provider=provider)
    session = app.new_session("chat", title=scenario.id)
    loop = app.loop(session, permission_mode="auto")
    result = loop.run_turn(scenario.task)
    events = list(TraceReader(str(ws.path("traces") / f"{session.session_id}.jsonl")))
    return events, result.text, app.evidence, result.stop_reason


def run_scenarios(scenarios: list[Scenario], root: Path | None = None) -> list[GateResult]:
    out = []
    for sc in scenarios:
        try:
            events, text, store, stop = run_scenario(sc, root)
        except Exception as e:  # noqa: BLE001 - a crashed scenario is a failed scenario, not a crashed run
            out.append(evaluate(sc, [], f"runner error: {type(e).__name__}: {e}", None, stop_reason="error"))
            continue
        out.append(evaluate(sc, events, text, store, stop))
    return out
