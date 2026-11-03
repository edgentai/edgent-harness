"""The heartbeat scheduler: time slots once per day, intervals after elapse, state persisted before the run, errors captured."""
from __future__ import annotations

import datetime as dt
import json
import threading
from typing import Any

import pytest

from edgent.application.heartbeat import DEFAULT_PROMPT, Heartbeat
from edgent.domain.errors import ConfigurationError
from edgent.domain.utils import FakeClock

T0 = dt.datetime(2026, 11, 3, 8, 0, tzinfo=dt.UTC)


class Recorder:
    def __init__(self, result: Any = "done", fail: bool = False) -> None:
        self.prompts: list[str] = []
        self.result = result
        self.fail = fail
        self.state_seen: list[dict[str, Any]] = []
        self.hb: Heartbeat | None = None

    def __call__(self, prompt: str) -> Any:
        self.prompts.append(prompt)
        if self.hb is not None:
            self.state_seen.append(self.hb.state())
        if self.fail:
            raise RuntimeError("the model is down")
        return self.result


class FakeTracer:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def record(self, kind: str, data: dict[str, Any] | None = None, **kw: Any) -> None:
        self.events.append((kind, dict(data or {})))


def at(hour: int, minute: int = 0, day_offset: int = 0) -> float:
    return (T0.replace(hour=hour, minute=minute) + dt.timedelta(days=day_offset)).timestamp()


def test_time_slot_fires_once_per_day(workspace) -> None:
    clock = FakeClock(at(8))
    run = Recorder()
    hb = Heartbeat(workspace, run, times=["21:00", "09:00"], clock=clock, tz="UTC")
    assert hb.times == [(9, 0), (21, 0)] and hb.prompt == DEFAULT_PROMPT and hb.configured()
    out = hb.tick()
    assert out == {"ran": False, "slot": None, "result": None, "next_due": "2026-11-03T09:00:00+00:00"} and run.prompts == []
    assert hb.next_due() == dt.datetime(2026, 11, 3, 9, 0, tzinfo=dt.UTC) and not hb.due()
    clock.advance(3600)
    assert hb.due()
    out = hb.tick()
    assert out["ran"] and out["slot"] == "2026-11-03 09:00" and out["result"] == "done" and out["next_due"] == "2026-11-03T21:00:00+00:00"
    assert run.prompts == [DEFAULT_PROMPT]
    clock.advance(60)
    assert not hb.due() and hb.tick()["ran"] is False and len(run.prompts) == 1
    clock.advance(12 * 3600 + 30 * 60)  # 21:31: the evening slot fires late, once
    assert hb.tick()["slot"] == "2026-11-03 21:00" and hb.tick()["ran"] is False
    clock.advance(11 * 3600 + 29 * 60)  # 09:00 next day
    out = hb.tick()
    assert out["slot"] == "2026-11-04 09:00" and len(run.prompts) == 3
    state = json.loads((workspace.path("state") / "heartbeat.json").read_text())
    assert state["last_slot"] == "2026-11-04 09:00" and state["fired"] == {"09:00": "2026-11-04", "21:00": "2026-11-03"} and state["runs"] == 3
    assert state["last_run_at"] == "2026-11-04T09:00:00+00:00" and state["last_error"] is None


def test_missed_slot_fires_late_then_waits_for_tomorrow(workspace) -> None:
    clock = FakeClock(at(14))
    run = Recorder()
    hb = Heartbeat(workspace, run, times=["09:00"], clock=clock)
    assert hb.tick()["slot"] == "2026-11-03 09:00"
    assert hb.next_due() == dt.datetime(2026, 11, 4, 9, 0, tzinfo=dt.UTC)
    clock.advance(5 * 3600)
    assert hb.tick()["ran"] is False
    hb2 = Heartbeat(workspace, run, times=["09:00"], clock=clock)  # a restart reads the persisted state
    assert hb2.tick()["ran"] is False and len(run.prompts) == 1


def test_interval_fires_after_elapse(workspace) -> None:
    clock = FakeClock(at(8))
    run = Recorder(result={"text": "nothing to report"})
    hb = Heartbeat(workspace, run, interval_minutes=30, clock=clock)
    out = hb.tick()  # never ran: fires at once
    assert out["ran"] and out["slot"] == "interval" and out["result"] == {"text": "nothing to report"} and out["next_due"] == "2026-11-03T08:30:00+00:00"
    clock.advance(29 * 60)
    assert not hb.due() and hb.tick()["ran"] is False
    clock.advance(60)
    assert hb.due() and hb.tick()["slot"] == "interval" and len(run.prompts) == 2
    assert hb.state()["last_run_at"] == "2026-11-03T08:30:00+00:00" and hb.state()["last_slot"] == "interval"


def test_slot_and_interval_together_prefer_the_slot(workspace) -> None:
    clock = FakeClock(at(9))
    run = Recorder()
    hb = Heartbeat(workspace, run, times=["09:00"], interval_minutes=60, clock=clock)
    out = hb.tick()
    assert out["slot"] == "2026-11-03 09:00" and len(run.prompts) == 1
    assert hb.tick()["ran"] is False  # the slot run also satisfied the interval
    clock.advance(3600)
    assert hb.tick()["slot"] == "interval"
    clock.advance(23 * 3600)
    assert hb.tick()["slot"] == "2026-11-04 09:00"


def test_state_persists_before_run_and_errors_are_captured(workspace) -> None:
    clock = FakeClock(at(9))
    tracer = FakeTracer()
    run = Recorder(fail=True)
    hb = Heartbeat(workspace, run, times=["09:00"], clock=clock, tracer=tracer)
    run.hb = hb
    out = hb.tick()
    assert out["ran"] and out["slot"] == "2026-11-03 09:00" and out["result"] is None and out["error"] == "RuntimeError: the model is down"
    assert run.state_seen[0]["last_slot"] == "2026-11-03 09:00" and run.state_seen[0]["last_run_at"] == "2026-11-03T09:00:00+00:00"  # written before run()
    state = hb.state()
    assert state["last_error"] == "RuntimeError: the model is down" and state["runs"] == 1
    assert tracer.events == [("heartbeat", {"slot": "2026-11-03 09:00", "ok": False, "error": "RuntimeError: the model is down", "at": "2026-11-03T09:00:00+00:00"})]
    assert hb.tick()["ran"] is False  # a failed run does not refire the slot
    run.fail = False
    clock.advance(24 * 3600)
    assert hb.tick()["error" if False else "slot"] == "2026-11-04 09:00" and hb.state()["last_error"] is None and tracer.events[-1][1]["ok"] is True


def test_run_forever_stops_on_the_event(workspace) -> None:
    clock = FakeClock(at(8))
    stop = threading.Event()
    seen: list[str] = []

    def run(prompt: str) -> str:
        seen.append(prompt)
        stop.set()
        return "ok"

    hb = Heartbeat(workspace, run, interval_minutes=5, prompt="  Heartbeat:   anything due?  ", clock=clock)
    assert hb.prompt == "Heartbeat: anything due?"
    t = threading.Thread(target=hb.run_forever, args=(stop,), kwargs={"poll_s": 0.01}, daemon=True)
    t.start()
    t.join(5)
    assert not t.is_alive() and seen == ["Heartbeat: anything due?"]
    hb.state_path = workspace.path("state") / "missing-dir" / "x" / "heartbeat.json"
    stop2 = threading.Event()
    timer = threading.Timer(0.2, stop2.set)
    timer.start()
    hb.run_forever(stop2, poll_s=0.01)  # a tick that cannot write state is traced, never raised


def test_no_schedule_means_never_due(workspace) -> None:
    hb = Heartbeat(workspace, Recorder(), clock=FakeClock(at(9)))
    assert not hb.configured() and hb.next_due() is None and not hb.due()
    assert hb.tick() == {"ran": False, "slot": None, "result": None, "next_due": None}


def test_from_settings_timezone_and_validation(workspace) -> None:
    clock = FakeClock(at(7, 59))
    run = Recorder()
    settings = {"heartbeat": {"times": "09:00, 18:30", "interval_minutes": "120", "prompt": "Daily brief.", "timezone": "Europe/Berlin"}, "web": {}}
    hb = Heartbeat.from_settings(workspace, run, settings, clock=clock)
    assert hb.times == [(9, 0), (18, 30)] and hb.interval_minutes == 120 and hb.prompt == "Daily brief." and str(hb.tz) == "Europe/Berlin"
    out = hb.tick()  # the interval has never run: fires at once; next slot is 09:00 Berlin = 08:00 UTC in November
    assert out["slot"] == "interval" and out["next_due"] == "2026-11-03T09:00:00+01:00"
    clock.advance(60)
    assert hb.tick()["slot"] == "2026-11-03 09:00" and hb.state()["fired"]["09:00"] == "2026-11-03"
    plain = Heartbeat.from_settings(workspace, run, {})
    assert not plain.configured() and plain.prompt == DEFAULT_PROMPT
    with pytest.raises(ConfigurationError):
        Heartbeat(workspace, run, times=["9am"])
    with pytest.raises(ConfigurationError):
        Heartbeat(workspace, run, times=["25:00"])
    with pytest.raises(ConfigurationError):
        Heartbeat(workspace, run, interval_minutes="soon")
    with pytest.raises(ConfigurationError):
        Heartbeat(workspace, run, interval_minutes=0.5)
    with pytest.raises(ConfigurationError):
        Heartbeat(workspace, run, tz="Mars/Olympus")
    with pytest.raises(ConfigurationError):
        Heartbeat(workspace, run, times=[f"{h:02d}:00" for h in range(24)] + ["00:30"])
    assert Heartbeat(workspace, run, times=None, interval_minutes=None, tz="utc").tz == dt.UTC


# ----------------------------------------------------------------------------- review additions
def test_several_missed_slots_are_satisfied_by_one_run(workspace) -> None:
    clock = FakeClock(at(14))
    run = Recorder()
    hb = Heartbeat(workspace, run, times=["09:00", "12:00", "18:00"], clock=clock)
    out = hb.tick()
    assert out["ran"] and out["slot"] == "2026-11-03 09:00" and out["slots"] == ["09:00", "12:00"] and len(run.prompts) == 1
    assert hb.tick()["ran"] is False and hb.state()["fired"] == {"09:00": "2026-11-03", "12:00": "2026-11-03"}
    assert hb.next_due() == dt.datetime(2026, 11, 3, 18, 0, tzinfo=dt.UTC)
    clock.advance(4 * 3600)
    out = hb.tick()
    assert out["slot"] == "2026-11-03 18:00" and "slots" not in out and len(run.prompts) == 2


def test_from_settings_rejects_a_non_object_heartbeat_setting(workspace) -> None:
    with pytest.raises(ConfigurationError):
        Heartbeat.from_settings(workspace, Recorder(), {"heartbeat": "09:00"})
    assert not Heartbeat.from_settings(workspace, Recorder(), {"heartbeat": None}).configured()
