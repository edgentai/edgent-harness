"""The heartbeat: wakes the agent with a fixed prompt on a schedule (after OpenClaw).

``edgent serve --heartbeat`` runs ``run_forever`` in a thread; ``edgent heartbeat`` runs one
``tick``. A time slot (HH:MM in the configured zone) fires once per day, at or after its
time; an interval fires once the interval has elapsed since the last run (and at once when
the agent has never run). State lives at ``.edgent/heartbeat.json`` and is written before
the run, so a crash mid-run never fires the same slot twice. Errors from the run are
captured into the result and the tracer, never raised, so the loop survives them.
"""
from __future__ import annotations

import datetime as _dt
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..domain.errors import ConfigurationError, Limit
from ..domain.utils import Clock, read_json, write_json

DEFAULT_PROMPT = (
    "Heartbeat. Check learner_schedule for due reviews, book_status for pending units, and anything worth "
    "reporting. Act within permissions; keep it brief; write nothing outside the workspace."
)
STATE_FILE = "heartbeat.json"
INTERVAL_MINUTES_LIMIT = Limit("heartbeat interval", 1, "minutes", setting_name="heartbeat.interval_minutes", floor=1)
TIME_SLOTS_LIMIT = Limit("heartbeat time slots", 24, "slots per day", setting_name="heartbeat.times")
POLL_DEFAULT_S = 30.0
POLL_FLOOR_S = 0.05
INTERVAL_SLOT = "interval"
_TIME_RE = re.compile(r"^(?P<h>[01]?\d|2[0-3]):(?P<m>[0-5]\d)$")


def _as_datetime(now: Any) -> _dt.datetime:
    if isinstance(now, _dt.datetime):
        return now if now.tzinfo is not None else now.replace(tzinfo=_dt.UTC)
    if isinstance(now, (int, float)) and not isinstance(now, bool):
        return _dt.datetime.fromtimestamp(float(now), tz=_dt.UTC)
    if isinstance(now, str):
        return _as_datetime(_dt.datetime.fromisoformat(now))
    raise TypeError(f"cannot read a time from {type(now).__name__}")


def _zone(name: Any) -> _dt.tzinfo:
    if name in (None, "", "UTC", "utc"):
        return _dt.UTC
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(str(name))
    except Exception as e:  # noqa: BLE001 - ZoneInfoNotFoundError or a missing tz database
        raise ConfigurationError(f"heartbeat.timezone {name!r} is not a known IANA zone (for example 'Europe/Berlin'). Correct it in .edgent/settings.json.") from e


def _parse_times(times: Any) -> list[tuple[int, int]]:
    if times in (None, "", []):
        return []
    raw = [t.strip() for t in times.split(",")] if isinstance(times, str) else list(times)
    out: set[tuple[int, int]] = set()
    for t in raw:
        m = _TIME_RE.match(str(t).strip())
        if not m:
            raise ConfigurationError(f"heartbeat.times entry {t!r} must be HH:MM in 24-hour time. Correct it in .edgent/settings.json.")
        out.add((int(m.group("h")), int(m.group("m"))))
    if len(out) > TIME_SLOTS_LIMIT.value:
        raise ConfigurationError(f"heartbeat.times has {len(out)} slots; {TIME_SLOTS_LIMIT.describe()} Use interval_minutes for anything more frequent.")
    return sorted(out)


def _parse_interval(value: Any) -> int | None:
    if value in (None, "", 0, False):
        return None
    try:
        minutes = int(value)
    except (TypeError, ValueError) as e:
        raise ConfigurationError(f"heartbeat.interval_minutes must be a whole number of minutes; got {value!r}. Correct it in .edgent/settings.json.") from e
    if minutes < INTERVAL_MINUTES_LIMIT.value:
        raise ConfigurationError(f"heartbeat.interval_minutes={minutes} is below the floor. {INTERVAL_MINUTES_LIMIT.describe()}")
    return minutes


class Heartbeat:
    """A scheduler over ``run(prompt)``; the workspace holds its state."""

    def __init__(
        self,
        workspace: Any,
        run: Callable[[str], Any],
        *,
        times: list[str] | str | None = None,
        interval_minutes: int | None = None,
        prompt: str | None = None,
        clock: Clock | None = None,
        tz: str = "UTC",
        tracer: Any = None,
    ) -> None:
        self.workspace = workspace
        self.run = run
        self.times = _parse_times(times)
        self.interval_minutes = _parse_interval(interval_minutes)
        self.prompt = " ".join(str(prompt).split()) if prompt and str(prompt).strip() else DEFAULT_PROMPT
        self.clock = clock or Clock()
        self.tz = _zone(tz)
        self.tracer = tracer
        self.state_path = Path(workspace.path("state")) / STATE_FILE
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, workspace: Any, run: Callable[[str], Any], settings: dict[str, Any] | None, *, clock: Clock | None = None, tracer: Any = None) -> Heartbeat:
        """Build from merged settings: ``settings['heartbeat']`` holds times, interval_minutes, prompt and timezone."""
        hb = (settings or {}).get("heartbeat") if isinstance(settings, dict) else None
        if hb is not None and not isinstance(hb, dict):
            raise ConfigurationError(f"settings.heartbeat must be an object with times, interval_minutes, prompt and timezone; got {type(hb).__name__}. Correct it in .edgent/settings.json.")
        hb = hb or {}
        return cls(workspace, run, times=hb.get("times"), interval_minutes=hb.get("interval_minutes"), prompt=hb.get("prompt"), clock=clock, tz=hb.get("timezone") or "UTC", tracer=tracer)

    # ------------------------------------------------------------------ state
    def configured(self) -> bool:
        return bool(self.times or self.interval_minutes)

    def state(self) -> dict[str, Any]:
        data = read_json(self.state_path, None)
        data = data if isinstance(data, dict) else {}
        fired = data.get("fired") if isinstance(data.get("fired"), dict) else {}
        return {"last_run_at": data.get("last_run_at"), "last_slot": data.get("last_slot"), "fired": dict(fired), "runs": int(data.get("runs") or 0), "last_error": data.get("last_error")}

    def _save(self, state: dict[str, Any]) -> None:
        write_json(self.state_path, state)

    def _now(self, now: Any = None) -> _dt.datetime:
        return _as_datetime(self.clock.now() if now is None else now).astimezone(self.tz)

    # ------------------------------------------------------------------ schedule
    def _candidates(self, now_dt: _dt.datetime, state: dict[str, Any]) -> list[tuple[_dt.datetime, str, str]]:
        """(when, slot key, label) for every configured trigger: the next time it would fire, past means due."""
        out: list[tuple[_dt.datetime, str, str]] = []
        today = now_dt.date().isoformat()
        for hh, mm in self.times:
            label = f"{hh:02d}:{mm:02d}"
            at = now_dt.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if state["fired"].get(label) == today:
                at = at + _dt.timedelta(days=1)
            out.append((at, f"{at.date().isoformat()} {label}", label))
        if self.interval_minutes:
            last = state.get("last_run_at")
            at = now_dt if not last else _as_datetime(last).astimezone(self.tz) + _dt.timedelta(minutes=self.interval_minutes)
            out.append((at, INTERVAL_SLOT, INTERVAL_SLOT))
        return out

    def next_due(self, now: Any = None) -> _dt.datetime | None:
        """When the next trigger fires (a past time means it is due now), or None without a schedule."""
        now_dt = self._now(now)
        cands = self._candidates(now_dt, self.state())
        return min(c[0] for c in cands) if cands else None

    def due(self, now: Any = None) -> bool:
        now_dt = self._now(now)
        return self._due_slot(now_dt, self.state()) is not None

    def _due_slot(self, now_dt: _dt.datetime, state: dict[str, Any]) -> tuple[str, str] | None:
        """The slot to fire now: the earliest due time slot, else the interval when it has elapsed."""
        due = sorted((c for c in self._candidates(now_dt, state) if c[0] <= now_dt), key=lambda c: (c[2] == INTERVAL_SLOT, c[0]))
        return (due[0][1], due[0][2]) if due else None

    # ------------------------------------------------------------------ running
    def tick(self, now: Any = None) -> dict[str, Any]:
        """Fire the due slot, if any. State is persisted before ``run``; a failing run is reported, not raised."""
        with self._lock:
            now_dt = self._now(now)
            state = self.state()
            slot = self._due_slot(now_dt, state)
            if slot is None:
                nxt = self.next_due(now_dt)
                return {"ran": False, "slot": None, "result": None, "next_due": nxt.isoformat() if nxt else None}
            key, label = slot
            consumed = [c[2] for c in self._candidates(now_dt, state) if c[0] <= now_dt and c[2] != INTERVAL_SLOT]
            state["last_run_at"] = now_dt.isoformat()
            state["last_slot"] = key
            state["runs"] += 1
            for lbl in consumed:  # one run satisfies every overdue slot of the day: an outage is not N model calls in a row
                state["fired"][lbl] = now_dt.date().isoformat()
            self._save(state)  # persist before proceeding: a crash in run() must not refire this slot
            out: dict[str, Any] = {"ran": True, "slot": key, "result": None}
            if len(consumed) > 1:
                out["slots"] = consumed
            try:
                out["result"] = self.run(self.prompt)
                state["last_error"] = None
            except Exception as e:  # noqa: BLE001 - the scheduler outlives any one run
                out["error"] = f"{type(e).__name__}: {e}"[:600]
                state["last_error"] = out["error"]
            self._save(state)
            self._trace("heartbeat", {"slot": key, "ok": "error" not in out, "error": out.get("error"), "at": state["last_run_at"]})
            nxt = self.next_due(now_dt)
            out["next_due"] = nxt.isoformat() if nxt else None
            return out

    def run_forever(self, stop_event: threading.Event, poll_s: float = POLL_DEFAULT_S) -> None:
        """Tick until ``stop_event`` is set; nothing raised by a tick escapes the loop."""
        poll_s = max(POLL_FLOOR_S, float(poll_s))
        while not stop_event.is_set():
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001 - a state write failure must not kill the serve loop
                self._trace("heartbeat_error", {"error": f"{type(e).__name__}: {e}"[:600]})
            stop_event.wait(poll_s)

    def _trace(self, kind: str, data: dict[str, Any]) -> None:
        if self.tracer is None:
            return
        try:
            self.tracer.record(kind, data)
        except Exception:  # noqa: BLE001 - tracing is best effort
            pass


__all__ = ["DEFAULT_PROMPT", "Heartbeat"]
