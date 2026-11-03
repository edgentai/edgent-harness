"""The learner model: mastery per concept, the progress ledger, misconceptions.

Everything lives under ``learner/``. ``mastery.json`` holds one SM-2 style
schedule per concept with the last few events as evidence; ``progress.json``
is an append-only ledger of events plus the gamification settings;
``misconceptions.md`` is a human-readable list of dated bullets.

Streaks, points and badges are never stored: ``progress`` recomputes them from
the ledger on every read, so nothing is awarded on a model's say-so and the
ledger can be audited by hand.
"""
from __future__ import annotations

import datetime as _dt
import re
import threading
from typing import Any

from .errors import Limit
from .outline import Outline
from .utils import atomic_write_text, read_json, truncate, write_json
from .workspace import Workspace

EVENT_TYPES: tuple[str, ...] = (
    "answered", "explained", "asked", "reviewed", "unit_done", "session_done", "interview_done", "film_done",
    "assessment_done", "misconception", "misconception_resolved",
)
QUALITY_MIN = 0.0
QUALITY_MAX = 5.0
PASS_QUALITY = 3.0  # SM-2: below this a recall is a lapse
EASE_DEFAULT = 2.5
EASE_FLOOR = 1.3
LEVEL_MAX = 5.0
CONCEPT_EVIDENCE_LIMIT = Limit("mastery evidence per concept", 20, "events")
INTERVAL_DAYS_LIMIT = Limit("review interval", 365, "days", does_not_raise=("ease", "repetitions"))  # SM-2 compounds; a year is the ceiling
RECENT_QUALITIES_WINDOW_EVENTS = 5
DUE_DEFAULT_LIMIT_CONCEPTS = 20
SCHEDULE_DEFAULT_LIMIT_CONCEPTS = 10
SUMMARY_DEFAULT_MAX_CHARS = 600
CONCEPT_NAME_MAX_CHARS = 120
DETAIL_MAX_CHARS = 500
COMEBACK_CORRECT_ANSWERS = 3
TEN_CORRECT = 10
HUNDRED_REVIEWS = 100
STREAK_BADGE_DAYS = {"seven_day_streak": 7, "thirty_day_streak": 30}

POINTS: dict[str, int] = {
    "answered": 10,  # only when the answer was correct (quality >= PASS_QUALITY)
    "unit_done": 50,
    "session_done": 20,
    "interview_done": 40,
    "reviewed": 5,
    "misconception_resolved": 15,
}
BADGE_NAMES: dict[str, str] = {
    "first_unit_done": "First unit done",
    "ten_correct_answers": "Ten correct answers",
    "seven_day_streak": "Seven-day streak",
    "thirty_day_streak": "Thirty-day streak",
    "first_interview": "First interview",
    "first_film": "First film",
    "hundred_reviews": "A hundred reviews",
    "comeback": "Comeback",
}
DEFAULT_SETTINGS: dict[str, Any] = {"streaks": True, "badges": True, "points": True, "timezone": "UTC"}
MISCONCEPTIONS_HEADER = "# Misconceptions\n\n"

_BULLET_RE = re.compile(
    r"^- (?P<date>\d{4}-\d{2}-\d{2}) (?:\[(?P<concept>[^\]]*)\] )?(?P<text>.*?)(?: \(resolved (?P<resolved>\d{4}-\d{2}-\d{2})\))?\s*$"
)
_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


class LearnerError(ValueError):
    """A malformed event or setting. Tools turn it into an invalid_input failure."""


# ----------------------------------------------------------------------------- time
def as_datetime(now: Any = None) -> _dt.datetime:
    """An aware datetime from None (wall clock), a datetime, an epoch number, an ISO string or a Clock."""
    if now is None:
        return _dt.datetime.now(tz=_dt.UTC)
    if isinstance(now, _dt.datetime):
        return now if now.tzinfo is not None else now.replace(tzinfo=_dt.UTC)
    if isinstance(now, bool):
        raise TypeError("now must be a datetime, an epoch number, an ISO string or a clock")
    if isinstance(now, (int, float)):
        return _dt.datetime.fromtimestamp(float(now), tz=_dt.UTC)
    if isinstance(now, str):
        return as_datetime(_dt.datetime.fromisoformat(now))
    if hasattr(now, "now") and callable(now.now):
        return as_datetime(now.now())
    raise TypeError(f"cannot read a time from {type(now).__name__}")


def _zone(name: Any, strict: bool = False) -> _dt.tzinfo:
    if not name or str(name).upper() == "UTC":
        return _dt.UTC
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(str(name))
    except Exception as e:  # noqa: BLE001 - ZoneInfoNotFoundError or a missing tz database
        if strict:
            raise LearnerError(f"unknown timezone {name!r}; use an IANA name such as 'Europe/Berlin' or 'UTC'") from e
        return _dt.UTC  # a zone already on disk must not break progress; UTC is the documented fallback


def _event_time(e: dict[str, Any]) -> _dt.datetime | None:
    """The event's instant, or None when ``at`` is missing or malformed: a hand-edited ledger must not break progress."""
    at = e.get("at")
    if not isinstance(at, str) or not at:
        return None
    try:
        return as_datetime(at)
    except ValueError:
        return None


def _norm_concept(name: Any) -> str:
    return " ".join(str(name).split()).lower()


# ----------------------------------------------------------------------------- the store
class LearnerStore:
    """Reads and writes ``learner/``. One instance per workspace; safe to share across threads."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.mastery_path = workspace.path("mastery")
        self.ledger_path = workspace.path("learner_progress")
        self.misconceptions_path = workspace.path("misconceptions")
        key = str(workspace.root)
        with _LOCKS_GUARD:
            self._lock = _LOCKS.setdefault(key, threading.RLock())

    # ------------------------------------------------------------------ files
    def mastery(self) -> dict[str, Any]:
        data = read_json(self.mastery_path, None) or {}
        concepts = data.get("concepts") if isinstance(data.get("concepts"), dict) else {}
        return {"version": 1, "concepts": dict(concepts)}

    def ledger(self) -> dict[str, Any]:
        data = read_json(self.ledger_path, None) or {}
        events = [e for e in (data.get("events") or []) if isinstance(e, dict) and e.get("at")]
        settings = {**DEFAULT_SETTINGS, **(data.get("settings") if isinstance(data.get("settings"), dict) else {})}
        return {"version": 1, "events": events, "settings": settings}

    @property
    def settings(self) -> dict[str, Any]:
        return self.ledger()["settings"]

    def update_settings(self, **changes: Any) -> dict[str, Any]:
        """Switch streaks, badges or points off, or set the timezone. Unknown keys are refused."""
        unknown = sorted(set(changes) - set(DEFAULT_SETTINGS))
        if unknown:
            raise LearnerError(f"unknown learner settings {unknown}; known: {sorted(DEFAULT_SETTINGS)}")
        if "timezone" in changes:
            _zone(changes["timezone"], strict=True)  # refused here rather than silently read back as UTC
        with self._lock:
            ledger = self.ledger()
            for k, v in changes.items():
                ledger["settings"][k] = str(v) if k == "timezone" else bool(v)
            write_json(self.ledger_path, ledger)
            return dict(ledger["settings"])

    # ------------------------------------------------------------------ events
    def record(self, event: dict[str, Any], now: Any = None) -> dict[str, Any]:
        """Validate, append to the ledger, update mastery (SM-2 when a quality is given), persist both files."""
        now_dt = as_datetime(now).astimezone(_dt.UTC)  # the ledger is kept in UTC so its ISO strings sort chronologically
        ev = self._normalise(event, now_dt)
        with self._lock:
            mastery = self.mastery()
            ledger = self.ledger()
            if ev["concept"]:
                self._touch_concept(mastery["concepts"], ev, now_dt)
            ledger["events"].append(ev)
            if ev["type"] == "misconception":
                self._append_misconception(ev, now_dt)
            elif ev["type"] == "misconception_resolved":
                self._resolve_misconceptions(ev, now_dt)
            write_json(self.mastery_path, mastery)
            write_json(self.ledger_path, ledger)
        return dict(ev)

    def validate(self, event: dict[str, Any], now: Any = None) -> dict[str, Any]:
        """The normalised event ``record`` would store, without storing it; raises LearnerError when malformed."""
        return self._normalise(event, as_datetime(now).astimezone(_dt.UTC))

    def _normalise(self, event: dict[str, Any], now_dt: _dt.datetime) -> dict[str, Any]:
        if not isinstance(event, dict):
            raise LearnerError("an event must be an object with at least a 'type'")
        etype = str(event.get("type") or "").strip()
        if etype not in EVENT_TYPES:
            raise LearnerError(f"unknown event type {etype!r}; use one of {', '.join(EVENT_TYPES)}")
        concept = _norm_concept(event["concept"]) if event.get("concept") not in (None, "") else ""
        if len(concept) > CONCEPT_NAME_MAX_CHARS:
            raise LearnerError(f"concept name longer than {CONCEPT_NAME_MAX_CHARS} chars")
        quality = event.get("quality")
        if quality is not None:
            if isinstance(quality, bool) or not isinstance(quality, (int, float)):
                raise LearnerError("quality must be a number from 0 to 5")
            if not QUALITY_MIN <= float(quality) <= QUALITY_MAX:
                raise LearnerError(f"quality {quality} is outside 0-5")
            quality = float(quality)
        detail = truncate(" ".join(str(event.get("detail") or "").split()), DETAIL_MAX_CHARS)
        ev: dict[str, Any] = {
            "at": now_dt.isoformat(),
            "type": etype,
            "concept": concept or None,
            "quality": quality,
            "points": _points_for(etype, quality),
            "detail": detail,
            "session_id": str(event["session_id"]) if event.get("session_id") else None,
        }
        if event.get("evidence_id"):
            ev["evidence_id"] = str(event["evidence_id"])
        return ev

    @staticmethod
    def _touch_concept(concepts: dict[str, Any], ev: dict[str, Any], now_dt: _dt.datetime) -> None:
        c = concepts.setdefault(ev["concept"], _new_concept())
        c.setdefault("evidence", [])
        c["evidence"].append({"at": ev["at"], "type": ev["type"], "quality": ev["quality"], "detail": ev["detail"], "evidence_id": ev.get("evidence_id")})
        c["evidence"] = c["evidence"][-int(CONCEPT_EVIDENCE_LIMIT.value):]
        c["last_seen"] = ev["at"]
        if ev["quality"] is not None:
            _apply_sm2(c, ev["quality"], now_dt)
        c["level"] = _level(c)

    # ------------------------------------------------------------------ misconceptions
    def misconceptions(self) -> list[dict[str, Any]]:
        try:
            text = self.misconceptions_path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return []
        out = []
        for line in text.splitlines():
            m = _BULLET_RE.match(line)
            if m:
                out.append({"date": m.group("date"), "concept": m.group("concept"), "text": m.group("text"), "resolved": m.group("resolved")})
        return out

    def open_misconceptions(self) -> list[dict[str, Any]]:
        return [m for m in self.misconceptions() if not m["resolved"]]

    def _append_misconception(self, ev: dict[str, Any], now_dt: _dt.datetime) -> None:
        text = ev["detail"] or f"a misconception about {ev['concept'] or 'the topic'}"
        tag = f"[{ev['concept']}] " if ev["concept"] else ""
        line = f"- {now_dt.date().isoformat()} {tag}{text}\n"
        if not self.misconceptions_path.exists():
            atomic_write_text(self.misconceptions_path, MISCONCEPTIONS_HEADER)
        with open(self.misconceptions_path, "a", encoding="utf-8") as f:
            f.write(line)

    def _resolve_misconceptions(self, ev: dict[str, Any], now_dt: _dt.datetime) -> int:
        """Mark open bullets as resolved: those matching the event's detail (substring), else every open bullet on its concept."""
        if not self.misconceptions_path.exists():
            return 0
        needle = (ev["detail"] or "").lower()
        lines = self.misconceptions_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        open_bullets: list[tuple[int, re.Match[str]]] = []
        for i, raw in enumerate(lines):
            m = _BULLET_RE.match(raw.rstrip("\n"))
            if m and not m.group("resolved"):
                open_bullets.append((i, m))
        hits = [i for i, m in open_bullets if needle and (needle in m.group("text").lower() or m.group("text").lower() in needle)]
        if not hits and ev["concept"]:  # the text named nothing on file: the concept is the next best key
            hits = [i for i, m in open_bullets if (m.group("concept") or "") == ev["concept"]]
        for i in hits:
            lines[i] = lines[i].rstrip("\n") + f" (resolved {now_dt.date().isoformat()})\n"
        if hits:
            atomic_write_text(self.misconceptions_path, "".join(lines))
        return len(hits)

    # ------------------------------------------------------------------ scheduling
    def due(self, now: Any = None, limit: int = DUE_DEFAULT_LIMIT_CONCEPTS) -> list[dict[str, Any]]:
        """Concepts whose next review is at or before ``now``, most overdue first."""
        now_dt = as_datetime(now)
        rows = []
        for name, c in self.mastery()["concepts"].items():
            nr = c.get("next_review")
            if not nr:
                continue
            try:
                nr_dt = as_datetime(nr)
            except ValueError:
                continue
            if nr_dt <= now_dt:
                rows.append({
                    "concept": name,
                    "level": float(c.get("level", 0.0)),
                    "overdue_days": round((now_dt - nr_dt).total_seconds() / 86400, 2),
                    "next_review": nr,
                    "interval_days": int(c.get("interval_days", 0)),
                    "lapses": int(c.get("lapses", 0)),
                })
        rows.sort(key=lambda r: (-r["overdue_days"], r["concept"]))
        return rows[: max(0, int(limit))]

    def outline_concepts(self) -> list[dict[str, str]]:
        """Concepts the outline says its units introduce (``meta.introduces``), in outline order, deduplicated."""
        try:
            outline = Outline(self.workspace.path("outline"))
        except Exception:  # noqa: BLE001 - a broken outline is not the learner's problem
            return []
        seen: set[str] = set()
        out: list[dict[str, str]] = []
        for u in outline.units:
            introduces = u.meta.get("introduces") if isinstance(u.meta, dict) else None
            if isinstance(introduces, str):
                introduces = [s for s in introduces.split(",")]
            for raw in introduces or []:
                name = _norm_concept(raw)
                if name and name not in seen:
                    seen.add(name)
                    out.append({"concept": name, "unit_id": u.id})
        return out

    def schedule(self, limit: int = SCHEDULE_DEFAULT_LIMIT_CONCEPTS, now: Any = None) -> list[dict[str, Any]]:
        """What to study next: due reviews first, then concepts the outline introduces that were never seen."""
        limit = max(0, int(limit))
        items = [dict(r, reason="due") for r in self.due(now, limit)]
        if len(items) < limit:
            known = set(self.mastery()["concepts"]) | {i["concept"] for i in items}
            for entry in self.outline_concepts():
                if entry["concept"] in known:
                    continue
                items.append({"concept": entry["concept"], "level": 0.0, "reason": "new", "unit_id": entry["unit_id"]})
                if len(items) >= limit:
                    break
        return items

    # ------------------------------------------------------------------ progress (computed, never stored)
    def progress(self, now: Any = None) -> dict[str, Any]:
        now_dt = as_datetime(now)
        ledger = self.ledger()
        settings = ledger["settings"]
        tz = _zone(settings.get("timezone"))
        timed = sorted(((t, e) for e in ledger["events"] if (t := _event_time(e)) is not None), key=lambda te: te[0])
        events = [e for _, e in timed]
        out: dict[str, Any] = {"events_total": len(events), "last_active": events[-1]["at"] if events else None}
        if settings.get("streaks", True):
            days = sorted({t.astimezone(tz).date() for t, _ in timed})
            out["streak_days"], out["longest_streak"] = _streaks(days, now_dt.astimezone(tz).date())
        if settings.get("points", True):
            out["points"] = sum(_points_for(e.get("type", ""), e.get("quality")) for e in events)
        if settings.get("badges", True):
            out["badges"] = _badges(timed, tz)
        return out

    # ------------------------------------------------------------------ context block
    def summary(self, max_chars: int = SUMMARY_DEFAULT_MAX_CHARS, now: Any = None) -> str:
        """A short text block for the system context: strongest and weakest concepts, due count, streak, misconceptions."""
        concepts = self.mastery()["concepts"]
        progress = self.progress(now)
        open_mis = self.open_misconceptions()
        if not concepts and not progress["events_total"]:
            return truncate("No learner data yet: nothing tracked, nothing due.", max_chars)
        lines = [f"Concepts tracked: {len(concepts)}."]
        if concepts:
            ranked = sorted(concepts.items(), key=lambda kv: (-float(kv[1].get("level", 0)), kv[0]))
            lines.append("Strongest: " + ", ".join(f"{n} ({float(c.get('level', 0)):.1f})" for n, c in ranked[:3]) + ".")
            lines.append("Weakest: " + ", ".join(f"{n} ({float(c.get('level', 0)):.1f})" for n, c in reversed(ranked[-3:])) + ".")
        due = self.due(now)
        lines.append(f"Due for review: {len(due)}" + (" (" + ", ".join(d["concept"] for d in due[:3]) + ")." if due else "."))
        bits = []
        if "streak_days" in progress:
            bits.append(f"streak {progress['streak_days']} days (longest {progress['longest_streak']})")
        if "points" in progress:
            bits.append(f"{progress['points']} points")
        if "badges" in progress:
            bits.append(f"{len(progress['badges'])} badges")
        if bits:
            lines.append("Progress: " + ", ".join(bits) + ".")
        lines.append(f"Open misconceptions: {len(open_mis)}" + (" (" + "; ".join(m["text"] for m in open_mis[:2]) + ")." if open_mis else "."))
        return truncate("\n".join(lines), max_chars)


# ----------------------------------------------------------------------------- rules
def _new_concept() -> dict[str, Any]:
    return {"level": 0.0, "ease": EASE_DEFAULT, "interval_days": 0, "repetitions": 0, "lapses": 0, "last_seen": None, "next_review": None, "evidence": []}


def _apply_sm2(c: dict[str, Any], quality: float, now_dt: _dt.datetime) -> None:
    """SM-2: intervals 1, 6, then interval*ease; ease moves with quality (floor 1.3); a lapse restarts at 1 day."""
    ease = float(c.get("ease", EASE_DEFAULT))
    if quality >= PASS_QUALITY:
        reps = int(c.get("repetitions", 0)) + 1
        if reps == 1:
            interval = 1
        elif reps == 2:
            interval = 6
        else:
            interval = round(int(c.get("interval_days", 1)) * ease)
        q = QUALITY_MAX - quality
        ease = max(EASE_FLOOR, ease + 0.1 - q * (0.08 + q * 0.02))
        c["repetitions"] = reps
    else:
        interval = 1
        c["repetitions"] = 0
        c["lapses"] = int(c.get("lapses", 0)) + 1
    c["ease"] = round(ease, 4)
    c["interval_days"] = max(1, min(int(interval), int(INTERVAL_DAYS_LIMIT.value)))
    c["next_review"] = (now_dt + _dt.timedelta(days=c["interval_days"])).isoformat()


def _level(c: dict[str, Any]) -> float:
    """0-5: the mean of recent qualities, discounted until a few successful repetitions back it."""
    qualities = [float(e["quality"]) for e in c.get("evidence", []) if e.get("quality") is not None]
    recent = qualities[-RECENT_QUALITIES_WINDOW_EVENTS:]
    if not recent:
        return 0.0
    weight = min(1.0, 0.4 + 0.2 * int(c.get("repetitions", 0)))
    return round(min(LEVEL_MAX, sum(recent) / len(recent) * weight), 2)


def _points_for(etype: str, quality: float | None) -> int:
    if etype == "answered":
        return POINTS["answered"] if quality is not None and quality >= PASS_QUALITY else 0
    return POINTS.get(etype, 0)


def _streaks(days: list[_dt.date], today: _dt.date) -> tuple[int, int]:
    """(current streak ending today or yesterday, longest run of consecutive study days)."""
    day_set = set(days)
    one = _dt.timedelta(days=1)
    cursor = today if today in day_set else today - one
    current = 0
    while cursor in day_set:
        current += 1
        cursor -= one
    longest = run = 0
    prev: _dt.date | None = None
    for d in sorted(day_set):
        run = run + 1 if prev is not None and d - prev == one else 1
        longest = max(longest, run)
        prev = d
    return current, longest


def _badges(timed: list[tuple[_dt.datetime, dict[str, Any]]], tz: _dt.tzinfo) -> list[dict[str, Any]]:
    """Badges from the ledger (events with their instants, oldest first), each earned once at the event that completed its rule."""
    earned: dict[str, str] = {}

    def earn(badge: str, at: str) -> None:
        earned.setdefault(badge, at)

    correct = reviews = run = 0
    prev_day: _dt.date | None = None
    comeback: dict[str, tuple[bool, int]] = {}  # concept -> (has lapsed, correct answers since)
    one = _dt.timedelta(days=1)
    for t, e in timed:
        etype, q, at, concept = e.get("type"), e.get("quality"), str(e.get("at")), e.get("concept")
        day = t.astimezone(tz).date()
        if prev_day is None or day != prev_day:
            run = run + 1 if prev_day is not None and day - prev_day == one else 1
            prev_day = day
        for badge, needed in STREAK_BADGE_DAYS.items():
            if run >= needed:
                earn(badge, at)
        if etype == "unit_done":
            earn("first_unit_done", at)
        elif etype == "interview_done":
            earn("first_interview", at)
        elif etype == "film_done":
            earn("first_film", at)
        elif etype == "reviewed":
            reviews += 1
            if reviews == HUNDRED_REVIEWS:
                earn("hundred_reviews", at)
        if etype == "answered" and q is not None and float(q) >= PASS_QUALITY:
            correct += 1
            if correct == TEN_CORRECT:
                earn("ten_correct_answers", at)
        if concept and q is not None:
            lapsed, since = comeback.get(concept, (False, 0))
            if float(q) < PASS_QUALITY:
                comeback[concept] = (True, 0)
            else:
                since = since + 1 if lapsed else 0
                comeback[concept] = (lapsed, since)
                if lapsed and since >= COMEBACK_CORRECT_ANSWERS:
                    earn("comeback", at)
    rows = [{"id": b, "name": BADGE_NAMES[b], "earned_at": at} for b, at in earned.items()]
    rows.sort(key=lambda r: r["earned_at"])  # stable: badges earned at the same instant keep the order their rules fired
    return rows


__all__ = ["BADGE_NAMES", "EVENT_TYPES", "POINTS", "LearnerError", "LearnerStore", "as_datetime"]
