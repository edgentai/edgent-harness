"""The learner model: SM-2 progression, due ordering, the schedule, progress computed from the ledger, badges, settings, summary."""
from __future__ import annotations

import datetime as dt
import json
from typing import Any

import pytest

from edgent.domain.learner import BADGE_NAMES, EVENT_TYPES, LearnerError, LearnerStore, as_datetime
from edgent.domain.outline import Outline, Unit
from edgent.domain.utils import FakeClock

T0 = dt.datetime(2026, 11, 3, 9, 0, tzinfo=dt.UTC)


def day(n: int, hour: int = 9) -> dt.datetime:
    return T0 + dt.timedelta(days=n, hours=hour - 9)


def answered(store: LearnerStore, concept: str, quality: float, when: dt.datetime, detail: str = "") -> dict[str, Any]:
    return store.record({"type": "answered", "concept": concept, "quality": quality, "detail": detail}, now=when)


# ----------------------------------------------------------------------------- SM-2
def test_sm2_progression_and_lapse(workspace) -> None:
    store = LearnerStore(workspace)
    seen = []
    for i, q in enumerate([5, 5, 4]):
        answered(store, "Vector Addition", q, day(i))
        c = store.mastery()["concepts"]["vector addition"]  # concept names are normalised
        seen.append((c["repetitions"], c["interval_days"], c["ease"]))
    assert [s[:2] for s in seen] == [(1, 1), (2, 6), (3, 16)]  # 1, 6, then round(6 * 2.7)
    assert seen[0][2] == 2.6 and seen[1][2] == 2.7 and seen[2][2] == 2.7  # +0.1, +0.1, +0 for quality 4
    c = store.mastery()["concepts"]["vector addition"]
    assert c["next_review"] == (day(2) + dt.timedelta(days=16)).isoformat() and c["last_seen"] == day(2).isoformat()
    assert c["lapses"] == 0 and c["level"] == pytest.approx(4.67, abs=0.01)
    answered(store, "vector addition", 2, day(3))  # a lapse
    c = store.mastery()["concepts"]["vector addition"]
    assert c["repetitions"] == 0 and c["interval_days"] == 1 and c["lapses"] == 1 and c["ease"] == 2.7
    assert c["next_review"] == day(4).isoformat() and c["level"] < 2.0
    answered(store, "vector addition", 5, day(4))
    c = store.mastery()["concepts"]["vector addition"]
    assert (c["repetitions"], c["interval_days"]) == (1, 1) and len(c["evidence"]) == 5
    assert c["evidence"][-1] == {"at": day(4).isoformat(), "type": "answered", "quality": 5.0, "detail": "", "evidence_id": None}


def test_ease_floor_and_evidence_cap(workspace) -> None:
    store = LearnerStore(workspace)
    for i in range(12):
        answered(store, "bayes", 3, day(i))  # quality 3 drops ease by 0.14 each time
    c = store.mastery()["concepts"]["bayes"]
    assert c["ease"] == 1.3 and c["repetitions"] == 12 and c["interval_days"] >= 6
    for i in range(12, 30):
        store.record({"type": "asked", "concept": "bayes", "detail": f"question {i}"}, now=day(i))
    c = store.mastery()["concepts"]["bayes"]
    assert len(c["evidence"]) == 20 and c["repetitions"] == 12  # asked events keep the schedule, only add evidence
    assert c["last_seen"] == day(29).isoformat()


def test_event_validation(workspace) -> None:
    store = LearnerStore(workspace)
    with pytest.raises(LearnerError):
        store.record({"type": "danced", "concept": "x"}, now=T0)
    with pytest.raises(LearnerError):
        store.record({"type": "answered", "concept": "x", "quality": 7}, now=T0)
    with pytest.raises(LearnerError):
        store.record({"type": "answered", "concept": "x", "quality": "five"}, now=T0)
    with pytest.raises(LearnerError):
        store.record("answered", now=T0)  # type: ignore[arg-type]
    with pytest.raises(LearnerError):
        store.record({"type": "answered", "concept": "y" * 200, "quality": 3}, now=T0)
    assert store.validate({"type": "reviewed", "concept": "  Dot   Product ", "quality": 4}, now=T0)["concept"] == "dot product"
    assert store.ledger()["events"] == []  # validate does not write
    ev = store.record({"type": "unit_done", "detail": "  chapter   one  ", "session_id": "s9", "evidence_id": "ev_1"}, now=T0)
    assert ev == {"at": T0.isoformat(), "type": "unit_done", "concept": None, "quality": None, "points": 50, "detail": "chapter one", "session_id": "s9", "evidence_id": "ev_1"}
    assert set(EVENT_TYPES) >= {"answered", "explained", "asked", "reviewed", "unit_done", "session_done", "interview_done", "film_done", "assessment_done", "misconception", "misconception_resolved"}


def test_record_persists_both_files(workspace) -> None:
    store = LearnerStore(workspace)
    answered(store, "norms", 4, T0)
    mastery = json.loads(workspace.read_text("learner/mastery.json"))
    ledger = json.loads(workspace.read_text("learner/progress.json"))
    assert mastery["version"] == 1 and "norms" in mastery["concepts"]
    assert ledger["version"] == 1 and len(ledger["events"]) == 1 and ledger["settings"]["timezone"] == "UTC"
    assert ledger["events"][0]["points"] == 10


# ----------------------------------------------------------------------------- scheduling
def test_due_ordering_and_limit(workspace) -> None:
    store = LearnerStore(workspace)
    answered(store, "a", 5, day(0))  # next review day 1
    answered(store, "b", 5, day(0))
    answered(store, "b", 5, day(1))  # next review day 7
    answered(store, "c", 5, day(5))  # next review day 6
    assert store.due(now=day(0)) == []
    due = store.due(now=day(10))
    assert [d["concept"] for d in due] == ["a", "c", "b"]
    assert due[0]["overdue_days"] == 9.0 and due[1]["overdue_days"] == 4.0 and due[2]["overdue_days"] == 3.0
    assert due[0]["interval_days"] == 1 and due[2]["interval_days"] == 6 and due[0]["lapses"] == 0
    assert [d["concept"] for d in store.due(now=day(10), limit=2)] == ["a", "c"]
    assert [d["concept"] for d in store.due(now=day(6, hour=10))] == ["a", "c"]


def test_schedule_includes_never_seen_outline_concepts(workspace) -> None:
    outline = Outline(workspace.path("outline"))
    outline.add(Unit(id="chapter-vectors", kind="chapter", title="Vectors", meta={"introduces": ["vector", "Vector Addition", "norm"]}))
    outline.add(Unit(id="chapter-dot", kind="chapter", title="Dot", meta={"introduces": "dot product, projection"}))
    outline.add(Unit(id="chapter-plain", kind="chapter", title="Plain"))
    store = LearnerStore(workspace)
    answered(store, "norm", 5, day(0))  # known, due on day 1
    answered(store, "projection", 5, day(0))
    answered(store, "projection", 5, day(1))  # not due until day 7
    items = store.schedule(limit=10, now=day(2))
    assert [(i["concept"], i["reason"]) for i in items] == [("norm", "due"), ("vector", "new"), ("vector addition", "new"), ("dot product", "new")]
    assert items[1]["unit_id"] == "chapter-vectors" and items[3]["unit_id"] == "chapter-dot" and items[1]["level"] == 0.0
    assert [i["concept"] for i in store.schedule(limit=2, now=day(2))] == ["norm", "vector"]
    assert store.outline_concepts()[0] == {"concept": "vector", "unit_id": "chapter-vectors"}


# ----------------------------------------------------------------------------- progress
def test_progress_streak_across_days(workspace) -> None:
    store = LearnerStore(workspace)
    assert store.progress(now=T0) == {"events_total": 0, "last_active": None, "streak_days": 0, "longest_streak": 0, "points": 0, "badges": []}
    for n in (0, 1, 2):
        answered(store, "a", 5, day(n))
    answered(store, "a", 5, day(2, hour=20))  # two events on one day count once
    p = store.progress(now=day(2, hour=23))
    assert p["streak_days"] == 3 and p["longest_streak"] == 3 and p["events_total"] == 4 and p["last_active"] == day(2, hour=20).isoformat()
    assert store.progress(now=day(3))["streak_days"] == 3  # yesterday still counts
    assert store.progress(now=day(4))["streak_days"] == 0 and store.progress(now=day(4))["longest_streak"] == 3
    answered(store, "a", 5, day(6))
    p = store.progress(now=day(6))
    assert p["streak_days"] == 1 and p["longest_streak"] == 3
    clock = FakeClock(day(7).timestamp())
    assert store.progress(now=clock)["streak_days"] == 1 and store.progress(now=clock.now())["streak_days"] == 1


def test_progress_uses_the_timezone_setting(workspace) -> None:
    store = LearnerStore(workspace)
    answered(store, "a", 5, dt.datetime(2026, 11, 3, 23, 30, tzinfo=dt.UTC))
    answered(store, "a", 5, dt.datetime(2026, 11, 4, 0, 30, tzinfo=dt.UTC))
    now = dt.datetime(2026, 11, 4, 1, 0, tzinfo=dt.UTC)
    assert store.progress(now=now)["streak_days"] == 2
    assert store.update_settings(timezone="America/New_York")["timezone"] == "America/New_York"
    assert store.progress(now=now)["streak_days"] == 1  # both events fall on the same evening in New York
    assert store.settings["timezone"] == "America/New_York"


def test_points_from_the_ledger(workspace) -> None:
    store = LearnerStore(workspace)
    expected = {"answered": 10, "unit_done": 50, "session_done": 20, "interview_done": 40, "reviewed": 5, "misconception_resolved": 15, "asked": 0, "explained": 0, "film_done": 0, "assessment_done": 0, "misconception": 0}
    for etype, pts in expected.items():
        ev = store.record({"type": etype, "concept": "c", "quality": 5 if etype in ("answered", "reviewed", "explained") else None}, now=T0)
        assert ev["points"] == pts, etype
    assert store.record({"type": "answered", "concept": "c", "quality": 2}, now=T0)["points"] == 0
    assert store.progress(now=T0)["points"] == sum(expected.values())
    ledger = store.ledger()
    ledger["events"][0]["points"] = 10_000  # the stored field is not the truth; progress recomputes
    workspace.write_json("learner/progress.json", ledger)
    assert store.progress(now=T0)["points"] == sum(expected.values())


def test_badges_are_earned_once(workspace) -> None:
    store = LearnerStore(workspace)
    store.record({"type": "unit_done", "detail": "u1"}, now=day(0))
    store.record({"type": "unit_done", "detail": "u2"}, now=day(0, hour=10))
    store.record({"type": "interview_done"}, now=day(0))
    store.record({"type": "film_done"}, now=day(0))
    for i in range(11):
        answered(store, "a", 5, day(0, hour=11) + dt.timedelta(minutes=i))
    badges = store.progress(now=day(0))["badges"]
    ids = [b["id"] for b in badges]
    assert ids == ["first_unit_done", "first_interview", "first_film", "ten_correct_answers"] and len(set(ids)) == len(ids)
    assert badges[0]["earned_at"] == day(0).isoformat() and badges[0]["name"] == BADGE_NAMES["first_unit_done"]
    assert badges[3]["earned_at"] == (day(0, hour=11) + dt.timedelta(minutes=9)).isoformat()


def test_streak_comeback_and_review_badges(workspace) -> None:
    store = LearnerStore(workspace)
    for n in range(8):
        store.record({"type": "reviewed", "concept": "a", "quality": 4}, now=day(n))
    badges = {b["id"]: b["earned_at"] for b in store.progress(now=day(7))["badges"]}
    assert badges == {"seven_day_streak": day(6).isoformat()}  # the seventh consecutive day
    answered(store, "b", 1, day(8))  # a lapse on b, then three correct on b
    answered(store, "b", 4, day(8, hour=10))
    answered(store, "a", 5, day(8, hour=11))  # a different concept does not count
    answered(store, "b", 5, day(8, hour=12))
    assert "comeback" not in {b["id"] for b in store.progress(now=day(8))["badges"]}
    answered(store, "b", 5, day(8, hour=13))
    badges = {b["id"]: b["earned_at"] for b in store.progress(now=day(8))["badges"]}
    assert badges["comeback"] == day(8, hour=13).isoformat()
    for i in range(92):
        store.record({"type": "reviewed", "concept": "a", "quality": 4}, now=day(9) + dt.timedelta(minutes=i))
    ids = [b["id"] for b in store.progress(now=day(9))["badges"]]
    assert "hundred_reviews" in ids and ids.count("hundred_reviews") == 1 and "thirty_day_streak" not in ids
    for n in range(10, 40):
        store.record({"type": "asked", "concept": "a"}, now=day(n))
    assert "thirty_day_streak" in {b["id"] for b in store.progress(now=day(39))["badges"]}


def test_settings_toggles_switch_features_off(workspace) -> None:
    store = LearnerStore(workspace)
    answered(store, "a", 5, T0)
    store.record({"type": "unit_done"}, now=T0)
    full = store.progress(now=T0)
    assert {"streak_days", "longest_streak", "points", "badges"} <= set(full)
    store.update_settings(streaks=False, badges=False)
    p = store.progress(now=T0)
    assert "streak_days" not in p and "longest_streak" not in p and "badges" not in p and p["points"] == 60 and p["events_total"] == 2
    store.update_settings(points=False)
    assert "points" not in store.progress(now=T0)
    assert json.loads(workspace.read_text("learner/progress.json"))["settings"] == {"streaks": False, "badges": False, "points": False, "timezone": "UTC"}
    with pytest.raises(LearnerError):
        store.update_settings(confetti=True)
    assert "streak" not in store.summary(now=T0) and "points" not in store.summary(now=T0)


# ----------------------------------------------------------------------------- misconceptions and summary
def test_misconceptions_file_and_resolution(workspace) -> None:
    store = LearnerStore(workspace)
    store.record({"type": "misconception", "concept": "Dot Product", "detail": "thinks the dot product is a vector"}, now=day(0))
    store.record({"type": "misconception", "concept": "bayes", "detail": "confuses P(A|B) with P(B|A)"}, now=day(1))
    store.record({"type": "misconception", "detail": "no concept given"}, now=day(1))
    text = workspace.read_text("learner/misconceptions.md")
    assert text.startswith("# Misconceptions") and "- 2026-11-03 [dot product] thinks the dot product is a vector\n" in text and "- 2026-11-04 no concept given\n" in text
    assert len(store.open_misconceptions()) == 3
    store.record({"type": "misconception_resolved", "detail": "dot product is a vector"}, now=day(5))  # by text
    store.record({"type": "misconception_resolved", "concept": "bayes"}, now=day(6))  # by concept when no text
    opened = store.open_misconceptions()
    assert [m["text"] for m in opened] == ["no concept given"]
    rows = store.misconceptions()
    assert rows[0]["resolved"] == "2026-11-08" and rows[1]["resolved"] == "2026-11-09" and rows[2]["resolved"] is None
    assert "(resolved 2026-11-08)" in workspace.read_text("learner/misconceptions.md")
    store.record({"type": "misconception_resolved", "detail": "nothing matches this"}, now=day(7))
    assert len(store.open_misconceptions()) == 1  # a non-matching resolution changes nothing


def test_summary_text(workspace) -> None:
    store = LearnerStore(workspace)
    assert store.summary(now=T0) == "No learner data yet: nothing tracked, nothing due."
    for i in range(3):
        answered(store, "vectors", 5, day(i))
    answered(store, "norms", 2, day(2))
    store.record({"type": "misconception", "concept": "norms", "detail": "thinks a norm can be negative"}, now=day(2))
    text = store.summary(now=day(4))
    assert "Concepts tracked: 2." in text and text.index("Strongest: vectors") < text.index("Weakest: norms")
    assert "Due for review: 1 (norms)." in text and "streak 0 days (longest 3)" in text and "30 points" in text
    assert "Open misconceptions: 1 (thinks a norm can be negative)." in text
    short = store.summary(max_chars=60, now=day(4))
    assert len(short) <= 60 and short.endswith("…[truncated]")


def test_as_datetime_accepts_the_usual_shapes() -> None:
    assert as_datetime(T0) == T0 and as_datetime(T0.timestamp()) == T0 and as_datetime(T0.isoformat()) == T0
    assert as_datetime(FakeClock(T0.timestamp())) == T0
    assert as_datetime(dt.datetime(2026, 11, 3, 9, 0)) == T0  # naive means UTC
    assert as_datetime().tzinfo is not None
    with pytest.raises(TypeError):
        as_datetime(True)


# ----------------------------------------------------------------------------- the learner tools
@pytest.fixture
def tools(registry, ctx):
    from edgent.tools.edu.learner import build_learner_registry

    ctx.extra["clock"] = FakeClock(T0.timestamp())
    return build_learner_registry(registry)


def call_ok(tools, ctx, name: str, **args: Any) -> dict[str, Any]:
    r = tools.call(name, args, ctx)
    assert r.ok, r.error
    return r.output


def call_err(tools, ctx, name: str, **args: Any) -> Any:
    r = tools.call(name, args, ctx)
    assert not r.ok, r.output
    assert r.error.remediation
    return r.error


def test_learner_tool_contracts(tools) -> None:
    from edgent.tools.edu.learner import LEARNER_TOOL_NAMES, build_learner_registry

    assert tools.names() == sorted(LEARNER_TOOL_NAMES)
    expected = {"learner_update": "write", "learner_schedule": "read", "learner_summary": "read", "learner_progress": "read", "learner_misconception": "write"}
    for name, permission in expected.items():
        spec = tools.get(name)
        assert spec is not None and spec.permission == permission and spec.group == "learner", name
        assert spec.parameters["additionalProperties"] is False and spec.when_not_to_use and len(spec.description) > 60
        for prop, schema in spec.parameters["properties"].items():
            assert schema.get("description"), f"{name}.{prop}"
    assert tools.get("learner_update").side_effects and not tools.get("learner_schedule").side_effects
    assert tools.get("learner_update").parameters["properties"]["events"]["items"]["properties"]["type"]["enum"] == list(EVENT_TYPES)
    assert tools.schema_bytes(compact=True) < 5000
    assert build_learner_registry(tools) is tools and len(tools.names()) == 5


def test_learner_update_is_all_or_nothing_and_writes_only_under_learner(tools, ctx, workspace) -> None:
    before = {p for p in workspace.glob("**/*") if not p.startswith("learner/")}
    e = call_err(tools, ctx, "learner_update", events=[{"type": "answered", "concept": "a", "quality": 4}, {"type": "answered", "concept": "b", "quality": 9}])
    assert e.category.value == "invalid_input" and "events[1]" in e.message and LearnerStore(workspace).ledger()["events"] == []
    e = call_err(tools, ctx, "learner_update", events=[{"type": "danced"}])
    assert e.category.value == "invalid_input" and "events[0].type" in e.message and "must be one of" in e.message  # the schema enum catches it first
    assert call_err(tools, ctx, "learner_update", events=[]).category.value == "invalid_input"
    out = call_ok(tools, ctx, "learner_update", events=[
        {"type": "answered", "concept": "Dot Product", "quality": 4, "detail": "computed (1,2)·(3,4)=11"},
        {"type": "unit_done", "detail": "chapter-vectors"},
        {"type": "misconception", "concept": "norms", "detail": "thinks a norm can be negative"},
    ])
    assert out["recorded"] == 3 and out["concepts"] == ["dot product", "norms"] and out["points_earned"] == 60
    assert out["progress"]["points"] == 60 and [b["id"] for b in out["progress"]["badges"]] == ["first_unit_done"] and out["progress"]["streak_days"] == 1
    assert out["paths"] == ["learner/mastery.json", "learner/progress.json", "learner/misconceptions.md"]
    events = LearnerStore(workspace).ledger()["events"]
    assert [e["session_id"] for e in events] == ["s1", "s1", "s1"] and events[0]["at"] == T0.isoformat()
    assert "thinks a norm can be negative" in workspace.read_text("learner/misconceptions.md")
    after = {p for p in workspace.glob("**/*") if not p.startswith("learner/")}
    assert after == before
    e = call_err(tools, ctx, "learner_update", events=[{"type": "asked"}] * 201)
    assert e.code == "limit:learner_update events per call" and "batches" in e.remediation


def test_learner_schedule_summary_progress_tools(tools, ctx, workspace) -> None:
    outline = Outline(workspace.path("outline"))
    outline.add(Unit(id="chapter-vectors", kind="chapter", title="Vectors", meta={"introduces": ["vector", "norm"]}))
    empty = call_ok(tools, ctx, "learner_schedule", limit=1)
    assert empty["count"] == 1 and empty["new"] == 1 and empty["items"][0] == {"concept": "vector", "level": 0.0, "reason": "new", "unit_id": "chapter-vectors"}
    call_ok(tools, ctx, "learner_update", events=[{"type": "answered", "concept": "norm", "quality": 5}])
    ctx.extra["clock"].advance(3 * 86400)
    out = call_ok(tools, ctx, "learner_schedule")
    assert [(i["concept"], i["reason"]) for i in out["items"]] == [("norm", "due"), ("vector", "new")] and out["due"] == 1 and out["new"] == 1 and "overdue" in out["hint"]
    s = call_ok(tools, ctx, "learner_summary", max_chars=200)
    summary = s["summary"]
    assert s["untrusted"] is True  # it quotes misconception text from the learner's files
    assert summary.startswith("Concepts tracked: 1.") and "Due for review: 1 (norm)." in summary and len(summary) <= 200
    p = call_ok(tools, ctx, "learner_progress")
    assert p["points"] == 10 and p["streak_days"] == 0 and p["due"] == 1 and p["open_misconceptions"] == 0 and p["settings"]["timezone"] == "UTC"
    LearnerStore(workspace).update_settings(badges=False)
    assert "badges" not in call_ok(tools, ctx, "learner_progress") and call_ok(tools, ctx, "learner_progress")["settings"]["badges"] is False


def test_learner_misconception_tool(tools, ctx, workspace) -> None:
    out = call_ok(tools, ctx, "learner_misconception", text="  thinks the dot product   is a vector ", concept="Dot Product")
    assert out["event"]["type"] == "misconception" and out["event"]["concept"] == "dot product" and out["open_misconceptions"] == 1 and out["path"] == "learner/misconceptions.md"
    assert "- 2026-11-03 [dot product] thinks the dot product is a vector" in workspace.read_text("learner/misconceptions.md")
    out = call_ok(tools, ctx, "learner_misconception", text="something unrelated", resolved=True)
    assert out["resolved_count"] == 0 and "no open bullet matched" in out["note"] and out["open_misconceptions"] == 1
    out = call_ok(tools, ctx, "learner_misconception", text="dot product is a vector", resolved=True)
    assert out["resolved_count"] == 1 and out["open_misconceptions"] == 0 and out["event"]["points"] == 15
    assert call_err(tools, ctx, "learner_misconception", text="   ").category.value == "invalid_input"
    assert call_err(tools, ctx, "learner_misconception", text="x" * 600).category.value == "invalid_input"


# ----------------------------------------------------------------------------- review additions
def test_record_stores_utc_and_progress_survives_a_hand_edited_ledger(workspace) -> None:
    store = LearnerStore(workspace)
    plus5 = dt.timezone(dt.timedelta(hours=5))
    local = dt.datetime(2026, 11, 3, 14, 0, tzinfo=plus5)  # 09:00 UTC
    assert answered(store, "a", 5, local)["at"] == T0.isoformat() and store.validate({"type": "asked"}, now=local)["at"] == T0.isoformat()
    answered(store, "a", 5, day(1))
    ledger = store.ledger()
    ledger["events"].append({"at": "not a time", "type": "unit_done", "points": 50})
    ledger["events"].insert(0, {"at": day(2).isoformat(), "type": "unit_done"})  # out of order on disk
    workspace.write_json("learner/progress.json", ledger)
    p = store.progress(now=day(2))
    assert p["events_total"] == 3 and p["last_active"] == day(2).isoformat() and p["streak_days"] == 3 and p["points"] == 70
    assert [b["id"] for b in p["badges"]] == ["first_unit_done"]


def test_update_settings_rejects_an_unknown_timezone(workspace) -> None:
    store = LearnerStore(workspace)
    with pytest.raises(LearnerError):
        store.update_settings(timezone="Mars/Olympus")
    assert store.settings["timezone"] == "UTC" and store.update_settings(timezone="utc")["timezone"] == "utc"
    ledger = store.ledger()
    ledger["settings"]["timezone"] = "Nowhere/Land"  # written by hand: read back as UTC rather than breaking progress
    workspace.write_json("learner/progress.json", ledger)
    answered(store, "a", 5, T0)
    assert store.progress(now=T0)["streak_days"] == 1


def test_resolution_matches_text_first_then_falls_back_to_the_concept(workspace) -> None:
    store = LearnerStore(workspace)
    store.record({"type": "misconception", "concept": "norms", "detail": "thinks a norm can be negative"}, now=day(0))
    store.record({"type": "misconception", "concept": "norms", "detail": "confuses the 1-norm with the 2-norm"}, now=day(0))
    store.record({"type": "misconception_resolved", "concept": "norms", "detail": "norm can be negative"}, now=day(1))
    assert [m["text"] for m in store.open_misconceptions()] == ["confuses the 1-norm with the 2-norm"]  # a text match is not widened by the concept
    store.record({"type": "misconception_resolved", "concept": "norms", "detail": "nothing on file says this"}, now=day(2))
    assert store.open_misconceptions() == []  # no text match: every open bullet on the concept
