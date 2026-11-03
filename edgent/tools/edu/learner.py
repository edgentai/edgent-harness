"""Learner tools: the observer's write path and the reads a tutor needs.

``learner_update`` is the tool through which evidence from a session (answers, teach-backs,
reviews, finished units) reaches the learner model; it writes nothing outside ``learner/``.
Streaks, points and badges are computed from the ledger by ``LearnerStore`` on every read,
so no tool here awards anything on its own.
"""
from __future__ import annotations

from typing import Any

from ...domain.errors import ErrorCategory, Limit, ToolFailure
from ...domain.learner import EVENT_TYPES, LearnerError, LearnerStore
from ...domain.workspace import FILES
from ..registry import B, I, N, S, ToolContext, ToolRegistry, ToolSpec, arr, obj, opt

LEARNER_TOOL_NAMES: tuple[str, ...] = ("learner_update", "learner_schedule", "learner_summary", "learner_progress", "learner_misconception")
UPDATE_EVENTS_LIMIT = Limit("learner_update events per call", 200, "events")
SCHEDULE_ITEMS_LIMIT = Limit("learner_schedule items", 50, "concepts")
SUMMARY_CHARS_LIMIT = Limit("learner_summary length", 2000, "chars")
MISCONCEPTION_TEXT_MAX_CHARS = 500


def _store(ctx: ToolContext) -> LearnerStore:
    return LearnerStore(ctx.workspace)


def _now(ctx: ToolContext) -> Any:
    return ctx.extra.get("clock")  # tests inject a clock; None means the wall clock


def learner_update(ctx: ToolContext, a: dict[str, Any]) -> dict[str, Any]:
    events = list(a["events"])
    if not events:
        raise ToolFailure("invalid_input", "events is empty.", ErrorCategory.invalid_input, remediation="Give at least one event with a type, and a concept and quality where they apply.")
    if len(events) > UPDATE_EVENTS_LIMIT.value:
        err = UPDATE_EVENTS_LIMIT.exceeded(len(events), "Send the events in batches.")
        raise ToolFailure(err.code, err.message, err.category, remediation=err.remediation, details=err.details)
    store = _store(ctx)
    now = _now(ctx)
    for i, ev in enumerate(events):  # validate everything before writing anything
        try:
            store.validate({**ev, "session_id": ev.get("session_id") or ctx.session_id}, now)
        except LearnerError as e:
            raise ToolFailure("invalid_input", f"events[{i}]: {e}", ErrorCategory.invalid_input, remediation=f"Fix that event. Types: {', '.join(EVENT_TYPES)}; quality is a number 0-5; concept is a short phrase.", index=i) from e
    recorded = [store.record({**ev, "session_id": ev.get("session_id") or ctx.session_id}, now) for ev in events]
    concepts = sorted({r["concept"] for r in recorded if r["concept"]})
    return {
        "recorded": len(recorded),
        "concepts": concepts,
        "points_earned": sum(r["points"] for r in recorded),
        "progress": store.progress(now),
        "paths": [FILES["mastery"], FILES["learner_progress"], FILES["misconceptions"]],
    }


def learner_schedule(ctx: ToolContext, a: dict[str, Any]) -> dict[str, Any]:
    limit = max(1, min(int(a.get("limit", 10)), int(SCHEDULE_ITEMS_LIMIT.value)))
    items = _store(ctx).schedule(limit, _now(ctx))
    due = [i for i in items if i["reason"] == "due"]
    return {
        "items": items,
        "count": len(items),
        "due": len(due),
        "new": len(items) - len(due),
        "hint": "review due concepts first (most overdue at the top); 'new' concepts come from the outline and have never been studied." if items else "nothing is due and the outline introduces no unseen concepts.",
    }


def learner_summary(ctx: ToolContext, a: dict[str, Any]) -> dict[str, Any]:
    max_chars = max(80, min(int(a.get("max_chars", 600)), int(SUMMARY_CHARS_LIMIT.value)))
    return {"summary": _store(ctx).summary(max_chars, _now(ctx)), "untrusted": True}  # quotes misconception text from the learner's files


def learner_progress(ctx: ToolContext, a: dict[str, Any]) -> dict[str, Any]:
    store = _store(ctx)
    out = store.progress(_now(ctx))
    out["settings"] = store.settings
    out["open_misconceptions"] = len(store.open_misconceptions())
    out["due"] = len(store.due(_now(ctx)))
    return out


def learner_misconception(ctx: ToolContext, a: dict[str, Any]) -> dict[str, Any]:
    text = " ".join(str(a["text"]).split())
    if not text:
        raise ToolFailure("invalid_input", "text is empty.", ErrorCategory.invalid_input, remediation="State the misconception in the learner's terms, for example 'thinks the dot product is a vector'.")
    if len(text) > MISCONCEPTION_TEXT_MAX_CHARS:
        raise ToolFailure("invalid_input", f"text is longer than {MISCONCEPTION_TEXT_MAX_CHARS} chars.", ErrorCategory.invalid_input, remediation="Shorten it to one sentence; details belong in the session notes.")
    resolved = bool(a.get("resolved", False))
    store = _store(ctx)
    now = _now(ctx)
    before = len(store.open_misconceptions())
    try:
        event = store.record({"type": "misconception_resolved" if resolved else "misconception", "concept": a.get("concept"), "detail": text, "session_id": ctx.session_id}, now)
    except LearnerError as e:
        raise ToolFailure("invalid_input", str(e), ErrorCategory.invalid_input, remediation="Give a short concept name and a one-sentence text.") from e
    after = len(store.open_misconceptions())
    out: dict[str, Any] = {"event": event, "open_misconceptions": after, "path": FILES["misconceptions"]}
    if resolved:
        out["resolved_count"] = before - after
        if before == after:
            out["note"] = "no open bullet matched the text (substring) or the concept; the event was recorded anyway."
    return out


# ----------------------------------------------------------------------------- registry
def _spec(name: str, fn: Any, description: str, parameters: dict[str, Any], *, permission: str, when_not_to_use: str, side_effects: bool = False, idempotent: bool = True, examples: list[dict[str, Any]] | None = None) -> ToolSpec:
    return ToolSpec(name=name, description=description, parameters=parameters, fn=fn, permission=permission, side_effects=side_effects, idempotent=idempotent, when_not_to_use=when_not_to_use, examples=list(examples or []), group="learner")


_EVENT = obj({
    "type": opt(S, "What happened: " + ", ".join(EVENT_TYPES) + ".", enum=list(EVENT_TYPES)),
    "concept": opt(S, "The concept involved, a short lower-case phrase (required for mastery to move)."),
    "quality": opt(N, "How well the learner recalled or explained it, 0 (blank) to 5 (perfect); below 3 counts as a lapse.", minimum=0, maximum=5),
    "detail": opt(S, "One line of evidence: what was asked and answered, or what was finished."),
    "evidence_id": opt(S, "An Evidence record id backing the event, when one exists."),
}, ["type"])


def learner_specs() -> list[ToolSpec]:
    return [
        _spec(
            "learner_update", learner_update,
            "Record evidence about the learner: answered (with quality 0-5), explained (a teach-back), asked, "
            "reviewed, unit_done, session_done, interview_done, film_done, assessment_done, misconception and "
            "misconception_resolved events. Mastery moves by an SM-2 rule from the quality; streaks, points and "
            "badges follow from the ledger. Writes only under learner/. Use it after an exchange, as the observer, "
            "with what the transcript shows, not with what you hope. Give each event a distinct detail: a call "
            "identical to an earlier one in this session is treated as a retry and replayed, not recorded again.",
            obj({"events": opt(arr(_EVENT, minItems=1), f"The events to record (at most {UPDATE_EVENTS_LIMIT.value:g} per call).")}, ["events"]),
            permission="write",
            side_effects=True,
            idempotent=False,
            when_not_to_use="you are grading a set (assess_grade records the answers itself) or noting a preference rather than evidence of learning (use remember).",
            examples=[{"events": [{"type": "answered", "concept": "dot product", "quality": 4, "detail": "computed (1,2)·(3,4)=11 with one slip"}]}],
        ),
        _spec(
            "learner_schedule", learner_schedule,
            "The spaced-repetition queue for a learning session: concepts due for review, most overdue first, "
            "then concepts the outline introduces that were never studied. Start a session with it.",
            obj({"limit": opt(I, f"Maximum concepts to return (default 10, capped at {SCHEDULE_ITEMS_LIMIT.value:g}).", minimum=1)}),
            permission="read",
            when_not_to_use="you need the whole mastery table (read learner/mastery.json) or the motivation stats (learner_progress).",
            examples=[{"limit": 5}],
        ),
        _spec(
            "learner_summary", learner_summary,
            "A short text block about the learner for your context: strongest and weakest concepts, how many "
            "reviews are due, the streak, open misconceptions. Use it to personalise explanations and items.",
            obj({"max_chars": opt(I, f"Length cap in characters (default 600, maximum {SUMMARY_CHARS_LIMIT.value:g}).", minimum=80)}),
            permission="read",
            when_not_to_use="you need the queue itself (learner_schedule) or exact numbers (learner_progress).",
        ),
        _spec(
            "learner_progress", learner_progress,
            "The motivation snapshot computed from the ledger: streak, longest streak, points, badges, totals, the "
            "number of due reviews and open misconceptions, plus the settings that switch features off.",
            obj({}),
            permission="read",
            when_not_to_use="you want the study queue (learner_schedule) or a prose summary (learner_summary).",
        ),
        _spec(
            "learner_misconception", learner_misconception,
            "Append a dated misconception bullet to learner/misconceptions.md (or mark one resolved when "
            "resolved is true: bullets whose text matches, else every open bullet on the concept) and record "
            "the event. Use it when the learner's words reveal a wrong model, and again when they correct it.",
            obj({
                "text": opt(S, "The misconception in one sentence, in the learner's terms.", maxLength=MISCONCEPTION_TEXT_MAX_CHARS),
                "concept": opt(S, "The concept it concerns, a short lower-case phrase."),
                "resolved": opt(B, "True to mark an existing bullet resolved instead of adding one (default false)."),
            }, ["text"]),
            permission="write",
            side_effects=True,
            idempotent=False,
            when_not_to_use="the learner simply answered wrongly without a systematic error (record an answered event with a low quality instead).",
            examples=[{"text": "thinks the dot product is a vector", "concept": "dot product"}],
        ),
    ]


def build_learner_registry(registry: ToolRegistry | None = None) -> ToolRegistry:
    """Register the learner tools (group 'learner') on ``registry`` (a new one when None); already-present names are kept."""
    reg = registry if registry is not None else ToolRegistry()
    for spec in learner_specs():
        if reg.get(spec.name) is None:
            reg.register(spec)
    return reg


__all__ = ["LEARNER_TOOL_NAMES", "build_learner_registry", "learner_specs"]
