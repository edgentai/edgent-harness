"""check_readability: is the prose pitched at the persona.

Code, math, front matter, citations and markup are stripped; the rest is
measured (Flesch-Kincaid grade, average sentence length, long-word ratio) and
compared with the band for the level named by the argument, the front
matter, the outline unit or the persona keywords. A sentence longer than the
level's ceiling is a defect on its own, because that is what a reader at that
level trips over first.
"""
from __future__ import annotations

import re
from typing import Any

from ...domain.errors import ErrorCategory, ToolFailure
from ...domain.obligations import CITATION_RE, PLACEHOLDER_RE, strip_code
from ...domain.outline import Outline
from ...domain.utils import parse_front_matter
from ..registry import ToolContext
from . import check_result, defect, record_evidence, resolve_text_file, trace_check, verdict

LEVELS: tuple[str, ...] = ("kg", "primary", "middle", "secondary", "undergraduate", "graduate", "professional")
GRADE_BANDS: dict[str, tuple[float, float]] = {
    "kg": (1, 2), "primary": (2, 5), "middle": (5, 8), "secondary": (8, 11), "undergraduate": (10, 15), "graduate": (12, 18), "professional": (10, 16),
}
SENTENCE_WORDS_CEILING: dict[str, int] = {"kg": 10, "primary": 15, "middle": 20, "secondary": 25, "undergraduate": 35, "graduate": 40, "professional": 35}
SENTENCE_WORDS_HARD_CEILING = 40  # applies when no level is known
MIN_WORDS_FOR_GRADE = 50  # below this the formula is noise and only sentence length is judged
LONG_WORD_SYLLABLES = 3
LONG_SENTENCES_SHOWN = 5
_LEVEL_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("kindergarten", "pre-school", "preschool", "kg"), "kg"),
    (("primary", "elementary", "grade school", "grades 1", "grade 1", "grade 2", "grade 3", "grade 4", "grade 5"), "primary"),
    (("middle school", "junior high", "middle", "grade 6", "grade 7", "grade 8"), "middle"),
    (("secondary", "high school", "high-school", "sixth form", "a-level", "grade 9", "grade 10", "grade 11", "grade 12", "school"), "secondary"),
    (("undergraduate", "undergrad", "bachelor", "college", "university", "first-year", "second-year", "sophomore", "freshman"), "undergraduate"),
    (("graduate", "postgraduate", "phd", "doctoral", "master", "msc", "researcher"), "graduate"),
    (("professional", "practitioner", "engineer", "industry", "developer", "analyst"), "professional"),
)
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n{2,}")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+(?:\.\d+)?")


def level_from_text(text: str | None) -> str | None:
    if not text:
        return None
    low = text.lower()
    if low.strip() in LEVELS:
        return low.strip()
    for keywords, level in _LEVEL_KEYWORDS:
        if any(re.search(r"(?<![a-z])" + re.escape(k) + r"s?(?![a-z])", low) for k in keywords):
            return level
    return None


def strip_markup(body: str) -> str:
    """Prose only: no code, math, citations, placeholders, links' targets, headings marks, tables or emphasis."""
    text = strip_code(body)
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\\)\$[^$\n]+?\$", " ", text)
    text = re.sub(r"`[^`\n]*`", " ", text)
    text = CITATION_RE.sub("", text)
    text = PLACEHOLDER_RE.sub("", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>\n]+>", " ", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s+.*$", "", text, flags=re.MULTILINE)  # headings are not sentences
    text = re.sub(r"^\s*([-*+]|\d+[.)])\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\|?\s*:?-{2,}.*$", "", text, flags=re.MULTILINE)
    text = text.replace("|", " ")
    text = re.sub(r"[*_]{1,3}", "", text)
    return text


def syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 1
    groups = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and groups > 1:
        groups -= 1
    return max(1, groups)


def text_stats(prose: str) -> dict[str, Any]:
    sentences = [s.strip() for s in _SENTENCE_END_RE.split(prose) if s and _WORD_RE.search(s)]
    words_per_sentence: list[tuple[int, str]] = []
    total_words = total_syllables = long_words = 0
    for s in sentences:
        words = _WORD_RE.findall(s)
        if not words:
            continue
        words_per_sentence.append((len(words), s))
        total_words += len(words)
        for w in words:
            syl = syllables(w)
            total_syllables += syl
            if syl >= LONG_WORD_SYLLABLES:
                long_words += 1
    n_sent = len(words_per_sentence)
    avg_len = total_words / n_sent if n_sent else 0.0
    asw = total_syllables / total_words if total_words else 0.0
    grade = max(0.0, 0.39 * avg_len + 11.8 * asw - 15.59) if total_words else 0.0  # the formula goes negative on very simple text
    longest = max(words_per_sentence, default=(0, ""))
    return {
        "words": total_words,
        "sentences": n_sent,
        "avg_sentence_words": round(avg_len, 1),
        "syllables_per_word": round(asw, 2),
        "long_word_ratio": round(long_words / total_words, 3) if total_words else 0.0,
        "fk_grade": round(grade, 1),
        "longest_sentence_words": longest[0],
        "long_sentences": [(n, s) for n, s in words_per_sentence],
    }


def resolve_level(ctx: ToolContext, level_arg: Any, front: dict[str, Any]) -> tuple[str | None, str]:
    """(level, where it came from): argument, front matter, the outline unit's meta, then the persona."""
    if isinstance(level_arg, str) and level_arg.strip():
        lvl = level_from_text(level_arg)
        if lvl is None:
            raise ToolFailure("invalid_input", f"unknown level {level_arg!r}", ErrorCategory.invalid_input, remediation=f"Use one of {', '.join(LEVELS)} or a persona phrase such as 'high school'.")
        return lvl, "argument"
    for key in ("level", "persona"):
        lvl = level_from_text(str(front.get(key))) if front.get(key) else None
        if lvl:
            return lvl, f"front matter {key}"
    try:
        outline = Outline(ctx.workspace.path("outline"))
    except Exception:  # noqa: BLE001 - an unreadable outline only removes one source of the level
        return None, "none"
    uid = front.get("unit_id")
    if isinstance(uid, str) and outline.has(uid):
        meta = outline.get(uid).meta
        for key in ("level", "persona"):
            lvl = level_from_text(str(meta.get(key))) if meta.get(key) else None
            if lvl:
                return lvl, f"outline unit {key}"
    lvl = level_from_text(outline.persona)
    return (lvl, "outline persona") if lvl else (None, "none")


def check_readability(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    _abs, rel, text = resolve_text_file(ctx, args.get("path"))
    front, body = parse_front_matter(text)
    level, level_source = resolve_level(ctx, args.get("level"), front)
    stats = text_stats(strip_markup(body))
    long_sentences = stats.pop("long_sentences")
    defects: list[dict[str, Any]] = []
    limitations: list[str] = []
    band = GRADE_BANDS.get(level or "")
    if stats["words"] < MIN_WORDS_FOR_GRADE:
        limitations.append(f"too short to grade ({stats['words']} words < {MIN_WORDS_FOR_GRADE}); only sentence length judged")
    elif band is None:
        limitations.append("no level known (argument, front matter, outline unit or persona); grade band not checked")
    elif stats["fk_grade"] > band[1] or (stats["fk_grade"] < band[0] and level != LEVELS[0]):  # nothing is too simple for kg
        direction = "above" if stats["fk_grade"] > band[1] else "below"
        defects.append(defect(rel, "medium", f"Flesch-Kincaid grade {stats['fk_grade']} is {direction} the {level} band {band[0]}-{band[1]} (average sentence {stats['avg_sentence_words']} words, long-word ratio {stats['long_word_ratio']})"))
    ceiling = SENTENCE_WORDS_CEILING.get(level or "", SENTENCE_WORDS_HARD_CEILING)
    too_long = [(n, s) for n, s in long_sentences if n > ceiling]
    for n, s in too_long[:LONG_SENTENCES_SHOWN]:
        line = _line_of(text, s)
        defects.append(defect(f"{rel}:L{line}" if line else rel, "medium", f"sentence of {n} words exceeds the {level or 'default'} ceiling of {ceiling}: {s[:80]!r}…"))
    if len(too_long) > LONG_SENTENCES_SHOWN:
        defects.append(defect(rel, "low", f"{len(too_long) - LONG_SENTENCES_SHOWN} more sentences exceed {ceiling} words"))
    passed = verdict(defects)
    detail = {**stats, "level": level, "level_source": level_source, "band": list(band) if band else None, "sentence_words_ceiling": ceiling, "sentences_over_ceiling": len(too_long), "untrusted": True}  # defects quote the chapter
    summary = {"passed": passed, "fk_grade": stats["fk_grade"], "level": level, "words": stats["words"], "sentences_over_ceiling": len(too_long)}
    eid = record_evidence(ctx, "check_readability", rel, text, summary, confidence="derived", authority="controlled_replay", limitations=limitations, parameters={"level": level})
    result = check_result(passed, "controlled_replay", "derived", defects, eid, detail, limitations)
    trace_check(ctx, "readability", rel, result)
    return result


def _line_of(body: str, sentence: str) -> int | None:
    head = " ".join(sentence.split()[:5])
    if not head:
        return None
    for n, line in enumerate(body.splitlines(), 1):
        if head in " ".join(line.split()):
            return n
    return None
