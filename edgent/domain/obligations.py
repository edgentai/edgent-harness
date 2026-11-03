"""Obligations: what a unit owes before it may be called done, and who may close each one.

A unit's obligations are derived mechanically from its kind and its content:
every fenced Python block owes an execution, every citation owes a resolution,
a chapter owes a render, a readability measurement and a prerequisite check,
a film owes a render, measurements and a look. Each obligation names the
closer (the tool that can close it) and the authority a verifier must have;
``close_obligation`` refuses a weaker verifier, so a model's opinion never
closes "the code runs". Advisory obligations (the judge) are recorded but do
not hold a unit at ``partial``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from .evidence import AUTHORITY_RANK, authority_at_least
from .outline import Outline, Unit
from .utils import parse_front_matter

ObligationKind = Literal["render", "execute", "cite", "readability", "prereqs", "measure", "look", "judge", "keys", "structure"]
ObligationStatus = Literal["open", "closed", "blocked"]
RequiredAuthority = Literal["source", "controlled_replay", "external_service", "model_inference"]

KINDS: tuple[str, ...] = ("render", "execute", "cite", "readability", "prereqs", "measure", "look", "judge", "keys", "structure")
TEXT_UNIT_KINDS: frozenset[str] = frozenset({"chapter", "section", "lesson"})
CONTENT_UNIT_KINDS: frozenset[str] = TEXT_UNIT_KINDS | {"research"}  # kinds whose target file is read to derive obligations
EXECUTABLE_LANGUAGES: frozenset[str] = frozenset({"python", "python3", "py"})
OUTPUT_LANGUAGE = "output"
CONTINUE_MARKER_RE = re.compile(r"^\s*#\s*edgent:\s*continue\s*$", re.MULTILINE)

REQUIRED_AUTHORITY: dict[str, str] = {
    "render": "controlled_replay",
    "execute": "controlled_replay",
    "cite": "source",
    "readability": "controlled_replay",
    "prereqs": "controlled_replay",
    "measure": "controlled_replay",
    "look": "controlled_replay",
    "judge": "model_inference",
    "keys": "controlled_replay",
    "structure": "controlled_replay",
}

# Closer per (unit kind, obligation kind). Chapter-like kinds share the check_* tools; media and
# assessment kinds name the tools of their modules; integration closes through the book build. The media
# tools measure as part of rendering (their ``ok`` is "no error-level measurement defect"), so one call
# closes both the render and the measure obligation; there is no separate measuring tool.
CLOSERS: dict[str, dict[str, str]] = {
    "chapter": {
        "structure": "check_render", "render": "check_render", "execute": "check_execute", "cite": "check_cite",
        "readability": "check_readability", "prereqs": "check_prereqs", "judge": "check_judge",
    },
    "film": {"render": "film_render", "measure": "film_render", "look": "film_look", "judge": "check_judge"},
    "widget": {"render": "widget_run", "measure": "widget_run", "judge": "check_judge"},
    "figure": {"render": "diagram_render", "measure": "diagram_render"},
    "exercise_set": {"keys": "assess_items_check", "judge": "check_judge"},
    "assessment": {"keys": "assess_items_check", "judge": "check_judge"},
    "research": {"structure": "check_cite", "cite": "check_cite", "judge": "check_judge"},
    "integration": {"structure": "book_build", "render": "book_build"},
}
CLOSERS["section"] = CLOSERS["chapter"]
CLOSERS["lesson"] = CLOSERS["chapter"]

# Obligation kind implied by a closer named in ``Unit.checks`` beyond the kind's defaults.
KIND_BY_CLOSER: dict[str, str] = {
    "check_render": "render", "check_execute": "execute", "check_cite": "cite", "check_readability": "readability",
    "check_prereqs": "prereqs", "check_judge": "judge", "film_look": "look",
    "assess_items_check": "keys", "widget_run": "render", "diagram_render": "render", "film_render": "render",
    "book_build": "render",
}

DEFAULT_TARGETS: dict[str, str] = {
    "film": "media/films/{id}",
    "widget": "media/widgets/{id}.html",
    "figure": "media/diagrams/{id}.svg",
    "exercise_set": "assessments/{id}/items.json",
    "assessment": "assessments/{id}/items.json",
    "integration": "book/build",
}

# Source ids are lowercase slugs (the ingest adapter lowercases them); a capitalised id in a chapter is the same source.
CITATION_RE = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9\-_]*):(p\.\d+|t=\d{1,2}:\d{2}(?::\d{2})?|L\d+)\]")
PLACEHOLDER_RE = re.compile(r"\{\{\s*(film|widget|exercises)\s*:\s*([a-zA-Z0-9\-_]+)\s*\}\}")
_FENCE_OPEN_RE = re.compile(r"^(\s*)(`{3,}|~{3,})\s*(.*?)\s*$")
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")


# ----------------------------------------------------------------------------- content parsing
@dataclass
class CodeBlock:
    """One fenced block: ``index`` counts every fence in the file, ``line`` is the opening fence (1-based)."""

    index: int
    language: str
    code: str
    line: int
    end_line: int
    closed: bool = True
    expected_output: str | None = None
    continues_previous: bool = False

    @property
    def executable(self) -> bool:
        return self.language in EXECUTABLE_LANGUAGES

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "language": self.language, "line": self.line, "end_line": self.end_line,
            "closed": self.closed, "chars": len(self.code), "has_expected_output": self.expected_output is not None,
            "continues_previous": self.continues_previous,
        }


def _language(info: str) -> str:
    info = info.strip()
    if info.startswith("{"):  # pandoc attribute form: {.python #id}
        info = info.strip("{}").strip()
        info = info[1:] if info.startswith(".") else info
    return info.split()[0].lower() if info else ""


def _closes(line: str, marker: str) -> bool:
    s = line.strip()
    return len(s) >= len(marker) and s == marker[0] * len(s)


def _dedent(line: str, indent: int) -> str:
    """Strip up to ``indent`` leading spaces: a fence inside a list item is indented, its code is not (CommonMark)."""
    k = 0
    while k < indent and k < len(line) and line[k] == " ":
        k += 1
    return line[k:]


def code_blocks(content: str) -> list[CodeBlock]:
    """Every fenced block with its language, expected ``output`` block and the continue marker."""
    lines = content.splitlines()
    blocks: list[CodeBlock] = []
    i = 0
    while i < len(lines):
        m = _FENCE_OPEN_RE.match(lines[i])
        if not m or not m.group(2):
            i += 1
            continue
        marker = m.group(2)
        indent = len(m.group(1).expandtabs(4))
        start = i
        i += 1
        body: list[str] = []
        while i < len(lines) and not _closes(lines[i], marker):
            body.append(_dedent(lines[i], indent))
            i += 1
        closed = i < len(lines)
        end = i if closed else len(lines) - 1
        i += 1
        code = "\n".join(body) + ("\n" if body else "")
        blocks.append(CodeBlock(len(blocks) + 1, _language(m.group(3)), code, start + 1, end + 1, closed))
    for prev, nxt in zip(blocks, blocks[1:], strict=False):
        between = lines[prev.end_line : nxt.line - 1]
        if nxt.language == OUTPUT_LANGUAGE and prev.language != OUTPUT_LANGUAGE and all(not s.strip() for s in between):
            prev.expected_output = nxt.code
    for b in blocks:
        b.continues_previous = bool(CONTINUE_MARKER_RE.search(b.code))
    return blocks


def strip_code(content: str) -> str:
    """The content with fenced blocks replaced by blank lines, so line numbers are preserved."""
    lines = content.splitlines()
    for b in code_blocks(content):
        for n in range(b.line - 1, min(b.end_line, len(lines))):
            lines[n] = ""
    return "\n".join(lines)


def citations(content: str) -> list[dict[str, Any]]:
    """Distinct citations outside code (fenced or inline), each with its first line: ``{"locator", "source_id", "ref", "line"}``."""
    seen: dict[str, dict[str, Any]] = {}
    for n, line in enumerate(strip_code(content).splitlines(), 1):
        for m in CITATION_RE.finditer(_INLINE_CODE_RE.sub("", line)):
            source_id = m.group(1).lower()
            loc = f"{source_id}:{m.group(2)}"
            if loc not in seen:
                seen[loc] = {"locator": loc, "source_id": source_id, "ref": m.group(2), "line": n, "count": 0}
            seen[loc]["count"] += 1
    return list(seen.values())


def placeholders(content: str) -> list[dict[str, Any]]:
    """``{{film: id}}``, ``{{widget: id}}`` and ``{{exercises: id}}`` lines outside code, with line numbers."""
    out: list[dict[str, Any]] = []
    for n, line in enumerate(strip_code(content).splitlines(), 1):
        for m in PLACEHOLDER_RE.finditer(_INLINE_CODE_RE.sub("", line)):
            out.append({"kind": m.group(1), "id": m.group(2), "line": n})
    return out


def normalise_concept(name: Any) -> str:
    return " ".join(str(name).lower().replace("_", " ").split())


def concept_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [v for v in re.split(r"[;,]", value) if v.strip()]
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for v in value:
        c = normalise_concept(v)
        if c and c not in out:
            out.append(c)
    return out


# ----------------------------------------------------------------------------- the obligation
@dataclass
class Obligation:
    id: str
    kind: str
    description: str
    target: str
    required_authority: str
    closer: str
    status: str = "open"
    reason: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    advisory: bool = False
    args: dict[str, Any] = field(default_factory=dict)  # what the closer needs: path, block index, locator, level

    @property
    def blocking(self) -> bool:
        return self.status == "open" and not self.advisory

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "description": self.description,
            "target": self.target,
            "required_authority": self.required_authority,
            "closer": self.closer,
            "status": self.status,
            "reason": self.reason,
            "evidence_ids": list(self.evidence_ids),
            "advisory": self.advisory,
            "args": dict(self.args),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Obligation:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _closer(unit_kind: str, kind: str) -> str:
    return CLOSERS.get(unit_kind, CLOSERS["chapter"]).get(kind, f"check_{kind}")


def _target(unit: Unit) -> str:
    if unit.path:
        return unit.path
    meta_path = unit.meta.get("path")
    if isinstance(meta_path, str) and meta_path:
        return meta_path
    return DEFAULT_TARGETS.get(unit.kind, f"{unit.kind}:{unit.id}").format(id=unit.id)


def _level(unit: Unit, front: dict[str, Any], outline: Outline | None) -> str:
    for cand in (front.get("level"), unit.meta.get("level"), front.get("persona"), unit.meta.get("persona"), outline.persona if outline else ""):
        if isinstance(cand, str) and cand.strip():
            return cand.strip()
    return ""


def derive_obligations(unit: Unit, content: str | None, outline: Outline | None = None) -> list[Obligation]:
    """The deterministic ledger a unit owes, in the order the closers should run."""
    target = _target(unit)
    front, _ = parse_front_matter(content) if content else ({}, "")
    kind = unit.kind
    uid = unit.id
    obs: list[Obligation] = []

    def add(ob_kind: str, description: str, suffix: str = "", status: str = "open", reason: str = "", advisory: bool = False, **args: Any) -> Obligation:
        ob = Obligation(
            id=f"{uid}:{ob_kind}" + (f":{suffix}" if suffix else ""),
            kind=ob_kind,
            description=description,
            target=target,
            required_authority=REQUIRED_AUTHORITY[ob_kind],
            closer=_closer(kind, ob_kind),
            status=status,
            reason=reason,
            advisory=advisory,
            args={"path": target, "unit_id": uid, **args},
        )
        obs.append(ob)
        return ob

    if kind in TEXT_UNIT_KINDS:
        add("structure", "front matter present, headings, balanced fences and $$ blocks")
        add("render", "the chapter renders to HTML without errors or broken links")
        for b in code_blocks(content or ""):
            if b.language == OUTPUT_LANGUAGE:
                continue
            if b.executable:
                add("execute", f"code block {b.index} (line {b.line}) runs" + (" and prints its shown output" if b.expected_output is not None else ""), suffix=str(b.index), block=b.index, line=b.line)
            else:
                lang = b.language or "(no language)"
                add("execute", f"code block {b.index} (line {b.line}, {lang}) runs", suffix=str(b.index), status="blocked", reason=f"no executor for {lang}", block=b.index, line=b.line)
        for c in citations(content or ""):
            add("cite", f"citation [{c['locator']}] resolves to the source text", suffix=c["locator"], locator=c["locator"], line=c["line"])
        level = _level(unit, front, outline)
        add("readability", f"reading level within the band for {level or 'the persona'}", level=level)
        introduces = concept_list(front.get("introduces")) or concept_list(unit.meta.get("introduces"))
        requires = concept_list(front.get("requires")) or concept_list(unit.meta.get("requires"))
        if introduces or requires:
            add("prereqs", "every required concept is introduced by an earlier unit; no forward references", unit_id=uid)
        add("judge", "a separate-context critic finds no high-severity defects against the rubric", advisory=True, unit_id=uid)
    elif kind == "film":
        add("render", "every beat compiles and the film renders")
        add("measure", "per-beat timings, overlap, clipping and font floor are clean")
        add("look", "frames of the render were looked at")
        add("judge", "two reviewers without author context accept the film", advisory=True, unit_id=uid)
    elif kind == "widget":
        add("render", "the widget opens in Chromium and its self-test passes")
        add("measure", "no console errors, layout and contrast within bounds")
        add("judge", "a critic accepts the widget against the rubric", advisory=True, unit_id=uid)
    elif kind == "figure":
        add("render", "the figure rasterises")
        add("measure", "labels do not overlap; contrast and size within bounds")
    elif kind in ("exercise_set", "assessment"):
        add("keys", "every answer key is recomputed mechanically where possible", set_id=uid)
        add("judge", "an item-quality critic accepts the set", advisory=True, unit_id=uid)
    elif kind == "research":
        add("structure", "the brief exists and its citations were checked")  # a missing brief is not a brief with nothing to cite
        for c in citations(content or ""):
            add("cite", f"citation [{c['locator']}] resolves and quotes the source", suffix=c["locator"], locator=c["locator"], line=c["line"])
        add("judge", "a contradiction check finds no unreconciled claims", advisory=True, unit_id=uid)
    elif kind == "integration":
        add("structure", "every film, widget and exercise set referenced by a chapter exists")
        add("render", "the whole book builds")
    for closer in unit.checks:
        if any(o.closer == closer for o in obs):
            continue
        ob_kind = KIND_BY_CLOSER.get(closer, "render")
        obs.append(Obligation(
            id=f"{uid}:{ob_kind}:{closer}", kind=ob_kind, description=f"required check {closer} passes", target=target,
            required_authority=REQUIRED_AUTHORITY[ob_kind], closer=closer, advisory=(ob_kind == "judge"), args={"path": target, "unit_id": uid},
        ))
    return obs


def close_obligation(ob: Obligation, verifier_authority: str, evidence_ids: list[str] | None, passed: bool, detail: str = "") -> Obligation:
    """Close ``ob`` when a verifier of sufficient authority passed; otherwise leave it open with the reason."""
    if verifier_authority not in AUTHORITY_RANK:
        ob.status = "open"
        ob.reason = f"unknown verifier authority {verifier_authority!r}"
        return ob
    if not authority_at_least(verifier_authority, ob.required_authority):
        ob.status = "open"
        ob.reason = f"verifier authority too weak: {verifier_authority} < {ob.required_authority}"
        return ob
    for eid in evidence_ids or []:
        if eid and eid not in ob.evidence_ids:
            ob.evidence_ids.append(eid)
    if passed:
        ob.status = "closed"
        ob.reason = detail or f"closed by {ob.closer} ({verifier_authority})"
    else:
        ob.status = "open"
        ob.reason = detail or f"{ob.closer} reported defects"
    return ob


def blocking_obligations(obs: list[Obligation]) -> list[Obligation]:
    return [o for o in obs if o.blocking]


def ledger_summary(obs: list[Obligation]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for o in obs:
        counts[o.status] = counts.get(o.status, 0) + 1
    return {
        "total": len(obs),
        "by_status": counts,
        "blocking_open": [o.id for o in obs if o.blocking],
        "advisory_open": [o.id for o in obs if o.status == "open" and o.advisory],
        "blocked": [o.id for o in obs if o.status == "blocked"],
    }
