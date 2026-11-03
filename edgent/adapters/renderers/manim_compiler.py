"""Storyboard to Manim scene: normalisation, per-beat syntax checks, lint, compilation and edit ops.

The model never writes a scene file. It emits a storyboard (``title``, ``persona``, ``prelude``, ``beats``;
each beat an ``id``, ``title``, ``narration``, ``code`` that is the body of a Manim method with ``self`` as the
Scene, and a target ``seconds``) and edits it with ops. ``compile_storyboard`` turns that into ``scene.py``:
the prelude, the storyboard's own prelude, and ``GeneratedFilm`` whose ``construct`` runs every beat body
inside ``with self.beat(id, title):`` so the render measures itself. Pure Python; Manim is never imported.
"""
from __future__ import annotations

import ast
import copy
import textwrap
from collections.abc import Iterable, Mapping
from typing import Any

from .manim_prelude import PRELUDE_SOURCE

STORYBOARD_VERSION = 1
SCENE_CLASS = "GeneratedFilm"
BEAT_INDENT = " " * 12  # inside class > construct > with
NARRATION_COMMENT_CHARS = 100
WORDS_PER_SECOND = 2.6  # spoken narration pace used for duration estimates
META_KEYS: tuple[str, ...] = ("title", "persona", "show_titles")
OP_NAMES: tuple[str, ...] = ("add_beat", "set_beat", "patch_beat", "move_beat", "remove_beat", "prelude", "set_meta")
_BEAT_CALLS = {"play", "wait", "add"}
_PROTECTED_NAMES = {"self", "beat"}


class CompileError(Exception):
    """A storyboard that cannot become a scene: which beat (None for the prelude or the structure) and which line."""

    def __init__(self, message: str, beat: int | None = None, line: int | None = None, text: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.beat = beat
        self.line = line
        self.text = text

    def __str__(self) -> str:
        where = f"beat {self.beat}" if self.beat is not None else "storyboard"
        if self.line is not None:
            where += f" line {self.line}"
        s = f"{where}: {self.message}"
        if self.text:
            s += f" ({self.text.strip()[:80]})"
        return s

    def to_dict(self) -> dict[str, Any]:
        return {"message": self.message, "beat": self.beat, "line": self.line, "text": self.text}


class _OpError(Exception):
    """An op that cannot be applied; the reason goes to the caller as a rejection, never as an exception."""


# ----------------------------------------------------------------------------- normalisation
def _text(value: Any, field: str, beat: int | None = None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    raise CompileError(f"{field} must be a string, got {type(value).__name__}", beat=beat)


def _seconds(value: Any, beat: int | None) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CompileError(f"seconds must be a number or null, got {type(value).__name__}", beat=beat)
    if value < 0:
        raise CompileError("seconds must not be negative", beat=beat)
    return float(value)


def _beat_id(value: Any, position: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CompileError(f"beats[{position}].id must be a positive integer, got {value!r}")
    return value


def normalise(storyboard: Mapping[str, Any] | None) -> dict[str, Any]:
    """A complete storyboard with defaults filled in and every beat carrying a stable id.

    Ids are allocated from ``next_beat_id`` and never reused: a removed beat's id stays retired in
    ``removed_beats``. Unknown top-level keys are kept so a richer storyboard survives a round trip.
    """
    if storyboard is None:
        storyboard = {}
    if not isinstance(storyboard, Mapping):
        raise CompileError(f"storyboard must be a JSON object, got {type(storyboard).__name__}")
    sb = copy.deepcopy(dict(storyboard))
    beats_in = sb.get("beats") or []
    if not isinstance(beats_in, list):
        raise CompileError("beats must be a list")
    removed_in = sb.get("removed_beats") or []
    if not isinstance(removed_in, list):
        raise CompileError("removed_beats must be a list")
    removed = [r for r in removed_in if isinstance(r, Mapping) and isinstance(r.get("id"), int)]

    seen: set[int] = set()
    pending: list[tuple[int | None, Mapping[str, Any]]] = []
    for i, b in enumerate(beats_in):
        if not isinstance(b, Mapping):
            raise CompileError(f"beats[{i}] must be an object, got {type(b).__name__}")
        raw_id = b.get("id")
        bid = None if raw_id is None else _beat_id(raw_id, i)
        if bid is not None:
            if bid in seen:
                raise CompileError(f"beat id {bid} is used twice; ids must be unique", beat=bid)
            seen.add(bid)
        pending.append((bid, b))

    retired = {int(r["id"]) for r in removed}
    next_id_in = sb.get("next_beat_id")
    next_id = next_id_in if isinstance(next_id_in, int) and not isinstance(next_id_in, bool) and next_id_in > 0 else 1
    next_id = max([next_id, *(i + 1 for i in seen), *(i + 1 for i in retired)])

    beats: list[dict[str, Any]] = []
    for bid, b in pending:
        if bid is None:
            bid, next_id = next_id, next_id + 1
        beats.append(
            {
                "id": bid,
                "title": _text(b.get("title"), "title", bid),
                "narration": _text(b.get("narration"), "narration", bid),
                "code": _text(b.get("code"), "code", bid),
                "seconds": _seconds(b.get("seconds"), bid),
            }
        )

    out: dict[str, Any] = {k: v for k, v in sb.items() if k not in ("beats", "removed_beats", "next_beat_id")}
    out["version"] = STORYBOARD_VERSION
    out["title"] = _text(sb.get("title"), "title") or "Untitled film"
    out["persona"] = _text(sb.get("persona"), "persona")
    out["prelude"] = _text(sb.get("prelude"), "prelude")
    out["show_titles"] = bool(sb.get("show_titles", True))
    out["beats"] = beats
    out["removed_beats"] = [{"id": int(r["id"]), "title": str(r.get("title") or ""), "reason": str(r.get("reason") or "")} for r in removed]
    out["next_beat_id"] = next_id
    return out


# ----------------------------------------------------------------------------- per-beat checks
def _parse_beat(code: str, beat_id: int | None) -> tuple[str, ast.Module]:
    body = textwrap.dedent(code or "").rstrip()  # leading blank lines stay, so reported lines match the author's
    inner = textwrap.indent(body, "    ") if body.strip() else "    pass"
    wrapped = f"def _beat(self):\n{inner}\n"
    try:
        tree = ast.parse(wrapped, filename=f"beat-{beat_id}")
    except SyntaxError as e:
        line = max(1, (e.lineno or 1) - 1)  # the wrapper's def line is line 0 of the beat
        raise CompileError(e.msg or "syntax error", beat=beat_id, line=line, text=e.text) from e
    fn = tree.body[0]
    assert isinstance(fn, ast.FunctionDef)
    for node in ast.walk(fn):
        if isinstance(node, (ast.Yield, ast.YieldFrom, ast.Await)) and _owner_function(fn, node) is fn:
            raise CompileError(f"{type(node).__name__.lower()} is not allowed in a beat (it would turn construct() into a coroutine)", beat=beat_id, line=node.lineno - 1)
    return body, tree


def _owner_function(root: ast.FunctionDef, target: ast.AST) -> ast.AST | None:
    """The innermost function or lambda that contains ``target`` (``root`` itself when not nested)."""
    owner: ast.AST | None = None

    def walk(node: ast.AST, current: ast.AST) -> bool:
        nonlocal owner
        if node is target:
            owner = current
            return True
        nxt = node if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and node is not root else current
        return any(walk(child, nxt) for child in ast.iter_child_nodes(node))

    walk(root, root)
    return owner


def check_beat_code(code: str, beat_id: int | None) -> str:
    """The dedented beat body, or a ``CompileError`` naming the beat and the line of the first syntax error."""
    body, _ = _parse_beat(code, beat_id)
    return body


def lint_beat(code: str, beat_id: int | None) -> list[str]:
    """Warnings about a beat that parses but will not do what the author meant."""
    body, tree = _parse_beat(code, beat_id)
    label = f"beat {beat_id}" if beat_id is not None else "beat"
    warnings: list[str] = []
    if not body.strip():
        warnings.append(f"{label}: the code is empty; the beat shows whatever the previous beat left on screen")
        return warnings
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"
    }
    if not calls & _BEAT_CALLS:
        warnings.append(f"{label}: no self.play, self.wait or self.add; nothing changes on screen")
    rebound = sorted({node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in _PROTECTED_NAMES})
    for name in rebound:
        warnings.append(f"{label}: rebinds {name!r}; the beat must leave self and beat untouched")
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and _owner_function(tree.body[0], node) is tree.body[0]:  # type: ignore[arg-type]
            warnings.append(f"{label}: return at line {node.lineno - 1} ends construct() and skips every later beat")
            break
    return warnings


# ----------------------------------------------------------------------------- compilation
def _one_line(text: str) -> str:
    """Text safe inside a ``#`` comment: no newline can end the comment and become code."""
    return " ".join((text or "").split())


def _narration_comment(narration: str) -> str:
    one_line = _one_line(narration)
    if len(one_line) > NARRATION_COMMENT_CHARS:
        one_line = one_line[: NARRATION_COMMENT_CHARS - 1] + "…"
    return one_line


def compile_storyboard(storyboard: Mapping[str, Any], only: Iterable[int] | None = None) -> tuple[str, list[str]]:
    """``(scene_source, warnings)`` for a storyboard; ``only`` restricts construct() to those beat ids.

    Raises ``CompileError`` for a beat or prelude that does not parse. The returned source always parses.
    """
    sb = normalise(storyboard)
    warnings: list[str] = []
    prelude = sb["prelude"].rstrip()
    if prelude.strip():
        try:
            ast.parse(prelude, filename="storyboard-prelude")
        except SyntaxError as e:
            raise CompileError(e.msg or "syntax error", beat=None, line=e.lineno, text=e.text) from e
    wanted = None if only is None else {int(i) for i in only}
    header = f"# Generated by Edgent from the storyboard {_one_line(sb['title'])!r}. Do not edit: change the storyboard and recompile."
    lines: list[str] = [header, "", PRELUDE_SOURCE.strip("\n"), "", "# ---- storyboard prelude ------------------------------------------------------------------------------------"]
    lines.append(prelude if prelude.strip() else "# (none)")
    lines += ["", "", f"class {SCENE_CLASS}(Film):", f"    show_titles = {bool(sb['show_titles'])!r}", "", "    def construct(self):"]
    emitted = 0
    for beat in sb["beats"]:
        bid = beat["id"]
        body = check_beat_code(beat["code"], bid)
        warnings.extend(lint_beat(beat["code"], bid))
        if wanted is not None and bid not in wanted:
            continue
        lines.append(f"        # beat {bid}: {_one_line(beat['title']) or '(untitled)'}")
        if beat["narration"].strip():
            lines.append(f"        # narration: {_narration_comment(beat['narration'])}")
        lines.append(f"        with self.beat({bid}, {beat['title']!r}):")
        lines.append(textwrap.indent(body if body.strip() else "pass", BEAT_INDENT))
        lines.append("")
        emitted += 1
    if not sb["beats"]:
        warnings.append("the storyboard has no beats")
    if wanted is not None:
        missing = sorted(wanted - {b["id"] for b in sb["beats"]})
        if missing:
            warnings.append(f"only={sorted(wanted)}: beats {missing} do not exist")
    if emitted == 0:
        lines.append("        pass")
    source = "\n".join(lines).rstrip() + "\n"
    try:
        ast.parse(source, filename="scene.py")
    except SyntaxError as e:  # pragma: no cover - every piece was parsed on its own; this is a compiler bug
        raise CompileError(f"internal: the generated scene does not parse: {e.msg}", beat=None, line=e.lineno, text=e.text) from e
    return source, warnings


# ----------------------------------------------------------------------------- ops
def _find(sb: dict[str, Any], beat_id: Any) -> tuple[int, dict[str, Any]]:
    if isinstance(beat_id, bool) or not isinstance(beat_id, int):
        raise _OpError(f"beat must be an integer id, got {beat_id!r}")
    for i, b in enumerate(sb["beats"]):
        if b["id"] == beat_id:
            return i, b
    ids = [b["id"] for b in sb["beats"]]
    raise _OpError(f"beat {beat_id} does not exist; existing ids: {ids}")


def _insert_index(sb: dict[str, Any], after: Any) -> int:
    if after is None:
        return len(sb["beats"])
    if after == 0:
        return 0
    i, _ = _find(sb, after)
    return i + 1


def _checked_code(code: Any, beat_id: int | None) -> str:
    if code is None:
        return ""
    if not isinstance(code, str):
        raise _OpError(f"code must be a string, got {type(code).__name__}")
    try:
        check_beat_code(code, beat_id)
    except CompileError as e:
        raise _OpError(f"code has a syntax error at line {e.line}: {e.message}") from e
    return code


def _string(op: Mapping[str, Any], key: str, default: str | None = "") -> str | None:
    value = op.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise _OpError(f"{key} must be a string, got {type(value).__name__}")
    return value


def _op_add_beat(sb: dict[str, Any], op: Mapping[str, Any]) -> dict[str, Any]:
    index = _insert_index(sb, op.get("after"))
    bid = sb["next_beat_id"]
    beat = {
        "id": bid,
        "title": _string(op, "title") or "",
        "narration": _string(op, "narration") or "",
        "code": _checked_code(op.get("code"), bid),
        "seconds": _op_seconds(op.get("seconds")),
    }
    sb["beats"].insert(index, beat)
    sb["next_beat_id"] = bid + 1
    return {"op": "add_beat", "beat": bid, "position": index}


def _op_seconds(value: Any) -> float | None:
    try:
        return _seconds(value, None)
    except CompileError as e:
        raise _OpError(e.message) from e


def _op_set_beat(sb: dict[str, Any], op: Mapping[str, Any]) -> dict[str, Any]:
    _, beat = _find(sb, op.get("beat"))
    fields = [k for k in ("title", "narration", "code", "seconds") if k in op]
    if not fields:
        raise _OpError("set_beat changes nothing: give title, narration, code or seconds")
    updates: dict[str, Any] = {}
    for key in fields:
        if key == "code":
            updates[key] = _checked_code(op.get("code"), beat["id"])
        elif key == "seconds":
            updates[key] = _op_seconds(op.get("seconds"))
        else:
            updates[key] = _string(op, key) or ""
    beat.update(updates)
    return {"op": "set_beat", "beat": beat["id"], "fields": fields}


def _op_patch_beat(sb: dict[str, Any], op: Mapping[str, Any]) -> dict[str, Any]:
    _, beat = _find(sb, op.get("beat"))
    find = _string(op, "find", None)
    replace = _string(op, "replace", "") or ""
    if not find:
        raise _OpError("find must be a non-empty string")
    count = beat["code"].count(find)
    if count != 1:
        raise _OpError(f"find occurs {count} times in the code of beat {beat['id']}; it must occur exactly once")
    new_code = beat["code"].replace(find, replace, 1)
    beat["code"] = _checked_code(new_code, beat["id"])
    return {"op": "patch_beat", "beat": beat["id"]}


def _op_move_beat(sb: dict[str, Any], op: Mapping[str, Any]) -> dict[str, Any]:
    i, beat = _find(sb, op.get("beat"))
    after = op.get("after")
    if after == beat["id"]:
        raise _OpError(f"beat {beat['id']} cannot be moved after itself")
    if after is None:
        raise _OpError("move_beat needs after: a beat id, or 0 for the front")
    sb["beats"].pop(i)
    try:
        index = _insert_index(sb, after)
    except _OpError:
        sb["beats"].insert(i, beat)
        raise
    sb["beats"].insert(index, beat)
    return {"op": "move_beat", "beat": beat["id"], "position": index}


def _op_remove_beat(sb: dict[str, Any], op: Mapping[str, Any]) -> dict[str, Any]:
    i, beat = _find(sb, op.get("beat"))
    reason = _string(op, "reason") or ""
    sb["beats"].pop(i)
    sb["removed_beats"].append({"id": beat["id"], "title": beat["title"], "reason": reason})
    return {"op": "remove_beat", "beat": beat["id"]}


def _op_prelude(sb: dict[str, Any], op: Mapping[str, Any]) -> dict[str, Any]:
    py = _string(op, "py", None)
    if py is None:
        raise _OpError("prelude needs py: the Python source to run once before the beats")
    try:
        ast.parse(py, filename="storyboard-prelude")
    except SyntaxError as e:
        raise _OpError(f"prelude has a syntax error at line {e.lineno}: {e.msg}") from e
    sb["prelude"] = py
    why = _string(op, "why") or ""
    if why:
        sb["prelude_why"] = why
    return {"op": "prelude", "lines": py.count("\n") + 1}


def _op_set_meta(sb: dict[str, Any], op: Mapping[str, Any]) -> dict[str, Any]:
    name = op.get("name")
    if name not in META_KEYS:
        raise _OpError(f"set_meta name must be one of {list(META_KEYS)}, got {name!r}")
    value = op.get("value")
    if name == "show_titles":
        if not isinstance(value, bool):
            raise _OpError("show_titles must be true or false")
    elif not isinstance(value, str):
        raise _OpError(f"{name} must be a string")
    elif name == "title" and not value.strip():
        raise _OpError("title must not be empty")
    sb[name] = value
    return {"op": "set_meta", "name": name}


_HANDLERS = {
    "add_beat": _op_add_beat,
    "set_beat": _op_set_beat,
    "patch_beat": _op_patch_beat,
    "move_beat": _op_move_beat,
    "remove_beat": _op_remove_beat,
    "prelude": _op_prelude,
    "set_meta": _op_set_meta,
}


def apply_ops(storyboard: Mapping[str, Any] | None, ops: Iterable[Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply edit ops in order. Returns ``(new_storyboard, applied, rejected)``; a rejected op changes nothing.

    Ops: ``add_beat {after?, title, narration, code, seconds?}`` (``after`` 0 inserts at the front, absent appends),
    ``set_beat {beat, title?, narration?, code?, seconds?}``, ``patch_beat {beat, find, replace}`` (``find`` must
    occur exactly once), ``move_beat {beat, after}``, ``remove_beat {beat, reason}``, ``prelude {py, why}`` and
    ``set_meta {name, value}`` for title, persona and show_titles. Code is syntax-checked before it is accepted.
    """
    sb = normalise(storyboard)
    applied: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, op in enumerate(ops or []):
        name = op.get("op") if isinstance(op, Mapping) else None
        try:
            if not isinstance(op, Mapping):
                raise _OpError(f"op must be an object with an 'op' field, got {type(op).__name__}")
            handler = _HANDLERS.get(str(name))
            if handler is None:
                raise _OpError(f"unknown op {name!r}; choose from {list(OP_NAMES)}")
            summary = handler(sb, op)
        except _OpError as e:
            rejected.append({"op": name if isinstance(name, str) else None, "index": index, "beat": op.get("beat") if isinstance(op, Mapping) else None, "reason": str(e)})
            continue
        summary["index"] = index
        applied.append(summary)
    return normalise(sb), applied, rejected


# ----------------------------------------------------------------------------- outline helpers
def narration_words(text: str) -> int:
    return len((text or "").split())


def estimate_seconds(beat: Mapping[str, Any]) -> float | None:
    """The beat's target duration: its ``seconds`` when set, else the narration at ``WORDS_PER_SECOND``."""
    if beat.get("seconds") is not None:
        return float(beat["seconds"])
    words = narration_words(str(beat.get("narration") or ""))
    return round(words / WORDS_PER_SECOND, 1) if words else None


def outline(storyboard: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One small row per beat for the model: id, title, target seconds, narration length, code size."""
    rows = []
    for b in normalise(storyboard)["beats"]:
        rows.append(
            {
                "id": b["id"],
                "title": b["title"] or "(untitled)",
                "seconds": b["seconds"],
                "estimated_seconds": estimate_seconds(b),
                "narration_words": narration_words(b["narration"]),
                "code_lines": len(b["code"].strip().splitlines()) if b["code"].strip() else 0,
            }
        )
    return rows


__all__ = [
    "META_KEYS",
    "OP_NAMES",
    "SCENE_CLASS",
    "WORDS_PER_SECOND",
    "CompileError",
    "apply_ops",
    "check_beat_code",
    "compile_storyboard",
    "estimate_seconds",
    "lint_beat",
    "narration_words",
    "normalise",
    "outline",
]
