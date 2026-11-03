"""check_render: does the chapter render at all, and is its structure sound.

Always runs the structural pass (front matter, headings, balanced fences and
math delimiters, relative links that resolve inside the workspace, placeholder
targets). When pandoc is installed the chapter is also rendered to standalone
HTML with KaTeX into ``book/build/<stem>.html``; without pandoc the result
carries the limitation ``no pandoc: structure only``.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from ...adapters.sandbox import run_subprocess
from ...domain.errors import Limit
from ...domain.obligations import code_blocks, placeholders, strip_code
from ...domain.utils import ensure_dir, parse_front_matter
from ..registry import ToolContext
from . import check_result, defect, last_line, record_evidence, resolve_text_file, trace_check, verdict

RENDER_TIMEOUT_LIMIT = Limit("check_render pandoc timeout", 120, "s")
PANDOC_MEMORY_BYTES = 2 << 30  # pandoc is a Haskell binary; it maps more address space than it uses
PANDOC_WARNINGS_SHOWN = 20
NO_PANDOC_LIMITATION = "no pandoc: structure only"
FRONT_MATTER_REQUIRED = ("title",)
FRONT_MATTER_RECOMMENDED = ("unit_id", "level")
_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_HEADING_RE = re.compile(r"^(#{1,6})\s+\S")
_EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "data:", "#", "tel:")
_PLACEHOLDER_TARGETS = {"film": "media/films/{id}", "widget": "media/widgets/{id}.html", "exercises": "assessments/{id}"}


def front_matter_lines(text: str) -> int:
    """How many leading lines the front matter block occupies (0 when there is none)."""
    if not text.startswith("---"):
        return 0
    end = text.find("\n---", 3)
    if end == -1:
        return 0
    return text[: end + 4].count("\n") + 1


def _prose_lines(text: str) -> list[str]:
    """Lines with code fences and the front matter blanked, so line numbers still match the file."""
    lines = strip_code(text).splitlines()
    for n in range(min(front_matter_lines(text), len(lines))):
        lines[n] = ""
    return lines


def structural_defects(ctx: ToolContext, abs_path: Path, rel: str, text: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Headings, fences, math delimiters, front matter, links and placeholders; nothing here needs a toolchain."""
    defects: list[dict[str, Any]] = []
    front, _body = parse_front_matter(text)
    if not text.startswith("---") or not front:
        defects.append(defect(f"{rel}:L1", "medium", "no front matter block (--- title / unit_id / level / introduces / requires / sources ---)"))
    else:
        for key in FRONT_MATTER_REQUIRED:
            if not front.get(key):
                defects.append(defect(f"{rel}:L1", "medium", f"front matter lacks {key!r}"))
        missing = [k for k in FRONT_MATTER_RECOMMENDED if k not in front]
        if missing:
            defects.append(defect(f"{rel}:L1", "low", "front matter lacks " + ", ".join(repr(k) for k in missing)))
    blocks = code_blocks(text)
    for b in blocks:
        if not b.closed:
            defects.append(defect(f"{rel}:L{b.line}", "high", f"code fence opened at line {b.line} is never closed"))
        elif not b.language:
            defects.append(defect(f"{rel}:L{b.line}", "low", f"fenced block at line {b.line} has no language"))
    prose = _prose_lines(text)
    headings = [(n, m.group(1)) for n, line in enumerate(prose, 1) if (m := _HEADING_RE.match(line))]
    if not headings:
        defects.append(defect(rel, "medium", "no headings"))
    prev_level = 0
    for n, hashes in headings:
        level = len(hashes)
        if prev_level and level > prev_level + 1:
            defects.append(defect(f"{rel}:L{n}", "low", f"heading level jumps from h{prev_level} to h{level}"))
        prev_level = level
    display = sum(line.count("$$") for line in prose)
    if display % 2:
        defects.append(defect(rel, "high", f"unbalanced $$ delimiters ({display} occurrences)"))
    in_display = False
    for n, line in enumerate(prose, 1):
        line = re.sub(r"`[^`]*`", "", line)
        if "$$" in line:
            in_display = not in_display if line.count("$$") % 2 else in_display
            line = line.replace("$$", "")
        if in_display:
            continue
        singles = len(re.findall(r"(?<!\\)\$", line))
        if singles % 2:
            defects.append(defect(f"{rel}:L{n}", "medium", f"unbalanced $ on line {n}"))
    links_checked = 0
    for n, line in enumerate(prose, 1):
        for m in _LINK_RE.finditer(line):
            href = m.group(1)
            if href.startswith(_EXTERNAL_PREFIXES):
                continue
            target = href.split("#", 1)[0].split("?", 1)[0]
            if not target:
                continue
            links_checked += 1
            candidate = (abs_path.parent / target).resolve() if not target.startswith("/") else (ctx.workspace.root / target.lstrip("/")).resolve()
            if not ctx.workspace.is_inside(candidate):
                defects.append(defect(f"{rel}:L{n}", "medium", f"link {href} escapes the workspace"))
            elif not candidate.exists():
                defects.append(defect(f"{rel}:L{n}", "medium", f"broken link {href} (expected {ctx.workspace.relpath(candidate)})"))
    refs = placeholders(text)
    for ph in refs:
        target = _PLACEHOLDER_TARGETS[ph["kind"]].format(id=ph["id"])
        if not (ctx.workspace.root / target).exists():
            defects.append(defect(f"{rel}:L{ph['line']}", "low", f"{{{{{ph['kind']}: {ph['id']}}}}} has no built target yet (expected {target})"))
    detail = {
        "front_matter_keys": sorted(front),
        "headings": len(headings),
        "code_blocks": [b.to_dict() for b in blocks],
        "display_math_delimiters": display,
        "links_checked": links_checked,
        "placeholders": refs,
        "untrusted": True,  # keys, ids and pandoc warnings quote the file
    }
    return defects, detail


def _pandoc_path(ctx: ToolContext) -> str | None:
    tc = getattr(ctx, "toolchain", None)
    if tc is not None:
        return tc.path("pandoc")
    return shutil.which("pandoc")


def check_render(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    abs_path, rel, text = resolve_text_file(ctx, args.get("path"))
    defects, detail = structural_defects(ctx, abs_path, rel, text)
    limitations: list[str] = []
    pandoc = _pandoc_path(ctx)
    confidence = "derived"
    if pandoc:
        build_dir = Path(ensure_dir(ctx.workspace.path("build")))
        out = build_dir / f"{abs_path.stem}.html"
        argv = [pandoc, str(abs_path), "--from", "markdown", "--to", "html5", "--standalone", "--katex", "--output", str(out)]
        res = run_subprocess(argv, cwd=str(ctx.workspace.root), timeout_s=RENDER_TIMEOUT_LIMIT.value, cpu_seconds=int(RENDER_TIMEOUT_LIMIT.value), memory_bytes=PANDOC_MEMORY_BYTES)
        warnings = [ln.strip() for ln in res.stderr.splitlines() if ln.startswith("[WARNING]")]
        detail["pandoc"] = {"returncode": res.returncode, "timed_out": res.timed_out, "output": ctx.workspace.relpath(out) if out.exists() else None, "warnings": warnings[:PANDOC_WARNINGS_SHOWN], "duration_s": round(res.duration_s, 3)}
        if res.timed_out:
            defects.append(defect(rel, "high", f"pandoc timed out after {RENDER_TIMEOUT_LIMIT.value:g} s; {RENDER_TIMEOUT_LIMIT.describe()}"))
        elif not res.ok:
            defects.append(defect(rel, "high", f"pandoc failed (exit {res.returncode}): {last_line(res.stderr) or 'no message'}"))
        else:
            confidence = "observed"
    else:
        limitations.append(NO_PANDOC_LIMITATION)
    passed = verdict(defects)
    summary = {"passed": passed, "defects": len(defects), "pandoc": bool(pandoc), "headings": detail["headings"], "code_blocks": len(detail["code_blocks"])}
    eid = record_evidence(ctx, "check_render", rel, text, summary, confidence=confidence, authority="controlled_replay", limitations=limitations)
    result = check_result(passed, "controlled_replay", confidence, defects, eid, detail, limitations)
    trace_check(ctx, "render", rel, result)
    return result
