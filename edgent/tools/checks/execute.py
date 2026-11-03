"""check_execute: every Python block in a chapter runs, and prints what the chapter shows.

Blocks run in order, each as its own sandboxed process, sharing one scratch
directory so a file written by one block can be read by the next. A block
whose first lines carry ``# edgent: continue`` is concatenated with the code
that ran before it, which is how a chapter builds on earlier definitions. When
a block is followed by an ``output`` fence, stdout is compared to it with
whitespace normalised.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

from ...adapters.sandbox import run_subprocess
from ...domain.errors import Limit, ToolFailure
from ...domain.obligations import CodeBlock, code_blocks
from ...domain.utils import ensure_dir, sha256_text
from ..registry import ToolContext
from . import check_result, defect, last_line, record_evidence, resolve_text_file, trace_check, verdict

DEFAULT_BLOCK_TIMEOUT_S = 60.0
BLOCK_TIMEOUT_LIMIT = Limit("check_execute block timeout", 300, "s", does_not_raise=("the registry timeout, which sits above it",))
BLOCKS_LIMIT = Limit("check_execute code blocks", 100, "blocks", does_not_raise=("splitting the chapter, which is the fix",))
BLOCK_MEMORY_BYTES = 1 << 30
STDOUT_PREVIEW_CHARS = 300
TOOL_TIMEOUT_S = 1800.0  # the registry ceiling for the whole file; BLOCKS_LIMIT * BLOCK_TIMEOUT_LIMIT can exceed it, so a long chapter is split


def normalise_output(text: str) -> list[str]:
    return [" ".join(line.split()) for line in text.strip().splitlines() if line.strip()]


def _scratch_dir(ctx: ToolContext, rel: str) -> Path:
    stem = Path(rel).stem
    d = Path(ctx.workspace.path("cache")) / "execute" / f"{stem}-{sha256_text(rel)[:8]}"  # one directory per file, not per stem
    shutil.rmtree(d, ignore_errors=True)  # a fresh directory keeps runs reproducible
    ensure_dir(d)
    return d


def _run_block(block: CodeBlock, code: str, scratch: Path, timeout_s: float) -> Any:
    script = scratch / f"block_{block.index:03d}.py"
    script.write_text(code, encoding="utf-8")
    cpu_seconds = max(1, int(timeout_s))
    return run_subprocess([sys.executable, "-I", str(script)], cwd=str(scratch), timeout_s=timeout_s, cpu_seconds=cpu_seconds, memory_bytes=BLOCK_MEMORY_BYTES)


def check_execute(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    abs_path, rel, text = resolve_text_file(ctx, args.get("path"))
    timeout_s = float(args.get("timeout_s") or DEFAULT_BLOCK_TIMEOUT_S)
    if timeout_s <= 0 or timeout_s > BLOCK_TIMEOUT_LIMIT.value:
        err = BLOCK_TIMEOUT_LIMIT.exceeded(timeout_s, "Pass a smaller timeout_s, or split a slow block.")
        raise ToolFailure(err.code, err.message, err.category, remediation=err.remediation, details=err.details)
    runnable = [b for b in code_blocks(text) if b.executable]
    if len(runnable) > BLOCKS_LIMIT.value:
        err = BLOCKS_LIMIT.exceeded(len(runnable), "Split the chapter into sections and check each.")
        raise ToolFailure(err.code, err.message, err.category, remediation=err.remediation, details=err.details)
    scratch = _scratch_dir(ctx, rel)
    defects: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    prev_code = ""
    prev_stdout = ""
    for b in runnable:
        continued = b.continues_previous and bool(prev_code)
        code = (prev_code.rstrip("\n") + "\n" + b.code) if continued else b.code
        res = _run_block(b, code, scratch, timeout_s)
        own_stdout = res.stdout[len(prev_stdout):] if continued and res.stdout.startswith(prev_stdout) else res.stdout
        entry: dict[str, Any] = {
            "block": b.index, "line": b.line, "ok": res.ok, "returncode": res.returncode, "timed_out": res.timed_out,
            "duration_s": round(res.duration_s, 3), "continued": continued, "stdout_preview": own_stdout[:STDOUT_PREVIEW_CHARS],
        }
        location = f"{rel}:L{b.line}"
        if res.timed_out:
            entry["error"] = f"timed out after {timeout_s:g} s"
            defects.append(defect(location, "high", f"block {b.index} timed out after {timeout_s:g} s; {BLOCK_TIMEOUT_LIMIT.describe()}"))
        elif not res.ok:
            entry["error"] = last_line(res.stderr) or f"exit status {res.returncode}"
            defects.append(defect(location, "high", f"block {b.index} failed: {entry['error']}"))
        elif b.expected_output is not None:
            matches = normalise_output(own_stdout) == normalise_output(b.expected_output)
            entry["output_matches"] = matches
            if not matches:
                defects.append(defect(location, "high", f"block {b.index} printed {own_stdout.strip()[:120]!r}, the chapter shows {b.expected_output.strip()[:120]!r}"))
        if res.truncated:
            entry["stdout_truncated"] = True
        entries.append(entry)
        prev_code, prev_stdout = code, res.stdout
    (scratch / "report.txt").write_text("\n".join(f"block {e['block']} line {e['line']}: {'ok' if e['ok'] else e.get('error', 'failed')}" for e in entries) + "\n", encoding="utf-8")
    passed = verdict(defects)
    detail = {"blocks": entries, "blocks_run": len(entries), "scratch_dir": ctx.workspace.relpath(scratch), "timeout_s": timeout_s, "untrusted": True}  # previews quote program output
    summary = {"passed": passed, "blocks_run": len(entries), "failed": [e["block"] for e in entries if not e["ok"] or e.get("output_matches") is False]}
    eid = record_evidence(ctx, "check_execute", rel, text, summary, confidence="observed", authority="controlled_replay", parameters={"timeout_s": timeout_s})
    result = check_result(passed, "controlled_replay", "observed", defects, eid, detail)
    if not entries:
        result["limitations"] = ["no python code blocks in the file"]
    trace_check(ctx, "execute", rel, result)
    return result
