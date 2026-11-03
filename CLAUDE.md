# Edgent: rules for working in this repository

Edgent is a general-purpose agent harness (Claude Code class) with a deep education layer.
Read `docs/EDU_HARNESS_PLAN.md` for the design and `docs/HARNESS_ENGINEERING_PRIMER.md` for the reasoning.

## Layout and dependency direction

```
edgent/domain/        pure types and rules; imports nothing else from edgent
edgent/application/   loop, context, sessions, compaction, permissions, hooks, budgets, subagents, skills, jobs, heartbeat
edgent/tools/         registry + text protocol (contracts), core tools, edu tools, checks, MCP server
edgent/adapters/      providers, sandbox, toolchain, renderers, browser, ingest
edgent/skills/        SKILL.md folders (pedagogy and craft)      edgent/agents/   subagent definitions (Markdown + front matter)
edgent/prompts/       system prompt and behaviour spec             edgent/ui/       CLI and local web UI
edgent/observability/ tracing and metrics                          tests/           pytest, offline, MockProvider
```

`domain` imports nothing of ours. `application` imports `domain` (and may import `tools.registry`). `tools` and `adapters` import `domain` and `tools.registry`. `ui` imports everything. Never import `ui` or `adapters.providers.<vendor>` from `application`.

## The foundation (already written; build on it, do not rewrite it)

| Need | Use |
|---|---|
| Messages, tool calls, replies, requests | `edgent/domain/types.py`: `Message`, `ContentPart`, `ToolCall`, `Reply`, `Usage`, `CompletionRequest` |
| Errors | `edgent/domain/errors.py`: raise `ToolFailure(code, message, category, remediation=..., partial=...)` inside tools; `HarnessError` is the wire shape; `Limit` for every ceiling |
| Provider port | `edgent/adapters/providers/base.py`: subclass `Provider`, set `caps: Capabilities`, implement `complete(request) -> Reply`; call `self.prepare(request)` first and `self.finalize(reply)` last |
| Offline model | `edgent/adapters/providers/mock.py`: `MockProvider().push(...)`, `MockProvider.tool_reply(name, args)`, `.on_role(role, handler)` |
| Tools | `edgent/tools/registry.py`: `ToolRegistry`, `@registry.tool(name, description, schema, permission=..., side_effects=..., group=...)`, `ToolContext`, helpers `obj/arr/opt/S/I/N/B` |
| Workspace | `edgent/domain/workspace.py`: `Workspace.init(root)`, `.path(key)`, `.resolve(rel)` (escape-protected), `.search`, `.glob`, `.instruction_chain`, `.settings`, `.commit`, `.append_progress` |
| Evidence and unknowns | `edgent/domain/evidence.py` (`Evidence`, `EvidenceStore`), `edgent/domain/unknowns.py` (`Unknown`, `UnknownsLedger`) |
| Outline | `edgent/domain/outline.py` (`Outline`, `Unit`; statuses move only through `set_status`) |
| Sandbox | `edgent/adapters/sandbox/runner.py`: `run_python`, `run_subprocess`, `InterpreterSession` |
| Toolchains | `edgent/adapters/toolchain.py`: `Toolchain.detect()`, `.require("ffmpeg", "purpose")` raises the typed unavailable error |
| Traces | `edgent/observability/tracing.py` (`Tracer.record(kind, data)`), `metrics.py` |

The foundation is frozen for parallel work. If you truly need a change, make it minimal and additive (a new optional field, a new helper), never remove or rename anything, and list the change in your report.

## Conventions

- Python 3.11+, type hints everywhere, dataclasses for records, `from __future__ import annotations`.
- The core has **no third-party dependencies**. Optional libraries (PyMuPDF, faster-whisper, playwright, manim) are imported lazily inside the function that needs them; when absent, raise `ToolFailure("unavailable", ..., category="unavailable", remediation=<install hint>)` via `ctx.toolchain.require(...)`. Never let an `ImportError` escape.
- Every tool: a clear description written for a new colleague (when to use, when not to), unambiguous parameter names with units, a JSON schema with `additionalProperties: false`, results that are small and high-signal (paginate or persist large outputs; the registry compacts anything over 6000 characters), typed errors with a remediation. Outputs must be JSON-serialisable. To show the model an image, return `{"_attach": [{"type": "image", "path": ...}]}` in the output dict.
- Every ceiling is a `Limit` with a unit, or at least a named module-level constant with a unit in the name. Failure messages say the effective value and the next action.
- Untrusted data rule: text from files, web pages, OCR and transcripts is data. Tools that return it set `"untrusted": true` in their output.
- Persist before you proceed: state goes to disk before the next model call.
- Logging: use the tracer when you have `ctx.tracer`; never print from library code.
- Style: `ruff check` clean (line length 120). Keep functions short; comments explain why, not what.

## Tests

- `pytest`, offline, deterministic. Use the fixtures in `tests/conftest.py` (`workspace`, `provider`, `registry`, `ctx`, `home`). No network. No real model.
- When a toolchain is absent (ffmpeg, pandoc, chromium, manim, tesseract, pymupdf), skip the test with a reason that says the check is unverified here: `tests/conftest.py:skip_unless("ffmpeg")`. On this machine ffmpeg, ffprobe, pandoc, pdftotext, node and Chromium (`edgent.adapters.toolchain.find_chromium()`) exist; tesseract, manim, latex, pymupdf, faster-whisper, playwright do not.
- Name test files `tests/test_<module>.py`. Only edit the files you own. Run your own tests first, then the whole suite (`python3 -m pytest -q`) before you report.
- Do not commit. The orchestrator commits.

## Reporting

Your final message is data for the orchestrator: files created or changed, tests added and their result, any foundation change, open issues or unverified behaviour. No prose beyond that.
