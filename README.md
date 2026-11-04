# Edgent

Edgent is a general-purpose agent harness, in the class of Claude Code and the Antigravity CLI, with a deep education layer. It runs a model in a loop over tools and a workspace of files; the same loop writes code, researches a question, tutors a learner over their own sources, builds a book of a few hundred pages, makes a Manim film, writes and verifies assignments, or runs a mock interview. It is a desktop tool for one person: no accounts, no server, no embeddings.

The model is the engine. Edgent is everything around it: the loop, the context, the tools, the workspace, permissions, sessions, compaction, subagents, skills, verification, a build job that survives a closed laptop, a local web reader, and an MCP server so other agents can use its tools.

- `docs/EDU_HARNESS_PLAN.md` is the design.
- `docs/HARNESS_ENGINEERING_PRIMER.md` is the course on why it is built this way.
- `CLAUDE.md` is the rulebook for working on the code.

## Install

Python 3.11 or newer; the core has no third-party dependencies.

```bash
pip install -e .              # the `edgent` command
pip install -e ".[dev]"       # plus pytest and ruff
```

Optional toolchains are discovered at run time and never required: `ffmpeg` and `ffprobe` (video, audio), `pandoc` (office documents), `pdftotext` or PyMuPDF (PDF), Chromium (rendering and measuring pages, screenshots), `manim` (films), `tesseract` (OCR), `faster-whisper` (transcription), `node`. `edgent doctor` lists what is present and what each missing one would add.

## Drivers

Edgent speaks to a model through one of four drivers. The first that applies is chosen; `edgent doctor` says which and why.

| kind | selected when | model |
|---|---|---|
| `gemini` | `GEMINI_API_KEY` (or `EDGENT_GEMINI_API_KEY`) is set | `EDGENT_GEMINI_MODEL`, default `gemini-2.5-pro` |
| `openai` | `OPENAI_BASE_URL` (or `EDGENT_OPENAI_BASE_URL`) points at any OpenAI-compatible server, or `OPENAI_API_KEY` is set | `OPENAI_MODEL` |
| `agy` | the Antigravity CLI `agy` is on the PATH | `EDGENT_MODEL` |
| `claude` | the Claude Code CLI `claude` is on the PATH | `EDGENT_MODEL` |
| `mock` | nothing applies (offline; tests and demos) | |

The CLI drivers use the subscription you already have, with no key: Edgent renders the context and the tool protocol into one prompt, runs `agy -p` or `claude -p`, and reads the reply back. They never pass `--dangerously-skip-permissions` or any flag that widens the host's permissions. Force a driver with `EDGENT_PROVIDER=<kind>` or `--provider`, and put keys in environment variables, not in files.

## Quick start

```bash
edgent init my-study                       # a study workspace: STUDY.md, sources/, book/, learner/, .edgent/
cd my-study
edgent sources add lecture.pdf https://example.org/notes.html talk.mp4
edgent chat                                # a long conversation over the sources, with citations that resolve
edgent run "write three exercises on eigenvalues from the sources, keyed and checked"
edgent build "a 150-page undergraduate book on linear algebra from these sources"
edgent serve                               # the local reader: chapters, chat about what is on screen, build and learner dashboards
```

Every answer about the sources cites a locator such as `textbook:p.12` or `lecture:t=00:14:32`; a locator that does not resolve is a defect the checks catch. What the sources do not cover is said to be beyond them.

Other commands: `edgent connectors` (Perplexity, Playwright, Firecrawl, PostHog, Composio and any MCP server; see Connectors below), `edgent status`, `edgent outline`, `edgent walkthrough` (the unknowns ledger), `edgent sessions`, `edgent trace <session>`, `edgent skills [agents]`, `edgent heartbeat`, `edgent eval evals/golden`, `edgent setup --host claude|agy|codex|cursor` and `edgent mcp`. Global options (`--workspace`, `--provider`, `--model`, `--effort low|medium|high`, `--mode`, `-v`) go before or after the command.

## The workspace

State lives in files you can read, grep and put under git:

```
STUDY.md              what this workspace is for; read first
sources/              raw/ (what you added), text/ (greppable views with [p.N] and [t=HH:MM:SS] locators), INDEX.md
book/                 outline.json (the feature list of a build), chapters/, build/ (walkthrough)
media/                films/, diagrams/, widgets/, audio/
assessments/          item sets with keys that were recomputed in a sandbox
learner/              mastery.json (spaced repetition), progress.json (streaks, badges), misconceptions.md
knowledge/house.md    the house style and decisions the builds keep to
evidence/             content-addressed records of every check that ran
unknowns.json         what could not be established, with the probe that would settle it
progress.md           the append-only log every session writes to
sessions/             transcripts (JSONL) and notes per conversation
.edgent/              settings.json, traces/, tool-results/, cache/, skills/ and agents/ overrides
```

A workspace can hold a codebase too: the same `read_file`, `edit_file`, `grep`, `glob`, `bash` and `run_python` tools work on it, under the same permission policy.

## Two modes

**Conversation.** `edgent chat`, `edgent run` and the web reader run one agent loop per turn: the context is assembled in cache-aware order (system prompt and behaviour spec, the role, the skills and subagent indexes, the workspace instructions, then the volatile orientation and the transcript), the model calls tools until it answers in prose, and the transcript is persisted before every model call. Long sessions are compacted in tiers (old tool results are cleared first, then blocks become notes, then notes roll up); anything verbatim is still recoverable with `session_recall`.

**Build.** `edgent build` plans an outline (an initializer session proposes units with objectives, dependencies and paths), shows it and asks before the workers start (`--plan-only` stops there, `--yes` skips the question), then runs ready units in worker sessions; one build at a time per workspace, guarded by `.edgent/build.lock`. A unit is done only when its obligations close with evidence: the code in it ran and produced the expected output, every citation resolves, the page renders and measures clean, prerequisites were introduced earlier. The gate, not the worker, moves a unit to `done`; a separate-context critic reviews new units and opens fix units for high-severity defects. Every status change is on disk and committed before the next step, so a crash or a closed laptop resumes from `outline.json`.

## Skills, agents, prompts

Pedagogy and craft live in Markdown, not in code. `edgent/skills/` holds sixteen skills (explaining, socratic tutoring, book authoring, course design, assessment design, mock interviews, learning sessions, deep research, Manim films, diagrams, interactive widgets, narration, reading levels, visual quality, the study workflow, connectors); the model sees a short index in every prompt and loads a body with the `skill` tool when it matches. `edgent/agents/` defines ten subagents (librarian, explorer, researcher, author, critic, examiner, filmmaker, illustrator, interviewer, observer), each with its own tool allowlist and effort; a subagent returns a summary, never its transcript. `edgent/prompts/` holds the system prompt, the behaviour spec and the role prompts. A workspace (`.edgent/skills`, `.edgent/agents`) or a user (`~/.edgent/`) overrides any of them by name.

## Settings

`.edgent/settings.json` in the workspace overrides `~/.edgent/settings.json`:

```json
{
  "mode": "ask",
  "permissions": {"allow": ["bash(pytest *)", "write_file(book/*)"], "deny": ["bash(rm *)"]},
  "budget": {"max_iterations": 60, "max_tool_calls": 300, "max_wall_s": 3600},
  "hooks": {"PreToolUse": [{"matcher": "bash", "command": "scripts/check-command.sh"}]},
  "heartbeat": {"times": ["07:30"], "prompt": "Review what is due and plan today's session.", "timezone": "Europe/Berlin"},
  "provider": {"kind": "openai", "model": "qwen3-32b", "base_url": "http://localhost:8000/v1", "api_key_env": "MY_KEY"}
}
```

Permission modes are `ask` (writes and commands need approval), `accept-edits` (writes inside the workspace go ahead, commands and the network ask), `auto` (everything but the network and deletes), `plan` (read-only) and `bypass`. `edgent chat` and `edgent run --interactive` ask on the console; `edgent run` without a console and the web reader have nobody to ask, so they use `accept-edits` unless a mode is configured. A `bash` call that asks for the network is judged as a network request. Writes into `.edgent/` (settings, hooks, skills, agents) always ask: the model never widens its own permissions. Deny patterns on `bash` match every command in a shell line, not only its start. Every ceiling is a named limit with a unit; a failure message says the effective value and the next action.

## Connectors

Connectors bring outside MCP servers into Edgent: their tools appear to the model as `mcp__<connector>__<tool>`, under the same permission policy as Edgent's own tools. Five ship as presets:

| Preset | What it adds | Runs | Needs |
|---|---|---|---|
| `perplexity` | web search with dates and citations, quick cited answers, reasoning, multi-source research reports | hosted (`https://api.perplexity.ai/mcp`), or local with `--variant local` | `PERPLEXITY_API_KEY` |
| `playwright` | a real browser driven through accessibility snapshots: navigate, click, type, fill forms, console, network, screenshots | local (`npx @playwright/mcp`), headless, isolated, pointed at the Chromium Edgent finds | Node.js 18+ |
| `firecrawl` | pages as clean Markdown (JavaScript included), site maps and crawls, web and paper search, document parsing | hosted (`https://mcp.firecrawl.dev/v2/mcp`), or `search` and `local` variants | nothing for the free tier; `FIRECRAWL_API_KEY` for crawl, map and agent |
| `posthog` | product analytics for a site or course: insights, SQL, dashboards, flags, experiments, errors, surveys | hosted (`https://mcp.posthog.com/mcp`), read-only and single-tool mode by default | `POSTHOG_API_KEY` (personal key, "MCP Server" preset) |
| `composio` | acting in 1000+ apps (Gmail, Drive, GitHub, Slack, Notion...) through seven meta-tools | hosted (`https://connect.composio.dev/mcp`), or a session URL with `--variant session` | `COMPOSIO_CONSUMER_KEY` |

```bash
export PERPLEXITY_API_KEY=pplx-...        # keys live in the environment, never in settings files
edgent connectors add perplexity          # writes a ${PERPLEXITY_API_KEY} reference, connects, lists the tools
edgent connectors add playwright --allow  # --allow: its tools run without asking (permissions.allow)
edgent connectors add firecrawl
edgent connectors                         # state of each: ready, needs a key, error
edgent connectors presets | test <name> | tools <name> | refresh | remove <name>
edgent connectors add mydocs --url https://example.com/mcp --header "Authorization=Bearer \${MY_TOKEN}"
edgent connectors add local-db --command npx --env "DB_URL=\${DB_URL}" -- -y some-mcp-server
```

The settings entry is small and holds no secret (`{"connectors": {"perplexity": {"preset": "perplexity"}}}`); `${VAR}` and `${VAR:-default}` are expanded when a connector starts, and every expanded value is redacted from errors and traces. Claude-style `mcpServers` entries pasted into `.edgent/settings.json` work too. A local server sees only the environment it needs (PATH, HOME, locale, proxy and CA settings) plus its own `env`, never your other keys.

How they behave:

- **Nothing starts at launch.** Tool lists are cached in `~/.edgent/cache/connectors/` (outside the workspace, because the annotations in them decide permissions); a server starts on its first call and stops when Edgent exits. A connector whose listing failed is tried again after five minutes in a long-running process. A connector that was never listed is listed before the first model call, bounded by a timeout.
- **Large tool sets are deferred.** A connector with more than eight tools (Firecrawl, Playwright, most custom servers) is listed by name only; the model loads the schemas it needs with `tool_search`, so a session does not pay for fifty schemas on every call. Set `"expose": "all"` or `"deferred"` per connector to override.
- **Permissions.** Read-only tools of the shipped presets run like `web_fetch` (allowed in build sessions, asked about in `ask` mode); everything else asks unless you allow it with a pattern such as `mcp__playwright` or `mcp__perplexity__perplexity_search`. Deny patterns work the same way (`"deny": ["mcp__composio"]`). Annotations from servers you configured yourself are not believed unless you set `"trust_annotations": true`.
- **Untrusted results.** Connector output is marked untrusted, images come back as attachments the model can see (saved under `.edgent/cache/connectors/`), large results are stored and previewed, and the `connectors` skill tells the model when each connector earns its cost, to cite pages only after adding them as sources, and to confirm before anything Composio or PostHog would send, post or change.
- **Limits.** A request carries at most 128 tool schemas (Edgent's own first); oversized connector schemas are shortened without dropping a parameter; redirects from a connector URL are refused, so a key never follows one to another host; `edgent setup` does not share Edgent's connectors with a host, which keeps its own.
- **Protocol.** The client speaks MCP over stdio and Streamable HTTP, opens with the `initialize` handshake (revisions 2024-11-05 to 2025-11-25, which every official server speaks today) and moves to the stateless 2026-07-28 revision when a server asks for it. It sends keys as headers and does not run OAuth; the deprecated HTTP+SSE transport is not supported.

## Hosting the tools elsewhere

`edgent setup --host claude` (or `agy`, `codex`, `cursor`) registers `edgent mcp` as an MCP server for that host and installs the skills and agents where the host looks for them, keeping backups. The host's own loop then drives Edgent's tools: ingestion, citations, checks, films, assessments, the learner model, with guided prompts (`build_book`, `tutor_session`, `mock_interview`, `make_assignments`, `audit_unknowns`, `deep_research`). `edgent doctor --host claude` verifies the registration.

## Verification

Nothing is believed because the model said so. Code runs in a sandbox (resource limits, scrubbed environment, no network unless allowed) and its output is compared with the expected block. Citations are resolved against the source text. Pages are rendered in Chromium and measured (overflow, contrast, text size, broken images) before anyone looks at them. Films are compiled, rendered at draft quality, measured and looked at frame by frame. Assessment keys are recomputed. Every check writes an evidence record with its confidence (observed, derived, inferred) and authority (source, controlled replay, external service, historical reference, model inference); a judge's opinion is advisory and never closes an obligation that a replay could close.

## Development

```bash
python3 -m pytest -q                    # offline, deterministic; skips say which toolchain was absent
ruff check edgent tests
edgent eval evals/golden --label local  # golden scenarios through the real loop and the evaluation gate
```

Layout and the dependency direction (`domain` imports nothing of ours; `application` imports `domain`; `tools` and `adapters` import both; `ui` imports everything) are in `CLAUDE.md`, with the conventions every tool and limit follows.
