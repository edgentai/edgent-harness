# Edgent Harness

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Dependencies: Zero](https://img.shields.io/badge/dependencies-stdlib%20only-brightgreen.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](pyproject.toml)

**The model is the engine. `edgent-harness` is everything around it.**

`edgent-harness` is a **general-purpose, zero-dependency Python agent runtime** built entirely on the Python standard library. It ships two complementary runtimes over a shared tool, sandbox, modality, and observability core:

1. **Interactive Session Runtime (`AgentLoop`)** — a Claude Code / Antigravity-style autonomous loop with native tool calling, real-time SSE streaming, multi-breakpoint prefix caching, sandboxed `bash` & background tasks, surgical file editing, web tools, bidirectional MCP (client & server), isolated fresh-context subagents, hierarchical `AGENTS.md` / `SKILL.md` discovery, LLM-backed compaction, per-turn checkpointing & `/undo`, HITL pauses, a streaming terminal REPL, a local HTTP/SSE daemon, and a periodic heartbeat scheduler.
2. **Verified Multimodal DAG Runtime (`Harness`)** — a crash-resumable SQLite task-DAG engine for multimodal ingestion (text, image, audio, video) into content-addressed citable artifacts, effort escalation, evidence verification, trust gating, four-tier memory, and golden-set / ablation evaluations.

No third-party Python packages are required (`dependencies = []`). Optional system binaries (`ffmpeg`/`ffprobe`, `tesseract`) are detected at runtime and reported honestly when absent.

---

## Architecture

```
                         ┌─────────────────────────────────────────┐
                         │              Interfaces                 │
                         │  Terminal REPL │ HTTP+SSE │ CLI │ MCP   │
                         └────────────────────┬────────────────────┘
                                              │
                 ┌────────────────────────────┴────────────────────────────┐
                 ▼                                                         ▼
┌────────────────────────────────────────┐      ┌────────────────────────────────────────┐
│     Interactive Runtime (AgentLoop)    │      │   Verified DAG Runtime (Harness)       │
│ • Hierarchical AGENTS.md / SKILL.md    │      │ • Multimodal ingestion & views         │
│ • Streaming + 3-breakpoint KV cache    │      │ • SQLite task DAG + crash resume       │
│ • Fresh-context subagents              │      │ • Evidence verifiers & trust gate      │
│ • LLM compaction + grep session_recall │      │ • 4-tier memory + Reflexion            │
│ • Per-turn checkpoints & undo()        │      │ • Golden evals & ablation suite        │
│ • HITL request_review + Heartbeat      │      │                                        │
└───────────────────┬────────────────────┘      └───────────────────┬────────────────────┘
                    └───────────────────────┬───────────────────────┘
                                            ▼
      ┌───────────────────────────────────────────────────────────────────────────┐
      │                    Providers, Tools & Safety Boundary                     │
      │ • Providers: OpenAICompatibleClient │ AnthropicMessagesClient │ Router    │
      │ • Tools: read_file (lines+images), edit_file, bash (+bg tasks),           │
      │          run_python, web_fetch (SSRF guard), web_search, MCPClient        │
      │ • Enforcement: JSON-schema validation, PermissionPolicy (ask/accept/auto/ │
      │                plan + glob allow/deny), Pre/Post hooks, UNTRUSTED_DATA    │
      │ • Sandbox: Workspace escape guard, scrubbed env, unshare -n / sandbox-exec│
      └───────────────────────────────────────────────────────────────────────────┘
```

---

## Key Capabilities

- **Multi-Provider, Streaming & Prefix Caching**
  - **OpenAI-compatible** (`/v1/chat/completions` for OpenAI, vLLM, SGLang, Ollama) and **Anthropic Messages API** (`/v1/messages`) over stdlib `urllib` with retries, typed `StreamEvent` SSE streaming, and per-stage/per-role `ModelRouter`.
  - **3-breakpoint Anthropic prompt caching** (`system`, last `tool`, latest `user` turn) paired with immutable per-session workspace orientation for high KV-cache hit rates.
  - Optional **sentinel/CDATA `<edgent_call>` text protocol** (`--text-tools`) and single-turn history flattening (`--single-turn-bridge`) for non-native-tool CLI bridges, preserving LaTeX, HTML, and SVG verbatim.
- **Claude-Code-Class General Toolset**
  - **Files**: line-range & multimodal `read_file` (attaches PNG/JPEG/GIF/WebP/BMP as native image blocks), surgical `edit_file` (exact/whitespace-tolerant anchor or `¶n` paragraph replacement with closest-match diagnostics), `write_file`, `delete_file`, `glob_files`, `search_files`, `read_numbered`.
  - **Execution**: sandboxed `bash` (persistent `cwd`, scrubbed env, Linux `unshare -n` / macOS `sandbox-exec` network isolation, process-group timeout kill, automatic output offloading to `.edgent/tool_outputs/`), background process tools (`task_status`, `task_output`, `task_stop`), `run_python`, `run_tests`, `lint_code`, `calculate`.
  - **Web**: `web_fetch` (SSRF private/loopback guard, stdlib HTML-to-text, `.edgent/web_cache/` persistence) and `web_search` — all tagged `UNTRUSTED_DATA`.
  - **Bidirectional MCP (JSON-RPC 2.0 over stdio)**: expose Edgent tools to external hosts via `harness mcp`, or mount any external stdio MCP server dynamically onto Edgent via `MCPClient` (`--mcp-server`).
- **Long-Horizon Reliability & Human-in-the-Loop**
  - **Isolated fresh-context subagents**: built-in `explore`, `coder`, `reviewer`, `planner`, and `researcher` roles (extensible via `.md` files in `.edgent/agents/`), with `require_tool_use` grounding guards on inspection roles.
  - **Memory without vectors**: verbatim JSONL transcripts, LLM-backed structured compaction into `notes.md`, turn-aware micro-compaction for single-turn bridges, and regex/keyword `session_recall`.
  - **Runtime guardrails**: doom-loop detection, truncation auto-continuation (`length`/`max_tokens`), mid-turn cancellation (`threading.Event`), per-turn content-addressed file checkpoints with `undo()`, and clean cross-process `request_review` pause/resume.
- **Built-In Interfaces**
  - **Interactive Terminal REPL** (`harness repl`): live token streaming, `[y/N/a]` permission prompts, Ctrl-C turn cancellation, and slash commands (`/mode`, `/undo`, `/checkpoints`, `/compact`, `/recall`, `/tasks`, `/skills`, `/subagents`, `/approve`, `/reject`, `/thinking`).
  - **Local HTTP + SSE Daemon** (`harness serve`): `127.0.0.1` REST + real-time `text/event-stream` API with optional Bearer auth and an OpenClaw-style periodic `HeartbeatScheduler` (`--heartbeat`).

---

## Quick Start

### 1. Clone & Verify (Zero Dependencies, Offline)

```bash
git clone https://github.com/edgentai/edgent-harness.git
cd edgent-harness

# Optional editable install (exposes the `harness` console script)
pip install -e . --no-build-isolation

# Run the full test suite (120 tests, offline, ~5 seconds)
python3 -m unittest discover -s tests -t .

# Check local binaries and sandbox isolation support
python3 -m harness doctor
```

### 2. Interactive Streaming Terminal REPL

**With the Anthropic API (native tools, SSE streaming, 3-breakpoint prompt caching):**

```bash
python3 -m harness \
  --backend anthropic \
  --base-url https://api.anthropic.com \
  --model claude-opus-4-6 \
  --api-key "$ANTHROPIC_API_KEY" \
  repl --workspace /path/to/your/project --mode ask
```

**With OpenAI / vLLM / SGLang:**

```bash
python3 -m harness \
  --backend openai \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen3-Coder-30B \
  repl --workspace /path/to/your/project --mode accept_edits
```

**Attach external MCP servers into the REPL:**

```bash
python3 -m harness \
  --backend anthropic --base-url https://api.anthropic.com --model claude-sonnet-4-6 --api-key "$ANTHROPIC_API_KEY" \
  repl --workspace . --mcp-server "docs=python3 -m harness mcp --workspace ../other-repo --mode plan"
```

### 3. Single-Turn Persistent CLI (`chat`)

```bash
# Send a turn (session state persists under <workspace>/.edgent/)
python3 -m harness --backend anthropic --base-url https://api.anthropic.com --model claude-sonnet-4-6 \
  --api-key "$ANTHROPIC_API_KEY" \
  chat "Audit auth.py for edge cases and add unit tests" --workspace . --session feat-auth --stream

# Resume a paused request_review or rewind the last turn's file edits
python3 -m harness chat "Looks good, proceed" --workspace . --session feat-auth --approve
python3 -m harness chat --workspace . --session feat-auth --undo 1
```

### 4. Local HTTP + SSE Streaming Daemon (`serve`)

```bash
python3 -m harness \
  --backend anthropic --base-url https://api.anthropic.com --model claude-sonnet-4-6 \
  --api-key "$ANTHROPIC_API_KEY" \
  serve --workspace /path/to/project --port 8765 --auth-token "$EDGENT_TOKEN"
```

Stream events from any client:

```bash
curl -N -X POST http://127.0.0.1:8765/v1/sessions/demo/messages \
  -H "Authorization: Bearer $EDGENT_TOKEN" \
  -H "Accept: text/event-stream" \
  -H "Content-Type: application/json" \
  -d '{"message": "Summarise the repository structure and run tests", "stream": true}'
```

Endpoints exposed: `GET /healthz`, `GET /v1/tools`, `GET /v1/skills`, `GET /v1/subagents`, `GET /v1/tasks`, `GET /v1/sessions`, `GET /v1/sessions/{id}`, `POST /v1/sessions/{id}/messages`, `/review`, `/cancel`, `/undo`, `/compact`, `GET /v1/heartbeat`, `POST /v1/heartbeat/{job_id}/trigger`.

### 5. Host Mode: Stdio MCP Server (`mcp`)

Expose Edgent's workspace tools, permission policy, and workflow prompts to Claude Code, Antigravity, Cursor, or any MCP client:

```bash
python3 -m harness mcp --workspace /path/to/project --mode accept_edits --compact
```

### 6. Structured Multimodal DAG Runs & Evaluations

```bash
# Run a multimodal evidence-verified task (offline mock or --backend openai|anthropic)
python3 -m harness run "What was the Q3 revenue?" --file report.txt --effort low

# Inspect runs, crash-resume, replay traces, run golden evals & ablations
python3 -m harness runs
python3 -m harness resume
python3 -m harness trace RUN_ID
python3 -m harness eval evals/golden_text.json
python3 -m harness ablate evals/golden_text.json
```

---

## Python SDK

### Interactive `AgentLoop` with Per-Role `ModelRouter` & Hooks

```python
from harness.core.providers import AnthropicMessagesClient, ModelRouter
from harness.sandbox.workspace import Workspace
from harness.session import AgentLoop, HookManager, PermissionMode, PermissionPolicy

fast = AnthropicMessagesClient("https://api.anthropic.com", "claude-sonnet-4-6", api_key="...")
deep = AnthropicMessagesClient("https://api.anthropic.com", "claude-opus-4-6", api_key="...", effort="high")

router = ModelRouter(fast)
router.set_route("reviewer", deep, effort="high", temperature=0.0)
router.set_route("planner", deep, effort="high")

policy = PermissionPolicy(
    mode=PermissionMode.ACCEPT_EDITS,
    deny=["bash(rm -rf*)", "write_file(.env*)"],
)

loop = AgentLoop(workspace=Workspace.open("./my-project", create_subdirs=False), client=router, policy=policy)
try:
    res = loop.send("session-1", "Implement caching in cache.py, run tests, and have `reviewer` audit the diff.")
    print(res.status, res.reply)
finally:
    loop.close()
```

### Multimodal DAG `Harness` with Optional Sidecars

```python
from harness.core.config import HarnessConfig
from harness.modality.sidecars import Sidecars, openai_transcription
from harness.orchestration import Harness

sidecars = Sidecars(stt=openai_transcription("http://127.0.0.1:8001/v1", model="whisper-large-v3"))
h = Harness(HarnessConfig.load("config.json"), backend="openai", sidecars=sidecars)
result = h.run("Summarise key decisions and action items.", files=["meeting.wav"])
print(result.status, result.answer, result.gaps)
```

---

## Customization & Extensibility

`edgent-harness` keeps its core strictly domain-agnostic and discovers project configuration automatically:

- **Project instructions (hierarchical)**: automatically loads `~/.edgent/AGENTS.md` → `<workspace>/AGENTS.md` / `CLAUDE.md` / `.edgent/AGENTS.md` → any `--context-file` arguments.
- **Skills (progressive disclosure)**: drop `<name>/SKILL.md` (with YAML front matter `name` and `description`) or `<name>.md` into `~/.edgent/skills/` or `<workspace>/.edgent/skills/` (or pass `--skills-dir`).
- **Custom subagents**: drop `<name>.md` files (front matter: `name`, `description`, `tools`, `model_role`, `max_turns`, `skills`) into `~/.edgent/agents/` or `<workspace>/.edgent/agents/` (or pass `--agents-dir`).
- **Permission modes**:
  - `plan`: strictly read-only + interactive.
  - `ask`: prompts for approval on `write`, `delete`, and `network` tools.
  - `accept_edits`: auto-approves workspace edits and `bash`; prompts on `delete` and `network`.
  - `auto`: auto-approves everything not matched by `policy.deny`.
- **Environment variables**: `HARNESS_BACKEND`, `HARNESS_BASE_URL`, `HARNESS_MODEL`, `HARNESS_API_KEY`, `HARNESS_STATE_DIR`, `HARNESS_EFFORT`, `CODE_RUNNER_PYTHON`.

---

## Repository Layout

| Package | Purpose |
|---|---|
| `harness/core` | Typed schemas (`Message`, `ContentPart`, `ToolCall`, `Usage`, `Plan`, `Verdict`), `OpenAICompatibleClient` & `AnthropicMessagesClient` (SSE streaming, prompt caching, retries), `ModelRouter`, `MockModelClient`, sentinel/CDATA `<edgent_call>` protocol, config |
| `harness/session` | `AgentLoop`, `SessionStore` (JSONL transcripts, LLM compaction, grep `recall`), `CheckpointManager` (per-turn file snapshots & `undo`), `PermissionPolicy` & `HookManager`, `SkillRegistry`, `SubagentRegistry`, `HeartbeatScheduler`, `run_doctor` |
| `harness/tools` | `ToolRegistry` (JSON-schema validation, permissions, hooks, idempotency, compaction, parallel read dispatch), general tools (`read_file`, `edit_file`, `bash` + background tasks, `web_fetch`, `web_search`, `todo_manage`, `request_review`, `session_recall`, `spawn_subagent`), stdio `MCPServer` & `MCPClient`, starter multimodal/code tools |
| `harness/ui` | `TerminalREPL` (streaming CLI with slash commands & Ctrl-C cancellation) and `HarnessServer` (stdlib `ThreadingHTTPServer` with REST + real-time SSE) |
| `harness/sandbox` | `Workspace` (path-escape protection, atomic writes, glob/search), subprocess sandbox (rlimits, scrubbed env, Linux `unshare -n` / macOS `sandbox-exec`), persistent `InterpreterSession` |
| `harness/modality` | Content-addressed `ArtifactStore`, magic-byte sniffing, image/video/audio adapters (`ffmpeg`/`ffprobe`), pluggable OCR/STT/caption/embedding `Sidecars`, modality budgeter |
| `harness/orchestration` | Persistent SQLite task DAG, classifier, planner, OODA/ReAct executor, parallel scheduler, crash-resumable `Harness` master loop |
| `harness/cognition` | Effort estimator & escalation, thinking budget control, self-consistency, best-of-N |
| `harness/verification` | Definition-of-done contracts, evidence relocation, numeric grounding, schema/test/critic verifiers, trust gate |
| `harness/memory` | Working, episodic, bi-temporal semantic, and versioned procedural memory with BM25 hybrid retrieval |
| `harness/observability` | Replayable JSONL traces, latency/cache/error metrics, cost accounting, redacted logs |
| `harness/evals` | Golden-set runner, adversarial suites, layer ablation runner |
| `harness/serving` | vLLM/SGLang launch helpers, crash-restarting supervisor, warm-up health checks |

Design references live in [`docs/HARNESS_ENGINEERING_PRIMER.md`](docs/HARNESS_ENGINEERING_PRIMER.md).

---

## Security Model & Current Limitations

- **Untrusted data isolation**: File reads, shell outputs, web fetches, and artifact views are wrapped in explicit `<<<UNTRUSTED_DATA>>>` delimiters; `web_fetch` blocks loopback, private, link-local, and reserved IPs by default.
- **Sandbox boundaries**: When `allow_network=False` (default), `bash` and `run_python` run with scrubbed credentials and network isolation via Linux `unshare -n` or macOS `sandbox-exec` (writes restricted to the workspace). For untrusted multi-tenant workloads in production, run inside a container or microVM.
- **Optional media sidecars**: PDF/Office extraction, OCR, and STT rely on external converters/sidecars (`tesseract`, Whisper endpoint, etc.) rather than bundled binaries; missing capabilities are reported explicitly rather than guessed.
- **Constrained decoding**: OpenAI/vLLM `response_format` is wired for structured outputs; `strict: true` grammar-constrained tool schemas and Anthropic `output_config.format` are not yet enabled.

---

## License

MIT
