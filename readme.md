# Edgent's general purpose harness

A plain-Python (standard library only, Python 3.10+) implementation of the build plan in
[`docs/archive/TUTORIAL_general_purpose_harness.md`](docs/archive/TUTORIAL_general_purpose_harness.md) (superseded by [`docs/HARNESS_ENGINEERING_PRIMER.md`](docs/HARNESS_ENGINEERING_PRIMER.md) and [`docs/EDU_HARNESS_PLAN.md`](docs/EDU_HARNESS_PLAN.md)): a production-style
multimodal agent harness around a self-hosted model served through an OpenAI-compatible API
(vLLM or SGLang running Qwen3.8-27B on an H100 is the target, but any compatible server works).

The model is the engine; this package is everything around it: ingestion of text, image, video and
audio into citable artifacts, effort budgeting, planning into a persistent task DAG, parallel
sub-agents with code-enforced tool permissions, verification against real evidence, a trust gate,
four-tier memory, tracing, evaluation, and crash-resume.

There are **no third-party dependencies**. `ffmpeg`/`ffprobe` and `tesseract` are optional sidecars
discovered at runtime; when they are missing the harness says so instead of guessing.

## Quick start

```bash
# run the test suite (100 tests, mock model, no GPU needed)
python -m unittest discover -s tests -t .

# ask a question about a file with the offline mock backend
python -m harness run "What was the revenue in Q3?" --file report.txt --effort low

# against a real OpenAI-compatible server
python -m harness --backend openai --base-url http://127.0.0.1:8000/v1 --model Qwen3.8-27B \
    run "When does the speaker first mention pricing, and what does the slide show?" --file talk.mp4

# other commands
python -m harness runs                 # list runs
python -m harness resume               # re-attach to every in-flight run (after a crash / new Colab session)
python -m harness trace RUN_ID         # replay the execution trace
python -m harness tools                # MCP-shaped tool manifest
python -m harness health               # model server, sidecars, state
python -m harness eval evals/golden_text.json      # golden set as CI, appends a row to evals/results.md
python -m harness ablate evals/golden_text.json    # with-and-without each layer
python -m harness memory list|search|forget --user U
```

Every run returns an answer, the claims with their citations, the trust-gate verdict
(`verified`, `partially_verified` or `failed`), the resolved evidence bundle, the declared gaps,
the cost report and the path of the JSONL trace. A failed run returns an honest failure report
instead of a confident guess.

Configuration comes from a JSON file (`--config`), `HARNESS_*` environment variables
(`HARNESS_BASE_URL`, `HARNESS_MODEL`, `HARNESS_API_KEY`, `HARNESS_STATE_DIR`, `HARNESS_EFFORT`,
`HARNESS_BACKEND`) or CLI flags. State (SQLite databases, artifacts, traces, workspaces, memory)
lives under `--state-dir` (default `.harness`); on Colab point it at mounted Drive.

## Layout (tutorial §16)

| Package | Tutorial | What it holds |
|---|---|---|
| `harness/core` | §4.5 | Typed schemas (messages, content parts, tool calls, plans, verdicts, budgets), `OpenAICompatibleClient` (urllib, retries, request IDs, token accounting, Qwen thinking switch, reasoning separation), `MockModelClient`, effort profiles, config, clock/ID utilities |
| `harness/serving` | §4.2–4.4 | vLLM/SGLang launch commands, crash-restarting supervisor that resumes in-flight runs, warm-up health checks (text, image, tool call, long context), environment manifest |
| `harness/modality` | §5 | Content-addressed artifact store with derived views and provenance; type sniffing by magic bytes; text/document, image (header parsing, preview/crop via ffmpeg, OCR sidecar), video (probe, scene detection, hierarchical keyframes, audio track), audio (normalise, silence-based splitting, STT sidecar); modality budgeter; prompt index with untrusted-data delimiters |
| `harness/cognition` | §6, §9.7, §10.2 | Difficulty estimator and effort escalation, thinking controller with budget forcing, self-consistency voting, best-of-N, sample diversification, coverage curves |
| `harness/orchestration` | §7, §8.1, §11, §13 | Persistent task DAG (SQLite, enforced transitions, stale marking, splice), planner with validation and fallback, classifier, roles with tool allowlists, OODA/ReAct sub-agent with output discipline and Reflexion, parallel scheduler with per-node checkpoints, master loop with resume |
| `harness/tools` | §8.5, §12.4 | Registry (JSON-schema validation, permission classes, timeouts, idempotency store, result compaction, typed errors, MCP export) and 26 starter tools |
| `harness/sandbox` | §12.1 | Subprocess sandbox (rlimits, scrubbed env, optional network namespace), per-run workspace with escape protection, persistent interpreter with snapshot/restore |
| `harness/verification` | §8.4, §11.4, §12.2 | Definition-of-done contract, verifiers (evidence relocation, numbers grounded, schema/linter, consistency, policy, recomputation, tests, grounded model critic, judge), trust gate and evidence bundle |
| `harness/memory` | §12.3 | Working memory with compaction, episodic store, bi-temporal semantic facts with gated writes, BM25 + optional vector hybrid retrieval, versioned procedural memory |
| `harness/observability` | §11.3, §14.1 | Replayable JSONL traces, metrics (latency percentiles, cache-hit rate, tool error rate), cost accounting, structured redacted logs |
| `harness/evals` | §14.2 | Golden-set runner with results table, adversarial cases, ablation runner |
| `harness/prompts` | §10.2 | Versioned behaviour spec and role prompts, cache-aware ordering |
| `harness/notebooks` | §4.1 | Colab launcher (mount Drive, start engine under supervisor, health check, resume) |

## How a run works (§13.2)

1. Files are ingested by content type into the artifact store; each gets derived text views (OCR,
   transcript, keyframe timeline, chunks) and is indexed for retrieval.
2. The task is classified (convergent / divergent / exploratory / structural) and an effort level is
   chosen (user override, classifier hint, or heuristic). The effort level fixes thinking budget,
   samples, verification passes, tool iterations, parallelism and the run budget.
3. A machine-checkable definition of done is written.
4. The planner emits a dependency graph, validated for cycles, unknown tools and unknown artifacts
   (a single-step fallback is used if the model's plan is invalid twice).
5. Ready nodes run in parallel as sub-agents. Each node's output is verified (schema, cited evidence
   resolved against real artifact content, numbers grounded, model critic at higher effort). Failed
   nodes trigger a Reflexion lesson, stale-marking of dependants and a bounded replan.
6. Results are synthesised, optionally self-probed, and passed through the trust gate against the
   contract. A failed gate escalates effort (bounded) and replans with the lessons learned.
7. Verified claims are committed to semantic memory with provenance; verified plans are saved as
   procedural memory. Everything is traced; state is checkpointed after every node so `resume`
   continues from the stored graph after a crash.

## Plugging in sidecars

`harness.modality.sidecars.Sidecars` holds replaceable callables: `ocr(path)`, `stt(path)`,
`caption(path)`, `embed(texts)`, `audio_tags(path)`. Defaults: tesseract CLI for OCR when present,
nothing for the others. `openai_transcription(base_url, ...)` builds an STT sidecar for any
OpenAI-compatible `/audio/transcriptions` endpoint (for example a Whisper server on the same GPU).

```python
from harness.core.config import HarnessConfig
from harness.modality.sidecars import Sidecars, openai_transcription
from harness.orchestration import Harness

side = Sidecars(stt=openai_transcription("http://127.0.0.1:8001/v1", model="whisper-large-v3"))
h = Harness(HarnessConfig.load("config.json"), backend="openai", sidecars=side)
result = h.run("Summarise decisions and action items.", files=["meeting.wav"])
print(result.status, result.answer, result.gaps)
```

Model capabilities from the §2 audit go in `HarnessConfig.capabilities` (`native_image`,
`native_audio`, `native_video`, `thinking_switch`, `native_tool_calls`, `context_length`,
`image_token_cost`); the client, budgeter and agent loop branch on them.

## What is and is not covered

- The complete architecture runs end to end offline against the mock model, including kill-and-resume,
  replanning, effort escalation, prompt-injection containment and honest failure.
- The real-model path (`--backend openai`) is implemented but was not exercised against a live
  Qwen server here; run `python -m harness health` and the M0 capability audit first.
- PDF and office document text extraction needs a converter sidecar (not bundled); such files are
  ingested and reported as "no text extractor configured" rather than silently skipped.
- Network isolation for the sandbox uses `unshare -n` when available and falls back to a scrubbed
  environment otherwise; use a container for hard isolation in production.
