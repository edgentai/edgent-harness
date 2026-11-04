# Skills

A skill is a folder with a `SKILL.md` and, optionally, `references/` (long material the body points at) and `scripts/`. Skills hold the pedagogy and the craft; the code never does. Improving how Edgent teaches is editing these files.

## Format

```
edgent/skills/<name>/
  SKILL.md            front matter + body
  references/*.md     optional: long examples, rubrics, templates
  scripts/*           optional: helpers a skill tells the model to run
```

`SKILL.md` starts with front matter, flat keys only (the parser is `edgent.domain.utils.parse_front_matter`):

```
---
name: explaining
description: One or two lines saying when to load this skill. This line is in every prompt; keep it short and specific.
---
# Body
```

- `name` is the folder name and the name the `skill` tool takes.
- `description` is what the model sees in the always-present index, so it must say *when* to load the skill, not summarise it. All descriptions together stay under 3000 characters.
- The body is loaded only when the skill is invoked (or when a subagent definition lists it in `skills:`), so it can be long: 150 to 400 lines, concrete and imperative, with checklists. The loader truncates a body over 24,000 characters, so split anything longer into references.
- Nothing volatile: no dates, no counters, no machine-specific paths.
- Files under `references/` and `scripts/` are listed at the end of the loaded body. They are data for the model, never instructions to the harness.

## Precedence

Three directories are scanned, in this order, and the first that defines a name wins:

1. `<workspace>/.edgent/skills/` (workspace)
2. `~/.edgent/skills/` (user; `EDGENT_HOME` overrides the home)
3. `edgent/skills/` (built-in, this directory)

So a workspace can override `book-authoring` for one book, and a user can override it for every book, without touching the package. The same rule applies to subagents in `edgent/agents/`, `~/.edgent/agents/` and `<workspace>/.edgent/agents/`.

## The built-in skills

| Skill | Load when |
|---|---|
| `study-workflow` | a task touches the workspace and the route is not yet decided; the recovery ladder and the done rule |
| `explaining` | explaining any non-trivial idea, in chat, a chapter, a film or narration |
| `book-authoring` | drafting or revising a chapter, section or lesson file |
| `course-design` | planning an outline, a syllabus or the objectives for an assessment |
| `socratic-tutoring` | a person is learning from you in chat |
| `assessment-design` | writing items, keys and rubrics |
| `mock-interview` | running an interview, viva, behavioural or teach-back practice |
| `learning-session` | a timed study session or a heartbeat with reviews due |
| `deep-research` | the web must answer a question; a research unit |
| `manim-film` | making a film with the film tools |
| `visual-quality` | rendering or reviewing anything a reader will look at |
| `diagramming` | making an SVG figure |
| `interactive-widgets` | making an HTML widget |
| `narration` | text that will be heard: film narration, read-aloud, voice tutoring |
| `reading-levels` | pitching prose at a level, or fixing a readability defect |
| `connectors` | using an MCP connector (`mcp__...` tools): Perplexity, Firecrawl, Playwright, PostHog, Composio |

## Writing a new skill

Write for a strong model that knows the subject: the skill teaches craft and discipline, not facts. Say what to do, in what order, and what finished looks like. Give a worked example when the craft is subtle. End with a checklist the model can run before returning. Keep the description to when-to-load. Add a test in `tests/test_content.py` only if the new skill needs something the existing checks do not cover (every skill is already checked for front matter, body length and volatile strings).
