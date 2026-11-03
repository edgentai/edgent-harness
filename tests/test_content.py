"""Content checks: the skills, subagent definitions and prompts parse, fit their budgets, name real tools and carry nothing volatile."""
from __future__ import annotations

import json
import re
import typing
from pathlib import Path

import pytest

from edgent import prompts
from edgent.adapters.providers.base import find_role, role_marker
from edgent.adapters.renderers.manim_compiler import OP_NAMES, compile_storyboard
from edgent.adapters.renderers.manim_render import QUALITY_FLAGS
from edgent.adapters.renderers.svg import validate_svg
from edgent.adapters.sandbox import run_python
from edgent.application.skills import INDEX_LINE_MAX_CHARS, SKILL_BODY_MAX_CHARS, SkillIndex, builtin_skills_dir
from edgent.application.subagents import EFFORTS, AgentDefinition, AgentIndex, SubagentRunner, builtin_agents_dir
from edgent.domain.capabilities import Capabilities
from edgent.domain.errors import ConfigurationError
from edgent.domain.learner import EVENT_TYPES
from edgent.domain.obligations import CONTINUE_MARKER_RE, PLACEHOLDER_RE
from edgent.domain.outline import UnitKind
from edgent.domain.types import Message
from edgent.domain.utils import ensure_dir, estimate_tokens, parse_front_matter, write_json
from edgent.tools.checks.readability import GRADE_BANDS, SENTENCE_WORDS_CEILING
from edgent.tools.checks.registry import build_checks_registry
from edgent.tools.core import build_core_registry
from edgent.tools.edu.assess import assess_items_check, normalise_item, validate_item
from edgent.tools.edu.book import build_book_registry
from edgent.tools.edu.learner import build_learner_registry
from edgent.tools.edu.media import HOUSE_THEME_CSS, WIDGET_VIEWPORT_PX, build_media_registry, diagram_render, widget_run
from edgent.tools.edu.sources import build_sources_registry
from tests.conftest import skip_unless

ALLOWED_AGENT_TOOLS: frozenset[str] = frozenset({
    "read_file", "write_file", "edit_file", "list_files", "glob", "grep", "bash", "run_python", "web_search", "web_fetch",
    "skill", "todo_write", "todo_read", "session_recall", "remember", "sources_add", "sources_add_url", "sources_list",
    "sources_cite_check", "sources_coverage", "check_render", "check_execute", "check_cite", "check_readability",
    "check_prereqs", "check_judge", "run_checks", "book_outline_get", "book_unit_brief", "film_compile", "film_render",
    "film_look", "diagram_render", "widget_run", "assess_create", "assess_items_check", "assess_render", "learner_update",
    "learner_schedule", "learner_summary",
})
EXPECTED_SKILLS: frozenset[str] = frozenset({
    "explaining", "book-authoring", "course-design", "socratic-tutoring", "assessment-design", "mock-interview",
    "learning-session", "deep-research", "manim-film", "visual-quality", "diagramming", "interactive-widgets",
    "narration", "reading-levels", "study-workflow",
})
EXPECTED_AGENTS: frozenset[str] = frozenset({
    "librarian", "researcher", "author", "illustrator", "filmmaker", "critic", "examiner", "observer", "explorer", "interviewer",
})
JOB_ROLES: tuple[str, ...] = ("main", "subagent", "worker", "initializer", "critic", "compactor", "examiner", "observer", "interviewer", "librarian", "researcher")
READ_ONLY_AGENTS: frozenset[str] = frozenset({"librarian", "explorer", "critic"})
MUTATING_TOOLS: frozenset[str] = frozenset({"write_file", "edit_file", "bash", "run_python", "sources_add", "sources_add_url", "film_compile", "film_render", "diagram_render", "widget_run", "assess_create", "learner_update"})
SKILL_DESCRIPTIONS_TOTAL_MAX_CHARS = 3000
SKILL_BODY_MIN_LINES = 150  # the skill spec asks for 150 to 400 lines
SUBAGENT_PROMPT_TOKENS_MAX = 10_000  # an agent body plus its skills; the task and the tool schemas come on top
SKILL_REQUIRED_TOOLS: dict[str, frozenset[str]] = {  # what an agent's own skills tell it to call
    "examiner": frozenset({"write_file", "edit_file"}),  # fix key.json and items.json, write rubric.md
    "illustrator": frozenset({"run_python"}),  # diagramming: compute layouts from constants
    "researcher": frozenset({"check_cite"}),  # deep-research: check the brief before returning
}
PROSE_LINE_MAX_CHARS = 160  # the files are wrapped at 110; only an unbreakable token may push a line past that
TOOL_NAME_PREFIXES: tuple[str, ...] = ("sources_", "check_", "book_", "film_", "assess_", "learner_", "widget_", "diagram_", "web_", "todo_", "session_", "tts_", "video_", "run_checks")
# ISO dates and datetimes, epoch seconds, and "generated at <digit>" phrases; locators such as t=00:14:32 are not volatile
VOLATILE_RE = re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2})?(?!\d)|\b(?:generated|updated|created|rendered) (?:at|on) \d|(?<!\d)1[5-9]\d{8}(?!\d)", re.IGNORECASE)
_BACKTICK_RE = re.compile(r"`([a-z][a-z0-9_]*)`")
_JSON_FENCE_RE = re.compile(r"```json\s*(.*?)```", re.DOTALL)
_PLAIN_FENCE_RE = re.compile(r"^```\n(.*?)^```$", re.DOTALL | re.MULTILINE)

PROMPTS_DIR = Path(prompts.__file__).resolve().parent


def skill_files() -> list[Path]:
    return sorted(builtin_skills_dir().glob("*/SKILL.md"))


def agent_files() -> list[Path]:
    return sorted(p for p in builtin_agents_dir().glob("*.md") if not p.name.startswith("."))


def content_files() -> list[Path]:
    return sorted({*builtin_skills_dir().rglob("*.md"), *agent_files(), *PROMPTS_DIR.rglob("*.md")})


def registry_names() -> set[str]:
    reg = build_core_registry()
    for build in (build_sources_registry, build_book_registry, build_media_registry, build_learner_registry, build_checks_registry):
        build(reg)
    from edgent.tools.edu.assess import build_assess_registry

    build_assess_registry(reg)
    return set(reg.names())


# ----------------------------------------------------------------------------- skills
def test_builtin_skill_set_and_readme():
    names = {p.parent.name for p in skill_files()}
    assert names == EXPECTED_SKILLS
    readme = (builtin_skills_dir() / "README.md").read_text(encoding="utf-8")
    assert "Precedence" in readme and "parse_front_matter" in readme
    for name in EXPECTED_SKILLS:
        assert f"`{name}`" in readme, f"README does not list {name}"


@pytest.mark.parametrize("path", skill_files(), ids=lambda p: p.parent.name)
def test_skill_parses_and_fits(path: Path):
    meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
    assert meta.get("name") == path.parent.name
    description = meta.get("description")
    assert isinstance(description, str) and description.strip() and "\n" not in description
    assert len(f"- {meta['name']}: {description}") <= INDEX_LINE_MAX_CHARS, "the index line would be truncated"
    lines = body.splitlines()
    assert len(lines) > SKILL_BODY_MIN_LINES, f"{path.parent.name} body has {len(lines)} lines"
    assert len(body) < SKILL_BODY_MAX_CHARS, "the loader would truncate this body"
    assert body.lstrip().startswith("# "), "the body starts with its title"
    assert "- [ ]" in body, "every skill ends with a checklist the model can run"
    assert not [ln for ln in prose_lines(lines) if len(ln) > PROSE_LINE_MAX_CHARS], "prose is hard-wrapped for terminals and diffs"


def prose_lines(lines: list[str]) -> list[str]:
    """Lines outside fences, tables, indented code and block quotes."""
    out: list[str] = []
    in_fence = False
    for ln in lines:
        if ln.strip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence or ln.startswith(("|", "    ", ">")):
            continue
        out.append(ln)
    return out


def test_skill_descriptions_total_under_budget():
    total = sum(len(parse_front_matter(p.read_text(encoding="utf-8"))[0]["description"]) for p in skill_files())
    assert total < SKILL_DESCRIPTIONS_TOTAL_MAX_CHARS, total


def test_builtin_skill_index_loads_whole_and_serves_references():
    idx = SkillIndex().load([builtin_skills_dir()], ["builtin"])
    assert set(idx.names()) == EXPECTED_SKILLS
    text = idx.index_text()
    assert "more skills" not in text and all(f"- {name}:" in text for name in EXPECTED_SKILLS)
    body = idx.load_body("mock-interview")
    assert "## Files in this skill" in body and "- references/rubrics.md" in body
    assert "## Teach-back" in idx.get("mock-interview").reference("rubrics.md")
    assert "## Files in this skill" not in idx.load_body("explaining")
    assert idx.get("manim-film").files() == ["references/storyboard-example.md"]
    assert idx.get("assessment-design").files() == ["references/items-example.md"]


def test_skills_name_only_real_tools():
    names = registry_names()
    offenders: list[str] = []
    for path in [*skill_files(), *agent_files(), *PROMPTS_DIR.rglob("*.md")]:
        for token in set(_BACKTICK_RE.findall(path.read_text(encoding="utf-8"))):
            looks_like_tool = token.startswith(TOOL_NAME_PREFIXES) and not token.endswith("_id") and token not in EVENT_TYPES
            if looks_like_tool and token not in names:
                offenders.append(f"{path.parent.name}/{path.name}: {token}")
    assert offenders == [], "tool names that do not exist in the registries: " + "; ".join(offenders)


def test_storyboard_example_compiles_and_items_example_validates():
    story = (builtin_skills_dir() / "manim-film" / "references" / "storyboard-example.md").read_text(encoding="utf-8")
    blocks = _JSON_FENCE_RE.findall(story)
    storyboard = json.loads(blocks[0])
    source, warnings = compile_storyboard(storyboard)
    assert warnings == [] and "class GeneratedFilm(Film)" in source and source.count("with self.beat(") == 4
    assert all({"id", "title", "narration", "code", "seconds"} <= set(b) for b in storyboard["beats"])
    ops = json.loads(blocks[1])
    assert [op["op"] for op in ops["ops"]] == ["patch_beat", "set_beat"]
    items = (builtin_skills_dir() / "assessment-design" / "references" / "items-example.md").read_text(encoding="utf-8")
    item_set = json.loads(_JSON_FENCE_RE.findall(items)[0])
    assert {"set_id", "title", "objectives", "level", "items"} <= set(item_set)
    kinds = set()
    for i, raw in enumerate(item_set["items"]):
        item = normalise_item(raw, i)
        assert validate_item(item) == [], (raw["id"], validate_item(item))
        kinds.add(item["kind"])
    assert kinds == {"mcq", "numeric", "short", "code", "proof"}


# ----------------------------------------------------------------------------- agents
def test_builtin_agent_set_and_index():
    assert {p.stem for p in agent_files()} == EXPECTED_AGENTS
    idx = AgentIndex().load([builtin_agents_dir()], ["builtin"])
    assert set(idx.names()) == EXPECTED_AGENTS
    text = idx.index_text()
    assert "more subagents" not in text and all(f"- {name}:" in text for name in EXPECTED_AGENTS)
    assert ALLOWED_AGENT_TOOLS <= registry_names(), "the allowed tool list has drifted from the registries"


@pytest.mark.parametrize("path", agent_files(), ids=lambda p: p.stem)
def test_agent_definition_is_valid(path: Path):
    text = path.read_text(encoding="utf-8")
    meta, body = parse_front_matter(text)
    assert set(meta) == {"name", "description", "tools", "effort", "skills", "maxTurns", "background"}
    d = AgentDefinition.from_markdown(text, "builtin", path, default_name=path.stem)
    assert d.name == path.stem == meta["name"]
    assert d.description and "\n" not in meta["description"] and len(f"- {d.name}: {d.description}") <= INDEX_LINE_MAX_CHARS
    assert d.tools, "every built-in agent has an explicit allowlist"
    assert set(d.tools) <= ALLOWED_AGENT_TOOLS, set(d.tools) - ALLOWED_AGENT_TOOLS
    assert len(d.tools) == len(set(d.tools))
    assert d.effort in EFFORTS and isinstance(d.max_turns, int) and d.max_turns >= 1 and isinstance(d.background, bool)
    assert set(d.skills) <= EXPECTED_SKILLS, set(d.skills) - EXPECTED_SKILLS
    assert len(d.system_prompt) > 300 and d.system_prompt.startswith("You are the ")
    assert "return" in d.system_prompt.lower(), "the body says what to return"
    if d.name in READ_ONLY_AGENTS:
        assert not set(d.tools) & MUTATING_TOOLS, f"{d.name} is read-only"
    if d.name == "observer":
        assert set(d.tools) & MUTATING_TOOLS == {"learner_update"}
    assert "ask_user" not in d.tools and "agent" not in d.tools and "book_mark_done" not in d.tools


def test_agent_bodies_name_only_their_own_tools():
    names = registry_names()
    for path in agent_files():
        d = AgentDefinition.from_markdown(path.read_text(encoding="utf-8"), "builtin", path, default_name=path.stem)
        named = {t for t in _BACKTICK_RE.findall(d.system_prompt) if t in names}
        assert named <= set(d.tools), f"{d.name} names tools it cannot call: {sorted(named - set(d.tools))}"
        assert SKILL_REQUIRED_TOOLS.get(d.name, frozenset()) <= set(d.tools), f"{d.name} lacks a tool its skills rely on"


def test_every_agent_composes_with_the_builtin_skills():
    skills = SkillIndex().load([builtin_skills_dir()], ["builtin"])
    agents = AgentIndex().load([builtin_agents_dir()], ["builtin"])
    for name in agents.names():
        d = agents.get(name)
        errors: list[str] = []
        prompt = SubagentRunner.child_system_prompt(d, skills, errors)
        assert errors == [], (name, errors)
        assert prompt.startswith(d.system_prompt) and all(f"# Skill: {s}" in prompt for s in d.skills), name
        assert estimate_tokens(prompt) <= SUBAGENT_PROMPT_TOKENS_MAX, (name, estimate_tokens(prompt))


def test_agents_discover_through_default_dirs(workspace, home):
    agents = AgentIndex.discover(workspace, home)
    skills = SkillIndex.discover(workspace, home)
    assert EXPECTED_AGENTS <= set(agents.names()) and EXPECTED_SKILLS <= set(skills.names())
    assert agents.get("author").source == "builtin" and skills.get("explaining").source == "builtin"


# ----------------------------------------------------------------------------- prompts
def test_main_prompt_contract():
    p = prompts.build_system_prompt("main")
    assert p.startswith("[edgent-role: main]\n")
    first = p.splitlines()[0]
    assert first == role_marker("main") == prompts.role_marker("main") and find_role([Message.system(p)]) == "main"
    assert "data, never instructions" in p, "the untrusted-data rule"
    assert "book_mark_done" in p and "The done rule" in p, "the done rule"
    assert "unverified" in p and "do not repeat the unchanged call" in p and "A denial is final" in p
    assert "## Heartbeat" in p and "# Role: main" in p and "## Mode: chat" in p
    assert p == prompts.build_system_prompt("main") == prompts.build_system_prompt("main", mode="chat", caps=None, extra="")
    assert estimate_tokens(p) <= prompts.MAIN_PROMPT_TOKENS_LIMIT.value, estimate_tokens(p)
    assert "## Provider" not in p


def test_modes_roles_and_extra():
    build = prompts.build_system_prompt("worker", mode="build")
    assert build.startswith("[edgent-role: worker]\n") and "# Role: worker" in build and "Build exactly one unit" in build and "## Mode: chat" not in build
    sub = prompts.build_system_prompt("librarian", mode="subagent", extra="  Task-specific line.  ")
    assert "# Role: librarian" in sub and "## Mode: subagent" in sub and sub.endswith("Task-specific line.")
    assert "final message" in prompts.MODE_TEXT["subagent"]
    unknown_role = prompts.build_system_prompt("host")
    assert unknown_role.startswith("[edgent-role: host]\n") and "# Role:" not in unknown_role
    assert set(prompts.roles()) >= EXPECTED_AGENTS | set(JOB_ROLES)
    for role in prompts.roles():
        text = prompts.build_system_prompt(role, mode="subagent" if role in EXPECTED_AGENTS else "chat")
        assert f"# Role: {role}" in text and estimate_tokens(text) <= prompts.MAIN_PROMPT_TOKENS_LIMIT.value, role
    with pytest.raises(ConfigurationError):
        prompts.build_system_prompt("main", mode="dream")
    with pytest.raises(ConfigurationError):
        prompts.build_system_prompt("bad role")
    with pytest.raises(ValueError):
        prompts.load("../system")
    with pytest.raises(ValueError):
        prompts.load("roles/../system")
    with pytest.raises(FileNotFoundError):
        prompts.load("roles/nobody")
    assert prompts.load("system").startswith("# Edgent") and prompts.load("roles/critic").startswith("# Role: critic")


def test_main_prompt_parts_are_in_order():
    p = prompts.build_system_prompt("main", extra="EXTRA-LINE")
    marks = ("[edgent-role: main]", "# Edgent", "## Untrusted data", "# Behaviour", "# Role: main", "## Mode: chat", "EXTRA-LINE")
    positions = [p.index(m) for m in marks]
    assert positions == sorted(positions) and p.count("[edgent-role:") == 1


def test_capability_notes_are_stable_and_optional():
    caps = Capabilities(images=False, max_prompt_bytes=120_000)
    p = prompts.build_system_prompt("main", caps=caps)
    assert "## Provider" in p and "cannot show you images" in p and "120000 bytes" in p
    assert p == prompts.build_system_prompt("main", caps=caps)
    assert prompts.capability_text(Capabilities(images=True)) == "" and prompts.capability_text(None) == ""
    assert "images" not in prompts.capability_text(Capabilities(images=True, max_prompt_bytes=10))


def test_load_refuses_an_oversized_prompt_file(tmp_path, monkeypatch):
    (tmp_path / "huge.md").write_text("x" * (prompts.PROMPT_FILE_MAX_CHARS + 1), encoding="utf-8")
    (tmp_path / "small.md").write_text("# Small\n", encoding="utf-8")
    monkeypatch.setattr(prompts, "PROMPTS_DIR", tmp_path)
    with pytest.raises(ConfigurationError, match="ceiling"):
        prompts.load("huge")
    assert prompts.load("small") == "# Small"
    assert prompts.roles() == []


def test_role_files_cover_every_role_and_say_what_to_return():
    for role in EXPECTED_AGENTS | set(JOB_ROLES):
        text = prompts.load(f"roles/{role}")
        assert text.startswith(f"# Role: {role}\n"), role
    assert "findings" in prompts.load("roles/subagent").lower() and "confidence" in prompts.load("roles/subagent").lower()
    assert "book_mark_done" in prompts.load("roles/worker") and "Stop." in prompts.load("roles/worker")
    assert "motion is the idea" in prompts.load("roles/initializer")
    assert "Do not rewrite" in prompts.load("roles/critic")
    assert "verbatim" in prompts.load("roles/compactor")


# ----------------------------------------------------------------------------- volatility
@pytest.mark.parametrize("path", content_files(), ids=lambda p: str(p.relative_to(PROMPTS_DIR.parent)))
def test_no_volatile_strings(path: Path):
    text = path.read_text(encoding="utf-8")
    m = VOLATILE_RE.search(text)
    assert m is None, f"{path.name} contains a volatile string: {m.group(0)!r}"
    assert "\t" not in text and not text.endswith("\n\n")


# ----------------------------------------------------------------------------- the examples the skills hand the model
def _skill_fences(name: str) -> list[str]:
    return _PLAIN_FENCE_RE.findall((builtin_skills_dir() / name / "SKILL.md").read_text(encoding="utf-8"))


def test_items_example_passes_the_real_key_check(ctx):
    text = (builtin_skills_dir() / "assessment-design" / "references" / "items-example.md").read_text(encoding="utf-8")
    item_set = json.loads(_JSON_FENCE_RE.findall(text)[0])
    set_dir = ctx.workspace.path("assessments") / item_set["set_id"]
    ensure_dir(set_dir)
    write_json(set_dir / "key.json", item_set)
    out = assess_items_check(ctx, {"set_id": item_set["set_id"]})
    assert out["passed"], [r for r in out["items"] if not r["ok"]]
    assert {r["id"] for r in out["items"] if r["verified"]} == {"q1", "q3", "q4"}, "numeric, code and regex keys recompute"
    assert sorted(out["unverifiable"]) == ["q2", "q5"], "the mcq is checked for distinctness only; the proof has a rubric"


def test_diagramming_examples_are_valid_svg(ctx):
    fences = _skill_fences("diagramming")
    example = next(f for f in fences if f.lstrip().startswith("<svg"))
    v = validate_svg(example)
    assert v.ok and not v.warnings and min(v.font_sizes) >= 12, (v.errors, v.warnings, v.font_sizes)
    script = next(f for f in fences if f.startswith("W, H, M"))
    r = run_python(script, str(ctx.workspace.root), timeout_s=30)
    assert r.ok and validate_svg(r.stdout).ok, r.stderr


@skip_unless("chromium")
def test_skill_examples_render_clean_in_chromium(ctx):
    svg = next(f for f in _skill_fences("diagramming") if f.lstrip().startswith("<svg"))
    out = diagram_render(ctx, {"diagram_id": "skill-example", "svg": svg})
    assert out["ok"] and out["defects"] == [] and out["png_path"], out["defects"]
    widget = _skill_fences("interactive-widgets")
    html = next(f for f in widget if f.lstrip().startswith("<div"))
    js = next(f for f in widget if f.lstrip().startswith("(function"))
    res = widget_run(ctx, {"widget_id": "bayes", "html": html, "js": js})
    assert res["tests_ran"] and res["test_failures"] == [] and res["console_errors"] == [], res["defects"]
    assert res["ok"] and res["defects"] == [], res["defects"]


# ----------------------------------------------------------------------------- the skills describe the code that exists
def _skill_text(name: str) -> str:
    return (builtin_skills_dir() / name / "SKILL.md").read_text(encoding="utf-8")


def test_reading_level_tables_match_the_checker():
    """The band table in reading-levels is parsed row by row; the other two skills quote the same numbers."""
    rows = re.findall(r"^\| (\w+) \| (\d+)-(\d+) \| (\d+) \|", _skill_text("reading-levels"), re.MULTILINE)
    bands = {name: (float(lo), float(hi)) for name, lo, hi, _ in rows}
    ceilings = {name: int(c) for name, _, _, c in rows}
    assert bands == {k: (float(lo), float(hi)) for k, (lo, hi) in GRADE_BANDS.items()}
    assert ceilings == SENTENCE_WORDS_CEILING
    for name in ("book-authoring", "course-design"):
        text = " ".join(_skill_text(name).split())  # the prose is hard-wrapped
        for level, (lo, hi) in GRADE_BANDS.items():
            assert f"{level} {lo:g}-{hi:g}" in text, (name, level)
    for level, ceiling in SENTENCE_WORDS_CEILING.items():
        assert f"{level} {ceiling}" in " ".join(_skill_text("book-authoring").split()), level


def test_chapter_contract_matches_the_obligation_parser():
    text = _skill_text("book-authoring")
    assert CONTINUE_MARKER_RE.search("# edgent: continue") and "`# edgent: continue`" in text
    for kind in ("film", "widget", "exercises"):
        assert PLACEHOLDER_RE.search(f"{{{{{kind}: abc-1}}}}") and f"{{{{{kind}: <" in text, kind
    assert "![caption](../../media/diagrams/<id>.svg)" in text
    kinds = re.search(r"- Kinds: (.+?)\.\n", _skill_text("course-design"), re.DOTALL).group(1)
    named = set(re.findall(r"`([a-z_]+)`", kinds))
    assert named and named <= set(typing.get_args(UnitKind)), named - set(typing.get_args(UnitKind))


def test_film_skill_names_real_ops_qualities_and_defects():
    film, visual = _skill_text("manim-film"), _skill_text("visual-quality")
    assert set(re.findall(r"`(\w+) \{", film)) == set(OP_NAMES)
    for quality in QUALITY_FLAGS:
        assert f'"{quality}"' in film, quality
    render_src = Path(__import__("edgent.adapters.renderers.manim_render", fromlist=["x"]).__file__).read_text(encoding="utf-8")
    for kind in ("beat_error", "still_card", "duration_mismatch", "text_overlap", "outside_safe_frame", "font_below_floor", "text_too_small", "missing_beat", "render_incomplete", "no_report"):
        assert f'"{kind}"' in render_src and f"`{kind}`" in film and f"`{kind}`" in visual, kind
    assert ".defects.json" in visual and "read_file" in visual


def test_widget_skill_matches_the_house_theme_and_viewport():
    text = _skill_text("interactive-widgets")
    for var in re.findall(r"`(--[a-z-]+)`", text):
        assert f"{var}:" in HOUSE_THEME_CSS, var
    for cls in re.findall(r"`(\.[a-z]+)`", text):
        assert cls in HOUSE_THEME_CSS, cls
    assert f"{WIDGET_VIEWPORT_PX[0]} x {WIDGET_VIEWPORT_PX[1]}" in text


def test_agent_bodies_do_not_claim_what_the_harness_cannot_do():
    """A subagent starts every call in a fresh context and read_file cannot show an image."""
    critic = (builtin_agents_dir() / "critic.md").read_text(encoding="utf-8")
    assert "<file>.defects.json" in critic and "cannot show" in critic
    interviewer = (builtin_agents_dir() / "interviewer.md").read_text(encoding="utf-8")
    assert "private log" not in interviewer and "fresh context" in interviewer
    for path in agent_files():
        d = AgentDefinition.from_markdown(path.read_text(encoding="utf-8"), "builtin", path, default_name=path.stem)
        if "screenshot" in d.system_prompt or "attached picture" in d.system_prompt:
            assert set(d.tools) & {"diagram_render", "widget_run", "film_look"}, f"{d.name} is told to look at a picture it has no tool to see"

