"""Core tools through the registry: files, search, sandbox, web (fake opener), delegation and session state."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from edgent.adapters import web
from edgent.tools import core
from edgent.tools.core import CORE_TOOL_NAMES, TODO_STATUSES, build_core_registry
from edgent.tools.registry import ReadOnly, ToolContext, ToolRegistry
from tests.test_web import DDG_HTML, PUBLIC_RESOLVER, FakeOpener, FakeResponse, html_page


class FakeSession:
    """The slice of application.sessions.Session the session tools rely on."""

    def __init__(self) -> None:
        self.todo: list[dict[str, Any]] = []
        self.stance: list[str] = []
        self.saves = 0
        self.exchanges = [
            {"turn": 3, "user": "explain the coin example again", "assistant": "Two coins, one biased 0.7: the posterior after HHT is 0.61."},
            {"turn": 7, "user": "what about dice", "assistant": "A fair die: each face 1/6."},
            {"turn": 9, "user": "back to the coins", "assistant": "Same coins; prior 0.5."},
        ]

    def recall(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        q = query.lower()
        return [e for e in self.exchanges if q in (e["user"] + " " + e["assistant"]).lower()][:k]

    def add_stance(self, text: str) -> None:
        self.stance.append(text)

    def save(self) -> None:
        self.saves += 1


class FakeSkills:
    def __init__(self, bodies: dict[str, str]) -> None:
        self.bodies = bodies

    def names(self) -> list[str]:
        return sorted(self.bodies)

    def load_body(self, name: str) -> str:
        return self.bodies[name]


@pytest.fixture
def tools(registry: ToolRegistry, ctx: ToolContext) -> ToolRegistry:
    ctx.session = FakeSession()
    return build_core_registry(registry)


def call(tools: ToolRegistry, ctx: ToolContext, tool: str, **args: Any) -> Any:
    return tools.call(tool, args, ctx)


def ok(tools: ToolRegistry, ctx: ToolContext, tool: str, **args: Any) -> dict[str, Any]:
    r = call(tools, ctx, tool, **args)
    assert r.ok, r.error
    return r.output


def err(tools: ToolRegistry, ctx: ToolContext, tool: str, **args: Any) -> Any:
    r = call(tools, ctx, tool, **args)
    assert not r.ok, r.output
    assert r.error.remediation
    return r.error


# ----------------------------------------------------------------------------- registry
def test_registry_contents_and_contracts(tools: ToolRegistry, workspace: Any) -> None:
    assert tools.names() == sorted(CORE_TOOL_NAMES) and len(CORE_TOOL_NAMES) == 17
    expected = {
        "read_file": "read", "write_file": "write", "edit_file": "write", "list_files": "read", "glob": "read", "grep": "read",
        "bash": "compute", "run_python": "compute", "web_search": "network", "web_fetch": "network", "agent": "spawn",
        "skill": "read", "ask_user": "interactive", "todo_write": "read", "todo_read": "read", "session_recall": "read", "remember": "read",
    }
    for name, permission in expected.items():
        spec = tools.get(name)
        assert spec is not None and spec.permission == permission and spec.group == "core", name
        assert spec.parameters.get("additionalProperties") is False and spec.when_not_to_use and len(spec.description) > 60, name
        assert spec.full_description().count("Do not use when") == 1
        for prop, schema in spec.parameters["properties"].items():
            assert schema.get("description"), f"{name}.{prop} has no description"
    assert tools.get("write_file").side_effects and tools.get("edit_file").side_effects and not tools.get("read_file").side_effects
    assert tools.get("bash").timeout_s > core.BASH_TIMEOUT_LIMIT.value and tools.get("agent").timeout_s >= 600
    assert tools.schema_bytes(compact=True) < 12_000
    assert tools.get("read_file").max_inline_chars > core.READ_LENGTH_LIMIT.value and tools.get("web_fetch").max_inline_chars > core.WEB_FETCH_CHARS_LIMIT.value
    assert tools.get("bash").max_inline_chars > 2 * core.SANDBOX_OUTPUT_CHARS and tools.get("todo_read").max_inline_chars is None
    sub = build_core_registry(include={"read_file", "grep"})
    assert sub.names() == ["grep", "read_file"]
    with pytest.raises(ValueError):
        build_core_registry(include={"read_file", "nope"})
    again = build_core_registry(tools)  # re-registering on the same registry is a no-op, not an error
    assert again is tools and len(tools.names()) == 17


# ----------------------------------------------------------------------------- read_file
def test_read_file_pages_with_next_offset(tools: ToolRegistry, ctx: ToolContext, workspace: Any) -> None:
    text = "".join(f"line {i:05d} of the big file\n" for i in range(1200))
    workspace.write_text("notes/big.txt", text)
    first = ok(tools, ctx, "read_file", path="notes/big.txt")
    assert first["untrusted"] is True and first["offset"] == 0 and first["length"] == core.READ_DEFAULT_LENGTH_CHARS
    assert first["total_chars"] == len(text) and first["next_offset"] == core.READ_DEFAULT_LENGTH_CHARS and first["start_line"] == 1
    pages, offset = [first["content"]], first["next_offset"]
    while offset is not None:
        page = ok(tools, ctx, "read_file", path="notes/big.txt", offset=offset, length=12_000)
        pages.append(page["content"])
        offset = page["next_offset"]
    assert "".join(pages) == text and len(pages) == 3
    capped = ok(tools, ctx, "read_file", path="notes/big.txt", length=50_000)
    assert capped["length"] == core.READ_LENGTH_LIMIT.value == 20_000 and "capped" in capped["note"] and capped["next_offset"] == 20_000
    mid = ok(tools, ctx, "read_file", path="notes/big.txt", offset=len("line 00000 of the big file\n") * 10, length=30)
    assert mid["start_line"] == 11 and mid["content"].startswith("line 00010")
    past = ok(tools, ctx, "read_file", path="notes/big.txt", offset=len(text) + 5)
    assert past["content"] == "" and past["next_offset"] is None and "past the end" in past["note"]
    assert ok(tools, ctx, "read_file", path=str(workspace.root / "notes" / "big.txt"), length=5)["path"] == "notes/big.txt"


def test_read_file_refusals(tools: ToolRegistry, ctx: ToolContext, workspace: Any) -> None:
    outside = workspace.root.parent / "outside.txt"
    outside.write_text("secret")
    for path in ("../outside.txt", str(outside), "/etc/passwd", "~/x", "notes/../../outside.txt"):
        e = err(tools, ctx, "read_file", path=path)
        assert e.category.value == "invalid_input" and "escapes the workspace" in e.message and "relative to the workspace root" in e.remediation, path
    assert err(tools, ctx, "read_file", path="missing.md").category.value == "not_found"
    assert err(tools, ctx, "read_file", path="sources").category.value == "invalid_input"
    (workspace.root / "blob.bin").write_bytes(b"abc\x00def" * 100)
    e = err(tools, ctx, "read_file", path="blob.bin")
    assert e.category.value == "unsupported" and "700 bytes" in e.message and "sources_add" in e.remediation
    (workspace.root / "paper.pdf").write_bytes(b"%PDF-1.4 text only header")
    assert err(tools, ctx, "read_file", path="paper.pdf").category.value == "unsupported"
    assert err(tools, ctx, "read_file", path="notes.md", offset=-1).category.value == "invalid_input"


# ----------------------------------------------------------------------------- write_file and edit_file
def test_write_file_creates_parents(tools: ToolRegistry, ctx: ToolContext, workspace: Any) -> None:
    out = ok(tools, ctx, "write_file", path="book/chapters/01-intro.md", content="# Intro\n\nCafé ☕\n")
    assert out == {"path": "book/chapters/01-intro.md", "bytes": len("# Intro\n\nCafé ☕\n".encode()), "created": True}
    assert workspace.read_text("book/chapters/01-intro.md") == "# Intro\n\nCafé ☕\n"
    out = ok(tools, ctx, "write_file", path="book/chapters/01-intro.md", content="# Intro v2\n")
    assert out["created"] is False and workspace.read_text("book/chapters/01-intro.md") == "# Intro v2\n"
    assert err(tools, ctx, "write_file", path="../escape.md", content="x").category.value == "invalid_input"
    assert err(tools, ctx, "write_file", path="book", content="x").category.value == "invalid_input"
    assert not (workspace.root.parent / "escape.md").exists()


def test_edit_file_exact_match_rules(tools: ToolRegistry, ctx: ToolContext, workspace: Any) -> None:
    workspace.write_text("src/app.py", "def f():\n    return 1\n\ndef g():\n    return 1\n\ndef h():\n    return 1\n")
    e = err(tools, ctx, "edit_file", path="src/app.py", old_string="return 2", new_string="return 3")
    assert e.category.value == "invalid_input" and "not found" in e.message and "read_file" in e.remediation
    e = err(tools, ctx, "edit_file", path="src/app.py", old_string="    return 1\n", new_string="    return 0\n")
    assert e.category.value == "invalid_input" and "3 times" in e.message and "lines 2, 5, 8" in e.message and e.details["lines"] == [2, 5, 8] and "replace_all" in e.remediation
    out = ok(tools, ctx, "edit_file", path="src/app.py", old_string="def g():\n    return 1\n", new_string="def g():\n    return 2\n")
    assert out == {"path": "src/app.py", "replacements": 1}
    assert workspace.read_text("src/app.py").count("return 1") == 2 and "return 2" in workspace.read_text("src/app.py")
    out = ok(tools, ctx, "edit_file", path="src/app.py", old_string="return 1", new_string="return 9", replace_all=True)
    assert out["replacements"] == 2 and workspace.read_text("src/app.py").count("return 9") == 2
    assert err(tools, ctx, "edit_file", path="src/app.py", old_string="", new_string="x").category.value == "invalid_input"
    assert err(tools, ctx, "edit_file", path="src/app.py", old_string="def", new_string="def").category.value == "invalid_input"
    assert err(tools, ctx, "edit_file", path="src/none.py", old_string="a", new_string="b").category.value == "not_found"
    assert err(tools, ctx, "edit_file", path="../x", old_string="a", new_string="b").category.value == "invalid_input"


def test_edit_file_keeps_crlf_and_refuses_non_utf8(tools: ToolRegistry, ctx: ToolContext, workspace: Any) -> None:
    (workspace.root / "win.txt").write_bytes(b"alpha\r\nbeta\r\ngamma\r\n")
    ok(tools, ctx, "edit_file", path="win.txt", old_string="beta", new_string="BETA")
    assert (workspace.root / "win.txt").read_bytes() == b"alpha\r\nBETA\r\ngamma\r\n"
    (workspace.root / "latin.txt").write_bytes(b"caf\xe9 au lait")
    e = err(tools, ctx, "edit_file", path="latin.txt", old_string="lait", new_string="noir")
    assert e.category.value == "unsupported" and "UTF-8" in e.message


# ----------------------------------------------------------------------------- list_files, glob, grep
def test_list_files_and_glob(tools: ToolRegistry, ctx: ToolContext, workspace: Any) -> None:
    for i in range(5):
        workspace.write_text(f"sources/text/doc{i}.md", f"# Doc {i}\nposterior {i}\n")
    workspace.write_text("sources/text/deep/inner.txt", "x")
    out = ok(tools, ctx, "list_files", path="sources/text")
    assert out["path"] == "sources/text" and out["count"] == 6 and not out["truncated"]
    assert {e["path"] for e in out["entries"]} >= {"sources/text/doc0.md", "sources/text/deep"} and out["entries"][0]["type"] == "dir"
    out = ok(tools, ctx, "list_files", path="sources/text", recursive=True, max_entries=3)
    assert out["count"] == 3 and out["truncated"] and "hint" in out
    assert ok(tools, ctx, "list_files")["path"] == "." and ok(tools, ctx, "list_files", path="STUDY.md")["entries"][0]["type"] == "file"
    assert all(".edgent" not in e["path"] for e in ok(tools, ctx, "list_files", recursive=True, max_entries=1000)["entries"])
    assert err(tools, ctx, "list_files", path="nope").category.value == "not_found"
    assert err(tools, ctx, "list_files", path="../").category.value == "invalid_input"
    out = ok(tools, ctx, "glob", pattern="sources/text/**/*.md")
    assert out["count"] == 5 and out["matches"][0] == "sources/text/doc0.md" and not out["truncated"]
    out = ok(tools, ctx, "glob", pattern="**/*.md", max_entries=2)
    assert out["count"] == 2 and out["truncated"]
    assert ok(tools, ctx, "glob", pattern="sources/text/*.txt")["matches"] == []
    for bad in ("/etc/*", "../*.md", ""):
        assert err(tools, ctx, "glob", pattern=bad).category.value == "invalid_input", bad


def test_grep_hits_context_truncation_and_errors(tools: ToolRegistry, ctx: ToolContext, workspace: Any) -> None:
    workspace.write_text("sources/text/a.md", "intro\nThe Posterior is proportional\nto the prior\nposterior again\n")
    workspace.write_text("sources/text/b.txt", "posterior in a txt\n")
    (workspace.root / "sources" / "text" / "c.bin").write_bytes(b"posterior\x00binary")
    out = ok(tools, ctx, "grep", pattern="posterior", path="sources/text")
    assert out["untrusted"] is True and out["count"] == 3 and not out["truncated"]
    assert out["hits"][0] == {"path": "sources/text/a.md", "line": 2, "text": "The Posterior is proportional"}
    assert {h["path"] for h in out["hits"]} == {"sources/text/a.md", "sources/text/b.txt"}
    assert ok(tools, ctx, "grep", pattern="posterior", path="sources/text", ignore_case=False)["count"] == 2
    assert ok(tools, ctx, "grep", pattern="posterior", path="sources/text", glob="*.txt")["count"] == 1
    out = ok(tools, ctx, "grep", pattern="posterior", path="sources/text", max_hits=2)
    assert out["count"] == 2 and out["truncated"] and "narrow" in out["hint"]
    out = ok(tools, ctx, "grep", pattern="prior$", path="sources/text/a.md", context=1)
    assert out["hits"][0]["line"] == 3 and out["hits"][0]["text"] == "The Posterior is proportional\nto the prior\nposterior again"
    e = err(tools, ctx, "grep", pattern="posterior(")
    assert e.category.value == "invalid_input" and "regular expression" in e.message
    assert err(tools, ctx, "grep", pattern="x", path="nowhere").category.value == "not_found"
    assert err(tools, ctx, "grep", pattern="x", path="../").category.value == "invalid_input"
    assert err(tools, ctx, "grep", pattern="x", context=99).category.value == "invalid_input"


# ----------------------------------------------------------------------------- bash and run_python
def test_bash_in_the_sandbox(tools: ToolRegistry, ctx: ToolContext, workspace: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDGENT_TEST_SECRET", "hunter2")
    workspace.write_text(".bash_profile", "echo PROFILE-RAN\n")  # a workspace file is data; a login shell would run it on every call
    workspace.write_text(".bashrc", "echo RC-RAN\n")
    out = ok(tools, ctx, "bash", command="echo hi; pwd; echo secret=$EDGENT_TEST_SECRET; echo home=$HOME")
    assert out["returncode"] == 0 and not out["timed_out"] and not out["truncated"] and out["duration_s"] >= 0
    lines = out["stdout"].splitlines()
    assert lines[0] == "hi" and Path(lines[1]).resolve() == workspace.root and lines[2] == "secret=" and lines[3] == f"home={workspace.root / core.BASH_HOME_REL}"
    assert "PROFILE-RAN" not in out["stdout"] and "RC-RAN" not in out["stdout"]
    out = ok(tools, ctx, "bash", command="echo oops >&2; exit 3")
    assert out["returncode"] == 3 and out["stderr"].strip() == "oops"
    out = ok(tools, ctx, "bash", command="sleep 5", timeout_s=1)
    assert out["timed_out"] and out["returncode"] == -9 and out["timeout_s"] == 1 and "timeout_s" in out["hint"] and out["duration_s"] < 4
    out = ok(tools, ctx, "bash", command="printf 'a%.0s' $(seq 1 30000)")
    assert out["truncated"] and len(out["stdout"]) <= core.SANDBOX_OUTPUT_CHARS + 40
    ok(tools, ctx, "bash", command="mkdir -p out && echo made > out/f.txt")
    assert workspace.read_text("out/f.txt") == "made\n"
    assert err(tools, ctx, "bash", command="   ").category.value == "invalid_input"
    assert ok(tools, ctx, "bash", command="true", timeout_s=10_000)["timeout_s"] == core.BASH_TIMEOUT_LIMIT.value
    assert ok(tools, ctx, "bash", command="definitely-not-a-command")["returncode"] == 127


def test_bash_reads_sandbox_settings(tools: ToolRegistry, ctx: ToolContext) -> None:
    ctx.settings = {"sandbox": {"cpu_seconds": 7, "memory_mb": 1500}}
    assert core._sandbox_limits(ctx, 60) == (7, 1500 << 20)
    ctx.settings = {}
    assert core._sandbox_limits(ctx, 45.5) == (46, core.DEFAULT_SANDBOX_MEMORY_MB << 20)
    out = ok(tools, ctx, "bash", command="python3 -c 'import resource; print(resource.getrlimit(resource.RLIMIT_CPU)[0])'", timeout_s=9)
    assert out["stdout"].strip() == "9"


def test_run_python_in_the_sandbox(tools: ToolRegistry, ctx: ToolContext, workspace: Any) -> None:
    out = ok(tools, ctx, "run_python", code="import os, sys\nprint(2 + 2)\nprint(os.getcwd())\nprint(sys.flags.isolated)")
    assert out["returncode"] == 0 and out["stdout"].splitlines() == ["4", str(workspace.root), "1"]
    out = ok(tools, ctx, "run_python", code="1/0")
    assert out["returncode"] == 1 and "ZeroDivisionError" in out["stderr"]
    out = ok(tools, ctx, "run_python", code="import time\ntime.sleep(5)", timeout_s=1)
    assert out["timed_out"] and "hint" in out
    assert err(tools, ctx, "run_python", code="\n").category.value == "invalid_input"
    assert not list((workspace.root / ".edgent" / "scratch").glob("cell_*.py"))


# ----------------------------------------------------------------------------- web tools
def test_web_search_tool(tools: ToolRegistry, ctx: ToolContext) -> None:
    opener = FakeOpener({web.SEARCH_URL: lambda url: FakeResponse(DDG_HTML, url)})
    ctx.extra["web_opener"], ctx.extra["web_resolver"] = opener, PUBLIC_RESOLVER
    out = ok(tools, ctx, "web_search", query="bayes theorem", max_results=2)
    assert out["untrusted"] is True and out["count"] == 2 and out["results"][0]["url"].startswith("https://en.wikipedia.org/")
    assert set(out["results"][0]) == {"title", "url", "snippet"} and out["query"] == "bayes theorem"
    assert ok(tools, ctx, "web_search", query="bayes theorem", max_results=99)["count"] == 3
    ctx.extra["web_opener"] = FakeOpener({})
    e = err(tools, ctx, "web_search", query="anything")
    assert e.category.value == "unavailable" and e.retryable is False and "try web_fetch on a known URL" in e.remediation
    assert err(tools, ctx, "web_search", query="  ").category.value == "invalid_input"


def test_web_fetch_tool_saves_markdown(tools: ToolRegistry, ctx: ToolContext, workspace: Any) -> None:
    opener = FakeOpener({"https://example.com/": lambda url: FakeResponse(html_page(body="<p>" + "word " * 3000 + "</p>"), url)})
    ctx.extra["web_opener"], ctx.extra["web_resolver"] = opener, PUBLIC_RESOLVER
    out = ok(tools, ctx, "web_fetch", url="https://example.com/docs/bayes")
    assert out["untrusted"] is True and out["title"] == "Bayes' theorem" and out["content"].startswith("# Bayes' theorem") and "**prior**" in out["content"]
    assert not out["truncated"] and out["total_chars"] == len(out["content"]) and out["content_type"] == "text/html" and "final_url" not in out
    assert out["saved_path"].startswith("sources/raw/web/example-com-docs-bayes-") and out["saved_path"].endswith(".md")
    saved = workspace.read_text(out["saved_path"])
    assert saved.startswith("---\nsource: https://example.com/docs/bayes\ntitle: \"Bayes' theorem\"\nfetched: ") and saved.endswith(out["content"])
    assert "evil()" not in saved and "menu" not in saved
    short = ok(tools, ctx, "web_fetch", url="https://example.com/docs/bayes", max_chars=100)
    assert short["truncated"] and len(short["content"]) == 100 and short["saved_path"] == out["saved_path"] and out["saved_path"] in short["hint"]
    assert ok(tools, ctx, "web_fetch", url="https://example.com/docs/bayes", max_chars=10**9)["content"] == out["content"]
    unsaved = ok(tools, ctx, "web_fetch", url="https://example.com/other", save=False)
    assert unsaved["saved_path"] is None and unsaved["title"] == "Bayes' theorem"
    assert len(list((workspace.root / "sources" / "raw" / "web").iterdir())) == 1


def test_web_fetch_tool_refuses_private_and_saves_pdf(tools: ToolRegistry, ctx: ToolContext, workspace: Any) -> None:
    opener = FakeOpener({"https://example.com/paper.pdf": lambda url: FakeResponse(b"%PDF-1.7\nfake", url, content_type="application/pdf")})
    ctx.extra["web_opener"], ctx.extra["web_resolver"] = opener, PUBLIC_RESOLVER
    for url in ("http://127.0.0.1:8000/secret", "http://localhost/", "file:///etc/passwd", "http://169.254.169.254/"):
        e = err(tools, ctx, "web_fetch", url=url)
        assert e.category.value == "invalid_input", url
    assert opener.calls == []
    out = ok(tools, ctx, "web_fetch", url="https://example.com/paper.pdf")
    assert out["saved_path"].startswith("sources/raw/web/example-com-paper-pdf-") and out["saved_path"].endswith(".pdf")
    assert (workspace.root / out["saved_path"]).read_bytes() == b"%PDF-1.7\nfake" and "sources_add" in out["content"] and out["title"] == "paper.pdf"
    assert err(tools, ctx, "web_fetch", url="https://example.com/missing").category.value == "not_found"
    assert err(tools, ctx, "web_fetch", url="").category.value == "invalid_input"


# ----------------------------------------------------------------------------- agent, skill, ask_user
def test_agent_tool(tools: ToolRegistry, ctx: ToolContext) -> None:
    e = err(tools, ctx, "agent", agent="librarian", task="find the coin example")
    assert e.category.value == "unavailable" and "subagents are not available in this context" in e.remediation
    calls: list[tuple[Any, ...]] = []

    def spawn(agent: str, task: str, **opts: Any) -> dict[str, Any]:
        calls.append((agent, task, opts))
        return {"agent": agent, "report": f"done: {task}", "background": opts.get("background")}

    ctx.spawn = spawn
    out = ok(tools, ctx, "agent", agent="librarian", task="find the coin example")
    assert out == {"agent": "librarian", "report": "done: find the coin example", "background": False} and calls[0][2] == {"background": False}
    assert ok(tools, ctx, "agent", agent="author", task="draft", background=True)["background"] is True
    ctx.spawn = lambda agent, task, **opts: "plain text report"
    assert ok(tools, ctx, "agent", agent="critic", task="review") == {"result": "plain text report"}


def test_skill_tool(tools: ToolRegistry, ctx: ToolContext) -> None:
    assert err(tools, ctx, "skill", name="chapter-writing").category.value == "unavailable"
    ctx.skills = FakeSkills({"chapter-writing": "# Chapter writing\n\nExample before definition.", "quiz": "# Quiz"})
    assert ok(tools, ctx, "skill", name="chapter-writing") == {"name": "chapter-writing", "body": "# Chapter writing\n\nExample before definition."}
    e = err(tools, ctx, "skill", name="storyboard")
    assert e.category.value == "not_found" and "chapter-writing, quiz" in e.message and e.details["available"] == ["chapter-writing", "quiz"]


def test_ask_user_tool(tools: ToolRegistry, ctx: ToolContext) -> None:
    e = err(tools, ctx, "ask_user", question="Which audience?")
    assert e.category.value == "unavailable" and e.remediation == "no interactive user; proceed with the most reasonable assumption and state it"
    prompts: list[str] = []

    def answer(prompt: str) -> str:
        prompts.append(prompt)
        return "undergraduate"

    ctx.ask_user = answer
    out = ok(tools, ctx, "ask_user", question="Which  audience?", options=["school", "undergraduate"])
    assert out == {"question": "Which audience?", "answer": "undergraduate", "options": ["school", "undergraduate"]}
    assert prompts == ["Which audience?\nOptions: school / undergraduate"]
    assert ok(tools, ctx, "ask_user", question="Go on?") == {"question": "Go on?", "answer": "undergraduate"}
    assert err(tools, ctx, "ask_user", question=" ").category.value == "invalid_input"


# ----------------------------------------------------------------------------- session state
def test_todo_write_and_read(tools: ToolRegistry, ctx: ToolContext) -> None:
    session = ctx.session
    out = ok(tools, ctx, "todo_write", items=[
        {"content": "Read the outline", "status": "completed"},
        {"id": "draft", "content": "  Draft chapter 2 ", "status": "in_progress"},
        {"content": "Run checks", "status": "pending"},
    ])
    assert out["count"] == 3 and out["open"] == 2
    assert out["items"] == [
        {"id": "t1", "content": "Read the outline", "status": "completed"},
        {"id": "draft", "content": "Draft chapter 2", "status": "in_progress"},
        {"id": "t2", "content": "Run checks", "status": "pending"},
    ]
    assert session.todo == out["items"] and session.saves == 1
    assert ok(tools, ctx, "todo_read") == {"items": out["items"], "count": 3, "open": 2}
    out = ok(tools, ctx, "todo_write", items=[{"id": "t1", "content": "a", "status": "pending"}, {"id": "t1", "content": "b", "status": "pending"}])
    assert [t["id"] for t in out["items"]] == ["t1", "t2"]
    e = err(tools, ctx, "todo_write", items=[{"content": "x", "status": "done"}])
    assert e.category.value == "invalid_input" and all(s in e.message for s in TODO_STATUSES)
    assert err(tools, ctx, "todo_write", items=[{"content": "  ", "status": "pending"}]).category.value == "invalid_input"
    assert err(tools, ctx, "todo_write", items=[{"content": "x"}]).category.value == "invalid_input"
    e = err(tools, ctx, "todo_write", items=[{"content": f"step {i}", "status": "pending"} for i in range(101)])
    assert e.category.value == "resource_constraint" and e.code == "limit:todo items" and "100 items" in e.message
    assert ok(tools, ctx, "todo_write", items=[])["items"] == [] and ok(tools, ctx, "todo_read")["count"] == 0


def test_session_recall_and_remember(tools: ToolRegistry, ctx: ToolContext) -> None:
    out = ok(tools, ctx, "session_recall", query="coins")
    assert out["count"] == 2 and out["matches"][0]["turn"] == 3 and "posterior after HHT is 0.61" in out["matches"][0]["assistant"]
    assert ok(tools, ctx, "session_recall", query="coins", k=1)["count"] == 1
    assert ok(tools, ctx, "session_recall", query="nothing here")["matches"] == []
    assert err(tools, ctx, "session_recall", query="  ").category.value == "invalid_input"
    out = ok(tools, ctx, "remember", text="Chapter 3 uses the frequentist framing; the learner asked for it.")
    assert out == {"stance": ["Chapter 3 uses the frequentist framing; the learner asked for it."]} and ctx.session.saves == 1
    assert "added" not in out  # the fake's add_stance returns None; the real Session returns a bool, surfaced as 'added'
    out = ok(tools, ctx, "remember", text="Second conclusion")
    assert out["stance"] == ["Chapter 3 uses the frequentist framing; the learner asked for it.", "Second conclusion"] and ctx.session.saves == 2
    assert err(tools, ctx, "remember", text="").category.value == "invalid_input"


def test_session_tools_without_a_session(tools: ToolRegistry, ctx: ToolContext) -> None:
    ctx.session = None
    for name, args in (("todo_write", {"items": []}), ("todo_read", {}), ("session_recall", {"query": "x"}), ("remember", {"text": "x"})):
        e = err(tools, ctx, name, **args)
        assert e.category.value == "unavailable" and "session" in e.message and "in your own reply" in e.remediation, name


# ----------------------------------------------------------------------------- policy interplay
def test_read_only_policy_blocks_the_right_tools(tools: ToolRegistry, ctx: ToolContext, workspace: Any) -> None:
    ctx.permissions = ReadOnly()
    workspace.write_text("a.txt", "hello")
    assert ok(tools, ctx, "read_file", path="a.txt")["content"] == "hello"
    assert ok(tools, ctx, "grep", pattern="hell")["count"] == 1
    assert ok(tools, ctx, "todo_write", items=[])["count"] == 0
    for name, args in (("write_file", {"path": "b.txt", "content": "x"}), ("edit_file", {"path": "a.txt", "old_string": "h", "new_string": "j"}),
                       ("bash", {"command": "true"}), ("web_fetch", {"url": "https://example.com/"}), ("agent", {"agent": "a", "task": "t"}),
                       ("ask_user", {"question": "q"})):
        assert call(tools, ctx, name, **args).error.code == "permission_denied", name
    assert not (workspace.root / "b.txt").exists() and workspace.read_text("a.txt") == "hello"


def test_tool_results_are_json_serialisable(tools: ToolRegistry, ctx: ToolContext, workspace: Any) -> None:
    import json

    workspace.write_text("j.txt", "x")
    for name, args in (("read_file", {"path": "j.txt"}), ("list_files", {}), ("glob", {"pattern": "*.txt"}), ("grep", {"pattern": "x"}),
                       ("bash", {"command": "echo 1"}), ("todo_read", {}), ("session_recall", {"query": "coins"})):
        r = call(tools, ctx, name, **args)
        assert r.ok and json.loads(r.as_text()) is not None, name
    big = ok(tools, ctx, "bash", command="printf 'a%.0s' $(seq 1 30000)")
    assert big["truncated"] and json.loads(json.dumps(big))["truncated"]  # a truncated sandbox result is still a plain JSON object


# ----------------------------------------------------------------------------- review additions: first-hour paths
def test_parallel_edits_to_one_file_do_not_lose_updates(tools: ToolRegistry, ctx: ToolContext, workspace: Any) -> None:
    """The loop dispatches a turn's tool calls with call_parallel; edits to one file must serialise, not interleave."""
    for trial in range(3):  # distinct arguments per trial: identical side-effecting calls would be replayed by the idempotency store
        workspace.write_text("par.txt", "".join(f"line {i} old\n" for i in range(8)))
        calls = [("edit_file", {"path": "par.txt", "old_string": f"line {i} old", "new_string": f"line {i} new{trial}"}, f"c{i}") for i in range(8)]
        results = tools.call_parallel(calls, ctx)
        assert all(r.ok and not r.replayed for r in results), [r.error.message for r in results if not r.ok]
        assert workspace.read_text("par.txt") == "".join(f"line {i} new{trial}\n" for i in range(8))


def test_read_file_keeps_escaped_pages_inline(tools: ToolRegistry, ctx: ToolContext, workspace: Any) -> None:
    """A page of JSON or CSV escapes to far more characters than it holds; it must still arrive inline and page cleanly."""
    import json

    text = json.dumps({f"key{i}": {"name": f"value {i}", "tags": ["a", "b"]} for i in range(900)})
    assert len(text) > 40_000
    workspace.write_text("data/q.json", text)
    r = call(tools, ctx, "read_file", path="data/q.json", length=20_000)
    assert r.ok and not r.compacted and 0 < r.output["length"] < 20_000 and "shortened" in r.output["note"]
    assert len(json.dumps(r.output["content"], ensure_ascii=False)) - 2 <= core.READ_ESCAPED_CHARS
    pages, offset = [], 0
    while offset is not None:
        r = call(tools, ctx, "read_file", path="data/q.json", offset=offset, length=20_000)
        assert r.ok and not r.compacted
        pages.append(r.output["content"])
        offset = r.output["next_offset"]
    assert "".join(pages) == text
    workspace.write_text("data/prose.txt", ("The posterior is proportional to the prior times the likelihood.\n" * 400)[:20_000])
    out = ok(tools, ctx, "read_file", path="data/prose.txt", length=20_000)
    assert out["length"] == 20_000 and "note" not in out  # plain text is never shortened
    assert core._fit_chunk('"' * 100, 50) == '"' * 25 and core._fit_chunk("abc", 3) == "abc" and core._fit_chunk("", 10) == ""


def test_read_file_size_limit(tools: ToolRegistry, ctx: ToolContext, workspace: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from edgent.domain.errors import Limit

    workspace.write_text("big.txt", "x" * 500)
    monkeypatch.setattr(core, "READ_FILE_BYTES_LIMIT", Limit("read_file file size", 100, "bytes"))
    e = err(tools, ctx, "read_file", path="big.txt")
    assert e.category.value == "resource_constraint" and e.code == "limit:read_file file size" and "500 bytes" in e.message and "grep" in e.remediation
    assert core.READ_FILE_BYTES_LIMIT.unit == "bytes"


def test_edit_file_matches_lf_old_string_in_crlf_file(tools: ToolRegistry, ctx: ToolContext, workspace: Any) -> None:
    (workspace.root / "crlf.py").write_bytes(b"def f():\r\n    return 1\r\n\r\ndef g():\r\n    pass\r\n")
    out = ok(tools, ctx, "edit_file", path="crlf.py", old_string="def f():\n    return 1\n", new_string="def f():\n    return 2\n")
    assert out["replacements"] == 1
    assert (workspace.root / "crlf.py").read_bytes() == b"def f():\r\n    return 2\r\n\r\ndef g():\r\n    pass\r\n"
    assert err(tools, ctx, "edit_file", path="crlf.py", old_string="def h():\n", new_string="x").category.value == "invalid_input"


def test_path_and_write_os_errors_are_typed(tools: ToolRegistry, ctx: ToolContext, workspace: Any) -> None:
    e = err(tools, ctx, "read_file", path="a\x00b")
    assert e.category.value == "invalid_input" and "unusable path" in e.message
    e = err(tools, ctx, "write_file", path="STUDY.md/child.md", content="x")
    assert e.category.value == "invalid_input" and "is a file, not a directory" in e.message and "list_files" in e.remediation
    e = err(tools, ctx, "edit_file", path="STUDY.md/child.md", old_string="a", new_string="b")
    assert e.category.value == "not_found"
    assert "." not in ok(tools, ctx, "glob", pattern="**")["matches"]


def test_sandbox_hints_explain_the_sandbox(tools: ToolRegistry, ctx: ToolContext) -> None:
    from edgent.adapters.sandbox import SandboxResult

    def hint(returncode: int, stderr: str = "", allow_network: bool = False) -> str:
        return core._sandbox_dict(SandboxResult(returncode, "", stderr, 0.1), 60, core.BASH_TIMEOUT_LIMIT, allow_network).get("hint", "")

    assert "CPU time" in hint(-24) and "cpu_seconds" in hint(152)
    assert "allow_network=true" in hint(6, "curl: (6) Could not resolve host: example.com")
    assert hint(6, "curl: (6) Could not resolve host: example.com", allow_network=True) == ""
    assert "git" in hint(128, "Author identity unknown\n\n*** Please tell me who you are.")
    assert hint(0, "Could not resolve host") == "" and "127" in hint(127)
    ctx.settings = {"sandbox": {"cpu_seconds": 1}}
    out = ok(tools, ctx, "bash", command="python3 -c 'while True: pass'", timeout_s=20)
    assert out["returncode"] in core._SIGXCPU_CODES and not out["timed_out"] and "CPU time" in out["hint"] and out["duration_s"] < 10


def test_session_recall_invalid_regex_and_the_real_session(tools: ToolRegistry, ctx: ToolContext, workspace: Any) -> None:
    """The session tools against application.sessions.Session itself, not only the fake."""
    from edgent.application.sessions import Session
    from edgent.domain.types import Message

    session = Session.create(workspace, "chat")
    ctx.session = session
    session.append(Message.user("explain the coin example again"))
    session.append(Message.assistant("Two coins, one biased 0.7: the posterior after HHT is 0.61."))
    e = err(tools, ctx, "session_recall", query="re:(unclosed")
    assert e.category.value == "invalid_input" and "re:" in e.remediation
    out = ok(tools, ctx, "session_recall", query="coins posterior")
    assert out["count"] >= 1 and "0.61" in out["matches"][0]["text"]
    assert ok(tools, ctx, "session_recall", query="re:HH[T]")["count"] == 1
    ok(tools, ctx, "todo_write", items=[{"content": "Draft chapter 2", "status": "in_progress"}])
    out = ok(tools, ctx, "remember", text="Chapter 3 uses the frequentist framing.")
    assert out == {"stance": ["Chapter 3 uses the frequentist framing."], "added": True}
    assert ok(tools, ctx, "remember", text="Chapter 3 uses the frequentist framing.")["added"] is False
    reloaded = Session.load(workspace, session.session_id)  # persisted before the tool returned
    assert reloaded.todo == [{"id": "t1", "content": "Draft chapter 2", "status": "in_progress"}] and reloaded.stance == ["Chapter 3 uses the frequentist framing."]
    assert "[in_progress] Draft chapter 2" in session.notes_path.read_text()


def test_skill_reference_argument(tools: ToolRegistry, ctx: ToolContext, workspace: Any) -> None:
    """The skill body tells the model to 'ask for one with the skill tool's reference argument'; it must exist."""
    from edgent.application.skills import SkillIndex

    d = workspace.root / ".edgent" / "skills" / "demo"
    (d / "references").mkdir(parents=True)
    (d / "SKILL.md").write_text("---\nname: demo\ndescription: Demo.\n---\n# Demo body\n")
    (d / "references" / "rubric.md").write_text("# Rubric\n\n- clear\n")
    ctx.skills = SkillIndex().load([workspace.root / ".edgent" / "skills"], ["workspace"])
    body = ok(tools, ctx, "skill", name="demo")["body"]
    assert body.startswith("# Demo body") and "references/rubric.md" in body and "reference argument" in body
    assert ok(tools, ctx, "skill", name="demo", reference="references/rubric.md") == {"name": "demo", "reference": "references/rubric.md", "body": "# Rubric\n\n- clear\n"}
    assert ok(tools, ctx, "skill", name="demo", reference="rubric.md")["body"].startswith("# Rubric")
    e = err(tools, ctx, "skill", name="demo", reference="../../SKILL.md")
    assert e.category.value == "not_found" and "outside" in e.message
    assert "available: references/rubric.md" in err(tools, ctx, "skill", name="demo", reference="nope.md").message
    assert err(tools, ctx, "skill", name="ghost", reference="x.md").category.value == "not_found"
    ctx.skills = FakeSkills({"plain": "# Plain"})  # an index without reference files
    assert err(tools, ctx, "skill", name="plain", reference="x.md").category.value == "unsupported"


def test_agent_keeps_the_category_of_host_errors(tools: ToolRegistry, ctx: ToolContext) -> None:
    """budgets.BudgetExhausted and friends are not ToolFailures but carry a HarnessError; the model must see it."""
    from edgent.domain.errors import ErrorCategory, HarnessError

    class Exhausted(Exception):
        def __init__(self, error: HarnessError) -> None:
            super().__init__(error.message)
            self.error = error

    def spawn(agent: str, task: str, **opts: Any) -> dict[str, Any]:
        raise Exhausted(HarnessError("limit:spawns", "spawn budget exceeded", ErrorCategory.resource_constraint, remediation="Do the task here.", details={"limit": "spawns"}))

    ctx.spawn = spawn
    e = err(tools, ctx, "agent", agent="librarian", task="find")
    assert e.category.value == "resource_constraint" and e.code == "limit:spawns" and e.remediation == "Do the task here." and e.details == {"limit": "spawns"}

    def boom(agent: str, task: str, **opts: Any) -> dict[str, Any]:
        raise RuntimeError("provider exploded")

    ctx.spawn = boom
    e = err(tools, ctx, "agent", agent="librarian", task="find")
    assert e.category.value == "execution_failure" and "provider exploded" in e.message


def test_web_fetch_keeps_escaped_content_inline(tools: ToolRegistry, ctx: ToolContext, workspace: Any) -> None:
    quotes = '"' * 30_000
    opener = FakeOpener({"https://example.com/": lambda url: FakeResponse(quotes, url, content_type="text/plain")})
    ctx.extra["web_opener"], ctx.extra["web_resolver"] = opener, PUBLIC_RESOLVER
    r = call(tools, ctx, "web_fetch", url="https://example.com/quotes.txt", max_chars=20_000)
    assert r.ok and not r.compacted and r.output["truncated"] and 0 < len(r.output["content"]) < 20_000
    assert f"only the first {len(r.output['content'])} of {len(quotes)} chars" in r.output["hint"] and r.output["saved_path"] in r.output["hint"]
    assert workspace.read_text(r.output["saved_path"]).endswith(quotes)
