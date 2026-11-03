"""The media tools through the registry: films with a fake manim, diagrams, widgets, narration and assembly."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from edgent.adapters.sandbox import SandboxResult
from edgent.adapters.toolchain import Toolchain
from edgent.domain.errors import Limit
from edgent.tools.edu import media
from edgent.tools.edu.media import HOUSE_THEME_CSS, MEDIA_TOOL_NAMES, build_media_registry, build_widget_document
from edgent.tools.registry import ToolContext, ToolRegistry, ToolResult
from tests.conftest import skip_unless
from tests.test_av import make_video, make_wav

STORYBOARD = {
    "title": "Pythagoras",
    "persona": "sophomore",
    "beats": [
        {"title": "Setup", "narration": "Here is a right triangle with legs a and b.", "seconds": 4, "code": "tri = Polygon(ORIGIN, RIGHT * 3, UP * 2, color=ACCENT)\nstage(tri)\nself.play(Create(tri))\nself.wait(1)\n"},
        {"title": "Squares", "narration": "Now we add squares on each side.", "code": "sq = Square(side_length=2)\nself.add(sq)\nself.play(Indicate(sq))\n"},
    ],
}

VALID_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 120"><rect width="300" height="120" fill="#fff"/>'
    '<text x="10" y="40" font-size="18" fill="#222">Prior</text><text x="20" y="44" font-size="18" fill="#222">Posterior</text>'
    '<text x="10" y="90" font-size="9" fill="#222">tiny</text><text x="250" y="100" font-size="16" fill="#222">clipped label</text></svg>'
)

WIDGET_HTML = '<div class="panel"><h1>Counter</h1><div class="controls"><button id="add">Add one</button> <output id="out">0</output></div><p><span style="color:#cccccc">faint note</span></p></div>'
WIDGET_JS = (
    "const out=document.getElementById('out');document.getElementById('add').onclick=()=>{out.textContent=String(Number(out.textContent)+1)};"
    "window.__edgentTest=()=>{document.getElementById('add').click();return out.textContent==='1'?[]:['click did not count: '+out.textContent]};"
    "console.error('deliberate');"
)


def good_report() -> dict[str, Any]:
    safe = {"left": -6.511, "right": 6.511, "top": 3.4, "bottom": -3.4, "margin": 0.6, "title_band": 1.2}
    config = {"pixel_width": 854, "pixel_height": 480, "frame_width": 14.222, "frame_height": 8.0, "frame_rate": 15, "quality": "low_quality", "font_floor": 24, "safe": safe}
    beats = [
        {
            "id": 1, "title": "Setup", "start_s": 0.0, "end_s": 1.2, "duration_s": 1.2, "plays": 1, "waits": 1, "adds": 0, "error": None,
            "mobjects": [
                {"kind": "text", "type": "Text", "text": "Setup", "font_size": 56, "bbox": [-1.2, 2.3, 1.2, 2.9], "opacity": 1.0, "role": "title"},
                {"kind": "shape", "type": "Polygon", "bbox": [-1.5, -1.0, 1.5, 1.0], "opacity": 1.0},
            ],
        },
        {
            "id": 2, "title": "Squares", "start_s": 1.2, "end_s": 1.9, "duration_s": 0.7, "plays": 0, "waits": 0, "adds": 1, "error": None,
            "mobjects": [{"kind": "shape", "type": "Square", "bbox": [-1.0, -1.0, 1.0, 1.0], "opacity": 1.0}],
        },
    ]
    return {"version": 1, "scene": "GeneratedFilm", "complete": True, "config": config, "beats": beats, "total_s": 1.9, "wall_s": 12.0}


class FakeManim:
    """Stands in for the manim CLI: writes the report the prelude would write and the mp4 manim would produce."""

    def __init__(self, report: dict[str, Any] | None, returncode: int = 0, stderr: str = "") -> None:
        self.report, self.returncode, self.stderr = report, returncode, stderr
        self.calls: list[dict[str, Any]] = []

    def __call__(self, argv: list[str], cwd: str, **kw: Any) -> SandboxResult:
        self.calls.append({"argv": argv, "cwd": cwd, **kw})
        if self.report is not None:
            Path(kw["env_extra"]["EDGENT_FILM_REPORT"]).write_text(json.dumps(self.report), encoding="utf-8")
        if self.returncode == 0:
            out = Path(argv[argv.index("--media_dir") + 1]) / "videos" / "scene" / "480p15" / "film.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            if Toolchain.detect().has("ffmpeg"):
                make_video(out)
            else:
                out.write_bytes(b"fake mp4")
        return SandboxResult(self.returncode, "", self.stderr, 0.5, argv=argv)


@pytest.fixture
def tools(registry: ToolRegistry, ctx: ToolContext) -> ToolRegistry:
    return build_media_registry(registry)


def call(tools: ToolRegistry, ctx: ToolContext, tool: str, **args: Any) -> ToolResult:
    return tools.call(tool, args, ctx)


def test_registry_contract(tools: ToolRegistry):
    assert set(MEDIA_TOOL_NAMES) <= set(tools.names())
    for name in MEDIA_TOOL_NAMES:
        spec = tools.get(name)
        assert spec is not None and spec.group == "media" and spec.parameters["additionalProperties"] is False and spec.description
    assert tools.get("film_render").cost_class == "expensive" and tools.get("film_render").timeout_s >= 1800
    assert tools.get("film_look").attaches_media and tools.get("film_compile").permission == "write"
    with pytest.raises(ValueError):
        build_media_registry(include={"nope"})
    assert build_media_registry(include={"film_outline"}).names() == ["film_outline"]


def test_film_compile_outline_and_ops(tools: ToolRegistry, ctx: ToolContext):
    r = call(tools, ctx, "film_compile", film_id="pythagoras", storyboard=STORYBOARD)
    assert r.ok, r.error
    out = r.output
    assert [b["id"] for b in out["beats"]] == [1, 2] and out["beat_count"] == 2 and out["warnings"] == [] and out["estimated_seconds"] == 6.7
    assert out["scene_path"] == "media/films/pythagoras/scene.py"
    scene = ctx.workspace.read_text(out["scene_path"])
    assert "class GeneratedFilm(Film)" in scene and "with self.beat(1, 'Setup'):" in scene and "from manim import *" in scene
    saved = ctx.workspace.read_json(out["storyboard_path"])
    assert saved["next_beat_id"] == 3 and saved["beats"][1]["title"] == "Squares"
    r = call(
        tools, ctx, "film_compile", film_id="pythagoras",
        ops=[
            {"op": "patch_beat", "beat": 2, "find": "Indicate(sq)", "replace": "Indicate(sq, scale_factor=1.1)"},
            {"op": "add_beat", "after": 1, "title": "Aside", "narration": "a short aside", "code": "self.wait(0.5)"},
            {"op": "remove_beat", "beat": 42, "reason": "nope"},
        ],
    )
    assert r.ok, r.error
    assert [a["op"] for a in r.output["applied"]] == ["patch_beat", "add_beat"] and r.output["rejected"][0]["op"] == "remove_beat"
    assert [b["id"] for b in r.output["beats"]] == [1, 3, 2]
    assert "scale_factor=1.1" in ctx.workspace.read_text("media/films/pythagoras/scene.py")
    o = call(tools, ctx, "film_outline", film_id="pythagoras").output
    assert o["title"] == "Pythagoras" and o["rendered"] is False and o["last_render"] is None and [b["id"] for b in o["beats"]] == [1, 3, 2]
    r = call(tools, ctx, "film_compile", film_id="pythagoras", storyboard={"title": "Replaced", "beats": [{"title": "Only", "code": "self.wait(1)"}]})
    assert r.ok and r.output["beats"][0]["id"] == 4  # a replacement never reuses a retired id
    r = call(tools, ctx, "film_compile", film_id="nothing-here")
    assert not r.ok and r.error.category.value == "invalid_input" and "no storyboard" in r.error.message
    r = call(tools, ctx, "film_outline", film_id="nothing-here")
    assert not r.ok and r.error.category.value == "not_found"
    for bad_id in ("../escape", "abc\n", "", "a b", "x" * 65):
        r = call(tools, ctx, "film_compile", film_id=bad_id, storyboard=STORYBOARD)
        assert not r.ok and r.error.category.value == "invalid_input", bad_id
    assert sorted(p.name for p in ctx.workspace.path("films").iterdir()) == ["pythagoras"]


def test_film_compile_syntax_error_names_beat_and_line(tools: ToolRegistry, ctx: ToolContext):
    bad = {"title": "Bad", "beats": [{"title": "ok", "code": "self.wait(1)"}, {"title": "broken", "code": "x = 1\ny = = 2\nself.wait(1)"}]}
    r = call(tools, ctx, "film_compile", film_id="bad", storyboard=bad)
    assert not r.ok
    e = r.error
    assert e.category.value == "invalid_input" and e.details["beat"] == 2 and e.details["line"] == 2 and "beat 2 at line 2" in e.message
    assert e.recovery == {"kind": "inspect_input_contract", "operation": "film_compile"} and "patch_beat" in e.remediation
    assert not ctx.workspace.file_exists("media/films/bad/storyboard.json")


def test_film_render_unavailable_without_manim(tools: ToolRegistry, ctx: ToolContext):
    call(tools, ctx, "film_compile", film_id="pythagoras", storyboard=STORYBOARD)
    ctx.toolchain.binaries["manim"] = None
    ctx.toolchain.modules["manim_py"] = False
    r = call(tools, ctx, "film_render", film_id="pythagoras")
    assert not r.ok
    e = r.error
    assert e.category.value == "unavailable" and e.details["status"] == "unrendered" and e.details["scene_path"] == "media/films/pythagoras/scene.py"
    assert "never blocks the book" in e.remediation and "edgent doctor" in e.remediation and "pip install edgent[media]" in e.remediation
    assert ctx.workspace.file_exists("media/films/pythagoras/storyboard.json")
    r = call(tools, ctx, "film_look", film_id="pythagoras")
    assert not r.ok and r.error.category.value == "unavailable" and "film_render" in r.error.remediation
    r = call(tools, ctx, "video_assemble", film_id="pythagoras")
    assert not r.ok and r.error.category.value == "unavailable"


def test_film_render_with_a_fake_manim_measures(tools: ToolRegistry, ctx: ToolContext):
    call(tools, ctx, "film_compile", film_id="pythagoras", storyboard=STORYBOARD)
    ctx.toolchain.binaries["manim"] = "/fake/bin/manim"
    fake = FakeManim(good_report())
    ctx.extra["media_runner"] = fake
    r = call(tools, ctx, "film_render", film_id="pythagoras", quality="draft")
    assert r.ok, r.error
    out = r.output
    assert out["video_path"] == "media/films/pythagoras/out.mp4" and out["report_path"] == "media/films/pythagoras/report.json"
    assert [t["id"] for t in out["beat_timings"]] == [1, 2] and out["duration_s"] == 1.9 and out["quality"] == "draft"
    kinds = {(d["beat"], d["kind"]) for d in out["defects"]}
    assert (2, "still_card") in kinds and (1, "duration_mismatch") in kinds and out["ok"] is False and out["defect_counts"]["error"] >= 1
    first = fake.calls[0]
    argv = first["argv"]
    assert argv[0] == "/fake/bin/manim" and argv[1:3] == ["-q", "l"] and argv[-2:] == ["scene.py", "GeneratedFilm"] and "--media_dir" in argv
    assert first["allow_network"] is False and first["env_extra"]["EDGENT_FILM_REPORT"].endswith("report.json") and first["timeout_s"] == 1800
    assert first["cwd"] == str(ctx.workspace.path("films") / "pythagoras")
    assert ctx.workspace.file_exists("media/films/pythagoras/out.mp4") and ctx.workspace.read_json("media/films/pythagoras/defects.json")["counts"]["error"] >= 1
    o = call(tools, ctx, "film_outline", film_id="pythagoras").output
    assert o["rendered"] is True and o["last_render"]["quality"] == "draft" and o["last_defects"]["error"] >= 1
    r = call(tools, ctx, "film_render", film_id="pythagoras", only=[2], quality="medium")
    assert r.ok and fake.calls[1]["argv"][1:3] == ["-q", "m"] and r.output["only"] == [2]
    r = call(tools, ctx, "film_render", film_id="pythagoras", only=[9])
    assert not r.ok and r.error.category.value == "invalid_input"
    r = call(tools, ctx, "film_render", film_id="pythagoras", quality="ultra")
    assert not r.ok and r.error.category.value == "invalid_input"


@skip_unless("ffmpeg")
def test_film_look_attaches_frames(tools: ToolRegistry, ctx: ToolContext):
    call(tools, ctx, "film_compile", film_id="pythagoras", storyboard=STORYBOARD)
    ctx.toolchain.binaries["manim"] = "/fake/bin/manim"
    ctx.extra["media_runner"] = FakeManim(good_report())
    assert call(tools, ctx, "film_render", film_id="pythagoras").ok
    ctx.extra.pop("media_runner")  # the frames come from the real ffmpeg
    r = call(tools, ctx, "film_look", film_id="pythagoras")
    assert r.ok, r.error
    frames = r.output["frames"]
    assert [f["beat"] for f in frames] == [1, 2] and frames[0]["time_s"] == 0.6 and frames[0]["path"].startswith("media/films/pythagoras/frames/")
    assert len(r.attachments) == 2 and r.attachments[0].type == "image" and Path(r.attachments[0].path).is_file() and "beat 1" in r.attachments[0].text
    r = call(tools, ctx, "film_look", film_id="pythagoras", beats=[2], max_frames=1)
    assert r.ok and [f["beat"] for f in r.output["frames"]] == [2] and len(r.attachments) == 1
    r = call(tools, ctx, "film_look", film_id="pythagoras", beats=[7])
    assert not r.ok and r.error.category.value == "invalid_input"


def test_failed_render_never_leaves_the_previous_film_behind(tools: ToolRegistry, ctx: ToolContext):
    call(tools, ctx, "film_compile", film_id="pythagoras", storyboard=STORYBOARD)
    ctx.toolchain.binaries["manim"] = "/fake/bin/manim"
    ctx.extra["media_runner"] = FakeManim(good_report())
    assert call(tools, ctx, "film_render", film_id="pythagoras").ok
    assert ctx.workspace.file_exists("media/films/pythagoras/out.mp4")
    partial = good_report()
    partial["complete"] = False
    ctx.extra["media_runner"] = FakeManim(partial, returncode=1, stderr="Traceback ...\nValueError: boom")
    r = call(tools, ctx, "film_render", film_id="pythagoras")
    assert not r.ok and r.error.code == "render_failed"
    assert not ctx.workspace.file_exists("media/films/pythagoras/out.mp4")
    r = call(tools, ctx, "film_look", film_id="pythagoras")
    assert not r.ok and r.error.category.value == "unavailable" and "produced no video" in r.error.message and "film_compile" in r.error.remediation
    o = call(tools, ctx, "film_outline", film_id="pythagoras").output
    assert o["rendered"] is False and o["last_render"]["ok"] is False
    r = call(tools, ctx, "video_assemble", film_id="pythagoras")
    assert not r.ok and r.error.category.value == "unavailable"


def test_only_render_keeps_scene_py_whole(tools: ToolRegistry, ctx: ToolContext):
    call(tools, ctx, "film_compile", film_id="pythagoras", storyboard=STORYBOARD)
    ctx.toolchain.binaries["manim"] = "/fake/bin/manim"
    fake = FakeManim(good_report())
    ctx.extra["media_runner"] = fake
    r = call(tools, ctx, "film_render", film_id="pythagoras", only=[2])
    assert r.ok, r.error
    assert fake.calls[0]["argv"][-2:] == ["scene_only.py", "GeneratedFilm"] and "--progress_bar" in fake.calls[0]["argv"] and fake.calls[0]["max_output"] >= 1_000_000
    whole = ctx.workspace.read_text("media/films/pythagoras/scene.py")
    part = ctx.workspace.read_text("media/films/pythagoras/scene_only.py")
    assert "with self.beat(1, 'Setup'):" in whole and "with self.beat(2, 'Squares'):" in whole
    assert "with self.beat(1, 'Setup'):" not in part and "with self.beat(2, 'Squares'):" in part


def test_film_tools_report_busy_instead_of_racing(tools: ToolRegistry, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch):
    call(tools, ctx, "film_compile", film_id="pythagoras", storyboard=STORYBOARD)
    ctx.toolchain.binaries["manim"] = "/fake/bin/manim"
    monkeypatch.setattr(media, "FILM_LOCK_WAIT_S", 0.2)
    started, release = threading.Event(), threading.Event()
    inner = FakeManim(good_report())

    def blocking(argv: list[str], cwd: str, **kw: Any) -> SandboxResult:
        started.set()
        assert release.wait(20)
        return inner(argv, cwd, **kw)

    ctx.extra["media_runner"] = blocking
    results: list[ToolResult] = []
    t = threading.Thread(target=lambda: results.append(call(tools, ctx, "film_render", film_id="pythagoras")))
    t.start()
    try:
        assert started.wait(10)
        r = call(tools, ctx, "film_compile", film_id="pythagoras", ops=[{"op": "set_beat", "beat": 1, "title": "Changed"}])
        assert not r.ok and r.error.code == "busy" and r.error.category.value == "resource_constraint" and r.error.retryable
        assert call(tools, ctx, "film_outline", film_id="pythagoras").output["beats"][0]["title"] == "Setup"  # nothing was applied
    finally:
        release.set()
        t.join(30)
    assert results and results[0].ok, results[0].error if results else "render never returned"
    assert call(tools, ctx, "film_compile", film_id="pythagoras", ops=[{"op": "set_beat", "beat": 1, "title": "Changed"}]).ok


def test_film_render_caps_inline_defects(tools: ToolRegistry, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch):
    call(tools, ctx, "film_compile", film_id="pythagoras", storyboard=STORYBOARD)
    ctx.toolchain.binaries["manim"] = "/fake/bin/manim"
    ctx.extra["media_runner"] = FakeManim(good_report())
    monkeypatch.setattr(media, "FILM_DEFECTS_INLINE_LIMIT", Limit("film_render inline defects", 1, "defects"))
    r = call(tools, ctx, "film_render", film_id="pythagoras")
    assert r.ok, r.error
    out = r.output
    assert len(out["defects"]) == 1 and out["defects_omitted"] >= 1 and out["defects_path"] == "media/films/pythagoras/defects.json"
    assert len(ctx.workspace.read_json(out["defects_path"])["defects"]) == 1 + out["defects_omitted"]
    assert out["defect_counts"]["error"] + out["defect_counts"]["warn"] == 1 + out["defects_omitted"]


def test_film_render_failure_names_the_beat(tools: ToolRegistry, ctx: ToolContext):
    call(tools, ctx, "film_compile", film_id="pythagoras", storyboard=STORYBOARD)
    ctx.toolchain.binaries["manim"] = "/fake/bin/manim"
    partial = good_report()
    partial["complete"] = False
    partial["beats"] = partial["beats"][:1] + [{**partial["beats"][1], "error": "NameError: name 'sq' is not defined", "mobjects": []}]
    ctx.extra["media_runner"] = FakeManim(partial, returncode=1, stderr="Traceback ...\nNameError: name 'sq' is not defined")
    r = call(tools, ctx, "film_render", film_id="pythagoras")
    assert not r.ok
    e = r.error
    assert e.code == "render_failed" and e.category.value == "execution_failure" and e.details["beat"] == 2 and "NameError" in e.message
    assert e.partial["failed_beat"]["beat"] == 2 and e.partial["beats_completed"] == [1, 2] and "NameError" in e.partial["log_tail"]
    assert r.output == e.partial  # the partial observation survives the failure
    ctx.extra["media_runner"] = lambda argv, cwd, **kw: SandboxResult(-9, "", "", 1800.0, timed_out=True, argv=argv)
    r = call(tools, ctx, "film_render", film_id="pythagoras")
    assert not r.ok and r.error.category.value == "timeout" and "only=[...]" in r.error.remediation


def test_diagram_render_valid_and_invalid(tools: ToolRegistry, ctx: ToolContext):
    r = call(tools, ctx, "diagram_render", diagram_id="bad", svg="<svg viewBox='0 0 10 10'><text>x</text>")
    assert not r.ok and r.error.category.value == "invalid_input" and "not well-formed" in r.error.message
    r = call(tools, ctx, "diagram_render", diagram_id="nons", svg="<svg viewBox='0 0 10 10'><script>alert(1)</script></svg>")
    assert not r.ok and "xmlns" in r.error.message and "<script>" in r.error.message
    r = call(tools, ctx, "diagram_render", diagram_id="handler", svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10" onload="alert(1)"><text font-size="16">x</text></svg>')
    assert not r.ok and r.error.category.value == "invalid_input" and "event handler" in r.error.message
    assert not ctx.workspace.file_exists("media/diagrams/bad.svg") and not ctx.workspace.file_exists("media/diagrams/nons.svg") and not ctx.workspace.file_exists("media/diagrams/handler.svg")
    r = call(tools, ctx, "diagram_render", diagram_id="bayes", svg=VALID_SVG)
    assert r.ok, r.error
    out = r.output
    assert out["svg_path"] == "media/diagrams/bayes.svg" and ctx.workspace.read_text(out["svg_path"]) == VALID_SVG
    assert out["validation"]["text_count"] == 4 and out["validation"]["view_box"] == [0, 0, 300, 120] and out["untrusted"] is True
    kinds = {d["kind"] for d in out["defects"]}
    assert "small_text" in kinds and out["ok"] is False
    if ctx.toolchain.has("chromium"):
        assert out["png_path"] == "media/diagrams/bayes.png" and Path(r.attachments[0].path).is_file() and out["limitations"] == []
        assert {"text_overlap", "clipped"} <= kinds and out["size"] == {"width": 1200, "height": 480}
    else:
        assert out["png_path"] is None and r.attachments == [] and "chromium" in out["limitations"][0]


def test_diagram_render_without_chromium_validates_only(tools: ToolRegistry, ctx: ToolContext):
    ctx.extra["chromium"] = None
    r = call(tools, ctx, "diagram_render", diagram_id="bayes", svg=VALID_SVG)
    assert r.ok, r.error
    assert r.output["png_path"] is None and r.output["limitations"] and r.attachments == []
    assert [d["kind"] for d in r.output["defects"]] == ["small_text"] and r.output["ok"] is False


@skip_unless("chromium")
def test_widget_run_through_the_registry(tools: ToolRegistry, ctx: ToolContext):
    r = call(tools, ctx, "widget_run", widget_id="counter", html=WIDGET_HTML, js=WIDGET_JS, title="Counter")
    assert r.ok, r.error
    out = r.output
    assert out["html_path"] == "media/widgets/counter.html" and out["screenshot_path"] == "media/widgets/counter.png"
    assert out["tests_ran"] is True and out["test_failures"] == [] and out["console_errors"] == ["deliberate"]
    assert out["measurements"]["low_contrast"] and out["measurements"]["overlaps"] == [] and out["measurements"]["overflow"] == []
    kinds = {d["kind"] for d in out["defects"]}
    assert {"console_error", "low_contrast"} <= kinds and out["ok"] is False and out["untrusted"] is True
    assert len(r.attachments) == 1 and r.attachments[0].type == "image" and Path(r.attachments[0].path).is_file()
    saved = ctx.workspace.read_text(out["html_path"])
    assert saved.startswith("<!doctype html>") and 'id="edgent-theme"' in saved and "edgent-pagelab" not in saved and "<title>Counter</title>" in saved
    r = call(tools, ctx, "widget_run", widget_id="quiet", html="<p>Nothing to test</p>")
    assert r.ok, r.error
    assert [d["kind"] for d in r.output["defects"]] == ["no_self_test"] and r.output["ok"] is True


def test_widget_run_without_chromium(tools: ToolRegistry, ctx: ToolContext):
    ctx.extra["chromium"] = None
    r = call(tools, ctx, "widget_run", widget_id="counter", html=WIDGET_HTML, js=WIDGET_JS)
    assert not r.ok and r.error.category.value == "unavailable" and r.output == {"html_path": "media/widgets/counter.html"}
    assert ctx.workspace.file_exists("media/widgets/counter.html")


def test_build_widget_document():
    doc = build_widget_document("<p>x</p>", css="p{color:red}", js="console.log(1)", title="T")
    assert doc.index(HOUSE_THEME_CSS) < doc.index("p{color:red}") < doc.index("<p>x</p>") < doc.index("console.log(1)")
    full = build_widget_document("<html><head><title>Own</title></head><body><p>y</p></body></html>", js="z()")
    assert full.index('id="edgent-theme"') < full.index("<title>Own</title>") and full.index("<script>z()</script>") < full.index("</body>")
    assert "<head>" in build_widget_document("<html><body></body></html>")


def test_tts_narrate_unavailable_and_fake(tools: ToolRegistry, ctx: ToolContext):
    ctx.extra["which"] = lambda name: None
    r = call(tools, ctx, "tts_narrate", text="Hello learner.", out_name="hello")
    assert not r.ok and r.error.category.value == "unavailable" and "espeak-ng" in r.error.remediation
    ctx.extra["which"] = lambda name: "/fake/espeak-ng" if name == "espeak-ng" else None

    def runner(argv: list[str], cwd: str, **kw: Any) -> SandboxResult:
        make_wav(Path(argv[argv.index("-w") + 1]), seconds=2.0)
        return SandboxResult(0, "", "", 0.01, argv=argv)

    ctx.extra["media_runner"] = runner
    r = call(tools, ctx, "tts_narrate", text="Hello learner, welcome back.", out_name="hello")
    assert r.ok, r.error
    assert r.output["path"] == "media/audio/hello.wav" and r.output["backend"] == "espeak-ng" and r.output["duration_s"] == 2.0 and r.output["words"] == 4
    assert ctx.workspace.file_exists("media/audio/hello.wav")
    r = call(tools, ctx, "tts_narrate", text="x", out_name="bad name")
    assert not r.ok and r.error.category.value == "invalid_input"


@skip_unless("ffmpeg")
def test_video_assemble_with_storyboard_captions(tools: ToolRegistry, ctx: ToolContext):
    call(tools, ctx, "film_compile", film_id="pythagoras", storyboard=STORYBOARD)
    make_video(ctx.workspace.path("films") / "pythagoras" / "out.mp4", seconds=2.0)
    ctx.workspace.write_json("media/films/pythagoras/report.json", good_report())
    make_wav(ctx.workspace.path("audio") / "narration.wav", seconds=1.0)
    r = call(tools, ctx, "video_assemble", film_id="pythagoras", narration="media/audio/narration.wav", captions="storyboard")
    assert r.ok, r.error
    out = r.output
    assert out["path"] == "media/films/pythagoras/final.mp4" and out["has_audio"] and out["has_captions"] and out["captions_source"] == "storyboard"
    srt = ctx.workspace.read_text(out["captions_path"])
    assert "00:00:00,000 --> 00:00:01,200" in srt and "right triangle" in srt and "00:00:01,200 --> 00:00:01,900" in srt
    assert abs(out["duration_s"] - 2.0) < 0.3
    r = call(tools, ctx, "video_assemble", film_id="pythagoras", captions=[{"start_s": 0, "end_s": 1, "text": "hi"}])
    assert r.ok and r.output["captions_source"] == "segments" and not r.output["has_audio"]
    r = call(tools, ctx, "video_assemble", film_id="pythagoras", narration="media/audio/missing.wav")
    assert not r.ok and r.error.category.value == "not_found"
    r = call(tools, ctx, "video_assemble", film_id="pythagoras", captions="media/audio/missing.srt")
    assert not r.ok and r.error.category.value == "not_found"
