"""Starter toolset (§12.4): files, code, image, video, audio, retrieval, sub-agent, ask-user, calculate."""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Optional

from ..core.types import Modality
from ..core.utils import json_dumps, truncate
from ..modality import audio as audio_mod
from ..modality import image as image_mod
from ..modality import video as video_mod
from ..sandbox.runner import run_python, run_subprocess, which
from .registry import ToolContext, ToolFailure, ToolRegistry, ToolSpec

S = {"type": "string"}
I = {"type": "integer"}
N = {"type": "number"}


def _obj(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


def _artifact(ctx: ToolContext, artifact_id: str):
    art = ctx.store.get(artifact_id)
    if art is None:
        raise ToolFailure("not_found", f"artifact {artifact_id} does not exist")
    return art


def _sbx(ctx: ToolContext) -> dict[str, Any]:
    sb = ctx.config.sandbox
    return {"cpu_seconds": sb.cpu_seconds, "memory_bytes": sb.memory_bytes, "allow_network": sb.allow_network}


# ---------------------------------------------------------------- files
def read_file(ctx: ToolContext, a: dict[str, Any]) -> Any:
    text = ctx.workspace.read_text(a["path"])
    off, ln = int(a.get("offset", 0)), int(a.get("length", 8000))
    return {"path": a["path"], "offset": off, "length": min(ln, len(text) - off), "total_chars": len(text), "content": text[off : off + ln], "untrusted": True}


def write_file(ctx: ToolContext, a: dict[str, Any]) -> Any:
    p = ctx.workspace.write_text(a["path"], a["content"])
    return {"path": a["path"], "bytes": os.path.getsize(p)}


def list_files(ctx: ToolContext, a: dict[str, Any]) -> Any:
    return {"files": ctx.workspace.list(a.get("path", "."))[:500]}


def search_files(ctx: ToolContext, a: dict[str, Any]) -> Any:
    try:
        return {"hits": ctx.workspace.search(a["pattern"], a.get("path", "."), int(a.get("max_hits", 50)))}
    except re.error as e:
        raise ToolFailure("invalid_arguments", f"bad regex: {e}")


# ---------------------------------------------------------------- code
def run_code(ctx: ToolContext, a: dict[str, Any]) -> Any:
    r = run_python(a["code"], ctx.workspace.path, timeout_s=min(float(a.get("timeout_s", 30)), ctx.config.sandbox.timeout_s), **_sbx(ctx))
    out = r.to_dict()
    out["ok"] = r.ok
    return out


def run_tests(ctx: ToolContext, a: dict[str, Any]) -> Any:
    target = a.get("target", "tests")
    argv = [sys.executable, "-I", "-m", "unittest", "discover", "-s", target, "-v"] if os.path.isdir(ctx.workspace.resolve(target)) else [sys.executable, "-I", "-m", "unittest", target, "-v"]
    r = run_subprocess(argv, ctx.workspace.path, timeout_s=ctx.config.sandbox.timeout_s, **_sbx(ctx))
    m = re.search(r"Ran (\d+) tests?", r.stderr)
    return {"ok": r.ok, "tests_run": int(m.group(1)) if m else 0, "returncode": r.returncode, "output": truncate(r.stderr or r.stdout, 6000), "timed_out": r.timed_out}


def lint_code(ctx: ToolContext, a: dict[str, Any]) -> Any:
    path = a["path"]
    r = run_subprocess([sys.executable, "-I", "-m", "py_compile", path], ctx.workspace.path, timeout_s=30, **_sbx(ctx))
    issues = [] if r.ok else [r.stderr.strip()]
    try:
        text = ctx.workspace.read_text(path)
        for i, line in enumerate(text.splitlines(), 1):
            if len(line) > 160:
                issues.append(f"{path}:{i}: line longer than 160 chars")
            if line.rstrip() != line:
                issues.append(f"{path}:{i}: trailing whitespace")
    except Exception:
        pass
    return {"ok": not issues, "issues": issues[:100]}


def calculate(ctx: ToolContext, a: dict[str, Any]) -> Any:
    """Anti-counting pattern (§9.7): arithmetic goes through code, not the model's head."""
    expr = a["expression"]
    if not re.fullmatch(r"[0-9a-zA-Z_\s\.\+\-\*/%\(\)\,\[\]<>=!]+", expr):
        raise ToolFailure("invalid_arguments", "expression contains unsupported characters")
    code = ("import math, statistics\n"
            "allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith('_')}\n"
            "allowed.update(sum=sum, min=min, max=max, len=len, abs=abs, round=round, sorted=sorted, mean=statistics.mean, median=statistics.median)\n"
            f"print(repr(eval(compile({expr!r}, '<calc>', 'eval'), {{'__builtins__': {{}}}}, allowed)))\n")
    r = run_python(code, ctx.workspace.path, timeout_s=10, **_sbx(ctx))
    if not r.ok:
        raise ToolFailure("execution_error", r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "evaluation failed")
    return {"expression": expr, "result": r.stdout.strip()}


def count_matches(ctx: ToolContext, a: dict[str, Any]) -> Any:
    art = _artifact(ctx, a["artifact_id"])
    view = ctx.store.get_view(art.id, a.get("view", "text"))
    if view is None or view.text is None:
        raise ToolFailure("not_found", "view has no text")
    try:
        rx = re.compile(a["pattern"], re.I if a.get("ignore_case", True) else 0)
    except re.error as e:
        raise ToolFailure("invalid_arguments", f"bad regex: {e}")
    hits = [{"offset": m.start(), "match": m.group(0)[:80]} for m in rx.finditer(view.text)]
    return {"artifact_id": art.id, "count": len(hits), "hits": hits[:200]}


# ---------------------------------------------------------------- image
def _image_path(ctx: ToolContext, art) -> str:
    if art.modality != Modality.IMAGE:
        raise ToolFailure("invalid_arguments", f"artifact {art.short} is {art.modality.value}, not an image")
    return art.path


def crop_zoom(ctx: ToolContext, a: dict[str, Any]) -> Any:
    art = _artifact(ctx, a["artifact_id"])
    src = _image_path(ctx, art)
    params = {k: a[k] for k in ("x", "y", "width", "height") if k in a}
    params["zoom"] = float(a.get("zoom", 2.0))
    cached = ctx.store.find_derived(art.id, "crop", params)
    if cached is None:
        dst = ctx.workspace.resolve(os.path.join("scratch", f"crop_{art.short}_{a['x']}_{a['y']}_{a['width']}_{a['height']}.png"))
        if not image_mod.crop(src, dst, int(a["x"]), int(a["y"]), int(a["width"]), int(a["height"]), params["zoom"]):
            raise ToolFailure("unavailable", "ffmpeg is required for cropping and is not available or failed", retryable=False)
        cached = ctx.store.put_file(dst, "image/png", Modality.IMAGE, parent_id=art.id, derived_kind="crop", derived_params=params, move=True)
        ctx.store.put_json_view(cached.id, "summary", {"modality": "image", "crop_of": art.id, **params})
    return {"artifact_id": cached.id, "parent": art.id, "box": [a["x"], a["y"], a["width"], a["height"]], "zoom": params["zoom"],
            "attach_image_artifact": cached.id, "hint": "the cropped region is attached as an image if the model accepts images; otherwise call ocr_region"}


def ocr_region(ctx: ToolContext, a: dict[str, Any]) -> Any:
    art = _artifact(ctx, a["artifact_id"])
    src = _image_path(ctx, art)
    if all(k in a for k in ("x", "y", "width", "height")):
        cropped = crop_zoom(ctx, {**a, "zoom": a.get("zoom", 2.0)})
        src = ctx.store.get(cropped["artifact_id"]).path
    res = ctx.sidecars.run_ocr(src)
    if not res.get("available"):
        raise ToolFailure("unavailable", res.get("reason", "OCR unavailable"))
    return {"artifact_id": art.id, "text": res.get("text", ""), "boxes": res.get("boxes", [])[:200], "error": res.get("error"), "untrusted": True}


def describe_region(ctx: ToolContext, a: dict[str, Any]) -> Any:
    art = _artifact(ctx, a["artifact_id"])
    _image_path(ctx, art)
    target = art
    if all(k in a for k in ("x", "y", "width", "height")):
        target = ctx.store.get(crop_zoom(ctx, {**a, "zoom": a.get("zoom", 1.0)})["artifact_id"])
    cap = ctx.sidecars.run_caption(target.path)
    return {"artifact_id": target.id, "caption": cap.get("caption") if cap.get("available") else None,
            "attach_image_artifact": target.id, "note": None if cap.get("available") else "no caption sidecar; the region is attached for native inspection"}


def compare_images(ctx: ToolContext, a: dict[str, Any]) -> Any:
    x, y = _artifact(ctx, a["artifact_a"]), _artifact(ctx, a["artifact_b"])
    if x.id == y.id:
        return {"identical": True, "ssim": 1.0}
    exe = which("ffmpeg")
    if not exe:
        return {"identical": False, "ssim": None, "note": "ffmpeg unavailable; only byte-identity compared"}
    r = run_subprocess([exe, "-hide_banner", "-i", x.path, "-i", y.path, "-lavfi", "[0:v][1:v]scale2ref[a][b];[a][b]ssim", "-f", "null", "-"],
                       ctx.workspace.path, timeout_s=60, cpu_seconds=60, memory_bytes=2 << 30)
    m = re.search(r"All:([0-9.]+)", r.stderr)
    ssim = float(m.group(1)) if m else None
    return {"identical": ssim is not None and ssim > 0.999, "ssim": ssim, "note": None if m else truncate(r.stderr, 300)}


# ---------------------------------------------------------------- video
def _video(ctx: ToolContext, artifact_id: str):
    art = _artifact(ctx, artifact_id)
    if art.modality != Modality.VIDEO:
        raise ToolFailure("invalid_arguments", f"artifact {art.short} is not a video")
    if not video_mod.available():
        raise ToolFailure("unavailable", "ffmpeg/ffprobe are required for video tools")
    return art


def video_probe(ctx: ToolContext, a: dict[str, Any]) -> Any:
    art = _video(ctx, a["artifact_id"])
    v = ctx.store.get_view(art.id, "metadata")
    return {"artifact_id": art.id, **(v.data() if v else video_mod.probe(art.path) or {})}


def video_scenes(ctx: ToolContext, a: dict[str, Any]) -> Any:
    art = _video(ctx, a["artifact_id"])
    v = ctx.store.get_view(art.id, "scenes")
    scenes = v.data() if v else None
    if scenes is None:
        times = video_mod.scene_changes(art.path, float(a.get("threshold", 0.4))) or [0.0]
        scenes = [{"start": s, "end": times[i + 1] if i + 1 < len(times) else None} for i, s in enumerate(times)]
    return {"artifact_id": art.id, "scenes": scenes}


def frame_at(ctx: ToolContext, a: dict[str, Any]) -> Any:
    art = _video(ctx, a["artifact_id"])
    t = round(float(a["t"]), 3)
    cached = ctx.store.find_derived(art.id, "frame", {"t": t})
    if cached is None:
        dst = ctx.workspace.resolve(os.path.join("scratch", f"frame_{art.short}_{int(t * 1000)}.jpg"))
        if not video_mod.frame_at(art.path, t, dst):
            raise ToolFailure("execution_error", f"could not extract frame at t={t}")
        cached = ctx.store.put_file(dst, "image/jpeg", Modality.IMAGE, parent_id=art.id, derived_kind="frame", derived_params={"t": t}, move=True)
        ctx.store.put_json_view(cached.id, "summary", {"modality": "image", "frame_of": art.id, "t": t})
    return {"artifact_id": cached.id, "video": art.id, "t": t, "attach_image_artifact": cached.id}


def frames_between(ctx: ToolContext, a: dict[str, Any]) -> Any:
    art = _video(ctx, a["artifact_id"])
    t1, t2, fps = float(a["t1"]), float(a["t2"]), float(a.get("fps", 1.0))
    if t2 <= t1:
        raise ToolFailure("invalid_arguments", "t2 must be greater than t1")
    out_dir = ctx.workspace.resolve(os.path.join("scratch", f"frames_{art.short}_{int(t1 * 1000)}_{int(t2 * 1000)}_{fps}"))
    frames = video_mod.frames_between(art.path, t1, t2, fps, out_dir, max_frames=int(a.get("max_frames", 16)))
    ids = []
    for fr in frames:
        child = ctx.store.put_file(fr["path"], "image/jpeg", Modality.IMAGE, parent_id=art.id, derived_kind="frame", derived_params={"t": fr["t"]}, move=True)
        ctx.store.put_json_view(child.id, "summary", {"modality": "image", "frame_of": art.id, "t": fr["t"]})
        ids.append({"t": fr["t"], "artifact_id": child.id})
    return {"video": art.id, "frames": ids, "artifact_ids": [f["artifact_id"] for f in ids], "attach_image_artifacts": [f["artifact_id"] for f in ids][:4]}


def transcript_between(ctx: ToolContext, a: dict[str, Any]) -> Any:
    art = _artifact(ctx, a["artifact_id"])
    target = art
    if art.modality == Modality.VIDEO:
        link = ctx.store.get_view(art.id, "audio_track")
        if not link:
            raise ToolFailure("not_found", "video has no audio track")
        target = ctx.store.get(link.data()["artifact_id"])
    segs_view = ctx.store.get_view(target.id, "transcript_segments")
    if segs_view is None or not segs_view.data():
        raise ToolFailure("unavailable", "no transcript available (speech-to-text sidecar not configured or no speech)")
    t1, t2 = float(a.get("t1", 0)), float(a.get("t2", 1e12))
    segs = [s for s in segs_view.data() if s.get("end", 0) >= t1 and s.get("start", 0) <= t2]
    return {"artifact_id": target.id, "t1": t1, "t2": t2, "segments": segs[:300], "text": audio_mod.transcript_text(segs), "untrusted": True}


# ---------------------------------------------------------------- audio
def _audio_wav(ctx: ToolContext, art) -> str:
    meta = ctx.store.get_view(art.id, "metadata")
    if meta and meta.data().get("normalised_path") and os.path.exists(meta.data()["normalised_path"]):
        return meta.data()["normalised_path"]
    return art.path


def transcribe_range(ctx: ToolContext, a: dict[str, Any]) -> Any:
    art = _artifact(ctx, a["artifact_id"])
    if art.modality == Modality.VIDEO:
        link = ctx.store.get_view(art.id, "audio_track")
        if not link:
            raise ToolFailure("not_found", "video has no audio track")
        art = ctx.store.get(link.data()["artifact_id"])
    if art.modality != Modality.AUDIO:
        raise ToolFailure("invalid_arguments", "artifact is not audio")
    src = _audio_wav(ctx, art)
    t1, t2 = float(a.get("t1", 0.0)), float(a.get("t2", 0.0))
    if t2 > t1:
        dst = ctx.workspace.resolve(os.path.join("scratch", f"aud_{art.short}_{int(t1 * 1000)}_{int(t2 * 1000)}.wav"))
        if not audio_mod.extract_range(src, dst, t1, t2):
            raise ToolFailure("unavailable", "ffmpeg is required to cut audio ranges")
        src = dst
    res = ctx.sidecars.run_stt(src)
    if not res.get("available"):
        raise ToolFailure("unavailable", res.get("reason", "speech-to-text unavailable"))
    segs = res.get("segments", [])
    for s in segs:
        s["start"] = s.get("start", 0) + (t1 if t2 > t1 else 0)
        s["end"] = s.get("end", 0) + (t1 if t2 > t1 else 0)
    return {"artifact_id": art.id, "segments": segs, "text": audio_mod.transcript_text(segs), "error": res.get("error"), "untrusted": True}


def diarise(ctx: ToolContext, a: dict[str, Any]) -> Any:
    art = _artifact(ctx, a["artifact_id"])
    v = ctx.store.get_view(art.id, "transcript_segments")
    segs = v.data() if v else []
    speakers = sorted({s.get("speaker") for s in segs if s.get("speaker")})
    if not speakers:
        raise ToolFailure("unavailable", "no speaker labels: the configured speech-to-text sidecar does not diarise")
    return {"artifact_id": art.id, "speakers": speakers, "segments": segs[:300]}


def tag_audio_events(ctx: ToolContext, a: dict[str, Any]) -> Any:
    art = _artifact(ctx, a["artifact_id"])
    res = ctx.sidecars.run_audio_tags(_audio_wav(ctx, art))
    if not res.get("available"):
        raise ToolFailure("unavailable", res.get("reason", "audio tagging unavailable"))
    return {"artifact_id": art.id, "events": res.get("events", [])}


# ---------------------------------------------------------------- retrieval
def fetch_artifact_view(ctx: ToolContext, a: dict[str, Any]) -> Any:
    art = _artifact(ctx, a["artifact_id"])
    name = a.get("view", "text")
    view = ctx.store.get_view(art.id, name)
    if view is None:
        names = [v.name for v in ctx.store.views(art.id)]
        raise ToolFailure("not_found", f"artifact {art.short} has no view '{name}'. Available: {names}")
    text = view.text or ""
    off, ln = int(a.get("offset", 0)), int(a.get("length", 6000))
    return {"artifact_id": art.id, "view": name, "kind": view.kind, "offset": off, "length": min(ln, max(0, len(text) - off)),
            "total_chars": len(text), "text": text[off : off + ln], "meta": {k: v for k, v in view.meta.items() if k != "boxes"}, "untrusted": True}


def search_memory(ctx: ToolContext, a: dict[str, Any]) -> Any:
    if ctx.memory is None:
        raise ToolFailure("unavailable", "memory system not attached")
    hits = ctx.memory.search(a["query"], k=int(a.get("k", 5)), tiers=a.get("tiers"))
    return {"query": a["query"], "hits": hits, "untrusted": True}


def list_artifacts(ctx: ToolContext, a: dict[str, Any]) -> Any:
    arts = ctx.store.list(roots_only=not a.get("include_derived", False))
    return {"artifacts": [{"artifact_id": x.id, "modality": x.modality.value, "mime": x.mime, "size": x.size, "source": x.source,
                           "views": [v.name for v in ctx.store.views(x.id)]} for x in arts[:200]]}


# ---------------------------------------------------------------- meta
def spawn_subagent(ctx: ToolContext, a: dict[str, Any]) -> Any:
    if ctx.spawn is None:
        raise ToolFailure("permission_denied", "sub-agents are not allowed at this effort level or for this role")
    return ctx.spawn(a["role"], a["task"], {"max_tool_calls": int(a.get("max_tool_calls", 8)), "artifacts": a.get("artifacts", [])})


def ask_user(ctx: ToolContext, a: dict[str, Any]) -> Any:
    if ctx.ask_user is None:
        return {"answered": False, "question": a["question"], "note": "non-interactive run: record this as a gap and proceed with the most reasonable assumption, stating it"}
    return {"answered": True, "question": a["question"], "answer": ctx.ask_user(a["question"]), "untrusted": True}


def build_default_registry(idempotency=None, ids=None, metrics=None) -> ToolRegistry:
    reg = ToolRegistry(idempotency=idempotency, ids=ids, metrics=metrics)
    box = {"x": I, "y": I, "width": {"type": "integer", "minimum": 1}, "height": {"type": "integer", "minimum": 1}}
    specs = [
        ToolSpec("read_file", "Read a text file from the run workspace, with optional offset/length paging.", _obj({"path": S, "offset": {"type": "integer", "minimum": 0}, "length": {"type": "integer", "minimum": 1, "maximum": 20000}}, ["path"]), read_file),
        ToolSpec("write_file", "Write a text file into the run workspace (outputs/ for deliverables).", _obj({"path": S, "content": S}, ["path", "content"]), write_file, permission="write", side_effects=True, idempotent=True),
        ToolSpec("list_files", "List files in the run workspace.", _obj({"path": S}, []), list_files),
        ToolSpec("search_files", "Regex search across workspace files.", _obj({"pattern": S, "path": S, "max_hits": I}, ["pattern"]), search_files),
        ToolSpec("run_python", "Run Python code in the sandbox (no network, CPU/memory/time limits). Use for arithmetic, counting, data processing and checking.", _obj({"code": S, "timeout_s": {"type": "number", "minimum": 1, "maximum": 300}}, ["code"]), run_code, permission="compute", timeout_s=320, cost_class="moderate", when_not_to_use="the question can be answered by reading a view; do not use for network access"),
        ToolSpec("run_tests", "Run unittest discovery (or a module) in the workspace sandbox.", _obj({"target": S}, []), run_tests, permission="compute", timeout_s=320, cost_class="moderate"),
        ToolSpec("lint_code", "Compile-check and lint a Python file in the workspace.", _obj({"path": S}, ["path"]), lint_code, permission="compute"),
        ToolSpec("calculate", "Evaluate an arithmetic/math expression exactly (sum, mean, sqrt, ...). Always use this instead of mental arithmetic.", _obj({"expression": S}, ["expression"]), calculate, permission="compute", examples=[{"expression": "sum([12.5, 3.25, 9]) * 1.2"}]),
        ToolSpec("count_matches", "Count regex matches in an artifact's text view (programmatic counting; models miscount).", _obj({"artifact_id": S, "pattern": S, "view": S, "ignore_case": {"type": "boolean"}}, ["artifact_id", "pattern"]), count_matches),
        ToolSpec("crop_zoom", "Crop a region of an image (pixel box) and zoom it, producing a new image artifact to look closer at small details.", _obj({"artifact_id": S, **box, "zoom": {"type": "number", "minimum": 0.1, "maximum": 8}}, ["artifact_id", "x", "y", "width", "height"]), crop_zoom, permission="compute", cost_class="moderate"),
        ToolSpec("ocr_region", "OCR the whole image or a pixel box of it. Returns text with word boxes.", _obj({"artifact_id": S, **box, "zoom": N}, ["artifact_id"]), ocr_region, permission="compute", timeout_s=120, cost_class="moderate"),
        ToolSpec("describe_region", "Attach an image (or a region of it) for close visual inspection; uses a caption sidecar when configured.", _obj({"artifact_id": S, **box, "zoom": N}, ["artifact_id"]), describe_region, permission="compute", cost_class="moderate"),
        ToolSpec("compare_images", "Structural similarity between two image artifacts.", _obj({"artifact_a": S, "artifact_b": S}, ["artifact_a", "artifact_b"]), compare_images, permission="compute"),
        ToolSpec("video_probe", "Duration, resolution, fps and audio presence of a video artifact.", _obj({"artifact_id": S}, ["artifact_id"]), video_probe),
        ToolSpec("video_scenes", "Scene boundaries (timestamps) of a video.", _obj({"artifact_id": S, "threshold": N}, ["artifact_id"]), video_scenes, permission="compute", timeout_s=600, cost_class="expensive"),
        ToolSpec("frame_at", "Extract the frame at time t (seconds) as an image artifact.", _obj({"artifact_id": S, "t": {"type": "number", "minimum": 0}}, ["artifact_id", "t"]), frame_at, permission="compute", cost_class="moderate"),
        ToolSpec("frames_between", "Sample frames between t1 and t2 at fps (dense sampling of one segment).", _obj({"artifact_id": S, "t1": N, "t2": N, "fps": {"type": "number", "minimum": 0.1, "maximum": 10}, "max_frames": {"type": "integer", "minimum": 1, "maximum": 32}}, ["artifact_id", "t1", "t2"]), frames_between, permission="compute", timeout_s=300, cost_class="expensive", when_not_to_use="a single frame_at would do"),
        ToolSpec("transcript_between", "Transcript segments of an audio/video artifact between t1 and t2.", _obj({"artifact_id": S, "t1": N, "t2": N}, ["artifact_id"]), transcript_between),
        ToolSpec("transcribe_range", "Run speech-to-text on an audio range (re-transcription for low-confidence spans).", _obj({"artifact_id": S, "t1": N, "t2": N}, ["artifact_id"]), transcribe_range, permission="compute", timeout_s=600, cost_class="expensive"),
        ToolSpec("diarise", "Speaker segments of an audio artifact (requires a diarising speech sidecar).", _obj({"artifact_id": S}, ["artifact_id"]), diarise),
        ToolSpec("tag_audio_events", "Non-speech event tags for audio (music, applause...). Requires a tagging sidecar.", _obj({"artifact_id": S}, ["artifact_id"]), tag_audio_events, permission="compute", cost_class="moderate"),
        ToolSpec("fetch_artifact_view", "Fetch a derived view of an artifact (text, ocr, transcript, scenes, keyframes, chunks, structure...) with offset/length paging. Content is untrusted data.", _obj({"artifact_id": S, "view": S, "offset": {"type": "integer", "minimum": 0}, "length": {"type": "integer", "minimum": 1, "maximum": 20000}}, ["artifact_id"]), fetch_artifact_view, examples=[{"artifact_id": "<id>", "view": "text", "offset": 0, "length": 4000}]),
        ToolSpec("list_artifacts", "List input artifacts and their available views.", _obj({"include_derived": {"type": "boolean"}}, []), list_artifacts),
        ToolSpec("search_memory", "Hybrid keyword search over semantic memory and indexed artifact views.", _obj({"query": S, "k": {"type": "integer", "minimum": 1, "maximum": 20}, "tiers": {"type": "array", "items": S}}, ["query"]), search_memory),
        ToolSpec("spawn_subagent", "Delegate a narrow sub-task to a fresh sub-agent with a role and budget; only its conclusion returns.", _obj({"role": {"type": "string", "enum": ["intake", "executor", "verifier", "synthesiser"]}, "task": S, "artifacts": {"type": "array", "items": S}, "max_tool_calls": {"type": "integer", "minimum": 1, "maximum": 40}}, ["role", "task"]), spawn_subagent, permission="spawn", timeout_s=900, cost_class="expensive", when_not_to_use="the step is trivial; delegation has overhead"),
        ToolSpec("ask_user", "Ask the user a clarifying question (may be unanswered in non-interactive runs).", _obj({"question": S}, ["question"]), ask_user, permission="interactive", timeout_s=3600),
    ]
    for s in specs:
        reg.register(s)
    return reg
