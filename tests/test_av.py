"""ffmpeg assembly on a synthesised clip (skipped without ffmpeg) and speech synthesis with fake backends."""
from __future__ import annotations

import math
import struct
import subprocess
import wave
from pathlib import Path

import pytest

from edgent.adapters.renderers import av, tts
from edgent.adapters.sandbox import SandboxResult
from edgent.adapters.toolchain import Toolchain
from edgent.domain.errors import ToolFailure
from tests.conftest import skip_unless

ffmpeg_available = skip_unless("ffmpeg")


def make_video(path: Path, seconds: float = 2.0, size: str = "320x240") -> Path:
    """A synthesised test clip: ffmpeg's colour source at 10 fps, no audio."""
    ffmpeg = Toolchain.detect().require("ffmpeg", "the test clip")
    path.parent.mkdir(parents=True, exist_ok=True)
    argv = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"color=c=blue:s={size}:d={seconds}:r=10", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(argv, check=True, timeout=120)
    return path


def make_wav(path: Path, seconds: float = 1.0, rate: int = 16000) -> Path:
    """A 440 Hz tone written with the stdlib, so no toolchain is needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = b"".join(struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * i / rate))) for i in range(int(seconds * rate)))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)
    return path


@ffmpeg_available
def test_probe_and_frames(tmp_path: Path):
    video = make_video(tmp_path / "clip.mp4")
    info = av.probe(video)
    assert abs(info["duration_s"] - 2.0) < 0.2 and info["width"] == 320 and info["height"] == 240
    assert info["has_video"] and not info["has_audio"] and info["fps"] == 10 and info["video_codec"] == "h264"
    frames = av.extract_frames(video, [0.5, 1.5, 9.0], tmp_path / "frames", width=160)
    assert [f["time_s"] for f in frames] == [0.5, 1.5, 9.0]
    assert Path(frames[0]["path"]).is_file() and Path(frames[1]["path"]).is_file()
    assert frames[2]["path"] is None and "past the end" in frames[2]["note"]
    with open(frames[0]["path"], "rb") as f:
        header = f.read(24)
    assert header[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">I", header[16:20])[0] == 160


@ffmpeg_available
def test_concat_mux_and_assemble(tmp_path: Path):
    a = make_video(tmp_path / "a.mp4")
    b = make_video(tmp_path / "b.mp4", seconds=1.0)
    joined = av.concat([a, b], tmp_path / "joined.mp4")
    assert abs(av.probe(joined)["duration_s"] - 3.0) < 0.3 and not (tmp_path / "joined.concat.txt").exists()
    tone = make_wav(tmp_path / "tone.wav", seconds=1.0)
    muxed = av.mux_audio(a, tone, tmp_path / "muxed.mp4")
    info = av.probe(muxed)
    assert info["has_audio"] and info["audio_codec"] == "aac" and abs(info["duration_s"] - 2.0) < 0.3  # audio padded to the picture
    segments = [{"start_s": 0, "end_s": 1, "text": "one"}, {"start_s": 1, "end_s": 2, "text": "two"}]
    out = av.assemble(a, tone, segments, tmp_path / "final.mp4")
    assert out["has_audio"] and out["has_captions"] and Path(out["captions_path"]).read_text(encoding="utf-8").startswith("1\n00:00:00,000 --> 00:00:01,000\none")
    probed = av.probe(out["path"])
    assert probed["has_audio"] and any(s["type"] == "subtitle" for s in probed["streams"]) and abs(out["duration_s"] - 2.0) < 0.3
    plain = av.assemble(a, None, None, tmp_path / "plain.mp4")
    assert not plain["has_audio"] and not plain["has_captions"] and av.probe(plain["path"])["has_video"]
    with pytest.raises(ToolFailure) as info_:
        av.assemble(tmp_path / "missing.mp4", None, None, tmp_path / "x.mp4")
    assert info_.value.error.category.value == "not_found"


def test_srt_and_limits(tmp_path: Path):
    assert av.format_srt_time(3661.5) == "01:01:01,500"
    text = av.srt_text([{"start_s": 0.5, "end_s": 2, "text": " two  words "}, {"start_s": 2, "end_s": 1, "text": "clamped"}, {"start_s": 3, "end_s": 4, "text": "   "}])
    assert text == "1\n00:00:00,500 --> 00:00:02,000\ntwo words\n\n2\n00:00:02,000 --> 00:00:02,500\nclamped\n\n"
    assert av.srt_text([]) == ""
    with pytest.raises(ToolFailure):
        av.srt_text([{"start_s": "soon", "end_s": 1, "text": "x"}])
    with pytest.raises(ToolFailure) as info:
        av.srt_text(["not a segment"])  # type: ignore[list-item]
    assert info.value.error.category.value == "invalid_input" and "start_s" in info.value.error.message
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not really")
    tc = Toolchain()
    tc.binaries["ffmpeg"] = "/fake/ffmpeg"
    with pytest.raises(ToolFailure) as info:
        av.extract_frames(clip, [0.0] * (int(av.FRAMES_PER_CALL_LIMIT.value) + 1), tmp_path / "f", toolchain=tc)
    assert info.value.error.code == "limit:frames per call"
    with pytest.raises(ToolFailure, match="non-negative"):
        av.extract_frames(clip, [-1.0], tmp_path / "f", toolchain=tc)
    none = Toolchain()
    with pytest.raises(ToolFailure) as info:
        av.probe(clip, toolchain=none)
    assert info.value.error.category.value == "unavailable" and "ffprobe" in info.value.error.message
    with pytest.raises(ToolFailure) as info:
        av.concat([clip], tmp_path / "out.mp4", toolchain=none)
    assert info.value.error.category.value == "unavailable" and "ffmpeg" in info.value.error.remediation


def test_relative_paths_reach_ffmpeg_with_an_absolute_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The sandbox pins HOME and TMPDIR to the cwd, so a relative one would point ffmpeg at a directory that is not there."""
    monkeypatch.chdir(tmp_path)
    Path("clip.mp4").write_bytes(b"x")
    tc = Toolchain()
    tc.binaries["ffmpeg"] = "/fake/ffmpeg"
    cwds: list[str] = []

    def runner(argv: list[str], cwd: str, **kw) -> SandboxResult:
        cwds.append(cwd)
        Path(argv[-1]).write_bytes(b"png")
        return SandboxResult(0, "", "", 0.01, argv=argv)

    av.extract_frames("clip.mp4", [0.0], "frames", toolchain=tc, runner=runner)
    assert cwds == [str(tmp_path / "frames")]


def test_ffmpeg_failure_is_typed(tmp_path: Path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    tc = Toolchain()
    tc.binaries["ffmpeg"] = "/fake/ffmpeg"
    tc.binaries["ffprobe"] = "/fake/ffprobe"

    def failing(argv: list[str], cwd: str, **kw) -> SandboxResult:
        assert kw["allow_network"] is False
        return SandboxResult(1, "", "Invalid data found when processing input", 0.01, argv=argv)

    with pytest.raises(ToolFailure) as info:
        av.extract_frames(clip, [0.0], tmp_path / "f", toolchain=tc, runner=failing)
    e = info.value.error
    assert e.code == "ffmpeg_failed" and e.category.value == "execution_failure" and "Invalid data" in e.message

    def slow(argv: list[str], cwd: str, **kw) -> SandboxResult:
        return SandboxResult(-9, "", "", 5.0, timed_out=True, argv=argv)

    with pytest.raises(ToolFailure) as info:
        av.mux_audio(clip, clip, tmp_path / "m.mp4", toolchain=tc, runner=slow, timeout_s=5)
    assert info.value.error.category.value == "timeout" and "5 s" in info.value.error.message


def test_tts_unavailable_lists_the_options(tmp_path: Path):
    with pytest.raises(ToolFailure) as info:
        tts.synthesize("hello", tmp_path / "out.wav", which=lambda name: None, environ={})
    e = info.value.error
    assert e.category.value == "unavailable" and "piper" in e.remediation and "espeak-ng" in e.remediation and "say" in e.remediation
    with pytest.raises(ToolFailure, match="no text"):
        tts.synthesize("   ", tmp_path / "out.wav", which=lambda name: None, environ={})
    with pytest.raises(ToolFailure) as info:
        tts.synthesize("x" * (int(tts.TTS_TEXT_LIMIT.value) + 1), tmp_path / "out.wav", which=lambda name: None, environ={})
    assert info.value.error.code == "limit:tts text"


def test_tts_fake_backends(tmp_path: Path):
    calls: list[tuple[list[str], dict]] = []

    def which(name: str) -> str | None:
        return {"piper": "/fake/piper", "espeak-ng": "/fake/espeak-ng"}.get(name)

    def runner(argv: list[str], cwd: str, **kw) -> SandboxResult:
        calls.append((argv, kw))
        make_wav(Path(argv[argv.index("-w") + 1]), seconds=1.5)
        return SandboxResult(0, "", "", 0.01, argv=argv)

    result = tts.synthesize("Hello there, narrator.", tmp_path / "n.wav", voice="en-us", which=which, runner=runner, environ={}, toolchain=Toolchain())
    assert result["backend"] == "espeak-ng" and result["path"] == str(tmp_path / "n.wav") and result["duration_s"] == 1.5
    argv, kw = calls[0]
    assert argv[0] == "/fake/espeak-ng" and argv[-1] == "--stdin" and argv[argv.index("-v") + 1] == "en-us"
    assert kw["stdin"] == b"Hello there, narrator." and kw["allow_network"] is False
    rows = {r["backend"]: r for r in tts.available_backends(which=which, environ={})}
    assert rows["piper"]["path"] and not rows["piper"]["usable"] and "voice model" in rows["piper"]["note"]
    assert rows["say"]["path"] is None and rows["espeak-ng"]["usable"]
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"onnx")
    piper_calls: list[list[str]] = []

    def piper_runner(argv: list[str], cwd: str, **kw) -> SandboxResult:
        piper_calls.append(argv)
        make_wav(Path(argv[argv.index("--output_file") + 1]), seconds=0.5)
        return SandboxResult(0, "", "", 0.01, argv=argv)

    result = tts.synthesize("Hi", tmp_path / "p.wav", which=which, runner=piper_runner, environ={tts.PIPER_VOICE_ENV: str(model)}, toolchain=Toolchain())
    assert result["backend"] == "piper" and result["voice"] == str(model) and piper_calls[0][:3] == ["/fake/piper", "--model", str(model)]

    def broken(argv: list[str], cwd: str, **kw) -> SandboxResult:
        return SandboxResult(1, "", "no voices", 0.01, argv=argv)

    with pytest.raises(ToolFailure) as info:
        tts.synthesize("Hi", tmp_path / "q.wav", which=which, runner=broken, environ={}, toolchain=Toolchain())
    e = info.value.error
    assert e.code == "tts_failed" and e.category.value == "execution_failure" and "espeak-ng failed" in e.message and "voice model" in e.message


def test_tts_say_is_only_the_macos_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def which(name: str) -> str | None:
        return "/usr/bin/say" if name == "say" else None

    monkeypatch.setattr(tts, "PLATFORM", "linux")
    rows = {r["backend"]: r for r in tts.available_backends(which=which, environ={})}
    assert rows["say"]["path"] is None and not rows["say"]["usable"] and "macOS" in rows["say"]["note"]
    with pytest.raises(ToolFailure) as info:
        tts.synthesize("hello", tmp_path / "out.wav", which=which, environ={}, runner=lambda *a, **k: SandboxResult(0, "", "", 0.0))
    assert info.value.error.category.value == "unavailable"  # a stray `say` is not a speech backend
    monkeypatch.setattr(tts, "PLATFORM", "darwin")
    calls: list[tuple[list[str], dict]] = []

    def runner(argv: list[str], cwd: str, **kw) -> SandboxResult:
        calls.append((argv, kw))
        make_wav(Path(argv[argv.index("-o") + 1]), seconds=0.25)
        return SandboxResult(0, "", "", 0.01, argv=argv)

    result = tts.synthesize("hello", tmp_path / "say.wav", voice="Samantha", which=which, environ={}, runner=runner, toolchain=Toolchain())
    assert result["backend"] == "say" and result["duration_s"] == 0.25
    argv, kw = calls[0]
    assert argv[0] == "/usr/bin/say" and argv[-2:] == ["-f", "-"] and argv[argv.index("-v") + 1] == "Samantha" and kw["stdin"] == b"hello"
