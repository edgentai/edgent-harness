"""Renderers: Manim films (prelude, storyboard compiler, render and measure), ffmpeg assembly, TTS, SVG.

Each is an adapter over an optional toolchain. Nothing here imports Manim, ffmpeg bindings or a browser
library; the toolchains run as sandboxed subprocesses and their absence is a typed ``unavailable`` error.
"""
