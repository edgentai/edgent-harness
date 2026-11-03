"""Discovery of optional toolchains, with honest 'unavailable' errors.

Nothing here installs anything. ``edgent doctor`` prints this report and the
one command that installs what is missing; tools call ``require`` and get a
typed ``unavailable`` error with the same remediation.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..domain.errors import ErrorCategory, ToolFailure

INSTALL_HINTS: dict[str, str] = {
    "ffmpeg": "Install ffmpeg (Debian: sudo apt install ffmpeg; macOS: brew install ffmpeg).",
    "ffprobe": "Install ffmpeg, which provides ffprobe.",
    "pandoc": "Install pandoc (Debian: sudo apt install pandoc; macOS: brew install pandoc).",
    "pdftotext": "Install poppler-utils (Debian: sudo apt install poppler-utils; macOS: brew install poppler), or pip install edgent[pdf].",
    "tesseract": "Install tesseract-ocr (Debian: sudo apt install tesseract-ocr; macOS: brew install tesseract).",
    "chromium": "Install Chromium or Google Chrome, or pip install edgent[browser] && playwright install chromium; or set EDGENT_CHROMIUM to the binary.",
    "node": "Install Node.js 18+ (https://nodejs.org).",
    "manim": "pip install edgent[media] and the Manim system dependencies (cairo, pango, ffmpeg, a TeX distribution with dvisvgm).",
    "latex": "Install a TeX distribution (Debian: sudo apt install texlive texlive-latex-extra dvisvgm; macOS: brew install --cask basictex).",
    "dvisvgm": "Install dvisvgm with your TeX distribution.",
    "pymupdf": "pip install edgent[pdf] (PyMuPDF).",
    "faster_whisper": "pip install edgent[stt] (faster-whisper) for on-device transcription.",
    "playwright": "pip install edgent[browser] && playwright install chromium.",
    "git": "Install git.",
}

_CHROMIUM_CANDIDATES = (
    "chromium",
    "chromium-browser",
    "google-chrome",
    "google-chrome-stable",
    "chrome",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
)


def find_chromium() -> str | None:
    env = os.environ.get("EDGENT_CHROMIUM")
    if env and Path(env).exists():
        return env
    for c in _CHROMIUM_CANDIDATES:
        p = shutil.which(c) if not c.startswith("/") else (c if Path(c).exists() else None)
        if p:
            return p
    pw = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    roots = [Path(pw)] if pw else []
    roots += [Path.home() / ".cache" / "ms-playwright", Path("/opt/pw-browsers")]
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in ("chromium*/chrome-linux/chrome", "chromium*/chrome-linux/headless_shell", "chromium*/chrome-mac*/Chromium.app/Contents/MacOS/Chromium", "chromium"):
            for p in sorted(root.glob(pattern), reverse=True):
                if p.exists() and (p.is_file() or p.is_symlink()):
                    return str(p)
    return None


@dataclass
class Toolchain:
    binaries: dict[str, str | None] = field(default_factory=dict)
    modules: dict[str, bool] = field(default_factory=dict)

    @classmethod
    def detect(cls) -> Toolchain:
        tc = cls()
        for name in ("ffmpeg", "ffprobe", "pandoc", "pdftotext", "tesseract", "node", "manim", "latex", "dvisvgm", "git"):
            tc.binaries[name] = shutil.which(name)
        tc.binaries["chromium"] = find_chromium()
        for mod, key in (("fitz", "pymupdf"), ("faster_whisper", "faster_whisper"), ("playwright", "playwright"), ("manim", "manim_py")):
            tc.modules[key] = importlib.util.find_spec(mod) is not None
        return tc

    def has(self, name: str) -> bool:
        return bool(self.binaries.get(name)) or bool(self.modules.get(name))

    def path(self, name: str) -> str | None:
        return self.binaries.get(name)

    def require(self, name: str, purpose: str = "") -> str:
        p = self.binaries.get(name)
        if p:
            return p
        if self.modules.get(name):
            return name
        hint = INSTALL_HINTS.get(name, f"Install {name}.")
        what = f" for {purpose}" if purpose else ""
        raise ToolFailure(
            "unavailable",
            f"{name} is required{what} and is not installed on this machine.",
            ErrorCategory.unavailable,
            remediation=f"{hint} Then run `edgent doctor` to confirm. Until then, continue with work that does not need it and record this as an unknown.",
            toolchain=name,
        )

    def report(self) -> dict[str, Any]:
        rows = []
        for name, p in sorted(self.binaries.items()):
            rows.append({"name": name, "available": bool(p), "path": p, "hint": None if p else INSTALL_HINTS.get(name)})
        for name, ok in sorted(self.modules.items()):
            rows.append({"name": name, "available": ok, "path": "python module" if ok else None, "hint": None if ok else INSTALL_HINTS.get(name)})
        return {"items": rows, "capabilities": self.capabilities()}

    def capabilities(self) -> dict[str, bool]:
        return {
            "pdf_text": self.has("pdftotext") or self.has("pymupdf"),
            "ocr": self.has("tesseract"),
            "audio_video_probe": self.has("ffprobe"),
            "transcription": self.has("faster_whisper"),
            "render_html_pdf": self.has("pandoc"),
            "browser_checks": self.has("chromium"),
            "manim_films": self.has("manim") or self.has("manim_py"),
            "latex": self.has("latex"),
            "git": self.has("git"),
        }
