"""The Manim prelude written into every generated scene file.

Edgent never imports Manim itself. ``PRELUDE_SOURCE`` is Python source that the storyboard compiler places at
the top of ``scene.py``; Manim runs it in the sandbox. It defines the house style (a safe frame, four text
sizes with a legibility floor, colour roles on a dark background), the layout helpers a beat may call, and
``Film``, the ``Scene`` subclass whose ``beat`` context manager measures what is on screen so a film can be
verified by code before anyone looks at it. The report goes to the file named by ``EDGENT_FILM_REPORT``.

Keep this module free of f-strings: the prelude is a literal, and braces inside it must survive verbatim.
"""
from __future__ import annotations

SAFE_MARGIN_UNITS = 0.6  # Manim units kept clear on every edge of the 14.22 x 8 frame
TITLE_BAND_UNITS = 1.2  # band at the top of the safe frame reserved for the beat title
FONT_FLOOR = 24  # Manim font size; CAPTION sits on it and nothing smaller is legible at 1080p
TEXT_SIZES: dict[str, int] = {"TITLE": 56, "BODY": 40, "LABEL": 32, "CAPTION": FONT_FLOOR}
MAX_REPORTED_MOBJECTS = 400  # per beat; a scene with more is already illegible
REPORT_ENV = "EDGENT_FILM_REPORT"

PRELUDE_SOURCE = r'''
# ---- Edgent prelude (generated; change the storyboard, not this file) -------------------------------------
from manim import *  # noqa: F401,F403  (the scene is run by Manim, never imported by Edgent)

import contextlib as _contextlib
import json as _json
import os as _os
import time as _time

import manim as _manim
import numpy as np

_config = getattr(_manim, "config", None)
FRAME_WIDTH = float(getattr(_config, "frame_width", 14.222222222222221))
FRAME_HEIGHT = float(getattr(_config, "frame_height", 8.0))

SAFE_MARGIN = 0.6  # units kept clear on every edge
SAFE_LEFT = -FRAME_WIDTH / 2 + SAFE_MARGIN
SAFE_RIGHT = FRAME_WIDTH / 2 - SAFE_MARGIN
SAFE_TOP = FRAME_HEIGHT / 2 - SAFE_MARGIN
SAFE_BOTTOM = -FRAME_HEIGHT / 2 + SAFE_MARGIN
SAFE_WIDTH = SAFE_RIGHT - SAFE_LEFT
SAFE_HEIGHT = SAFE_TOP - SAFE_BOTTOM
TITLE_BAND = 1.2  # units at the top of the safe frame reserved for the beat title
CONTENT_TOP = SAFE_TOP - TITLE_BAND
CONTENT_CENTER = np.array([0.0, (CONTENT_TOP + SAFE_BOTTOM) / 2, 0.0])

# Text sizes are Manim font sizes. CAPTION sits on the floor; nothing smaller is legible at 1080p.
TITLE = 56
BODY = 40
LABEL = 32
CAPTION = 24
FONT_FLOOR = 24

# Colour roles on a dark background. INK for primary text, MUTED for secondary, DIM for what is out of focus.
BACKGROUND = "#0b0f14"
INK = "#f2f4f8"
MUTED = "#aab2bf"
DIM = "#596273"
ACCENT = "#4cc9f0"
ACCENT_2 = "#f9a03f"
DIM_OPACITY = 0.25

MAX_REPORTED_MOBJECTS = 400
_TEXT_TYPES = {
    "Text", "MarkupText", "Paragraph", "MathTex", "Tex", "SingleStringMathTex", "Title", "BulletedList", "Code",
    "DecimalNumber", "Integer", "Variable",
}
_TEXT_ATTRS = ("original_text", "text", "tex_string", "code_string")

if _config is not None:
    try:
        _config.background_color = BACKGROUND
    except Exception:  # noqa: BLE001 - an exotic config object must not stop the render
        pass


def bbox(mobject):
    """(x0, y0, x1, y1) of a mobject in scene units."""
    return (
        float(mobject.get_left()[0]),
        float(mobject.get_bottom()[1]),
        float(mobject.get_right()[0]),
        float(mobject.get_top()[1]),
    )


def clamp_to_safe(mobject, left=SAFE_LEFT, right=SAFE_RIGHT, top=SAFE_TOP, bottom=SAFE_BOTTOM):
    """Scale down (never up) until the mobject fits the safe frame, then shift it inside. Returns the mobject."""
    width, height = right - left, top - bottom
    if mobject.width > 0 and mobject.height > 0:
        factor = min(1.0, width / mobject.width, height / mobject.height)
        if factor < 1.0:
            mobject.scale(factor)
    x0, y0, x1, y1 = bbox(mobject)
    dx = (left - x0) if x0 < left else (right - x1) if x1 > right else 0.0
    dy = (bottom - y0) if y0 < bottom else (top - y1) if y1 > top else 0.0
    if dx or dy:
        mobject.shift(np.array([dx, dy, 0.0]))
    return mobject


def stage(mobject, full=False):
    """Centre a mobject or group in the content band below the title (``full``: the whole safe frame)."""
    top = SAFE_TOP if full else CONTENT_TOP
    clamp_to_safe(mobject, top=top)
    mobject.move_to(np.array([0.0, (top + SAFE_BOTTOM) / 2, 0.0]))
    return mobject


def _members(group):
    subs = getattr(group, "submobjects", None)
    return list(subs) if subs else [group]


def focus(group, keep, dim=DIM_OPACITY):
    """Animations that dim every member of ``group`` except ``keep`` (a member, a list of members, or a descendant).

    Use it inside play: ``self.play(focus(diagram, diagram[2]))``. ``unfocus`` restores the group.
    """
    kept = list(keep) if isinstance(keep, (list, tuple, set)) else [keep]
    anims = []
    for member in _members(group):
        family = member.get_family()
        is_kept = any(k is member or any(k is f for f in family) for k in kept)
        anims.append(member.animate.set_opacity(1.0 if is_kept else dim))
    return AnimationGroup(*anims) if anims else Wait(0.1)


def unfocus(group):
    """Animations that restore full opacity to every member of ``group``."""
    anims = [member.animate.set_opacity(1.0) for member in _members(group)]
    return AnimationGroup(*anims) if anims else Wait(0.1)


class Film(Scene):
    """A Scene whose beats measure themselves. Subclassed by the generated film; never instantiated by hand."""

    show_titles = True
    report_path = _os.environ.get("EDGENT_FILM_REPORT") or ""

    def __init__(self, *args, **kwargs):
        self._edgent_beats = []
        self._edgent_title = None
        self._edgent_title_text = ""
        self._edgent_plays = 0
        self._edgent_waits = 0
        self._edgent_adds = 0
        self._edgent_counting = True
        self._edgent_started = _time.time()
        super().__init__(*args, **kwargs)

    # ---- counting ------------------------------------------------------------------------------------------
    def play(self, *args, **kwargs):
        if getattr(self, "_edgent_counting", False):
            wait_cls = getattr(_manim, "Wait", None)
            if args and wait_cls is not None and all(isinstance(a, wait_cls) for a in args):
                self._edgent_waits += 1
            else:
                self._edgent_plays += 1
        return super().play(*args, **kwargs)

    def add(self, *mobjects):
        if getattr(self, "_edgent_counting", False):
            self._edgent_adds += 1
        return super().add(*mobjects)

    def _play_uncounted(self, *anims, **kwargs):
        self._edgent_counting = False
        try:
            self.play(*anims, **kwargs)
        finally:
            self._edgent_counting = True

    # ---- titles --------------------------------------------------------------------------------------------
    def set_title(self, title):
        """Show ``title`` in the title band, replacing the current one. Not counted as a beat animation."""
        if self._edgent_title is not None and self._edgent_title_text == title:
            return self._edgent_title
        new = Text(title, font_size=TITLE, color=INK)
        clamp_to_safe(new)
        new.move_to(np.array([0.0, SAFE_TOP - TITLE_BAND / 2, 0.0]))
        old = self._edgent_title
        self._edgent_title, self._edgent_title_text = new, title
        if old is None:
            self._play_uncounted(FadeIn(new, run_time=0.4))
        else:
            self._play_uncounted(FadeOut(old, run_time=0.3), FadeIn(new, run_time=0.3))
        return new

    def clear_title(self):
        if self._edgent_title is not None:
            old, self._edgent_title, self._edgent_title_text = self._edgent_title, None, ""
            self._play_uncounted(FadeOut(old, run_time=0.3))

    # ---- beats ---------------------------------------------------------------------------------------------
    @_contextlib.contextmanager
    def beat(self, beat_id, title=""):
        """Run one beat: records timing, animation counts and the screen contents into the report."""
        entry = {
            "id": int(beat_id),
            "title": str(title or ""),
            "start_s": self._scene_time(),
            "end_s": None,
            "duration_s": None,
            "plays": 0,
            "waits": 0,
            "adds": 0,
            "mobjects": [],
            "error": None,
        }
        self._edgent_plays = self._edgent_waits = self._edgent_adds = 0
        if title and self.show_titles:
            self.set_title(str(title))
        try:
            yield entry
        except BaseException as e:
            entry["error"] = (type(e).__name__ + ": " + str(e))[:500]
            raise
        finally:
            entry["end_s"] = self._scene_time()
            entry["duration_s"] = round(entry["end_s"] - entry["start_s"], 3)
            entry["plays"], entry["waits"], entry["adds"] = self._edgent_plays, self._edgent_waits, self._edgent_adds
            try:
                entry["mobjects"] = self._snapshot()
            except Exception as e:  # noqa: BLE001 - measurement must never break the render
                entry["snapshot_error"] = (type(e).__name__ + ": " + str(e))[:300]
            self._edgent_beats.append(entry)
            self._write_report(complete=False)

    def tear_down(self):
        try:
            super().tear_down()
        finally:
            self._write_report(complete=True)

    # ---- measurement ---------------------------------------------------------------------------------------
    def _scene_time(self):
        t = getattr(getattr(self, "renderer", None), "time", None)
        try:
            return round(float(t), 3)
        except (TypeError, ValueError):
            return 0.0

    def _snapshot(self):
        out = []
        title = self._edgent_title

        def visit(m, depth):
            if depth > 12 or len(out) >= MAX_REPORTED_MOBJECTS:
                return
            names = {c.__name__ for c in type(m).__mro__}
            if names & _TEXT_TYPES:
                out.append(self._describe(m, "text", role="title" if m is title else None))
                return
            subs = getattr(m, "submobjects", None) or []
            if subs:
                for s in subs:
                    visit(s, depth + 1)
                has_points = getattr(m, "has_points", None)
                if callable(has_points) and has_points():  # a container that also draws itself (an Arrow and its tip)
                    out.append(self._describe(m, "shape"))
            else:
                out.append(self._describe(m, "shape"))

        for m in list(self.mobjects):
            visit(m, 0)
        return [d for d in out if d is not None]

    def _describe(self, m, kind, role=None):
        x0, y0, x1, y1 = bbox(m)
        if x1 - x0 <= 0 and y1 - y0 <= 0:
            return None
        d = {"kind": kind, "type": type(m).__name__, "bbox": [round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)], "opacity": self._opacity(m)}
        if kind == "text":
            d["text"] = self._text_of(m)
            size = getattr(m, "font_size", None)
            try:
                d["font_size"] = round(float(size), 2) if size is not None else None
            except (TypeError, ValueError):
                d["font_size"] = None
        if role:
            d["role"] = role
        return d

    @staticmethod
    def _text_of(m):
        for attr in _TEXT_ATTRS:
            value = getattr(m, attr, None)
            if isinstance(value, str) and value:
                return value[:200]
        return ""

    @staticmethod
    def _opacity(m):
        values = []
        for getter in ("get_fill_opacity", "get_stroke_opacity"):
            fn = getattr(m, getter, None)
            if callable(fn):
                try:
                    values.append(float(fn()))
                except Exception:  # noqa: BLE001
                    pass
        return round(max(values), 3) if values else 1.0

    def _config_info(self):
        info = {}
        for key in ("pixel_width", "pixel_height", "frame_width", "frame_height", "frame_rate", "quality"):
            value = getattr(_config, key, None)
            info[key] = value if isinstance(value, (int, float, str)) or value is None else str(value)
        info["safe"] = {"left": SAFE_LEFT, "right": SAFE_RIGHT, "top": SAFE_TOP, "bottom": SAFE_BOTTOM, "margin": SAFE_MARGIN, "title_band": TITLE_BAND}
        info["font_floor"] = FONT_FLOOR
        return info

    def _write_report(self, complete):
        path = self.report_path
        if not path:
            return
        report = {
            "version": 1,
            "scene": type(self).__name__,
            "complete": bool(complete),
            "config": self._config_info(),
            "beats": self._edgent_beats,
            "total_s": self._scene_time(),
            "wall_s": round(_time.time() - self._edgent_started, 3),
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            _json.dump(report, f, indent=1, default=str)
        _os.replace(tmp, path)
# ---- end of Edgent prelude ----------------------------------------------------------------------------------
'''

__all__ = ["FONT_FLOOR", "MAX_REPORTED_MOBJECTS", "PRELUDE_SOURCE", "REPORT_ENV", "SAFE_MARGIN_UNITS", "TEXT_SIZES", "TITLE_BAND_UNITS"]
