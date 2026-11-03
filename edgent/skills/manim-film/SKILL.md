---
name: manim-film
description: Making a Manim film with film_compile, film_render and film_look: the storyboard JSON contract, prelude helpers and colour roles, pacing and narration, the compile-render-look loop.
---
# Manim film

Use this to make a film for an idea that moves. You never write a scene file: you write a storyboard,
`film_compile` turns it into `media/films/<id>/scene.py` with the house prelude, `film_render` renders it in
the sandbox and measures it, `film_look` shows you frames. Load `narration` for the spoken text and
`visual-quality` for judging the frames.

## The storyboard contract

One JSON object, sent whole to `film_compile` the first time, then edited with ops:

```
{
  "title": "Bayes on a grid",
  "persona": "first-year undergraduate, knows fractions",
  "prelude": "",
  "show_titles": true,
  "beats": [
    {"id": 1, "title": "A thousand people", "narration": "Start with a thousand people...", "code": "...", "seconds": 12},
    ...
  ]
}
```

- `title` and `persona` are strings. `prelude` is optional Python placed before the scene class (helpers and
  constants shared by beats); it must parse.
- `show_titles` (default true): the beat's `title` is shown in the title band at the top of each beat,
  automatically.
- Each beat: `id` (a positive integer, stable; omit on new beats and the compiler allocates one), `title`,
  `narration` (what the narrator says during the beat), `code`, `seconds` (the target duration, number or
  null).
- `code` is the body of a Manim method with `self` as the Scene: lines such as `g = Square();
  self.play(Create(g))`. It is wrapped as `with self.beat(id, title): <code>`. Rules the compiler enforces or
  lints: no `yield`, `await` or `return` in a beat; do not rebind `self` or `beat`; a beat with no
  `self.play`, `self.wait` or `self.add` changes nothing on screen and is a lint warning.
- Mobjects persist across beats (it is one `construct`); what beat 3 leaves on screen is what beat 4 starts
  with. Clear or fade what the next beat does not need.

Ops, applied in order to the saved storyboard: `add_beat {after?, title, narration, code, seconds?}` (`after:
0` for the front), `set_beat {beat, title?, narration?, code?, seconds?}`, `patch_beat {beat, find, replace}`
(`find` must occur exactly once), `move_beat {beat, after}`, `remove_beat {beat, reason}`, `prelude {py,
why}`, `set_meta {name, value}`. A rejected op is reported with its reason and changes nothing; fix it, do not
resend the same op.

## The prelude: what every beat can use

Frame and layout:
- `SAFE_LEFT`, `SAFE_RIGHT`, `SAFE_TOP`, `SAFE_BOTTOM`, `SAFE_WIDTH`, `SAFE_HEIGHT`: the safe frame, 0.6 units
  inside every edge of the 14.22 x 8 frame. Nothing legible lives outside it.
- `TITLE_BAND` (1.2 units at the top of the safe frame, where the beat title goes), `CONTENT_TOP`,
  `CONTENT_CENTER`: the band below the title where content belongs.
- `stage(mobject, full=False)`: scale down (never up) until it fits, then centre it in the content band
  (`full=True`: the whole safe frame). Use it on every group you build.
- `clamp_to_safe(mobject)`: scale down and shift until it is inside the safe frame; returns the mobject.
- `bbox(mobject)`: `(x0, y0, x1, y1)` in scene units, for your own layout arithmetic.
- `focus(group, keep)`: an animation that dims every member of `group` except `keep` (a member, a list, or a
  descendant) to 25 percent opacity: `self.play(focus(diagram, diagram[2]))`. `unfocus(group)` restores it.
- `self.set_title(text)` and `self.clear_title()` when you need a title change inside a beat (the beat's own
  title is set for you).

Text sizes (Manim font sizes): `TITLE` 56, `BODY` 40, `LABEL` 32, `CAPTION` 24. `CAPTION` sits on the floor
(`FONT_FLOOR` 24); anything smaller is a measured defect. Pass them as `font_size=BODY`.

Colour roles on the dark background `BACKGROUND`: `INK` for primary text and the main object, `MUTED` for
secondary text and guides, `DIM` for what is out of focus, `ACCENT` for the one thing the narration is about,
`ACCENT_2` for the second thing when two are compared. No other colours unless the subject has them (a colour
map, a flag); never meaning by hue alone.

## Pacing

- Six to twelve beats for a two- to four-minute film. One idea per beat; the title names it in three to six
  words.
- Narration sets the seconds: about 2.6 words per second. A beat of 30 words runs about 12 seconds; set
  `seconds` to that, and make the animation fill it (`run_time` on plays, `self.wait()` between). The
  measurement flags a beat whose duration misses its target by more than 25 percent (minimum 1.5 s), or, when
  `seconds` is null, misses the narration estimate by more than 35 percent (minimum 2 s).
- Something moves in every beat: at least one `self.play`. A beat with no play that lasts half a second or
  more is a still card, a measured error. Reveal with `Create`, `Write`, `FadeIn`; change with `Transform`,
  `.animate`; point with `Indicate`, `focus`.
- The picture changes under the narration, in the order the narration mentions things. When the narrator says
  "the ten who have it", those ten light up then, not two seconds earlier.
- First beat: the picture's parts, empty of the idea. Last beat: the idea stated as the picture, held for a
  `self.wait(2)`.
- Build, do not replace: add to the picture across beats so the viewer never re-reads a layout. Clear it only
  at a real change of scene.

## Layout rules the measurement enforces

- Everything visible inside the safe frame: text outside is an error, a shape outside is a warning. `stage()`
  and `clamp_to_safe()` exist so you never compute margins by hand.
- Text does not overlap text (an overlap of more than 0.02 units in both directions is an error). Arrange
  labels with `.next_to`, `.arrange`, `VGroup(...).arrange(DOWN, buff=0.4)`, and stage the group.
- Font sizes at or above `CAPTION`; and a text that was scaled down (by `stage` on a big group) below about 17
  px at 1080p is flagged as too small: use fewer words, or split the group.
- At most a few text objects on screen at once; a crowded frame is a measured failure and a pedagogical one.

## The loop

1. Storyboard first, as text: the one idea (`explaining`), the beats with narration written before any code,
   the picture per beat. Write `seconds` from the narration.
2. `film_compile` with the whole storyboard. Read the result: syntax errors name the beat and line; lint
   warnings name beats that do nothing or rebind names. Fix with ops until the compile is clean and the
   outline shows target seconds per beat.
3. `film_render` with `quality: "draft"` (480p, fastest). Read the defects by beat: `beat_error`,
   `still_card`, `duration_mismatch`, `text_overlap`, `outside_safe_frame`, `font_below_floor`,
   `text_too_small`, `missing_beat` (the render stopped before it), `render_incomplete`, `no_report` (the
   scene never reached a beat; read the log).
4. Fix one cause at a time with ops; re-render with `only: [beat ids]` to check a single beat, then the whole
   film. A beat error is a Manim exception: read it, it names the line.
5. `film_look` at the changed beats, and at least once at the whole film (one frame from the middle of each
   beat; the default is six frames, `max_frames` allows twelve, and the result names the beats it skipped, so
   call again with `beats` until every beat was seen). Judge what the measurements cannot: does the picture
   show the idea; is the emphasis where the narration points; is anything cluttered or ambiguous. Fix and
   repeat.
6. When the defect list is empty and the frames read well: `film_render` with `quality: "final"` (1080p), one
   more look, done. (`"medium"`, 720p, exists for a mid-way look at type and spacing; it is not a
   deliverable.) `video_assemble` adds narration audio and captions when the pipeline has them; without a
   synthesiser the captions still come from the storyboard.

Without manim, `film_render` returns `unavailable` with the install hint: stop at the compiled storyboard, say
so, and never block the chapter on it.

## A beat, as an example

```
narration: "Of the ten who have it, the test catches nine. One slips through."
seconds: 9
code:
  sick = VGroup(*[Dot(color=ACCENT) for _ in range(10)]).arrange(RIGHT, buff=0.25)
  label = Text("10 people have it", font_size=LABEL, color=INK).next_to(sick, UP, buff=0.4)
  group = VGroup(sick, label)
  stage(group)
  self.play(FadeIn(sick), Write(label), run_time=2)
  caught = sick[:9]
  self.play(focus(sick, list(caught)), run_time=1.5)
  tag = Text("9 caught", font_size=CAPTION, color=MUTED).next_to(sick, DOWN, buff=0.4)
  self.play(Write(tag), run_time=1.5)
  self.play(Indicate(sick[9], color=ACCENT_2), run_time=1.5)
  self.wait(2)
```

Nine seconds of narration, about nine seconds of animation, everything staged, two text objects, the thing the
narration names lit when it is named.

## Anti-patterns

- Writing code before narration; a film is narration with pictures, not pictures with a voice-over.
- A beat that is a slide: text appears, nothing moves.
- Explaining in on-screen text what the narrator says; the screen shows, the voice tells.
- Positions computed in raw coordinates instead of `stage`, `next_to`, `arrange`.
- Many colours; meaning by colour alone; a new colour per beat.
- Re-rendering the whole film at final quality to fix one beat.
- Shipping a film you measured but never looked at.

## Checklist

- [ ] One idea; six to twelve beats; narration written first; seconds from the words.
- [ ] Every beat: title, narration, code with at least one play, seconds; mobjects staged.
- [ ] Sizes from `TITLE`/`BODY`/`LABEL`/`CAPTION`; colours from the roles; `ACCENT` on the thing being
      discussed.
- [ ] Compile clean (no syntax errors, no lint warnings).
- [ ] Draft render clean: no beat errors, still cards, overlaps, safe-frame crossings, small text, duration
      mismatches.
- [ ] Looked at every beat; emphasis matches narration; no clutter.
- [ ] Final render clean and looked at; or unrendered with the reason recorded.
