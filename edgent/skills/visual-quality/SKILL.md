---
name: visual-quality
description: Judging and fixing rendered output: frame, type and colour rules, the defects the harness measures, the ones that need eyes, and the checklist before a final render.
---
# Visual quality

Use this whenever you render something a reader will look at, and whenever you review one. The harness
measures what it can (overlap, frame, size, contrast, timing, errors) and attaches a picture for what it
cannot. Your job is to fix the measured defects first and then look, honestly, at the picture.

## The frame

- Everything legible lives inside a safe area. Films: 0.6 units inside every edge of the 14.22 x 8 frame, with
  a 1.2-unit title band at the top; use `stage()` and `clamp_to_safe()`. Pages and widgets: inside the
  viewport (1280 x 800 for widgets), nothing overflowing sideways or below. Diagrams: inside the `viewBox`.
- One focal point per frame. The eye goes to the biggest, brightest or only moving thing; put the idea there.
- Reading order is left to right, top to bottom; a layout that asks the eye to jump back is a defect even when
  nothing overlaps.
- Whitespace is not empty: a frame with four objects and air reads faster than one with eight and none. Three
  to five labelled things at once is the ceiling for a learner.
- Alignment: things that belong together share an edge or a centre line (`arrange`, `next_to` with a buff,
  grid layouts). Near-alignment (off by a little) looks like an error.

## Type

- Floors: films, Manim font size 24 (`CAPTION`); pages, diagrams and widgets, 12 px, and 16 px for anything
  the reader must read rather than glance at. Measured; below the floor is a defect.
- Four sizes, no more: title, body, label, caption. Films: `TITLE` 56, `BODY` 40, `LABEL` 32, `CAPTION` 24.
  Pages: the house theme's h1/h2/h3, body 16 px, caption 14 px.
- Line length: under seventy characters on a page (`max-width: 70ch` in the theme); on a film, a text object
  of more than about eight words is two objects or a caption.
- Text does not touch text: a buff of at least 0.3 units on a film, 8 px on a page. Measured as overlap when
  it fails.
- Scaling down a group to fit shrinks its text with it: the measurement flags text under about 17 px tall at
  1080p. Use fewer words instead of a smaller group.
- Math as rendered math (KaTeX on pages, `MathTex` on films), never as a screenshot of math.

## Colour

- Roles, not palettes. Films: `INK` for primary, `MUTED` for secondary, `DIM` for out of focus, `ACCENT` for
  the one thing being discussed, `ACCENT_2` for a second thing when two are compared, on the dark
  `BACKGROUND`. Pages: the theme's `--ink`, `--muted`, `--accent`, `--ok`, `--warn` on `--bg`, in light and
  dark schemes.
- Contrast: 4.5:1 for normal text, 3:1 for large text (24 px, or bold 18.66 px), measured against the
  effective background. Light grey on white and mid-blue on dark fail; check before you choose.
- Never meaning by hue alone: a colour difference carries meaning only with a second channel (a label, a
  shape, a position, a pattern). One reader in twelve does not see the difference between red and green.
- At most two accents on screen. A third colour means a legend, and a legend means the picture is not
  explaining itself.
- Focus by dimming, not by adding: `focus(group, keep)` on a film, opacity on a page. The thing you want seen
  stays as it was; everything else fades.

## The defects the harness measures

Six families; the measurement names the object, the beat or the element, and the amount.

| Family | Film (`film_render`) | Page, widget, diagram (`widget_run`, `diagram_render`) | Fix |
|---|---|---|---|
| Overlap | `text_overlap`: two text objects overlap by more than 0.02 units both ways | `text_overlap`: two text nodes overlap, in px | more buff, `arrange`, fewer words, stage the group |
| Outside the frame | `outside_safe_frame`: an object crosses the safe margin (error for text, warn for shapes) | `overflow_x`, `overflow_y` past the viewport; `clipped` beyond the viewBox | `stage`/`clamp_to_safe`; smaller content; a viewBox that contains the drawing |
| Too small | `font_below_floor` (size under 24); `text_too_small` (scaled under about 17 px at 1080p) | `small_text` under 12 px | the size roles; fewer words; split the group |
| Low contrast | not measured on films (the roles guarantee it; keep to them) | `low_contrast` with the ratio and the requirement | the theme colours; darker ink; lighter background |
| Timing | `still_card` (no play in a beat of 0.5 s or more); `duration_mismatch` (off target by over 25 percent, or off the narration estimate by over 35 percent) | not applicable | animate the idea; adjust `seconds` or `run_time`; split or merge beats |
| Errors | `beat_error` (an exception; the message names the line); `missing_beat`, `render_incomplete`, `no_report` | `console_error`; `test_failure` (the self-test returned strings); `no_self_test`; `pagelab_error` | read the message; fix the cause; never suppress |

Diagrams additionally fail validation before rendering: not well-formed XML, no `xmlns`, no `viewBox` (or
size), a `<script>` or an event handler attribute. Labels under 12 px are a measured `small_text` defect.

Fix measured defects before looking. One cause at a time; re-render the one beat (`only`) or the one page;
then the whole.

## What needs eyes

The measurement passes and the picture can still be wrong. Look for:

- Does the picture show the mechanism, or decorate it? A reader who sees only the frame should be able to say
  what the idea is.
- Is the emphasis where the narration or the caption points? The accent must be on the thing being discussed
  at the moment it is discussed.
- Clutter: anything on screen the current idea does not need. Remove it or dim it.
- Ambiguity: an arrow that could point at two things; a label equidistant from two objects; two shapes that
  differ only by colour.
- Motion that distracts (films): things moving that are not the idea; a transform that reads as a different
  change than intended; a reveal that pre-empts the narration.
- Legible but wrong: the right size and contrast on a label that says the wrong thing, a value that does not
  match the text, an axis without units.
- Layout that reads in the wrong order; a title that is a topic ("Bayes") instead of the beat's idea ("Most
  positives are healthy").
- Light and dark: a page is viewed in both schemes; a diagram sits on a white book page (give it its own
  background rect when it is designed dark).

Say what you looked at: which frames or screenshot, and what you judged. "Measured clean" and "looked at" are
two different claims.

## Reading a measurement report

The render and page tools return a flat list of defects, each `{"kind", "severity", "detail", ...}` with the
beat id or the element that caused it. Work it in this order, because a defect higher in the list makes the
ones below it unreliable:

1. Errors that stop the run: `beat_error`, `no_report`, `render_incomplete`, `console_error`, `test_failure`,
   `pagelab_error`. Read the message; it names the line or the element. Nothing else is trustworthy until
   these are gone.
2. Geometry: `text_overlap`, `outside_safe_frame`, `overflow_x`, `overflow_y`, `clipped`. Fix by layout
   (`stage`, `arrange`, `next_to`, a viewBox that contains the drawing), never by shrinking text below its
   role.
3. Size: `font_below_floor`, `text_too_small`, `small_text`. Fix by fewer words or a split group, not a
   smaller font.
4. Contrast: `low_contrast`. Fix by returning to the theme's colour roles.
5. Timing (films): `still_card`, `duration_mismatch`. Fix by animating the idea or by matching `seconds` to
   the narration; never by speeding the narration.

One cause per round. Re-render only the beat or the page that changed (`only` for films), then the whole
artifact once before looking.

## A worked fix

A draft render of beat 3 of a Bayes film returns three defects:

- `text_overlap` (error): "9 caught" overlaps "50 flagged" by 0.41 x 0.22 units. Cause: both tags were placed
  with `next_to(..., DOWN)` on groups that sit close together. Fix: arrange the two labelled groups with a
  larger `buff` and stage the parent group; do not shrink the tags.
- `outside_safe_frame` (error): the caption crosses the bottom margin. Cause: the caption was placed below a
  group that already touched the safe frame. Fix: `stage(group)` after adding the caption, or
  `clamp_to_safe(caption)`; check that the result is still readable, not just inside.
- `duration_mismatch` (warn): the beat ran 4.2 s; the storyboard targets 10 s. Cause: two short plays and no
  wait. Fix: `run_time=2` on the plays and `self.wait(2)` at the end, so the picture holds while the narrator
  finishes; do not cut narration to fit the animation.

After the three ops, re-render beat 3 with `only: [3]`, confirm an empty list, re-render the whole film, then
`film_look` at beats 2 to 4: the "50 flagged" tag must light up when the narrator says fifty.

## Anti-patterns

- Fixing an overlap by shrinking the text instead of moving it.
- Adding a colour to show emphasis instead of dimming everything else.
- A title that names the topic rather than the idea of the frame.
- Re-rendering the whole film at final quality to fix one beat.
- Treating "measured clean" as "looked at".
- Reporting taste as a defect when reviewing; or reporting nothing because the measurements passed.

## Before a final render

- [ ] Measured defect list empty, or each remaining item named with why it is acceptable (a shape
      intentionally bleeding off the frame; a decorative label at 12 px).
- [ ] Looked at every beat (one frame each) or the screenshot at the viewport size; emphasis matches narration
      or caption.
- [ ] Four type sizes at most; nothing under the floor; lines under seventy characters.
- [ ] Two accents at most; meaning never by hue alone; contrast checked.
- [ ] One focal point per frame; reading order left to right, top to bottom; three to five labelled things at
      once.
- [ ] Titles state the idea of the beat; captions say what to see.
- [ ] Final render at the shipping quality (films: `final`, 1080p), looked at once more.

## Reviewing someone else's render (as critic)

Read the measurements first, then look: a film keeps them in `media/films/<id>/defects.json` and `film_look`
shows the frames; a diagram or widget keeps its last tool result beside its file, in `<file>.defects.json`
(the defects, counts and measurements), which `read_file` returns (say so when it is missing). Its picture is
the `.png` beside it; `read_file` cannot show an image, so you see it only when the task attached it, and
"measured clean" without "looked at" is what you report otherwise. Report each defect with its location (beat
id, element, line), its family, and the fix; high when it misleads (wrong emphasis, wrong value, ambiguous
arrow), medium when it costs clarity (clutter, order, a weak title), low for polish. Do not report taste. Do
not edit.
