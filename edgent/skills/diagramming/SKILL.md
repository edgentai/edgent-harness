---
name: diagramming
description: Making an SVG diagram for diagram_render: when a picture earns its place, viewBox and text sizes, label discipline, contrast, no meaning by hue alone, the render-measure-look loop.
---
# Diagramming

Use this to make a figure for a chapter or an answer. A diagram is an SVG file at `media/diagrams/<id>.svg`,
saved and checked by `diagram_render`, embedded in a chapter as `![caption](../../media/diagrams/<id>.svg)`.
It earns its place only when it shows a relationship faster than words can, and it is finished only when it
measures clean and you have looked at it.

## When a picture earns its place

Draw when the idea is a relationship the reader must see at once:
- parts of a whole (the thousand people split into four groups);
- a mapping or a flow (inputs to outputs, states and transitions, a pipeline);
- a geometric fact (vectors adding, a projection, an angle);
- a comparison of shapes or sizes (two distributions, before and after);
- a structure (a tree, a graph, a layered architecture).

Do not draw when:
- the idea is a sequence of steps the reader must follow slowly (prose, or a film if it moves);
- the picture would need a paragraph to decode (it is two pictures, or it is a table);
- the relationship is numeric and the numbers are the point (a table);
- it would only repeat the text ("a box labelled Bayes' theorem").

One idea per diagram. A figure that shows two things shows neither.

## Decide what it must show

Before any SVG, write three lines:
1. The relationship the reader should see, as one sentence ("the positives are a thin slice of the sick and a
   wide slice of the healthy").
2. The two or three labels that carry it, in the reader's words.
3. What to leave out (everything the sentence does not need: axes that are not read, decorations, a legend).

Then the caption, which says what to see, not what the figure is: "The positives (outlined) are 9 of the 10
sick and 50 of the 990 healthy" rather than "Figure 3: the population".

## SVG mechanics for diagram_render

The tool validates, rasterises in Chromium on a white page the size of the viewBox, measures and attaches the
picture. What it requires:

- Root `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 W H">`. The `viewBox` is required (preferred over
  width and height); choose round numbers in user units that you will treat as pixels at render size (800 x
  450 for a wide figure, 600 x 600 square). Rendering is at 1200 px wide by default.
- No `<script>` (a diagram is static; interactive means a widget). Avoid `<foreignObject>`: its text is not
  measured.
- Text as `<text>` elements, with `font-size` in px, 12 px floor, 16 px for labels the reader must read, 20-24
  px for a title inside the figure. Set `font-family="system-ui, sans-serif"`.
- Everything inside the viewBox: anything drawn beyond it is clipped and reported.
- A white background is assumed (the book page). A diagram designed dark carries its own `<rect width="100%"
  height="100%" fill="...">` first.
- Keep the file small: under a hundred elements for most figures; reuse with `<defs>` and `<use>`; no embedded
  raster images.

## Label discipline

- Every shape the reader must identify has a label, next to it, not in a legend. A legend is a sign the
  picture needs decoding.
- Labels are short noun phrases in the reader's vocabulary (the chapter's words, same spelling, same case).
- A label belongs unambiguously to one thing: closer to it than to anything else, or connected by a short
  leader line.
- Labels do not overlap each other or the shapes they name; the measurement reports overlap in px. Move the
  label, shorten it, or add a leader; do not shrink it below the floor.
- Numbers carry units; axes carry a title and units; arrows carry a verb when the direction means something
  ("flows to", "maps to").
- At most seven labelled things; past that, it is two figures.

## Colour and contrast

- Meaning by position, shape, outline and label; colour is a second channel, never the only one. Two groups
  differ by fill *and* by a label or a pattern (`stroke-dasharray`, hatching).
- Two accent colours at most, plus black ink and grey. The accent marks the thing the caption is about.
- Contrast 4.5:1 for text against what it sits on (3:1 for 24 px or larger); measured and reported. Dark ink
  on white; white text only on a dark fill that passes.
- Thin lines (under 1.5 px at render size) vanish in print; use 2 px strokes for structure and 1 px for grids.
- No gradients, shadows or 3D effects; they add nothing and hurt contrast.

## Layout

- Reading order left to right, top to bottom; the cause on the left or top, the effect on the right or bottom.
- Align: shared edges and centre lines; equal gaps. Compute positions from a few constants rather than typing
  coordinates.
- Margin inside the viewBox of at least 16 units so strokes and labels at the edge are not clipped.
- Sizes mean something or are equal: if two boxes differ in size, the reader will look for why.

## The loop

1. Write the three lines and the caption.
2. Write the SVG with constants for the layout (compute coordinates in your head or with `run_python` when the
   figure has many elements).
3. `diagram_render` with the id and the SVG. Read: validation errors (fix before anything else), then the
   measured defects (`text_overlap`, `clipped`, `small_text`, `low_contrast`, `overflow`), then look at the
   attached picture.
4. Fix one cause at a time, re-render. Change the drawing, not the measurement.
5. Look once more at the clean render: does it show the sentence from step 1? Would the persona read it
   without the caption? Is the accent on the thing the caption names?
6. Embed with the caption; the chapter's `check_render` verifies the path exists.

Without Chromium the tool validates only and says so: the look is unverified; record it.

## Computing a layout

Type coordinates from constants, not by eye. For anything with more than a handful of elements, generate the
SVG with `run_python` from a few numbers and paste the result into `diagram_render`:

```
W, H, M = 640, 360, 24                      # viewBox and margin
total, sick, caught, flagged = 1000, 10, 9, 50
bar_w = W - 2 * M
sick_w = bar_w * sick / total               # widths proportional to counts
caught_w = sick_w * caught / sick
flagged_w = (bar_w - sick_w) * flagged / (total - sick)
rows = [f'<rect x="{M}" y="60" width="{bar_w}" height="200" fill="#f4f6f8" stroke="#1b1f24" stroke-width="2"/>',
        f'<rect x="{M}" y="60" width="{sick_w:.1f}" height="200" fill="#0b67c2"/>',
        f'<rect x="{M}" y="60" width="{caught_w:.1f}" height="200" fill="none" stroke="#f9a03f" stroke-width="3"/>',
        f'<rect x="{M + sick_w:.1f}" y="60" width="{flagged_w:.1f}" height="200" fill="none" stroke="#f9a03f" stroke-width="3"/>']
print(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui, sans-serif">' + "".join(rows) + "</svg>")
```

Rules that follow from computing: every size means something (here, a count); equal gaps come from one
constant; a change of data regenerates the figure instead of nudging rectangles; the margin keeps strokes
inside the viewBox. Add the labels afterwards, by hand, at positions derived from the same constants, and let
the measurement tell you if two collide.

## A small example

```
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 360" font-family="system-ui, sans-serif">
  <rect x="40" y="60" width="540" height="200" fill="#f4f6f8" stroke="#1b1f24" stroke-width="2"/>
  <rect x="40" y="60" width="6" height="200" fill="#0b67c2"/>
  <text x="40" y="44" font-size="20" fill="#1b1f24">1000 people</text>
  <text x="56" y="285" font-size="16" fill="#0b67c2">10 have it (thin bar)</text>
  <text x="300" y="285" font-size="16" fill="#1b1f24">990 do not</text>
  <rect x="40" y="60" width="5" height="200" fill="none" stroke="#f9a03f" stroke-width="3"/>
  <rect x="46" y="60" width="27" height="200" fill="none" stroke="#f9a03f" stroke-width="3"/>
  <text x="90" y="330" font-size="16" fill="#1b1f24">outlined: the 59 who test positive (9 sick, 50 healthy)</text>
</svg>
```

One relationship, three labels, two accents, nothing to decode. The widths are proportional (10 and 990 of
1000; 9 of 10 and 50 of 990), which is the point.

## Anti-patterns

- A legend; a title that names the topic instead of the idea; decorative icons.
- Labels under 12 px; grey text on white; meaning carried by red versus green.
- A diagram that repeats the sentence beside it.
- Dozens of elements, every one labelled; or a diagram with no labels.
- Coordinates typed by hand until it looks right, then nothing aligned.
- Shipping a validation-only render as if it were looked at.

## Checklist

- [ ] One relationship, written as a sentence; two or three labels; a caption that says what to see.
- [ ] `xmlns`, `viewBox`, no script, text as `<text>` at 12 px or more (16 for labels), inside the viewBox
      with a margin.
- [ ] Labels next to their things, unambiguous, non-overlapping, in the chapter's words.
- [ ] Meaning by position, shape and label; at most two accents; contrast measured.
- [ ] Aligned layout from constants; reading order left to right, top to bottom.
- [ ] `diagram_render` clean (validation and measurements); looked at the picture; embedded with the caption.
