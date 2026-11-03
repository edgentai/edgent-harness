# Example storyboard

A complete storyboard for `film_compile`: four beats, narration written first, seconds set from the words, a
small prelude shared by the beats, every beat staged and animated. Copy the shape, not the subject.

```json
{
  "title": "Bayes on a grid",
  "persona": "first-year undergraduate, knows fractions",
  "prelude": "CATCH = 0.9\nFLAG = 0.05\n\n\ndef people(n, color):\n    return VGroup(*[Dot(color=color, radius=0.08) for _ in range(n)]).arrange_in_grid(rows=2, buff=0.12)\n",
  "show_titles": true,
  "beats": [
    {
      "id": 1,
      "title": "A thousand people",
      "narration": "Start with a thousand people. Ten of them have the condition. The other nine hundred and ninety do not.",
      "code": "sick = people(10, ACCENT)\nhealthy = Rectangle(width=6, height=2.2, color=MUTED)\nlabel_s = Text(\"10 have it\", font_size=LABEL, color=INK).next_to(sick, UP, buff=0.4)\nlabel_h = Text(\"990 do not\", font_size=LABEL, color=INK).next_to(healthy, UP, buff=0.4)\ngroup = VGroup(VGroup(sick, label_s), VGroup(healthy, label_h)).arrange(RIGHT, buff=1.0)\nstage(group)\nself.play(FadeIn(sick), Write(label_s), run_time=2)\nself.play(Create(healthy), Write(label_h), run_time=2)\nself.wait(2)\n",
      "seconds": 8
    },
    {
      "id": 2,
      "title": "The test on the sick",
      "narration": "Of the ten who have it, the test catches nine. One slips through.",
      "code": "self.play(focus(sick, list(sick[:9])), run_time=1.5)\ntag_s = Text(\"9 caught\", font_size=CAPTION, color=MUTED).next_to(sick, DOWN, buff=0.4)\nself.play(Write(tag_s), run_time=1.5)\nself.play(Indicate(sick[9], color=ACCENT_2), run_time=1.5)\nself.wait(1.5)\n",
      "seconds": 6
    },
    {
      "id": 3,
      "title": "The test on the healthy",
      "narration": "Of the nine hundred and ninety who do not, the test wrongly flags about fifty. Fifty is small next to nine hundred and ninety, and large next to nine.",
      "code": "flagged = Rectangle(width=6 * FLAG, height=2.2, color=ACCENT_2, fill_color=ACCENT_2, fill_opacity=0.6)\nflagged.align_to(healthy, LEFT).align_to(healthy, UP)\ntag_h = Text(\"50 flagged\", font_size=CAPTION, color=MUTED).next_to(healthy, DOWN, buff=0.4)\nself.play(FadeIn(flagged), Write(tag_h), run_time=2)\nself.play(Indicate(tag_s, color=ACCENT), Indicate(tag_h, color=ACCENT_2), run_time=2)\nself.wait(2)\n",
      "seconds": 10
    },
    {
      "id": 4,
      "title": "Most positives are healthy",
      "narration": "Fifty-nine people test positive. Only nine of them have the condition. A positive test moves your belief, but not nearly to certainty, because the condition is rare.",
      "code": "self.play(FadeOut(label_s), FadeOut(label_h), run_time=1)\nresult = Text(\"9 of 59 positives have it\", font_size=BODY, color=INK)\nresult.next_to(group, DOWN, buff=0.6)\nclamp_to_safe(result)\nself.play(Write(result), run_time=2)\nself.play(unfocus(sick), run_time=1)\nself.wait(3)\n",
      "seconds": 12
    }
  ]
}
```

What to notice:

- The narration of each beat was written first; `seconds` is roughly the word count divided by 2.6, rounded up
  to leave the picture time to move.
- The prelude holds the constants and a helper the beats share; nothing else.
- Beat 1 builds the picture and stages it once; later beats add to it (`tag_s`, `flagged`, `result`) and clear
  only what the final beat does not need.
- The accent is on the thing being discussed in that beat: the sick in beat 1, the one who slipped through in
  beat 2, the flagged healthy in beat 3.
- Every beat has at least one `self.play`; the last beat holds the result for three seconds.
- Sizes come from `LABEL`, `CAPTION` and `BODY`; nothing is typed as a raw font size.

Edits after the first render go through ops, for example:

```json
{"ops": [
  {"op": "patch_beat", "beat": 3, "find": "fill_opacity=0.6", "replace": "fill_opacity=0.8"},
  {"op": "set_beat", "beat": 4, "seconds": 14}
]}
```
