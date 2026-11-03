---
name: interactive-widgets
description: Building a self-contained HTML widget for widget_run: prefixed ids, no external scripts, the window.__edgentTest self-test, the house theme, what the harness measures.
---
# Interactive widgets

Use this to make something the reader plays with: a parameter they change and a consequence they see. A widget
is one self-contained HTML file at `media/widgets/<id>.html`, saved and run by `widget_run` in headless
Chromium with no network, and embedded in a chapter as `{{widget: <id>}}`. It is finished when its self-test
passes, nothing errors, the measurements are clean, and you have looked at the screenshot.

## When a widget earns its place

Build one when the learning is in the reader's own action: moving a slider and watching the posterior change,
choosing a sample size and seeing the estimate wobble, editing an input and seeing the algorithm's steps. Do
not build one for a static picture (`diagramming`), for a process best watched (`manim-film`), or for a
calculator nobody asked for. One idea per widget; one or two controls; one output the reader watches.

## What widget_run takes and does

Arguments: `widget_id`, `html` (body markup, or a full document), optional `css`, optional `js`, optional
`title`. The tool wraps the markup in a page with the house theme first (your CSS overrides it), runs it at
1280 x 800 with scripts allowed and the network blocked, and reports:

- `console_error`: any uncaught error or `console.error`.
- `test_failure`: each string returned by `window.__edgentTest`; `no_self_test` when it is missing.
- `overflow_x`, `overflow_y`: elements extending past the viewport.
- `text_overlap`: text nodes overlapping, in px.
- `small_text`: text under 12 px.
- `low_contrast`: text under 4.5:1 against its effective background (3:1 for 24 px, or bold 18.66 px).
- A screenshot, attached as an image.

Without Chromium the tool saves the file and reports `unavailable`; the widget is unverified. Record it; do
not block the chapter.

## The contract

1. Self-contained: no external scripts, stylesheets, fonts or images; no `fetch`, no network of any kind.
   Everything inline. Libraries are not available; write the maths and the drawing yourself (a canvas or
   inline SVG is enough).
2. Prefixed ids and classes: every id and every custom class starts with the widget id (`bayes-rate`,
   `bayes-out`), so two widgets on one page cannot collide and the theme's classes stay the theme's.
3. The house theme is applied first and provides: CSS variables `--bg`, `--surface`, `--ink`, `--muted`,
   `--accent`, `--accent-ink`, `--ok`, `--warn`, `--border`, `--radius`, `--font`, `--mono`; classes `.panel`
   (a bordered surface), `.controls` (a flex row of controls), `.row`, `.caption`, `.ok`, `.warn`; sensible
   `button`, `input`, `label`, `h1`-`h3`, `p`, `code` styling; light and dark schemes via
   `prefers-color-scheme`. Use them; add your own rules only for what the theme does not cover.
4. A self-test: `window.__edgentTest` is a function that drives the controls and returns `[]` when everything
   is right, or a list of strings each naming one failure. It may return a promise. The harness calls it after
   the page loads and treats every string as a defect.
5. Accessible by default: every control has a `<label>`; the output is an `<output>` or a live region
   (`aria-live="polite"`); keyboard works (buttons and inputs, not clickable divs); focus is visible (the
   theme does this).
6. Deterministic where it can be: a widget that uses randomness exposes a seed or a "same again" path so the
   self-test can check a known case.

## Structure of a widget

```
<div class="panel" id="bayes-widget">
  <h2>How rare is it?</h2>
  <p class="caption">Move the rate and watch what fraction of positives are real.</p>
  <div class="controls">
    <label for="bayes-rate">Rate of the condition (per 1000)</label>
    <input id="bayes-rate" type="range" min="1" max="500" value="10">
    <output id="bayes-rate-out" for="bayes-rate">10</output>
  </div>
  <div class="row">
    <svg id="bayes-grid" viewBox="0 0 400 120" width="400" height="120" aria-label="positives split"></svg>
    <p>Fraction of positives with the condition: <output id="bayes-out" aria-live="polite">0.15</output></p>
  </div>
</div>
```

And the script, after the markup:

```
(function () {
  const rate = document.getElementById('bayes-rate');
  const rateOut = document.getElementById('bayes-rate-out');
  const out = document.getElementById('bayes-out');
  const grid = document.getElementById('bayes-grid');
  const CATCH = 0.9, FLAG = 0.05;
  function posterior(r) { const tp = r * CATCH, fp = (1 - r) * FLAG; return tp / (tp + fp); }
  function draw(r) {
    const p = posterior(r);
    rateOut.textContent = String(Math.round(r * 1000));
    out.textContent = p.toFixed(2);
    grid.innerHTML = '<rect x="0" y="20" width="' + (400 * p) + '" height="80" fill="var(--accent)"/>'
      + '<rect x="' + (400 * p) + '" y="20" width="' + (400 * (1 - p)) + '" height="80" fill="var(--border)"/>'
      + '<text x="8" y="70" font-size="16" fill="var(--accent-ink)">real</text>';
  }
  rate.addEventListener('input', () => draw(rate.value / 1000));
  draw(rate.value / 1000);
  window.__edgentTest = function () {
    const failures = [];
    rate.value = 10; rate.dispatchEvent(new Event('input'));
    if (out.textContent !== '0.15') failures.push('rate 10/1000 should give 0.15, got ' + out.textContent);
    rate.value = 100; rate.dispatchEvent(new Event('input'));
    if (out.textContent !== '0.67') failures.push('rate 100/1000 should give 0.67, got ' + out.textContent);
    if (!grid.querySelector('rect')) failures.push('the grid did not draw');
    rate.value = 10; rate.dispatchEvent(new Event('input'));  // back to the starting view for the screenshot
    return failures;
  };
})();
```

Everything prefixed, the theme's variables for colour, a label per control, the output in an `<output>`, and a
test that drives the control through two known cases, inspects the result and resets the control.

## Writing the self-test

- Drive the real controls (set the value, dispatch the event, click the button); do not call internal
  functions only.
- Check the visible result (text content, an attribute, an element's existence), not internal state.
- Two or three known cases with exact expected output, including an edge (minimum, maximum, zero, empty).
- Return strings that say what was expected and what happened; the harness shows them verbatim.
- Leave the widget in its initial state at the end (reset the controls), so the screenshot shows the starting
  view.
- Keep it fast: the page has a few seconds of virtual time; no long timers, no animations the test waits on.

## Layout and legibility

- Fits the viewport: a widget is at most 1200 px wide and sits in a `.panel`; nothing scrolls sideways. Height
  under 700 px so the screenshot shows it whole.
- Text at 16 px (the theme's default) for anything read; captions at 14 px; nothing under 12 px, including SVG
  text and canvas text.
- Contrast: use the theme's `--ink` on `--bg` or `--surface`, `--accent-ink` on `--accent`; anything else is
  measured and usually fails. Check both schemes: the theme switches in dark mode and your custom colours must
  still pass.
- Controls at least 40 px tall (the theme's buttons are); sliders full width; labels visible, not
  placeholders.
- One output the reader watches; secondary numbers in a `.caption`.
- No meaning by colour alone in the drawing; add a label or a pattern.

## The loop

1. Write the idea in one sentence, the one or two controls, and the output. Decide the two known cases the
   test will check.
2. Write markup, script and (rarely) CSS as above.
3. `widget_run`. Read, in order: console errors, test failures, measurements; then look at the screenshot.
4. Fix one cause at a time; run again. A console error first: nothing else is trustworthy while the script is
   broken.
5. When clean and the screenshot reads well, embed with `{{widget: <id>}}` and a sentence in the chapter that
   says what to try.

## Anti-patterns

- An external library or font; a `fetch`; anything that needs a server.
- Unprefixed ids (`out`, `btn`); styles that override the theme globally (`body { ... }`).
- A self-test that returns `[]` without driving anything, or none at all.
- Controls without labels; a clickable `div`; a result shown only by colour.
- A dashboard of six controls; a widget that is a calculator for the chapter's exercise.
- Custom grey text; light text on `--surface`; 10 px axis labels.
- Shipping with a console error or a failing test "that does not matter".

## Checklist

- [ ] One idea; one or two labelled controls; one watched output; two known cases decided.
- [ ] Self-contained: no external resources, no network; ids and classes prefixed with the widget id.
- [ ] Theme variables and classes used; custom colours pass contrast in both schemes.
- [ ] `window.__edgentTest` drives the controls, checks visible results on known cases and an edge, returns
      `[]` or specific strings, resets.
- [ ] Fits 1280 x 800 without overflow; text 16 px body, nothing under 12 px; controls 40 px tall.
- [ ] `widget_run` clean: no console errors, no test failures, no measured defects; screenshot looked at.
- [ ] Embedded with `{{widget: <id>}}` and a sentence saying what to try.
