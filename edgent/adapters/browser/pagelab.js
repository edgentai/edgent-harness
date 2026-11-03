/*
 * Edgent page lab. Injected into the <head> of a wrapper page before any content script runs. After load it
 * runs the page's own self-test (window.__edgentTest, returning an array of failure strings, possibly through a
 * promise), measures the layout, and writes a JSON report into <script id="edgent-report"> appended to <body>
 * for --dump-dom to pick up. It touches no network and leaves one global, window.__edgentPagelab.
 *
 * Measurements: elements overflowing the viewport (outermost only, ignoring regions an ancestor clips or
 * scrolls), SVG content cut off by its drawing's viewBox, text that overlaps other text (per line box, leaf
 * text nodes only), text below 12 px, and text whose contrast against its effective background is below WCAG
 * AA (the background walks up the ancestors; a label inside a drawing is measured against the filled shape
 * under it). Dead controls are not measurable from outside (a button without a handler is indistinguishable
 * from one wired by delegation), so they are not reported.
 */
(function () {
  'use strict';
  var VERSION = 1;
  var MAX_ITEMS = 25;            // per measurement list
  var MAX_TEXT_ELEMENTS = 600;   // the overlap check is pairwise
  var MAX_RECTS_PER_ELEMENT = 40;
  var MIN_FONT_PX = 12;
  var MIN_CONTRAST = 4.5;        // WCAG AA, normal text
  var MIN_CONTRAST_LARGE = 3.0;  // WCAG AA, large text: >= 24 px, or bold >= 18.66 px
  var SETTLE_MS = 50;            // let post-load layout finish before measuring
  var TEST_TIMEOUT_MS = 2000;    // a self-test that never settles is itself a failure
  var SVG_NS = 'http://www.w3.org/2000/svg';
  var SKIP_TAGS = { script: 1, style: 1, noscript: 1, title: 1, template: 1, head: 1, meta: 1, link: 1 };
  var SHAPE_TAGS = { rect: 1, circle: 1, ellipse: 1, path: 1, polygon: 1 };  // filled drawing shapes a label can sit on

  var consoleErrors = [];
  var pagelabErrors = [];

  function errText(e) {
    if (e == null) return 'unknown error';
    if (e instanceof Error) return (e.name || 'Error') + ': ' + (e.message || '');
    try { return typeof e === 'string' ? e : JSON.stringify(e); } catch (x) { return String(e); }
  }
  function joinArgs(args) {
    var parts = [];
    for (var i = 0; i < args.length; i++) parts.push(errText(args[i]));
    return parts.join(' ').slice(0, 500);
  }
  var originalError = console.error;
  console.error = function () {
    try { consoleErrors.push(joinArgs(arguments)); } catch (e) { /* never break the page */ }
    try { originalError.apply(console, arguments); } catch (e) { /* ignore */ }
  };
  window.addEventListener('error', function (ev) {
    var msg = ev && ev.message ? ev.message : 'error';
    if (ev && ev.filename) msg += ' (' + String(ev.filename).split('/').pop() + ':' + ev.lineno + ')';
    consoleErrors.push(msg.slice(0, 500));
  });
  window.addEventListener('unhandledrejection', function (ev) {
    consoleErrors.push(('unhandled rejection: ' + errText(ev && ev.reason)).slice(0, 500));
  });

  // ---- helpers --------------------------------------------------------------------------------------------
  function isSvg(el) { return el.namespaceURI === SVG_NS; }
  function tagOf(el) { return (el.tagName || '').toLowerCase(); }
  function describe(el) {
    var s = tagOf(el);
    if (el.id) s += '#' + el.id;
    var cls = typeof el.className === 'string' ? el.className : (el.className && el.className.baseVal) || '';
    var names = cls.trim().split(/\s+/).filter(Boolean).slice(0, 2);
    if (names.length) s += '.' + names.join('.');
    var text = (el.textContent || '').replace(/\s+/g, ' ').trim();
    if (text) s += ' "' + text.slice(0, 40) + (text.length > 40 ? '…' : '') + '"';
    return s;
  }
  function round1(x) { return Math.round(x * 10) / 10; }
  function styleOf(el) { try { return getComputedStyle(el); } catch (e) { return null; } }
  function visible(el) {
    var cs = styleOf(el);
    if (!cs) return false;
    return cs.display !== 'none' && cs.visibility !== 'hidden' && parseFloat(cs.opacity) > 0.05;
  }
  function viewport() {
    return { width: window.innerWidth || document.documentElement.clientWidth, height: window.innerHeight || document.documentElement.clientHeight };
  }
  function pageRect(r) {
    return { left: r.left + window.scrollX, top: r.top + window.scrollY, right: r.right + window.scrollX, bottom: r.bottom + window.scrollY, width: r.width, height: r.height };
  }

  // ---- text boxes -----------------------------------------------------------------------------------------
  function textBoxes() {
    var boxes = [];
    var byElement = new Map();
    var walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null);
    var node;
    while ((node = walker.nextNode())) {
      if (!node.nodeValue || !node.nodeValue.trim()) continue;
      var el = node.parentElement;
      if (!el || SKIP_TAGS[tagOf(el)] || !visible(el)) continue;
      var rects = [];
      try {
        var range = document.createRange();
        range.selectNodeContents(node);
        var list = range.getClientRects();
        for (var i = 0; i < list.length && i < MAX_RECTS_PER_ELEMENT; i++) if (list[i].width > 0 && list[i].height > 0) rects.push(pageRect(list[i]));
      } catch (e) { /* fall through to the element box */ }
      if (!rects.length) {
        var r = el.getBoundingClientRect();
        if (r.width > 0 && r.height > 0) rects.push(pageRect(r));
      }
      if (!rects.length) continue;
      var box = byElement.get(el);
      if (!box) {
        if (boxes.length >= MAX_TEXT_ELEMENTS) break;
        box = { el: el, rects: [] };
        byElement.set(el, box);
        boxes.push(box);
      }
      for (var j = 0; j < rects.length && box.rects.length < MAX_RECTS_PER_ELEMENT; j++) box.rects.push(rects[j]);
    }
    return boxes;
  }

  function overlaps(boxes) {
    var out = [];
    for (var i = 0; i < boxes.length && out.length < MAX_ITEMS; i++) {
      for (var j = i + 1; j < boxes.length && out.length < MAX_ITEMS; j++) {
        var a = boxes[i], b = boxes[j];
        if (a.el.contains(b.el) || b.el.contains(a.el)) continue;
        var best = null;
        for (var p = 0; p < a.rects.length; p++) {
          for (var q = 0; q < b.rects.length; q++) {
            var w = Math.min(a.rects[p].right, b.rects[q].right) - Math.max(a.rects[p].left, b.rects[q].left);
            var h = Math.min(a.rects[p].bottom, b.rects[q].bottom) - Math.max(a.rects[p].top, b.rects[q].top);
            if (w > 1 && h > 1 && (!best || w * h > best[0] * best[1])) best = [w, h];
          }
        }
        if (best) out.push({ a: describe(a.el), b: describe(b.el), overlap_px: [Math.round(best[0]), Math.round(best[1])] });
      }
    }
    return out;
  }

  // ---- overflow -------------------------------------------------------------------------------------------
  function clippedBy(el) {
    for (var n = el.parentElement; n && n !== document.body && n !== document.documentElement; n = n.parentElement) {
      var cs = styleOf(n);
      if (cs && (cs.overflowX !== 'visible' || cs.overflowY !== 'visible')) return true;
    }
    return false;
  }
  function overflow() {
    var vp = viewport();
    var out = [];
    var flagged = new Set();
    var all = document.body.getElementsByTagName('*');
    for (var i = 0; i < all.length && out.length < MAX_ITEMS; i++) {
      var el = all[i];
      if (SKIP_TAGS[tagOf(el)] || !visible(el)) continue;
      if (isSvg(el) && el.ownerSVGElement) continue;  // drawing content is judged against its viewBox instead
      var r = el.getBoundingClientRect();
      if (r.width === 0 && r.height === 0) continue;
      var box = pageRect(r);
      var dx = Math.max(box.right - vp.width, -box.left);
      var dy = Math.max(box.bottom - vp.height, -box.top);
      if (dx <= 1 && dy <= 1) continue;
      var ancestorFlagged = false;
      for (var n = el.parentElement; n; n = n.parentElement) if (flagged.has(n)) { ancestorFlagged = true; break; }
      flagged.add(el);
      if (ancestorFlagged || clippedBy(el)) continue;
      out.push({ element: describe(el), axis: dx > 1 && dy > 1 ? 'xy' : dx > 1 ? 'x' : 'y', overflow_px: [Math.max(0, Math.round(dx)), Math.max(0, Math.round(dy))] });
    }
    return out;
  }

  // ---- clipped drawing content ---------------------------------------------------------------------------
  function clipped() {
    var out = [];
    var all = document.body.getElementsByTagName('*');
    for (var i = 0; i < all.length && out.length < MAX_ITEMS; i++) {
      var el = all[i];
      var root = isSvg(el) ? el.ownerSVGElement : null;
      if (!root || el.children.length || !visible(el)) continue;
      var rs = styleOf(root);
      if (rs && rs.overflow === 'visible') continue;
      var r = el.getBoundingClientRect();
      if (r.width === 0 && r.height === 0) continue;
      var box = root.getBoundingClientRect();
      var dx = Math.max(box.left - r.left, r.right - box.right);
      var dy = Math.max(box.top - r.top, r.bottom - box.bottom);
      if (dx > 1 || dy > 1) out.push({ element: describe(el), beyond_px: [Math.max(0, Math.round(dx)), Math.max(0, Math.round(dy))] });
    }
    return out;
  }

  // ---- font size and contrast ----------------------------------------------------------------------------
  function smallText(boxes) {
    var out = [];
    for (var i = 0; i < boxes.length && out.length < MAX_ITEMS; i++) {
      var cs = styleOf(boxes[i].el);
      var px = cs ? parseFloat(cs.fontSize) : NaN;
      if (px && px < MIN_FONT_PX) out.push({ element: describe(boxes[i].el), font_px: round1(px), min_px: MIN_FONT_PX });
    }
    return out;
  }
  function parseColor(s) {
    var m = /rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)(?:\s*[,/]\s*([\d.]+%?))?\s*\)/.exec(s || '');
    if (!m) return null;
    var a = 1;
    if (m[4] !== undefined) a = m[4].slice(-1) === '%' ? parseFloat(m[4]) / 100 : parseFloat(m[4]);
    return { r: parseFloat(m[1]), g: parseFloat(m[2]), b: parseFloat(m[3]), a: a };
  }
  var WHITE = { r: 255, g: 255, b: 255, a: 1 };
  function composite(fg, bg) {
    var a = fg.a;
    return { r: fg.r * a + bg.r * (1 - a), g: fg.g * a + bg.g * (1 - a), b: fg.b * a + bg.b * (1 - a), a: 1 };
  }
  function effectiveBackground(el) {
    for (var n = el; n; n = n.parentElement) {
      var cs = styleOf(n);
      if (!cs) return null;
      if (cs.backgroundImage && cs.backgroundImage !== 'none') return null;  // gradients and images: unknown
      var c = parseColor(cs.backgroundColor);
      if (c && c.a > 0) {
        if (c.a >= 1) return c;
        var below = n.parentElement ? effectiveBackground(n.parentElement) : WHITE;
        return below ? composite(c, below) : null;
      }
    }
    return WHITE;
  }
  function shapeBehind(el, rect) {
    // A label inside a drawing sits on whatever filled shape is under it, not on the page: hit-test its centre.
    if (!isSvg(el) || !document.elementsFromPoint || !rect) return null;
    var x = (rect.left + rect.right) / 2 - window.scrollX, y = (rect.top + rect.bottom) / 2 - window.scrollY;
    var list;
    try { list = document.elementsFromPoint(x, y); } catch (e) { return null; }
    for (var i = 0; i < list.length; i++) {
      var n = list[i];
      if (n === el || n.contains(el) || !isSvg(n) || !SHAPE_TAGS[tagOf(n)]) continue;
      var cs = styleOf(n);
      var c = cs ? parseColor(cs.fill) : null;
      if (!c || c.a === 0) continue;
      var fo = cs ? parseFloat(cs.fillOpacity) : 1;
      if (!isNaN(fo) && fo < 1) c = { r: c.r, g: c.g, b: c.b, a: c.a * fo };
      if (c.a <= 0) continue;
      return c.a >= 1 ? c : composite(c, effectiveBackground(n) || WHITE);
    }
    return null;
  }
  function luminance(c) {
    function chan(v) { v = v / 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); }
    return 0.2126 * chan(c.r) + 0.7152 * chan(c.g) + 0.0722 * chan(c.b);
  }
  function contrastRatio(a, b) {
    var la = luminance(a), lb = luminance(b);
    return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
  }
  function cssColor(c) { return 'rgb(' + Math.round(c.r) + ', ' + Math.round(c.g) + ', ' + Math.round(c.b) + ')'; }
  function lowContrast(boxes) {
    var out = [];
    for (var i = 0; i < boxes.length && out.length < MAX_ITEMS; i++) {
      var el = boxes[i].el;
      var cs = styleOf(el);
      if (!cs) continue;
      var fg = parseColor(isSvg(el) ? cs.fill : cs.color);
      if (!fg || fg.a === 0) continue;
      var bg = shapeBehind(el, boxes[i].rects[0]) || effectiveBackground(el);
      if (!bg) continue;
      if (fg.a < 1) fg = composite(fg, bg);
      var ratio = contrastRatio(fg, bg);
      var px = parseFloat(cs.fontSize) || 16;
      var bold = parseInt(cs.fontWeight, 10) >= 700 || cs.fontWeight === 'bold';
      var required = px >= 24 || (bold && px >= 18.66) ? MIN_CONTRAST_LARGE : MIN_CONTRAST;
      if (ratio < required) out.push({ element: describe(el), ratio: round1(ratio), required: required, fg: cssColor(fg), bg: cssColor(bg) });
    }
    return out;
  }

  // ---- the self-test --------------------------------------------------------------------------------------
  function normaliseFailures(value) {
    if (value == null || value === true) return [];
    if (value === false) return ['__edgentTest returned false'];
    if (Array.isArray(value)) return value.filter(function (v) { return v != null && v !== ''; }).map(errText).slice(0, 50);
    return [errText(value)];
  }
  function runTests(done) {
    if (typeof window.__edgentTest !== 'function') { done({ ran: false, failures: [] }); return; }
    var finished = false;
    function finish(failures) { if (finished) return; finished = true; done({ ran: true, failures: failures }); }
    var timer = setTimeout(function () { finish(['__edgentTest did not finish within ' + TEST_TIMEOUT_MS + ' ms']); }, TEST_TIMEOUT_MS);
    try {
      var result = window.__edgentTest();
      Promise.resolve(result).then(
        function (value) { clearTimeout(timer); finish(normaliseFailures(value)); },
        function (err) { clearTimeout(timer); finish(['__edgentTest rejected: ' + errText(err)]); }
      );
    } catch (err) {
      clearTimeout(timer);
      finish(['__edgentTest threw: ' + errText(err)]);
    }
  }

  // ---- report ---------------------------------------------------------------------------------------------
  function measure() {
    var boxes = textBoxes();
    var images = document.images ? document.images.length : 0;
    var noAlt = 0;
    for (var i = 0; i < images; i++) if (!document.images[i].hasAttribute('alt')) noAlt++;
    return {
      measurements: { overflow: overflow(), clipped: clipped(), overlaps: overlaps(boxes), small_text: smallText(boxes), low_contrast: lowContrast(boxes) },
      counts: { elements: document.body.getElementsByTagName('*').length, text_elements: boxes.length, images: images, images_without_alt: noAlt },
      document: { width: document.documentElement.scrollWidth, height: document.documentElement.scrollHeight }
    };
  }
  function build(tests) {
    var report = { version: VERSION, title: document.title || '', viewport: viewport(), console_errors: consoleErrors.slice(0, 50), tests_ran: tests.ran, test_failures: tests.failures, pagelab_errors: pagelabErrors };
    try {
      var m = measure();
      report.measurements = m.measurements;
      report.counts = m.counts;
      report.document = m.document;
    } catch (e) {
      pagelabErrors.push('measurement failed: ' + errText(e));
      report.measurements = { overflow: [], clipped: [], overlaps: [], small_text: [], low_contrast: [] };
    }
    return report;
  }
  function write(report) {
    var old = document.getElementById('edgent-report');
    if (old && old.parentNode) old.parentNode.removeChild(old);
    var script = document.createElement('script');
    script.id = 'edgent-report';
    script.type = 'application/json';
    script.textContent = JSON.stringify(report).replace(/</g, '\\u003c');
    (document.body || document.documentElement).appendChild(script);
  }
  function start() {
    setTimeout(function () {
      runTests(function (tests) {
        try { write(build(tests)); } catch (e) { write({ version: VERSION, console_errors: consoleErrors, pagelab_errors: ['pagelab failed: ' + errText(e)], tests_ran: tests.ran, test_failures: tests.failures }); }
      });
    }, SETTLE_MS);
  }
  window.__edgentPagelab = { version: VERSION, consoleErrors: consoleErrors };
  if (document.readyState === 'complete') start(); else window.addEventListener('load', start);
})();
