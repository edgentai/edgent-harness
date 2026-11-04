from edgent.adapters.markdown import markdown_to_html, page

DOC = """---
title: Vectors
unit_id: chapter-vectors
introduces: [vector, norm]
---
# Vectors

A vector is an arrow [textbook:p.12]. Inline $x^2$ and **bold** and `code` and [link](http://x).

## Norm

$$
\\|v\\| = \\sqrt{v \\cdot v}
$$

- one
- two
  1. nested a
  2. nested b

```python
print(1 + 1)
```

| a | b |
|---|---|
| 1 | 2 |

> a quote

{{film: f1}}

---

Last para with <script>alert(1)</script>.
"""


def test_markdown_to_html():
    body, meta = markdown_to_html(DOC)
    assert meta["title"] == "Vectors" and meta["introduces"] == ["vector", "norm"]
    assert '<section id="vectors" data-section="vectors">' in body and '<h1 id="h-vectors">Vectors</h1>' in body
    assert 'data-locator="textbook:p.12"' in body and "$x^2$" in body and "<strong>bold</strong>" in body and "<code>code</code>" in body
    assert '<a href="http://x">link</a>' in body
    assert '<div class="math-block">' in body and "sqrt" in body
    assert "<ul>" in body and "<ol>" in body and "<li>nested a</li>" in body
    assert '<pre><code class="language-python">print(1 + 1)</code></pre>' in body
    assert "<table>" in body and "<th>a</th>" in body and "<td>2</td>" in body
    assert "<blockquote><p>a quote</p></blockquote>" in body
    assert 'class="placeholder placeholder-film" data-kind="film" data-id="f1"' in body
    assert "<hr>" in body and "&lt;script&gt;" in body and "<script>" not in body
    assert page("T", body).startswith("<!doctype html>") and "</section>" in body


def test_attribute_values_are_escaped_and_schemes_filtered():
    body, _ = markdown_to_html('![x](x"onerror="alert(1)) and [a](http://x/?q="1") and ![alt "q"](i.png)')
    assert 'src="x&quot;onerror=&quot;alert(1"' in body and 'onerror="' not in body
    assert '<a href="http://x/?q=&quot;1&quot;">a</a>' in body and 'alt="alt &quot;q&quot;"' in body
    body, _ = markdown_to_html("[j](javascript:alert(1)) [v](vbscript:x) [d](data:text/html,hi) ![s](javascript:alert(1))")
    assert body.count('href="#"') == 3 and 'src="#"' in body and "javascript:" not in body
    body, _ = markdown_to_html("[r](../ch/02.md#sec) [m](mailto:a@b.c) [h](https://x/y) ![p](data:image/png;base64,AAAA) ![q](./fig.svg)")
    assert 'href="../ch/02.md#sec"' in body and 'href="mailto:a@b.c"' in body and 'href="https://x/y"' in body
    assert 'src="data:image/png;base64,AAAA"' in body and 'src="./fig.svg"' in body
