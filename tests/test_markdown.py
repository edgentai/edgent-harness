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
