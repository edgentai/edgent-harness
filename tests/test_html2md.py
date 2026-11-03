from edgent.adapters.html2md import html_to_markdown


def test_html_to_markdown_basic():
    html = """<html><head><title>T</title><style>x{}</style><script>bad()</script></head>
    <body><nav>skip me</nav><h1>Head</h1><p>Para with <a href="http://x">link</a> and <code>c</code>.</p>
    <ul><li>one</li><li>two</li></ul><pre>raw  code</pre><table><tr><th>a</th><th>b</th></tr><tr><td>1</td><td>2</td></tr></table></body></html>"""
    md, title = html_to_markdown(html)
    assert title == "T" and "# Head" in md and "[link](http://x)" in md and "`c`" in md
    assert "- one" in md and "- two" in md and "raw  code" in md and "| a | b |" in md
    assert "bad()" not in md and "skip me" not in md
