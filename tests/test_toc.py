import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bs4 import BeautifulSoup
from jinja2.filters import do_striptags
from pelican import contents


spec = importlib.util.spec_from_file_location(
    "site_toc", Path(__file__).resolve().parents[1] / "my-plugins" / "toc.py"
)
toc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(toc)


def make_content(html, **metadata):
    return SimpleNamespace(
        _content=html, metadata=metadata, settings={"TOC": toc.TOC_DEFAULT.copy()}
    )


class TocTests(unittest.TestCase):
    def assert_matches_legacy(self, html, **metadata):
        original = make_content(html, **metadata)
        optimized = make_content(html, **metadata)
        toc._generate_toc_legacy(original)
        toc.generate_toc(optimized)
        self.assertEqual(getattr(optimized, "toc", None), getattr(original, "toc", None))
        self.assertEqual(
            BeautifulSoup(optimized._content, "html.parser").decode(formatter="html"),
            original._content,
        )
        return optimized

    def test_heading_text_ids_and_duplicate_suffixes(self):
        content = self.assert_matches_legacy(
            '<h1>Hello <em>world</em> &amp; friends<!-- ignored --></h1>'
            '<h2 id="same">First</h2><h3 id="same">Second</h3>'
            '<h4 id="same_1">Third</h4><h5></h5><h6></h6>'
        )
        self.assertEqual(content.toc, [
            {"level": 1, "anchor": "hello world & friends", "text": "Hello world & friends"},
            {"level": 2, "anchor": "same", "text": "First"},
            {"level": 3, "anchor": "same_1", "text": "Second"},
            {"level": 4, "anchor": "same_2", "text": "Third"},
            {"level": 5, "anchor": "_1", "text": ""},
            {"level": 6, "anchor": "_2", "text": ""},
        ])

    def test_body_is_preserved(self):
        before = '<div data-z="1" data-a="2">π&nbsp;🙂\n<img src="a.svg"></div>\n'
        after = '\n<pre>  a &lt; b\n\n</pre><svg><path d="M 0 0"/></svg>'
        content = self.assert_matches_legacy(before + '<h2>Heading</h2>' + after)
        self.assertEqual(content._content, before + '<h2 id="heading">Heading</h2>' + after)

    def test_entities_inline_tags_and_duplicate_attributes(self):
        for html in [
            '<h2>Café &#x1F642; &nbsp; &lt; &#169; &unknown;</h2>',
            '<h2><code>x_y</code><br/>line <img alt="not heading text"/></h2>',
            '<H2 ID="first" id="last">Case</H2>',
            '<h2>A<script>ignore()</script><style>.ignore{}</style>B</h2>',
            '<h2>A<![CDATA[B]]>C</h2>',
            '<h2><span class="katex"><math><mi>x</mi></math></span></h2>',
        ]:
            with self.subTest(html=html):
                self.assert_matches_legacy(html)

    def test_fake_headings_are_ignored(self):
        content = self.assert_matches_legacy(
            '<!-- <h2>Comment</h2> --><script>"<h2>Script</h2>"</script>'
            '<style>/* <h2>Style</h2> */</style><h2>Real</h2>'
        )
        self.assertEqual(content.toc, [{"level": 2, "anchor": "real", "text": "Real"}])

    def test_nonbreaking_spaces_preserve_generated_previews(self):
        html = '<p>Before <span class="katex">Set\xa0of\xa0days</span></p><h2>Math</h2>'
        content = self.assert_matches_legacy(html)
        original = make_content(html)
        toc._generate_toc_legacy(original)
        self.assertEqual(do_striptags(content._content), do_striptags(original._content))
        self.assertIn('Set&nbsp;of&nbsp;days', content._content)
        raw = '<script>const space = "\xa0";</script><style>p::after{content:"\xa0"}</style>'
        untouched = make_content(raw)
        toc.generate_toc(untouched)
        self.assertEqual(untouched._content, raw)

    def test_custom_heading_filter_only_reserves_selected_ids(self):
        content = self.assert_matches_legacy(
            '<h1 id="same">Ignored</h1><h2 id="same">Selected</h2>'
            '<h3>Ignored too</h3><h4 id="same">Also selected</h4>',
            toc_headers="^h[24]$",
        )
        self.assertEqual([entry["anchor"] for entry in content.toc], ["same", "same_1"])

    def test_ambiguous_fragments_fall_back_to_original_parser(self):
        for html in [
            '<h2>Unclosed',
            '<h2>Outer<h3>Inner</h3></h2>',
            '<h2/>After',
            '<div><h2>Crossing</div>after</h2>',
            '<template><h2>Template <em>text</em></h2></template>',
            '<p>Paragraph</p></p><h2>Heading</h2>',
            '<p>Paragraph</p></br><h2>Heading</h2>',
        ]:
            with self.subTest(html=html):
                content = make_content(html)
                with patch.object(toc, "_generate_toc_legacy", wraps=toc._generate_toc_legacy) as legacy:
                    toc.generate_toc(content)
                    legacy.assert_called_once_with(content)
                self.assert_matches_legacy(html)

    def test_disabled_and_static_content_are_untouched(self):
        for flag in ("false", False, True):
            content = make_content('<h2>Disabled</h2>', toc_run=flag)
            toc.generate_toc(content)
            self.assertEqual(content._content, '<h2>Disabled</h2>')
            self.assertFalse(hasattr(content, "toc"))
        toc.generate_toc(contents.Static.__new__(contents.Static))

    def test_no_headings_preserves_body_and_existing_toc(self):
        content = make_content('<p title="x">No &nbsp; headings</p>')
        content.toc = [{"level": 1, "anchor": "existing", "text": "Existing"}]
        toc.generate_toc(content)
        self.assertEqual(content._content, '<p title="x">No &nbsp; headings</p>')
        self.assertEqual(content.toc[0]["anchor"], "existing")


if __name__ == "__main__":
    unittest.main()
