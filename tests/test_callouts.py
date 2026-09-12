import unittest
from unittest.mock import patch
from xml.etree import ElementTree

from markdown import Markdown
from markdown_obsidian_callouts.obsidian_callouts import ObsidianCalloutsBlockProcessor

from markdown_fast_callouts import FastCalloutsBlockProcessor


class CalloutTests(unittest.TestCase):
    def test_long_block_without_marker_never_runs_regex(self):
        markdown = Markdown(extensions=["markdown_fast_callouts"])
        processor = markdown.parser.blockprocessors["obsidian-callouts"]
        block = '<svg><path d="' + 'M 0 0 L 1 1 ' * 10000 + '"/></svg>'
        with patch.object(ObsidianCalloutsBlockProcessor, "CALLOUT_PATTERN") as pattern:
            self.assertFalse(processor.test(ElementTree.Element("div"), block))
            pattern.search.assert_not_called()

    def test_rendering_matches_original_extension(self):
        examples = [
            "A paragraph with **bold** and a [link](example.html).",
            "> An ordinary blockquote.\n> Another line.",
            "> [!note]\n> A note with **bold** content.",
            "Before\n\n> [!question]- A collapsed question\n> An answer.\n\nAfter",
            "> [!tip]+ An expanded tip\n> First line.\n> Second line.",
            "> [!warning] Outer\n> Outer content.\n> > [!note] Inner\n> > Nested content.",
            "* Item\n\n    > [!note] In a list\n    > List callout.",
            "```text\n> [!note] This stays code\n```",
            "Literal [! text is not a callout.\n\n> [!unsupported-kind] Custom\n> Body.",
            "> [ !note] Invalid marker\n> Still a quote.",
        ]
        for source in examples:
            with self.subTest(source=source):
                original = Markdown(extensions=["obsidian-callouts", "extra"]).convert(source)
                optimized = Markdown(extensions=["markdown_fast_callouts", "extra"]).convert(source)
                self.assertEqual(optimized, original)

    def test_callouts_still_respect_recursion_limit(self):
        markdown = Markdown()
        processor = FastCalloutsBlockProcessor(markdown.parser)
        with patch("markdown_obsidian_callouts.obsidian_callouts.util.nearing_recursion_limit", return_value=True):
            self.assertFalse(processor.test(ElementTree.Element("div"), "> [!note]\n> Content."))


if __name__ == "__main__":
    unittest.main()
