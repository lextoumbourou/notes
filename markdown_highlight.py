"""Render Obsidian-style ==highlight== as <mark>highlight</mark>."""

from markdown.extensions import Extension
from markdown.inlinepatterns import SimpleTagInlineProcessor

# Same shape as Obsidian: no space just inside either pair of equals signs.
HIGHLIGHT_RE = r"(==)(?![\s=])(.+?)(?<![\s=])=="


class HighlightExtension(Extension):
    def extendMarkdown(self, md):
        # Below code spans (190) and escapes, so `==` inside code is untouched.
        md.inlinePatterns.register(
            SimpleTagInlineProcessor(HIGHLIGHT_RE, "mark"), "obsidian-highlight", 65
        )


makeExtension = HighlightExtension
