"""Keep prices as prose while using pelican-katex for math rendering."""

from markdown.extensions import Extension
from markdown.inlinepatterns import InlineProcessor
from pelican import signals
from pelican_katex import plugin as upstream
from pelican_katex.markdown import KatexExtension, KatexPattern


# Retain the existing display-math syntax. For inline math, use Pandoc-style
# boundaries: no whitespace just inside the delimiters, and no digit after the
# closing dollar. Stop at unescaped dollars rather than skipping currency signs
# to find a later closing delimiter. Escaped dollars inside LaTeX stay intact.
PATTERN = (
    r"(?P<preceding>\s?)(?P<delimiter>(?<!\$)\$(?P<display>\$)?)"
    r"(?P<latex>(?(display).+?|(?!\s)(?:\\[\s\S]|[^\\$])+?(?<!\s)))"
    r"(?P=delimiter)(?(display)|(?![\d$]))"
)


class CurrencyKatexPattern(KatexPattern):
    def __init__(self, md=None):
        InlineProcessor.__init__(self, PATTERN, md)


class CurrencyKatexExtension(Extension):
    def extendMarkdown(self, md):
        md.ESCAPED_CHARS.append("$")
        # As upstream: run after code spans, before backslash escapes, so LaTeX
        # commands are not altered by Markdown before reaching KaTeX.
        md.inlinePatterns.register(CurrencyKatexPattern(md), "katex", 186)


def configure_pelican(pelican):
    upstream.configure_pelican(pelican)
    extensions = pelican.settings["MARKDOWN"]["extensions"]
    extensions[:] = [
        CurrencyKatexExtension() if isinstance(extension, KatexExtension) else extension
        for extension in extensions
    ]


def register():
    # Keep upstream's options, reStructuredText support and preamble lifecycle;
    # replace only its Markdown integration. No installed package is patched.
    upstream.register()
    signals.initialized.disconnect(upstream.configure_pelican)
    signals.initialized.connect(configure_pelican)
