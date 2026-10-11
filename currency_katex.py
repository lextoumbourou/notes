"""Keep prices as prose while using pelican-katex for math rendering."""

from xml.etree import ElementTree

from markdown.extensions import Extension
from markdown.extensions.attr_list import AttrListTreeprocessor
from markdown.inlinepatterns import InlineProcessor
from markdown.util import AtomicString
from pelican import signals
from pelican_katex import plugin as upstream
from pelican_katex import markdown as upstream_markdown
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
    def __init__(self, md=None, stash_html=True):
        InlineProcessor.__init__(self, PATTERN, md)
        self.stash_html = stash_html

    def handleMatch(self, match, data):
        # Unlike upstream, math may open after punctuation such as "(" or a quote,
        # but still not inside a word ("US$5", "a$x$"), which keeps currency and
        # identifiers as prose. Consume one character so the parser moves on.
        preceding = match.group("preceding")
        before = data[match.start() - 1] if match.start() > 0 else ""
        if not preceding and before and (before.isalnum() or before == "\\"):
            return None, match.start() + 1, match.start() + 1
        start, end = match.start() + len(preceding), match.end()
        latex = match.group("latex")
        display = match.group("delimiter") == "$$"
        if display and latex.startswith("@"):
            upstream_markdown.push_preamble(latex[1:])
            return "", start, end

        rendered = upstream_markdown.render_latex(latex, {"displayMode": display})

        # Inline attribute lists need a real element to attach their attributes.
        # The tree path is also available for extensions that inspect math nodes.
        if not self.stash_html or AttrListTreeprocessor.INLINE_RE.match(data[match.end():]):
            node = ElementTree.fromstring(rendered)
            upstream_markdown.revert_xmlns_resolution(node)
            for elem in node.iter():
                if elem.text is not None:
                    elem.text = AtomicString(elem.text)
            return node, start, end

        # KaTeX already produced HTML. Store it like fenced-code output rather
        # than parsing XML and repeatedly walking its MathML and SVG descendants.
        return self.md.htmlStash.store(rendered), start, end


class CurrencyKatexExtension(Extension):
    def __init__(self, **kwargs):
        self.config = {"stash_html": [True, "Preserve KaTeX HTML without Markdown tree processing."]}
        super().__init__(**kwargs)

    def extendMarkdown(self, md):
        md.ESCAPED_CHARS.append("$")
        # As upstream: run after code spans, before backslash escapes, so LaTeX
        # commands are not altered by Markdown before reaching KaTeX.
        md.inlinePatterns.register(CurrencyKatexPattern(md, self.getConfig("stash_html")), "katex", 186)


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
