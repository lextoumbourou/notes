"""Render {% notebook path [cells[start:end]] %} tags as Jupyter notebooks.

pelican_jupyter's liquid tag plugin no longer loads with Python-Markdown 3.4+
(its vendored extension uses the removed extendMarkdown(md, md_globals) API),
so this extension handles the tag itself with pelican_jupyter's nbconvert export.
Paths are relative to content_dir, as with the old LIQUID_CONFIGS CONTENT_DIR.
"""

import os
import re

from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor
from pelican_jupyter.core import get_html_from_filepath, parse_css


NOTEBOOK_TAG = re.compile(
    r"^[ \t]*\{%\s*notebook\s+(?P<src>\S+)"
    r"(?:\s+cells\[(?P<start>-?\d*):(?P<end>-?\d*)\])?\s*%\}[ \t]*$",
    re.MULTILINE,
)


ANCHOR_LINK = re.compile(r'<a class="anchor-link" href="#[^"]*">(?:¶|&para;|&#182;)</a>')


class NotebookEmbedPreprocessor(Preprocessor):
    def __init__(self, md, content_dir):
        super().__init__(md)
        self.content_dir = content_dir

    def run(self, lines):
        def replace(match):
            start, end = match.group("start"), match.group("end")
            content, info = get_html_from_filepath(
                os.path.join(self.content_dir, match.group("src")),
                start=int(start) if start else 0,
                end=int(end) if end else None,
            )
            # The theme styles the notebook markup, so skip Jupyter's own CSS.
            html = parse_css(content, info, ignore_css=True)
            # nbconvert's heading permalinks would show up as "¶" entries in the page contents.
            html = ANCHOR_LINK.sub("", html)
            block = f'<div class="notebook-embed">{html}</div>'
            return "\n" + self.md.htmlStash.store(block) + "\n"

        return NOTEBOOK_TAG.sub(replace, "\n".join(lines)).split("\n")


class NotebookEmbedExtension(Extension):
    def __init__(self, **kwargs):
        self.config = {"content_dir": ["notes", "Directory notebook paths are relative to."]}
        super().__init__(**kwargs)

    def extendMarkdown(self, md):
        md.registerExtension(self)
        # Before fenced code (25), notebook fences (26) and Mermaid (27).
        md.preprocessors.register(
            NotebookEmbedPreprocessor(md, self.getConfig("content_dir")), "notebook_embed", 28
        )


def makeExtension(**kwargs):
    return NotebookEmbedExtension(**kwargs)
