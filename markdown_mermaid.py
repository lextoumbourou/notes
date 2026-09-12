"""Render Mermaid fences to embedded SVGs using the native Rust renderer."""

import base64

from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor
from markdown_inline_mermaid import BLOCK_RE

from mermaid_renderer import render_svg


class MermaidPreprocessor(Preprocessor):
    def run(self, lines):
        # Keep the existing extension's fence syntax and HTML placement unchanged.
        def replace(match):
            svg = render_svg(match.group("content"))
            encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
            image = f'<img src="data:image/svg+xml;base64,{encoded}">'
            return "\n" + self.md.htmlStash.store(image) + "\n"

        return BLOCK_RE.sub(replace, "\n".join(lines)).split("\n")


class MermaidExtension(Extension):
    def extendMarkdown(self, md):
        md.registerExtension(self)
        md.preprocessors.register(MermaidPreprocessor(md), "mermaid_block", 27)


def makeExtension(**kwargs):
    return MermaidExtension(**kwargs)
