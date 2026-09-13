"""Retain Markdown headings so Pelican can avoid reparsing the article body."""

from markdown.extensions import Extension
from markdown.extensions.footnotes import FootnotePostprocessor
from markdown.extensions.md_in_html import MarkdownInHTMLPostprocessor
from markdown.extensions.toc import run_postprocessors
from markdown.postprocessors import AndSubstitutePostprocessor, RawHtmlPostprocessor, UnescapePostprocessor
from markdown.treeprocessors import Treeprocessor


class CaptureHeadings(Treeprocessor):
    def run(self, root):
        self.md.site_toc_headings = None
        # Unknown later processors may add headings that are absent here.
        if list(self.md.treeprocessors)[-1] is not self or any(
            type(processor) not in (RawHtmlPostprocessor, MarkdownInHTMLPostprocessor,
                                   AndSubstitutePostprocessor, UnescapePostprocessor,
                                   FootnotePostprocessor)
            for processor in self.md.postprocessors
        ):
            return
        # Raw HTML can introduce headings or malformed markup outside the tree.
        # Keep the existing HTML scanner for those documents (and code fences).
        if any('<' in block for block in self.md.htmlStash.rawHtmlBlocks):
            return
        headings = []
        for element in root.iter():
            if element.tag in ('script', 'style', 'template'):
                return
            if element.tag not in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
                continue
            tail = element.tail
            try:
                element.tail = None
                fragment = self.md.serializer(element)
            finally:
                element.tail = tail
            headings.append(run_postprocessors(fragment, self.md))
        self.md.site_toc_headings = headings


class TocHeadingsExtension(Extension):
    def extendMarkdown(self, md):
        md.registerExtension(self)
        self.md = md
        self.reset()
        # Capture final heading IDs after Markdown's TOC processor (priority 5).
        md.treeprocessors.register(CaptureHeadings(md), 'site_toc_headings', 0)

    def reset(self):
        self.md.site_toc_headings = None


def makeExtension(**kwargs):
    return TocHeadingsExtension(**kwargs)
