"""Let the browser defer below-the-fold images.

Every Markdown image after the first in a note gets loading="lazy" (the first is
usually the lead picture, so it stays eager). Raw HTML images are left as written.
"""

from markdown.extensions import Extension
from markdown.treeprocessors import Treeprocessor


class LazyImages(Treeprocessor):
    def run(self, root):
        for index, img in enumerate(root.iter('img')):
            if index == 0:
                continue
            img.set('loading', img.get('loading', 'lazy'))
            img.set('decoding', img.get('decoding', 'async'))


class LazyImagesExtension(Extension):
    def extendMarkdown(self, md):
        # After the TOC (5), before markdown_toc_headings captures the final tree (0).
        md.treeprocessors.register(LazyImages(md), 'lazy_images', 1)


def makeExtension(**kwargs):
    return LazyImagesExtension(**kwargs)
