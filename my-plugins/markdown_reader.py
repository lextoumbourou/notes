"""Read Markdown frontmatter and retain headings for the site's TOC."""

from frontmark.reader import FrontmarkReader
from pelican import signals


class SiteMarkdownReader(FrontmarkReader):
    def read(self, source_path):
        content, metadata = super().read(source_path)
        headings = getattr(self._md, 'site_toc_headings', None)
        if headings is not None:
            # Other plugins may alter the HTML before the TOC runs.
            metadata['_site_toc_headings'] = (content, headings)
        return content, metadata


def add_reader(readers):
    readers.reader_classes['md'] = SiteMarkdownReader


def register():
    signals.readers_init.connect(add_reader)
