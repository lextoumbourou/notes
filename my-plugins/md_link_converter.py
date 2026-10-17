from pelican.utils import slugify
from pelican import signals
from pelican.generators import ArticlesGenerator
from functools import lru_cache
from urllib.parse import unquote
import os
import re
from logging import getLogger

logger = getLogger(__name__)


@lru_cache(maxsize=None)
def frontmatter_slug(path):
    """The `slug:` from a linked note's frontmatter, so notes whose slug
    differs from their filename still get the right URL."""
    try:
        with open(path, encoding='utf-8') as f:
            if f.readline().strip() != '---':
                return None
            for line in f:
                if line.strip() == '---':
                    return None
                key, sep, value = line.partition(':')
                if sep and key.strip() == 'slug':
                    return value.strip().strip('"\'') or None
    except OSError:
        return None
    return None


def convert_md_links(instance):
    if not instance._content:
        return

    try:
        # Find <a> tags with .md hrefs, excluding external URLs
        html_links = re.compile(r'<a href="(?!https?://)([^"]+\.md)">([^<]+)</a>')

        source_dir = os.path.dirname(getattr(instance, 'source_path', None) or '')

        # Replace the .md links with slug.html links
        def replace_link(match):
            url_path = match.group(1)
            file_stem = url_path.split('/')[-1]
            file_stem = file_stem[:-3]  # Remove .md from the file stem

            link_text = match.group(2)

            target = os.path.normpath(os.path.join(source_dir, unquote(url_path)))
            slug = frontmatter_slug(target) or slugify(file_stem)
            return f'<a href="{slug}.html">{link_text}</a>'

        original = instance._content
        instance._content = html_links.sub(replace_link, original)
        prepared = instance.metadata.get('_site_toc_headings')
        if prepared is not None and prepared[0] == original:
            # This rewrite only changes link destinations. Keep captured
            # heading fragments aligned with the same change in the body.
            instance.metadata['_site_toc_headings'] = (
                instance._content,
                [html_links.sub(replace_link, heading) for heading in prepared[1]],
            )
    except Exception as e:
        logger.error('Exception occurred: %s', e, exc_info=True)

def register():
    signals.content_object_init.connect(convert_md_links)
