from pelican.utils import slugify
from pelican import signals
from pelican.settings import DEFAULT_CONFIG
from pelican.generators import ArticlesGenerator
from functools import lru_cache
from urllib.parse import unquote
import os
import re
import yaml
from logging import getLogger

logger = getLogger(__name__)


@lru_cache(maxsize=None)
def frontmatter(path):
    """A linked note's frontmatter, or {} when it has none or can't be read."""
    try:
        with open(path, encoding='utf-8') as f:
            if f.readline().strip() != '---':
                return {}
            lines = []
            for line in f:
                if line.strip() == '---':
                    break
                lines.append(line)
        meta = yaml.safe_load(''.join(lines))
    except (OSError, yaml.YAMLError):
        return {}
    return meta if isinstance(meta, dict) else {}


def target_slug(path, file_stem, settings):
    """The slug Pelican gives a linked note: its `slug:`, else its title
    slugified as Pelican does it, else (no title) its filename."""
    meta = frontmatter(path)
    if meta.get('slug'):
        return str(meta['slug'])
    source = meta.get('title') if settings.get('SLUGIFY_SOURCE', 'title') == 'title' else None
    return slugify(
        str(source or file_stem),
        regex_subs=settings.get('SLUG_REGEX_SUBSTITUTIONS', []),
        preserve_case=settings.get('SLUGIFY_PRESERVE_CASE', False),
        use_unicode=settings.get('SLUGIFY_USE_UNICODE', False),
    )


def convert_md_links(instance):
    if not instance._content:
        return

    try:
        # Find <a> tags with .md hrefs, excluding external URLs. A section
        # anchor (note.md#heading) is kept and moved onto the .html link.
        html_links = re.compile(r'<a href="(?!https?://)([^"#]+\.md)(#[^"]*)?">([^<]+)</a>')

        source_dir = os.path.dirname(getattr(instance, 'source_path', None) or '')

        # Replace the .md links with slug.html links
        def replace_link(match):
            url_path = match.group(1)
            file_stem = url_path.split('/')[-1]
            file_stem = file_stem[:-3]  # Remove .md from the file stem

            anchor = match.group(2) or ''
            link_text = match.group(3)

            target = os.path.normpath(os.path.join(source_dir, unquote(url_path)))
            slug = target_slug(target, file_stem, getattr(instance, 'settings', None) or DEFAULT_CONFIG)
            return f'<a href="{slug}.html{anchor}">{link_text}</a>'

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
