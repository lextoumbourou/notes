from pelican.utils import slugify
from pelican import signals
from pelican.generators import ArticlesGenerator
import re
from logging import getLogger

logger = getLogger(__name__)


def convert_md_links(instance):
    if not instance._content:
        return

    try:
        # Find <a> tags with .md hrefs
        html_links = re.compile(r'<a href="([^"]+\.md)">([^<]+)</a>')

        # Replace the .md links with file_stem.html links
        def replace_link(match):
            url_path = match.group(1)
            file_stem = url_path.split('/')[-1]
            file_stem = file_stem[:-3]  # Remove .md from the file stem

            link_text = match.group(2)

            slug = slugify(file_stem)
            return f'<a href="{slug}.html">{link_text}</a>'

        instance._content = html_links.sub(replace_link, instance._content)
    except Exception as e:
        logger.error('Exception occurred: %s', e, exc_info=True)

def register():
    signals.content_object_init.connect(convert_md_links)
