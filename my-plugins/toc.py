'''
toc
===================================

This plugin generates a structured TOC for pages and articles.
Sets content.toc to a list of dicts: [{level, anchor, text}, ...]
'''

from __future__ import unicode_literals

import logging
import re

from bs4 import BeautifulSoup, Comment

from pelican import contents, signals
from pelican.utils import slugify


logger = logging.getLogger(__name__)
TOC_DEFAULT = {
    'TOC_HEADERS': '^h[1-6]',
    'TOC_RUN': 'true',
}
TOC_KEY = 'TOC'

IDCOUNT_RE = re.compile(r'^(.*)_([0-9]+)$')


def unique(id, ids):
    while id in ids or not id:
        m = IDCOUNT_RE.match(id)
        if m:
            id = '%s_%d' % (m.group(1), int(m.group(2)) + 1)
        else:
            id = '%s_%d' % (id, 1)
    ids.add(id)
    return id


def init_default_config(pelican):
    from pelican.settings import DEFAULT_CONFIG

    def update_settings(settings):
        temp = TOC_DEFAULT.copy()
        if TOC_KEY in settings:
            temp.update(settings[TOC_KEY])
        settings[TOC_KEY] = temp
        return settings

    DEFAULT_CONFIG = update_settings(DEFAULT_CONFIG)
    if pelican:
        pelican.settings = update_settings(pelican.settings)


def generate_toc(content):
    if isinstance(content, contents.Static):
        return

    _toc_run = content.metadata.get(
        'toc_run',
        content.settings[TOC_KEY]['TOC_RUN'])
    if not _toc_run == 'true':
        return

    all_ids = set()
    soup = BeautifulSoup(content._content, 'html.parser')
    entries = []

    try:
        header_re = re.compile(content.metadata.get(
            'toc_headers', content.settings[TOC_KEY]['TOC_HEADERS']))
    except re.error as e:
        logger.error("TOC_HEADERS '%s' is not a valid re",
                     content.settings[TOC_KEY]['TOC_HEADERS'])
        raise e

    for header in soup.findAll(header_re):
        text = header.get_text()
        raw_id = header.attrs.get('id') or slugify(text, ())
        anchor = unique(raw_id, all_ids)
        header.attrs['id'] = anchor
        level = int(header.name[1])  # h2 -> 2
        entries.append({'level': level, 'anchor': anchor, 'text': text})

    if entries:
        content.toc = entries
    content._content = soup.decode(formatter='html')


def register():
    signals.initialized.connect(init_default_config)
    signals.content_object_init.connect(generate_toc)
