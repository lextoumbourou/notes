'''
toc
===================================

This plugin generates a structured TOC for pages and articles.
Sets content.toc to a list of dicts: [{level, anchor, text}, ...]
'''

from __future__ import unicode_literals

from html.parser import HTMLParser
import logging
import re

from bs4 import BeautifulSoup

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


def _generate_toc_legacy(content):
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

    for header in soup.find_all(header_re):
        text = header.get_text()
        raw_id = header.attrs.get('id') or slugify(text, ())
        anchor = unique(raw_id, all_ids)
        header.attrs['id'] = anchor
        level = int(header.name[1])  # h2 -> 2
        entries.append({'level': level, 'anchor': anchor, 'text': text})

    if entries:
        content.toc = entries
    content._content = soup.decode(formatter='html')


VOID_TAGS = frozenset(
    'area base br col embed hr img input link meta param source track wbr'.split()
)


class _HeadingScanner(HTMLParser):
    """Locate complete headings without building a tree for the article body."""
    def __init__(self, html, header_re):
        super().__init__(convert_charrefs=False)
        self.html = html
        self.header_re = header_re
        self.line_offsets = [0] + [m.end() for m in re.finditer('\n', html)]
        self.ranges = []
        self.replacements = []
        self.current = None
        self.children = []
        self.open_tags = []
        self.template_depth = 0
        self.fallback_reason = None

    def source_offset(self):
        line, column = self.getpos()
        return self.line_offsets[line - 1] + column

    def handle_starttag(self, tag, attrs):
        if self.fallback_reason:
            return
        if tag not in VOID_TAGS:
            self.open_tags.append(tag)
        if tag == 'template':
            self.template_depth += 1
        selected = self.header_re.search(tag)
        if self.current:
            if selected:
                self.fallback_reason = 'nested selected headings'
            elif tag not in VOID_TAGS:
                self.children.append(tag)
        elif selected:
            # BeautifulSoup treats text under a template differently.
            if self.template_depth:
                self.fallback_reason = 'heading inside a template'
            else:
                self.current = (tag, self.source_offset())
                self.children = []

    def handle_startendtag(self, tag, attrs):
        if self.header_re.search(tag):
            self.fallback_reason = 'self-closing selected heading'

    def handle_data(self, data):
        if self.current or self.fallback_reason or '\xa0' not in data:
            return
        if self.open_tags and self.open_tags[-1] in ('script', 'style'):
            return
        # Jinja's striptags collapses literal NBSP before decoding entities.
        # Keep the old spelling so generated social previews retain their text.
        start = self.source_offset()
        self.replacements.append((start, start + len(data), data.replace('\xa0', '&nbsp;')))

    def handle_endtag(self, tag):
        if self.fallback_reason:
            return
        # BeautifulSoup drops stray closing tags. Browsers can turn </p> or
        # </br> into extra elements, so preserve the old repair for such input.
        if tag not in self.open_tags:
            self.fallback_reason = 'unmatched closing tag'
            return
        index = len(self.open_tags) - 1 - self.open_tags[::-1].index(tag)
        del self.open_tags[index:]
        if self.current:
            if tag == self.current[0]:
                end = self.html.find('>', self.source_offset()) + 1
                self.ranges.append((self.current[1], end))
                self.current = None
                self.children = []
            elif tag in self.children:
                index = len(self.children) - 1 - self.children[::-1].index(tag)
                del self.children[index:]
            else:
                self.fallback_reason = 'heading closes an enclosing tag'
        if tag == 'template':
            self.template_depth = max(0, self.template_depth - 1)


def generate_toc(content):
    if isinstance(content, contents.Static):
        return
    settings = content.settings[TOC_KEY]
    if content.metadata.get('toc_run', settings['TOC_RUN']) != 'true':
        return
    if not isinstance(content._content, str):
        return _generate_toc_legacy(content)
    try:
        header_re = re.compile(content.metadata.get('toc_headers', settings['TOC_HEADERS']))
    except re.error:
        logger.error("TOC_HEADERS '%s' is not a valid re", settings['TOC_HEADERS'])
        raise
    scanner = _HeadingScanner(content._content, header_re)
    try:
        scanner.feed(content._content)
        scanner.close()
    except (AssertionError, ValueError):
        return _generate_toc_legacy(content)
    # A partial or overlapping fragment could change the old parser's behavior.
    if scanner.fallback_reason or scanner.current:
        logger.debug('Using legacy TOC parser: %s', scanner.fallback_reason or 'unclosed heading')
        return _generate_toc_legacy(content)

    entries, all_ids = [], set()
    for start, end in scanner.ranges:
        # Retain BeautifulSoup's heading text/entity handling and serialization.
        soup = BeautifulSoup(content._content[start:end], 'html.parser')
        header = soup.find(header_re)
        if header is None:
            return _generate_toc_legacy(content)
        text = header.get_text()
        anchor = unique(header.attrs.get('id') or slugify(text, ()), all_ids)
        header.attrs['id'] = anchor
        entries.append({'level': int(header.name[1]), 'anchor': anchor, 'text': text})
        scanner.replacements.append((start, end, header.decode(formatter='html')))
    if entries:
        content.toc = entries
    pieces, previous = [], 0
    for start, end, replacement in sorted(scanner.replacements):
        pieces.extend((content._content[previous:start], replacement))
        previous = end
    pieces.append(content._content[previous:])
    content._content = ''.join(pieces)


def register():
    signals.initialized.connect(init_default_config)
    signals.content_object_init.connect(generate_toc)
