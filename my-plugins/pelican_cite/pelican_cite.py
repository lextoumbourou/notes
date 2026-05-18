# -*- coding: utf-8 -*-
"""
pelican-cite
==============

A Pelican plugin that provides a BibTeX-style reference system within
pelican sites.

Originally by the pelican-cite contributors: https://github.com/cmacmackin/pelican-cite
Based on the Pelican BibTeX plugin written by Vlad Niculae <vlad@vene.ro>

Inlined here with local modifications (author-year citation format).
"""

import logging
import re
import sys

try:
    from pybtex.database.input.bibtex import Parser
    from pybtex.database.output.bibtex import Writer
    from pybtex.database import BibliographyData, PybtexError, Entry
    from pybtex.backends import html
    from pybtex.style.formatting import toplevel
    from pybtex.style.formatting.unsrt import dashify, Style as UnsrtStyle
    from pybtex.style.template import (
        join, words, field, optional, first_of, sentence, tag, optional_field,
    )
    from pybtex.plugin import find_plugin

    pyb_imported = True
except ImportError:
    pyb_imported = False

from pelican import signals
from pelican.contents import Static
from .author_year import LabelStyle

__version__ = '1.0.0'

JUMP_BACK = '<a class="cite-backref" href="#ref-{0}-{1}" title="Jump back to reference {1}">{2}</a>'
CITE_RE = re.compile(r"\[&#64;(&#64;)?\s*(\w.*?)\s*\]")
DATE_RE = re.compile(r"(?P<y>\d{4})(?:-(?P<m>\d{1,2})(?:-(?P<d>\d{1,2}))?)?")
CITE_2_RE = re.compile(r">\s*\(\s*(.*?),\s*(.*?)\s*\)\s*<")


class Style(UnsrtStyle):
    name = 'inline'
    default_sorting_style = 'author_year_title'
    default_label_style = 'author_year'

    def __init__(self, label_style=None, name_style=None, sorting_style=None, abbreviate_names=False, **kwargs):
        self.name_style = find_plugin('pybtex.style.names', name_style or self.default_name_style)()
        self.label_style = LabelStyle()
        self.sorting_style = find_plugin('pybtex.style.sorting', sorting_style or self.default_sorting_style)()
        self.format_name = self.name_style.format
        self.format_labels = self.label_style.format_labels
        self.sort = self.sorting_style.sort
        self.abbreviate_names = abbreviate_names

    def get_article_template(self, e):
        pages = field('pages', apply_func=dashify)
        date = words[optional_field('month'), field('year')]
        volume_and_pages = first_of[
            # volume and pages, with optional issue number
            optional[
                join[
                    field('volume'),
                    optional['(', field('number'), ')'],
                    ':', pages
                ],
            ],
            # pages only
            words['pages', pages],
        ]
        template = toplevel[
            self.format_names('author'),
            self.format_title(e, 'title'),
            sentence[
                tag('em')[first_of[
                              optional[field('journal')],
                              optional[field('journaltitle')],
                          ],
                ],
                optional[volume_and_pages],
                date],
            sentence[optional_field('note')],
            self.format_web_refs(e),
        ]
        return template


logger = logging.getLogger(__name__)
global_bib = None
bibliography_start = '<hr>\n<h2>Bibliography</h2>\n'
bibliography_end = ''
if pyb_imported:
    style = Style()
    backend = html.Backend()
else:
    style = None
    backend = None


def get_bib_file(article):
    """
    If a bibliography file is specified for this article/page, parse
    it and return the parsed object.
    """
    if 'publications_src' in article.metadata:
        refs_file = article.metadata['publications_src']
        try:
            local_bib = Parser().parse_file(refs_file)
            return local_bib
        except PybtexError as e:
            logger.warning('`pelican_bibtex` failed to parse file %s: %s' % (
                refs_file,
                str(e)))
            return global_bib
    else:
        return global_bib


def process_content(article):
    """
    Substitute the citations and add a bibliography for an article or
    page, using the local bib file if specified or the global one otherwise.
    """
    data = get_bib_file(article)
    if not data:
        return
    content = article._content
    content = content.replace("@", "&#64;")

    # Scan post to figure out what citations are needed
    cite_count = {}
    replace_count = {}
    for citation in CITE_RE.findall(content):
        if citation[1] not in cite_count:
            cite_count[citation[1]] = 1
            replace_count[citation[1]] = 1
        else:
            cite_count[citation[1]] += 1

    # Get formatted entries for the appropriate bibliographic entries
    cited = []
    for key in data.entries.keys():
        if key in cite_count:
            cited.append(data.entries[key])
    if len(cited) == 0:
        return

    # Patch entries, adding missing things to workaround style expecting fields
    # that are not there
    for entry in cited:  # type: Entry
        # Zotero at least exports with "date" instead of separate "year" etc.
        if 'year' not in entry.fields and 'date' in entry.fields:
            date_parse = DATE_RE.match(entry.fields['date'])
            if date_parse:
                groups = date_parse.groupdict()
                if groups['y']:
                    entry.fields['year'] = groups['y']
                if groups['m']:
                    entry.fields['month'] = groups['m']
                if groups['d']:
                    entry.fields['day'] = groups['d']
    formatted_entries = style.format_entries(cited)

    # Get the data for the required citations and append to content
    labels = {}
    content += bibliography_start
    for formatted_entry in formatted_entries:
        key = formatted_entry.key
        ref_id = key.replace(' ', '')
        label = ("<a href='#" + ref_id + "' id='ref-" + ref_id + "-{0}'>"
                 + formatted_entry.label + "</a>")
        t = formatted_entry.text.render(backend)
        t = t.replace('\\{', '&#123;')
        t = t.replace('\\}', '&#125;')
        t = t.replace('{', '')
        t = t.replace('}', '')
        text = ("<p id='" + ref_id + "'>" + t)
        for i in range(cite_count[key]):
            if i == 0:
                text += ' ' + JUMP_BACK.format(ref_id, 1, '↩')
                if cite_count[key] > 1:
                    text += JUMP_BACK.format(ref_id, 1, ' <sup>1</sup> ')
            else:
                text += JUMP_BACK.format(ref_id, i + 1, '<sup>' + str(i + 1) + '</sup> ')
        text += '</p>'
        content += text + '\n'
        labels[key] = label

    content += bibliography_end

    # Replace citations in article/page
    cite_count = {}

    def replace_cites(match):
        label = match.group(2)
        if label in labels:
            if label not in cite_count:
                cite_count[label] = 1
                replace_count[label] = 1
            else:
                cite_count[label] += 1
            lab = labels[label].format(cite_count[label])
            if '&#64;&#64;' in match.group():
                m = CITE_2_RE.search(lab)
                lab = lab[0:m.start()] + '>' + m.group(1) + ' (' + m.group(2) + ')<' + lab[m.end():]
                return lab
            else:
                return lab
        else:
            logger.warning('No BibTeX entry found for key "{}"'.format(label))
            return match.group(0)

    content = CITE_RE.sub(replace_cites, content)
    article._content = content


def add_citations(content):
    if isinstance(content, Static):
        return

    global global_bib
    if not pyb_imported:
        logger.warning('`pelican-cite` failed to load dependency `pybtex`')
        return

    process_content(content)


def init(pelican_instance):
    global global_bib, bibliography_start, bibliography_end
    if not pyb_imported:
        logger.warning('`pelican-cite` failed to load dependency `pybtex`')
        return

    if 'BIBLIOGRAPHY_START' in pelican_instance.settings:
        bibliography_start = pelican_instance.settings['BIBLIOGRAPHY_START']
    if 'BIBLIOGRAPHY_END' in pelican_instance.settings:
        bibliography_end = pelican_instance.settings['BIBLIOGRAPHY_END']

    if 'PUBLICATIONS_SRC' in pelican_instance.settings:
        refs_file = pelican_instance.settings['PUBLICATIONS_SRC']
        try:
            global_bib = Parser().parse_file(refs_file)
        except PybtexError as e:
            logger.warning('`pelican_bibtex` failed to parse file %s: %s' % (
                refs_file,
                str(e)))


def register():
    signals.initialized.connect(init)
    signals.content_object_init.connect(add_citations)
