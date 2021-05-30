from functools import partial

import frontmark

MARKUP = ('md', )

from pelican_jupyter import liquid as nb_liquid

AUTHOR = 'Lex Toumbourou'

DEFAULT_LANG = u'en'

ARTICLE_URL = '{slug}.html'

SITENAME = 'Notes by Lex'

THEME = "theme"

PLUGIN_PATHS = ['pelican-plugins', 'pelican-cite/src']
STATIC_PATHS = ['_media']

DRAFT_URL = u'{slug}.html'
DRAFT_SAVE_AS = u'{slug}.html'

USE_FOLDER_AS_CATEGORY = True

def build_url(label, base, end):
    label_parts = label.split()
    return base + '-'.join([l.lower() for l in label_parts]) + end

MARKDOWN = {
    "extension_configs": {
        "wikilinks": {
            "base_url": "/",
            "end_url": ".html",
            "build_url": build_url
        },
        "markdown.extensions.codehilite": {
            "css_class": "highlight",
        },
        "markdown.extensions.extra": {},
        "markdown.extensions.meta": {},
    },
    "output_format": "html5",
}

PLUGINS = ['latex', 'pelican_cite', frontmark, 'subcategory', nb_liquid]

PUBLICATIONS_SRC = 'notes/citations.bib'

BIBLIOGRAPHY_START = '<section id="bib"><h4>References</h4>'
BIBLIOGRAPHY_END = '</section>'

DEFAULT_PAGINATION = 10

JINJA_FILTERS = {
    'sort_by_article_count': partial(
        sorted,
        key=lambda tags: len(tags[1]),
        reverse=True)} # reversed for descending order

SUMMARY_MAX_LENGTH = 25

IGNORE_FILE = ['.ipynb_checkpoints']

LIQUID_CONFIGS = (("CONTENT_DIR", "notes", ""), )
