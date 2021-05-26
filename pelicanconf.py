import frontmark

AUTHOR = 'Lex Toumbourou'

DEFAULT_LANG = u'en'

ARTICLE_URL = '{slug}.html'

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

PLUGINS = ['latex', 'pelican_cite', frontmark, 'subcategory']

PUBLICATIONS_SRC = 'citations.bib'

BIBLIOGRAPHY_START = '<section id="bib"><h4>References</h4>'
BIBLIOGRAPHY_END = '</section>'
