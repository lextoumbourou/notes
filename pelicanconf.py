import os

from functools import partial

import frontmark

MARKUP = ("md",)

from pelican_jupyter import liquid as nb_liquid

AUTHOR = "Lex Toumbourou"

DEFAULT_LANG = "en"

ARTICLE_URL = "{slug}.html"

SITENAME = "Notes by Lex"

THEME = "theme"

PLUGIN_PATHS = ["pelican-plugins", "pelican-cite/src", "my-plugins"]
STATIC_PATHS = ["_media"]

DRAFT_URL = "{slug}.html"
DRAFT_SAVE_AS = "{slug}.html"

USE_FOLDER_AS_CATEGORY = True

ENV = os.environ.get("ENV", "prod")

MARKDOWN = {
    "extension_configs": {
        "markdown.extensions.codehilite": {
            "css_class": "highlight",
        },
        "markdown.extensions.extra": {},
        "markdown.extensions.meta": {},
    },
    "output_format": "html5",
}

PLUGINS = [
    "pelican_katex",
    "pelican_cite",
    frontmark,
    "subcategory",
    nb_liquid,
    "md_link_converter",
    "linked_mentions",
]

PUBLICATIONS_SRC = "notes/citations.bib"

BIBLIOGRAPHY_START = '<section id="bib"><h4>References</h4>'
BIBLIOGRAPHY_END = "</section>"

DEFAULT_PAGINATION = 10

JINJA_FILTERS = {
    "sort_by_article_count": partial(
        sorted, key=lambda tags: len(tags[1]), reverse=True
    )
}  # reversed for descending order

SUMMARY_MAX_LENGTH = 25

IGNORE_FILE = [".ipynb_checkpoints"]

LIQUID_CONFIGS = (("CONTENT_DIR", "notes", ""),)

ARTICLE_EXCLUDES = ["journal", "posts", "templates", "notebooks", ".env", "output", "permanent/notebooks"]

RELATIVE_URLS = True

TIMEZONE = "Australia/Brisbane"

SITEURL = "https://notesbylex.com"
if ENV == "local":
    SITEURL = "http://localhost:8000"
