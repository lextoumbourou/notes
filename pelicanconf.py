import os

from functools import partial

MARKUP = ("md",)

AUTHOR = "Lex Toumbourou"
SITE_DESCRIPTION = "A collection of notes by Lex Toumbourou."
GOOGLE_ANALYTICS_ID = "G-GLND6HLD4D"

HERO_TITLE = "Notes By Lex Toumbourou"
HERO_LEDE = "A digital garden about AI, software, learning, and other assorted topics."

SOCIAL = [
    ("🦋", "Bluesky",  "https://bsky.app/profile/notesbylex.com"),
    ("🐘", "Mastodon", "https://fedi.notesbylex.com/@lex"),
    ("💼", "LinkedIn", "https://www.linkedin.com/in/lextoumbourou/"),
    ("⌥",  "GitHub",   "https://github.com/lextoumbourou"),
]

EXTRA_CSS_URLS = [
    "https://cdn.jsdelivr.net/gh/lextoumbourou/markdown-obsidian-callouts@e54d4dfcbce9791f24fc8c9ae39d0f4909e5ae8f/markdown_obsidian_callouts/static/callouts.min.css",
]
EXTRA_JS_URLS = [
    "https://cdn.jsdelivr.net/gh/lextoumbourou/markdown-obsidian-callouts@e54d4dfcbce9791f24fc8c9ae39d0f4909e5ae8f/markdown_obsidian_callouts/static/callouts.min.js",
]

DEFAULT_LANG = "en"

ARTICLE_URL = "{slug}.html"

SITENAME = "NotesByLex.com"

THEME = "theme"

PLUGIN_PATHS = ["pelican-plugins", "my-plugins"]
STATIC_PATHS = ["_media"]

DRAFT_URL = "{slug}.html"
DRAFT_SAVE_AS = "{slug}.html"

USE_FOLDER_AS_CATEGORY = True

ENV = os.environ.get("ENV", "prod")

MARKDOWN = {
    "extension_configs": {
        "markdown_inline_mermaid": {},
        "obsidian-callouts": {},
        "markdown.extensions.codehilite": {
            "css_class": "highlight",
        },
        "markdown.extensions.extra": {},
        "markdown.extensions.meta": {},
        "markdown.extensions.toc": {
            "permalink": False,
        },
    },
    "output_format": "html5",
}

PLUGINS = [
    "pelican_alias",
    "pelican_katex",
    "frontmark",
    "pelican_jupytext",
    "subcategory",
    "md_link_converter",
    "bluesky_comments",
    "pelican_cite",
    "pelican_graph_view",
    "toc",
]

GRAPH_VIEW = {"include_hidden": True}

PUBLICATIONS_SRC = "notes/citations.bib"

BIBLIOGRAPHY_START = '<section id="bib"><h4>References</h4>'
BIBLIOGRAPHY_END = "</section>"

DEFAULT_PAGINATION = 10

DIRECT_TEMPLATES = ["index", "notes", "tags", "categories", "archives"]
PAGINATED_TEMPLATES = {"index": None, "notes": None, "tag": None, "category": None, "author": None}

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
