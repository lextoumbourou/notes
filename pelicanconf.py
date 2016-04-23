#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = u'Lex Toumbourou'
SITENAME = u'LexToumbourou.com'
SITEURL = 'http://lextoumbourou.github.io'

TIMEZONE = 'Australia/Melbourne'

DEFAULT_LANG = u'en'

ARTICLE_URL = 'blog/posts/{slug}/'
ARTICLE_SAVE_AS = 'blog/posts/{slug}/index.html'

# Feed generation is usually not desired when developing
FEED_DOMAIN = SITEURL
FEED_ATOM = 'atom.xml'

# Social widget
SOCIAL = (('You can add links in your config file', '#'),
          ('Another social link', '#'),)

DEFAULT_PAGINATION = None

THEME = "themes/lextoumbourou-theme"

STATIC_PATHS = ['images', 'extra']

EXTRA_PATH_METADATA  = {
    'extra/robots.txt': {'path': 'robots.txt'},
    'extra/favicon.ico': {'path': 'favicon.ico'},
}

DISQUS_SITENAME = 'lextoumbouroucom'
