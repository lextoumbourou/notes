#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = u'Lex Toumbourou'
SITENAME = u'LexToumbourou.com'
SITEURL = ''

TIMEZONE = 'Australia/Melbourne'

DEFAULT_LANG = u'en'

ARTICLE_URL = 'blog/posts/{slug}/'
ARTICLE_SAVE_AS = 'blog/posts/{slug}/index.html'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None

# Social widget
SOCIAL = (('You can add links in your config file', '#'),
          ('Another social link', '#'),)

DEFAULT_PAGINATION = None

THEME = "themes/lextoumbourou-theme"

STATIC_PATHS = ['images']

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True
