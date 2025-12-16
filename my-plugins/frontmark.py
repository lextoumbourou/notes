"""
Frontmark - CommonMark Markdown reader with YAML frontmatter for Pelican

This is a local copy fixed to work with current Pelican versions.
Based on https://github.com/lextoumbourou/pelican-frontmark
"""

import collections
import logging
import re

try:
    import commonmark
    from commonmark.common import escape_xml
    from commonmark.render.html import potentially_unsafe
except ImportError:
    commonmark = False

from markdown import Markdown

try:
    import yaml
except ImportError:
    yaml = False

from blinker import signal
from pelican import signals
from pelican.readers import BaseReader
from pelican.utils import pelican_open

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import TextLexer, get_lexer_by_name

log = logging.getLogger(__name__)

# Custom signal for YAML type registration
frontmark_yaml_register = signal('frontmark_yaml_register')

DELIMITER = '---'
BOUNDARY = re.compile(r'^{0}$'.format(DELIMITER), re.MULTILINE)
STR_TAG = 'tag:yaml.org,2002:str'

INTERNAL_LINK = re.compile(r'^%7B(\\w+)%7D')


class HtmlRenderer(commonmark.HtmlRenderer):
    """An altered CommonMark HTML rendered taking reader settings in account."""

    linkable_tags = ('a', 'img')
    linkable_attrs = ('href', 'src')

    def __init__(self, reader):
        self.reader = reader
        super().__init__()

    @property
    def use_pygments(self):
        return bool(self.reader.pygments_options)

    @property
    def pygments_options(self):
        if isinstance(self.reader.pygments_options, dict):
            return self.reader.pygments_options
        return {}

    def tag(self, name, attrs=None, selfclosing=None):
        """Helper function to produce an HTML tag."""
        if self.disable_tags > 0:
            return

        if name in self.linkable_tags and attrs and len(attrs) > 0:
            for attrib in attrs:
                if attrib[0] in self.linkable_attrs:
                    attrib[1] = INTERNAL_LINK.sub(r'{\g<1>}', attrib[1])

        super().tag(name, attrs, selfclosing)

    def escape(self, text):
        escaped = escape_xml(text)
        return INTERNAL_LINK.sub(r'{\g<1>}', escaped)

    def image(self, node, entering):
        """Handle image nodes"""
        if entering:
            if self.disable_tags == 0:
                if self.options.get('safe') and potentially_unsafe(node.destination):
                    self.lit('<img src="" alt="')
                else:
                    self.lit('<img src="' + self.escape(node.destination) + '" alt="')
                self.disable_tags += 1
        else:
            self.disable_tags -= 1
            if self.disable_tags == 0:
                if node.title:
                    self.lit('" title="' + self.escape(node.title))
                self.lit('" />')

    def code_block(self, node, entering):
        """Output Pygments if required else use default html5 output"""
        if self.use_pygments:
            self.cr()
            info_words = node.info.split() if node.info else []

            if len(info_words) > 0 and len(info_words[0]) > 0:
                try:
                    lexer = get_lexer_by_name(info_words[0])
                except ValueError:
                    lexer = TextLexer()
            else:
                lexer = TextLexer()

            formatter = HtmlFormatter(**self.pygments_options)
            parsed = highlight(node.literal, lexer, formatter)
            self.lit(parsed)
            self.cr()
        else:
            super().code_block(node, entering)


class FrontmarkReader(BaseReader):
    """Reader for CommonMark Markdown files with YAML metadata"""

    enabled = bool(commonmark) and bool(yaml)
    file_extensions = ['md']

    def read(self, source_path):
        self._source_path = source_path
        self._md = Markdown(**self.settings['MARKDOWN'])

        with pelican_open(source_path) as text:
            metadata, content = self._parse(text)

        content = self._md.convert(content)

        return content, self._parse_metadata(metadata)

    def _parse(self, text):
        """Parse text with frontmatter, return metadata and content.
        If frontmatter is not found, returns an empty metadata dictionary
        and original text content."""
        text = str(text).strip()

        if not text.startswith(DELIMITER):
            return {}, text

        try:
            _, fm, content = BOUNDARY.split(text, 2)
        except ValueError:
            return {}, text

        metadata = yaml.load(fm, Loader=self.loader_class)
        metadata = metadata if (isinstance(metadata, dict)) else {}
        return metadata, content

    def _parse_metadata(self, meta):
        """Return the dict containing document metadata"""
        formatted_fields = self.settings['FORMATTED_FIELDS']

        output = collections.OrderedDict()
        for name, value in meta.items():
            name = name.lower()
            if name in formatted_fields:
                rendered = self._render(value).strip()
                output[name] = self.process_metadata(name, rendered)
            else:
                output[name] = self.process_metadata(name, value)
        return output

    @property
    def pygments_options(self):
        """Optional Pygments options"""
        return self.settings.get('FRONTMARK_PYGMENTS')

    def _render(self, text):
        """Render CommonMark with settings taken in account"""
        parser = commonmark.Parser()
        ast = parser.parse(text)
        renderer = HtmlRenderer(self)
        html = renderer.render(ast)
        return html

    def yaml_markdown_constructor(self, loader, node):
        """Allows to optionally parse Markdown in multiline literals"""
        value = loader.construct_scalar(node)
        return self._render(value).strip()

    def yaml_multiline_as_markdown_constructor(self, loader, node):
        """Allows to optionally parse Markdown in multiline literals"""
        value = loader.construct_scalar(node)
        return self._render(value).strip() if node.style == '|' else value

    @property
    def loader_class(self):
        reader = self

        class FrontmarkLoader(yaml.Loader):
            """Custom YAML Loader for frontmark
            - Mapping order is respected (with OrderedDict)"""

            def construct_mapping(self, node, deep=False):
                """Use OrderedDict as default for mappings"""
                return collections.OrderedDict(self.construct_pairs(node))

        FrontmarkLoader.add_constructor('!md', self.yaml_markdown_constructor)
        if self.settings.get('FRONTMARK_PARSE_LITERAL', True):
            FrontmarkLoader.add_constructor(STR_TAG, self.yaml_multiline_as_markdown_constructor)
        for _, pair in frontmark_yaml_register.send(self):
            if not len(pair) == 2:
                log.warning('Ignoring YAML type (%s), expected a (tag, handler) tuple', pair)
                continue
            tag, constructor = pair
            FrontmarkLoader.add_constructor(tag, constructor)

        return FrontmarkLoader


def add_reader(readers):
    for k in FrontmarkReader.file_extensions:
        readers.reader_classes[k] = FrontmarkReader


def register():
    signals.readers_init.connect(add_reader)
