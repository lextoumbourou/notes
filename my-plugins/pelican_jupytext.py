"""
Pelican Jupytext Plugin

Detects markdown files paired with Jupyter notebooks via jupytext.
When a paired article is detected (by reading the jupytext metadata in the YAML header),
it renders the corresponding Jupyter notebook instead of the markdown content.
"""

import logging
import os
import re

import yaml
from nbconvert import HTMLExporter
from pelican import signals
from pelican.readers import BaseReader
from pelican_katex.rendering import render_latex

log = logging.getLogger(__name__)


def render_latex_in_html(html_content):
    """Replace $$...$$ and $...$ with KaTeX-rendered HTML."""
    def replace_display_math(match):
        latex = match.group(1)
        try:
            return render_latex(latex, {"displayMode": True})
        except Exception as e:
            log.warning(f"KaTeX failed to render display math: {e}")
            return match.group(0)

    def replace_inline_math(match):
        latex = match.group(1)
        try:
            return render_latex(latex, {"displayMode": False})
        except Exception as e:
            log.warning(f"KaTeX failed to render inline math: {e}")
            return match.group(0)

    # Render display math first ($$...$$), then inline ($...$)
    # Use negative lookbehind/ahead to avoid matching $$ as two $
    content = re.sub(r'\$\$(.*?)\$\$', replace_display_math, html_content, flags=re.DOTALL)
    content = re.sub(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', replace_inline_math, content, flags=re.DOTALL)
    return content

YAML_FRONTMATTER_PATTERN = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)

# Store reference to the original reader class
_original_md_reader = None


def is_jupytext_paired(metadata):
    """Check if metadata indicates a jupytext-paired file with an ipynb."""
    if not metadata:
        return False

    jupyter_meta = metadata.get('jupyter', {})
    if not jupyter_meta:
        return False

    jupytext_meta = jupyter_meta.get('jupytext', {})
    if not jupytext_meta:
        return False

    formats = jupytext_meta.get('formats', '')
    return isinstance(formats, str) and 'ipynb' in formats


def get_notebook_path(md_filepath):
    """Get the path to the paired notebook file."""
    base_path = os.path.splitext(md_filepath)[0]
    return base_path + '.ipynb'


def parse_yaml_frontmatter(content):
    """Parse YAML frontmatter from markdown content."""
    match = YAML_FRONTMATTER_PATTERN.match(content)
    if not match:
        return None, content

    try:
        yaml_content = match.group(1)
        metadata = yaml.safe_load(yaml_content)
        remaining = content[match.end():]
        return metadata, remaining
    except yaml.YAMLError as e:
        log.warning(f"Failed to parse YAML frontmatter: {e}")
        return None, content


class JupytextMarkdownReader(BaseReader):
    """
    Wrapper reader that detects jupytext-paired files and renders the notebook.
    Delegates to the original markdown reader for non-paired files.
    """

    enabled = True
    file_extensions = ['md']

    def __init__(self, settings):
        super().__init__(settings)
        # Create instance of original reader for delegation
        if _original_md_reader:
            self._original_reader = _original_md_reader(settings)
        else:
            from pelican.readers import MarkdownReader
            self._original_reader = MarkdownReader(settings)

    def read(self, source_path):
        with open(source_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()

        yaml_meta, _ = parse_yaml_frontmatter(raw_content)

        if is_jupytext_paired(yaml_meta):
            notebook_path = get_notebook_path(source_path)

            if os.path.exists(notebook_path):
                log.debug(f"Jupytext paired file detected: {source_path} -> {notebook_path}")
                return self._read_notebook(source_path, notebook_path, yaml_meta)
            else:
                log.warning(
                    f"Jupytext paired notebook not found: {notebook_path}. "
                    f"Falling back to markdown rendering."
                )

        # Delegate to original reader for non-jupytext files
        return self._original_reader.read(source_path)

    def _read_notebook(self, md_path, notebook_path, yaml_meta):
        """Render notebook to HTML and extract metadata from markdown frontmatter."""
        exporter = HTMLExporter(
            template_name='basic',
            exclude_raw=True,  # Skip raw cells (including jupytext frontmatter)
            exclude_input_prompt=True,
            exclude_output_prompt=True,
        )

        content, resources = exporter.from_filename(notebook_path)

        # Render LaTeX math expressions with KaTeX
        content = render_latex_in_html(content)

        # Build metadata from YAML frontmatter
        metadata = {}
        standard_fields = ['title', 'date', 'modified', 'category', 'tags',
                          'slug', 'authors', 'author', 'summary', 'status',
                          'template', 'save_as', 'url', 'lang']

        for field in standard_fields:
            if field in yaml_meta:
                value = yaml_meta[field]
                metadata[field] = self.process_metadata(field, value)

        metadata['jupyter_notebook'] = True
        metadata['jupytext_paired'] = True

        return content, metadata


def add_reader(readers):
    """Register the JupytextMarkdownReader, preserving the original reader."""
    global _original_md_reader

    # Store the original reader class before replacing (but not ourselves!)
    current_reader = readers.reader_classes.get('md')
    if current_reader is not JupytextMarkdownReader:
        _original_md_reader = current_reader

    # Register our wrapper reader
    readers.reader_classes['md'] = JupytextMarkdownReader


def register():
    """Register the plugin with Pelican."""
    signals.readers_init.connect(add_reader)
