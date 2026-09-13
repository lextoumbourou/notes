import copy
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from frontmark.reader import FrontmarkReader
from pelican.settings import DEFAULT_CONFIG


spec = importlib.util.spec_from_file_location(
    'site_markdown_reader', Path(__file__).resolve().parents[1] / 'my-plugins' / 'markdown_reader.py'
)
reader_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader_module)


class MarkdownReaderTests(unittest.TestCase):
    def test_frontmatter_and_saved_outputs_match_frontmark(self):
        settings = copy.deepcopy(DEFAULT_CONFIG)
        settings['MARKDOWN']['extensions'] = ['extra', 'toc', 'markdown_notebook_fences', 'markdown_toc_headings']
        source = '''---
title: Stored output
date: '2026-09-13 00:00'
notebook:
  markdownLinks: true
---

## Example

```python {format=html}
print('example')
```
<!-- nb-output hash="example" format="html" -->
<div class="nb-output"><pre>example</pre></div>
<!-- /nb-output -->
'''
        with TemporaryDirectory() as directory:
            path = Path(directory, 'example.md')
            path.write_text(source)
            actual = reader_module.SiteMarkdownReader(settings).read(str(path))
            expected = FrontmarkReader(settings).read(str(path))
        self.assertEqual(actual, expected)
        self.assertEqual(actual[1]['notebook'], {'markdownLinks': True})

    def test_plain_markdown_passes_headings_without_leaking_between_reads(self):
        settings = copy.deepcopy(DEFAULT_CONFIG)
        settings['MARKDOWN']['extensions'] = ['toc', 'markdown_toc_headings']
        reader = reader_module.SiteMarkdownReader(settings)
        with TemporaryDirectory() as directory:
            path = Path(directory, 'example.md')
            path.write_text('---\ntitle: First\n---\n\n## First')
            content, metadata = reader.read(str(path))
            self.assertEqual(metadata['_site_toc_headings'], (content, ['<h2 id="first">First</h2>']))
            path.write_text('---\ntitle: Second\n---\n\nNo heading.')
            content, metadata = reader.read(str(path))
            self.assertEqual(metadata['_site_toc_headings'], (content, []))
