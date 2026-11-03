import unittest

from bs4 import BeautifulSoup
from markdown import Markdown


def render(source, default=None):
    md = Markdown(extensions=['extra', 'codehilite', 'markdown_notebook_fences'])
    md.notebook_input = default
    return BeautifulSoup(md.convert(source), 'html.parser')


class NotebookFenceTests(unittest.TestCase):
    def test_folds_only_source_and_leaves_failed_output_visible(self):
        page = render('''```python {id=example input="collapsed"}
print(42)
```
<!-- nb-output id="example" format="html" -->
<div class="nb-output"><pre class="nb-status-error">A failure</pre></div>
<!-- /nb-output -->
''')
        folded = page.select_one('details.nb-input')
        self.assertIsNotNone(folded)
        self.assertFalse(folded.has_attr('open'))
        self.assertEqual(folded.summary.text, 'Show code')
        self.assertIn('print(42)', folded.get_text())
        output = page.select_one('.nb-output')
        self.assertIsNone(output.find_parent('details'))
        self.assertIn('A failure', output.text)

    def test_note_default_allows_cell_override_and_preserves_output_fences(self):
        page = render('''```python
print(1)
```
<!-- nb-output format="python" -->
```python
print("saved output")
```
<!-- /nb-output -->

```python {input=show}
print(2)
```

```text
A static prompt
```
''', default='collapsed')
        self.assertEqual(len(page.select('details.nb-input')), 1)
        for source in ['print("saved output")', 'print(2)', 'A static prompt']:
            block = next(pre for pre in page.select('pre') if source in pre.text)
            self.assertIsNone(block.find_parent('details'))

    def test_existing_attributes_and_nested_fences_keep_their_rendering(self):
        plain = render('```python\nprint(1)\n```')
        annotated = render('```python {id=x run=false format=html}\nprint(1)\n```')
        self.assertEqual(str(plain), str(annotated))
        nested = render('````markdown\n```python {input=collapsed}\nprint(1)\n```\n````')
        self.assertIsNone(nested.select_one('details'))
        self.assertIn('input=collapsed', nested.text)

    def test_alias_tilde_fences_and_invalid_values_follow_the_same_default(self):
        for opening, closing in [('```js {input=invalid}', '```'), ('~~~shell', '~~~')]:
            with self.subTest(opening=opening):
                page = render(f'{opening}\nhello\n{closing}', default='collapsed')
                self.assertIsNotNone(page.select_one('details.nb-input'))


if __name__ == '__main__':
    unittest.main()
