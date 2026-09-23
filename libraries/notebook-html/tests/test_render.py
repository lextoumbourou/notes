import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from notebook_html import load_html, read_article, render_anthropic_html, render_openai_html, render_html

PAGE = '<!doctype html><html lang="en"><body><script>window.ready=true</script>Quiz</body></html>'


class RenderingTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'quiz.html'

    def test_completed_message_saves_only_text_and_usage(self):
        message = SimpleNamespace(
            model='claude-opus-5-5', stop_reason='end_turn',
            content=[SimpleNamespace(type='thinking', thinking='private', signature='secret'),
                     SimpleNamespace(type='text', text=PAGE)],
            usage=SimpleNamespace(model_dump=lambda **kw: {'input_tokens': 30, 'output_tokens': 50}),
        )
        preview = render_anthropic_html(message, self.path)
        self.assertEqual(self.path.read_text(), PAGE)
        receipt = self.path.with_suffix('.json').read_text()
        self.assertNotIn('private', receipt)
        self.assertNotIn('secret', receipt)
        self.assertEqual(json.loads(receipt)['usage']['input_tokens'], 30)
        self.assertIn('sandbox="allow-scripts"', preview._repr_html_())
        self.assertNotIn('allow-same-origin', preview._repr_html_())
        self.assertIn('&lt;script&gt;', preview._repr_html_())

    def test_incomplete_response_preserves_existing_artifact(self):
        self.path.write_text('original')
        for reason in ('max_tokens', 'tool_use', None):
            with self.subTest(reason=reason), self.assertRaises(ValueError):
                render_anthropic_html({'stop_reason': reason, 'content': [{'type': 'text', 'text': PAGE}]}, self.path)
        self.assertEqual(self.path.read_text(), 'original')
        self.assertFalse(self.path.with_suffix('.json').exists())

    def test_invalid_html_cannot_overwrite_page(self):
        self.path.write_text('original')
        for text in ('', '<!doctype html><html><body>unfinished', 'Here is your page: ' + PAGE):
            with self.subTest(text=text), self.assertRaises(ValueError):
                render_html(text, self.path)
        self.assertEqual(self.path.read_text(), 'original')

    def test_openai_saves_final_text_and_billing_details_only(self):
        response = {
            'model': 'gpt-6-sol', 'status': 'completed',
            'reasoning': {'effort': 'high'}, 'service_tier': 'default',
            'output': [
                {'type': 'reasoning', 'encrypted_content': 'secret'},
                {'type': 'message', 'content': [{'type': 'output_text', 'text': PAGE}]},
            ],
            'usage': {'input_tokens': 2000, 'input_tokens_details': {
                'cached_tokens': 500, 'cache_write_tokens': 1000},
                'output_tokens': 100, 'output_tokens_details': {'reasoning_tokens': 60}},
        }
        render_openai_html(response, self.path)
        self.assertEqual(self.path.read_text(), PAGE)
        record = json.loads(self.path.with_suffix('.json').read_text())
        self.assertEqual(record['reasoning_effort'], 'high')
        self.assertEqual(record['usage'], response['usage'])
        self.assertNotIn('secret', json.dumps(record))

    def test_incomplete_or_refused_openai_response_preserves_artifacts(self):
        self.path.write_text('original')
        self.path.with_suffix('.json').write_text('original receipt')
        for status in ('incomplete', 'failed', 'cancelled', None, 'completed'):
            with self.subTest(status=status), self.assertRaises(ValueError):
                render_openai_html({'status': status, 'output': [
                    {'type': 'message', 'content': [{'type': 'refusal', 'refusal': 'No'}]},
                ]}, self.path)
        self.assertEqual(self.path.read_text(), 'original')
        self.assertEqual(self.path.with_suffix('.json').read_text(), 'original receipt')

    def test_json_response_and_fenced_html(self):
        render_anthropic_html({'model': 'test', 'stop_reason': 'end_turn', 'usage': {},
                              'content': [{'type': 'text', 'text': '```html\n' + PAGE + '\n```'}]}, self.path)
        self.assertEqual(load_html(self.path).html, PAGE)

    def test_article_reader_excludes_frontmatter_and_notebook(self):
        self.path.write_text('---\ntitle: Test\n---\n\nArticle.\n\n## Hands-on example\nAPI code and output')
        self.assertEqual(read_article(self.path), 'Article.')
        with self.assertRaises(ValueError):
            read_article(self.path, before='Missing')

    def test_title_is_escaped_and_height_validated(self):
        preview = render_html(PAGE, self.path, title='A "quiz" <preview>')
        self.assertIn('A &quot;quiz&quot; &lt;preview&gt;', preview._repr_html_())
        with self.assertRaises(ValueError):
            render_html(PAGE, self.path, height=-1)

    def test_input_marker_survives_heading_edits(self):
        for heading in ('Real example', 'Try it yourself', 'Build a game'):
            with self.subTest(heading=heading):
                self.path.write_text(
                    '---\ntitle: Test\n---\n\nArticle.\n\n'
                    '<!-- notebook-input-end -->\n\n'
                    f'## {heading}\nAPI code and saved output'
                )
                self.assertEqual(read_article(self.path), 'Article.')
                self.assertEqual(read_article(self.path, before='Old heading'), 'Article.')

    def test_explicit_heading_still_works_without_a_marker(self):
        self.path.write_text('Article.\n\n## Real example\nAPI code and output')
        self.assertEqual(read_article(self.path, before='Real example'), 'Article.')


if __name__ == '__main__':
    unittest.main()
