import tempfile
import unittest
from pathlib import Path

import nbformat
from markdown import Markdown


def write_notebook(directory):
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell("## A heading"),
        nbformat.v4.new_code_cell("first = 1", execution_count=1),
        nbformat.v4.new_code_cell(
            "second = 2",
            execution_count=2,
            outputs=[nbformat.v4.new_output("stream", name="stdout", text="hello\n")],
        ),
    ]
    nbformat.write(nb, str(Path(directory) / "example.ipynb"))


def render(text, content_dir):
    md = Markdown(extensions=["markdown_notebook_embed", "fenced_code"],
                  extension_configs={"markdown_notebook_embed": {"content_dir": content_dir}})
    return md.convert(text)


class NotebookEmbedTests(unittest.TestCase):
    def test_renders_the_whole_notebook(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_notebook(tmp)
            html = render("Before.\n\n{% notebook example.ipynb %}\n\nAfter.", tmp)
            self.assertIn('<div class="notebook-embed">', html)
            self.assertIn("first", html)
            self.assertIn("hello", html)
            self.assertIn("<p>After.</p>", html)
            self.assertNotIn("{% notebook", html)
            self.assertNotIn("anchor-link", html)

    def test_renders_a_slice_of_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_notebook(tmp)
            html = render("{% notebook example.ipynb cells[2:3] %}", tmp)
            self.assertIn("second", html)
            self.assertNotIn("first", html)
            self.assertNotIn("A heading", html)


if __name__ == "__main__":
    unittest.main()
