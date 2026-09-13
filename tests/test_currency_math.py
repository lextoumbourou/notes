import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from xml.sax.saxutils import escape

from bs4 import BeautifulSoup
from markdown import Markdown
import nbformat
from pelican.settings import DEFAULT_CONFIG

from currency_katex import CurrencyKatexExtension, configure_pelican


class CurrencyMathTests(unittest.TestCase):
    def render(self, source, expected):
        calls = []

        def render_latex(latex, options):
            calls.append((latex, options["displayMode"]))
            return '<span class="katex">' + escape(latex) + '</span>'

        with patch("pelican_katex.markdown.render_latex", side_effect=render_latex):
            result = Markdown(extensions=["extra", CurrencyKatexExtension()]).convert(source)
        self.assertEqual(calls, expected)
        return result

    def test_currency_stays_prose(self):
        examples = [
            "$20,000 and $30,000",
            "From $5 to below $5.",
            "$10 per million input tokens and $50 per million output tokens.",
            "Game A: 66% chance of winning $2400, 33% chance of winning $2500 "
            "and 1% chance of $0 (EV = $2409)",
            "Costs $0.25, $1,200.50 or $-10.",
        ]
        for source in examples:
            with self.subTest(source=source):
                self.assertEqual(self.render(source, []), '<p>' + source + '</p>')

    def test_currency_next_to_real_math(self):
        self.render("Pay $100, then calculate $x+1$ and $100$.", [
            ("x+1", False), ("100", False)
        ])
        self.render("Use $x$ to price $100 and $200; then $$x^2$$.", [
            ("x", False), ("x^2", True)
        ])

    def test_inline_boundaries_do_not_consume_other_dollars(self):
        for source in ["$ x$", "$x $", "$x$2", "$$x$", "$x$$", "$myhostname, $mydomain"]:
            with self.subTest(source=source):
                self.render(source, [])
        self.render("$x $ and $y$", [("y", False)])
        self.render("$x$, $y$. $a\nb$", [("x", False), ("y", False), ("a\nb", False)])

    def test_escaped_dollars_and_latex_commands(self):
        output = self.render(r"Literal \$x\$, price \$100 and maths $x$.", [("x", False)])
        self.assertIn("Literal $x$, price $100", output)
        self.render(r"$\text{cost: \$5} + x$", [(r"\text{cost: \$5} + x", False)])
        self.render(r"$a\\$", [(r"a\\", False)])
        self.render(r"$$\begin{aligned}a &= b \\ c &= d\end{aligned}$$", [
            (r"\begin{aligned}a &= b \\ c &= d\end{aligned}", True)
        ])

    def test_code_is_not_math(self):
        output = self.render("`$x$`\n\n```python\nprice = '$20 and $30'\n```", [])
        self.assertIn('<code>$x$</code>', output)
        self.assertIn("price = '$20 and $30'", output)

    def test_display_math_and_preamble_preserve_upstream_behavior(self):
        self.render("$$\nx + y\n$$", [("\nx + y\n", True)])
        self.render(r"$$ \mathbf{w}_{-j} $$and $$\mathbf{w}_j $$", [
            (r" \mathbf{w}_{-j} ", True), (r"\mathbf{w}_j ", True)
        ])
        with patch("pelican_katex.markdown.push_preamble") as preamble:
            self.render(r"$$@\newcommand{\R}{\mathbb{R}}$$", [])
        preamble.assert_called_once_with(r"\newcommand{\R}{\mathbb{R}}")

    def test_pelican_uses_currency_extension(self):
        pelican = SimpleNamespace(settings={"MARKDOWN": {"extensions": ["extra"]}})
        configure_pelican(pelican)
        with patch("pelican_katex.markdown.render_latex") as render:
            output = Markdown(**pelican.settings["MARKDOWN"]).convert("$20 and $30")
        render.assert_not_called()
        self.assertEqual(output, "<p>$20 and $30</p>")

    def test_notebooks_use_same_rules_without_touching_code(self):
        spec = importlib.util.spec_from_file_location(
            "site_jupytext", Path(__file__).resolve().parents[1] / "my-plugins" / "pelican_jupytext.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        notebook = nbformat.v4.new_notebook(cells=[
            nbformat.v4.new_markdown_cell(r"Pay $20 or $30. Literal \$x\$. Formula $x+1$."),
            nbformat.v4.new_code_cell("print('$code$')", outputs=[
                nbformat.v4.new_output("stream", name="stdout", text="$output$")
            ]),
        ])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "note.ipynb"
            nbformat.write(notebook, path)
            reader = module.JupytextMarkdownReader(DEFAULT_CONFIG.copy())
            with patch("pelican_katex.markdown.render_latex", return_value='<span class="katex">x+1</span>') as render:
                output, _ = reader._read_notebook("note.md", str(path), {"title": "Notebook"})
        render.assert_called_once_with("x+1", {"displayMode": False})
        soup = BeautifulSoup(output, "html.parser")
        self.assertIn("Pay $20 or $30. Literal $x$.", soup.get_text())
        self.assertIn("$code$", soup.get_text())
        self.assertIn("$output$", soup.get_text())


if __name__ == "__main__":
    unittest.main()
