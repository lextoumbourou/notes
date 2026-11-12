"""Keep a styling hook for plain-text blocks rendered by CodeHilite."""

from pygments.formatters import HtmlFormatter


class CodeBlockFormatter(HtmlFormatter):
    def __init__(self, lang_str="", **options):
        if lang_str in {"language-text", "language-txt", "language-plaintext"}:
            options["cssclass"] = f'{options.get("cssclass", "highlight")} plain-text'
        elif lang_str == "language-output":
            # Program output with columns, such as a table: keep every line intact and scroll sideways.
            options["cssclass"] = f'{options.get("cssclass", "highlight")} wide-output'
        super().__init__(**options)
