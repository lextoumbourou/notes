"""Keep a styling hook for plain-text blocks rendered by CodeHilite."""

from pygments.formatters import HtmlFormatter


class CodeBlockFormatter(HtmlFormatter):
    def __init__(self, lang_str="", **options):
        if lang_str in {"language-text", "language-txt", "language-plaintext"}:
            options["cssclass"] = f'{options.get("cssclass", "highlight")} plain-text'
        super().__init__(**options)
