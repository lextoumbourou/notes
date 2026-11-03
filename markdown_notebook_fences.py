"""Render notebook fences, optionally folding source while leaving output visible."""

import re

from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor


FENCE_RE = re.compile(
    r"^(?P<indent> {0,3})(?P<fence>`{3,}|~{3,})(?P<info>.*)$"
)
NOTEBOOK_INFO_RE = re.compile(
    r"^(?P<language>[A-Za-z0-9_+.-]+)\s+\{(?P<args>[^{}]*)\}\s*$"
)
ARG_RE = re.compile(r'(\w+)=(?:"([^"]*)"|(\S+))')
NOTEBOOK_LANGUAGES = {"python", "javascript", "js", "bash", "sh", "shell", "r", "sql", "duckdb"}


def input_visibility(value):
    return value if value in ("show", "collapsed") else None


class NotebookFencePreprocessor(Preprocessor):
    def run(self, lines: list[str]) -> list[str]:
        fence_character: str | None = None
        fence_length = 0
        collapsed = False
        in_output = False
        rendered = []
        default_input = input_visibility(getattr(self.md, "notebook_input", None))

        for line in lines:
            if fence_character is not None:
                closing_fence = re.compile(
                    rf"^ {{0,3}}{re.escape(fence_character)}{{{fence_length},}}\s*$"
                )
                if closing_fence.match(line):
                    fence_character = None
                    fence_length = 0
                    rendered.append(line)
                    if collapsed:
                        rendered.extend(["", "</details>", ""])
                    collapsed = False
                else:
                    rendered.append(line)
                continue

            # Persisted output may itself contain fences (format=python, etc.).
            # Its presentation is independent of the source-code default.
            if in_output:
                rendered.append(line)
                if line.strip() == "<!-- /nb-output -->":
                    in_output = False
                continue
            if line.startswith("<!-- nb-output "):
                in_output = True
                rendered.append(line)
                continue

            match = FENCE_RE.match(line)
            if not match:
                rendered.append(line)
                continue

            fence = match.group("fence")
            info = match.group("info").strip()
            notebook_info = NOTEBOOK_INFO_RE.match(info)
            language = info
            args = {}
            if notebook_info:
                language = notebook_info.group("language")
                args = {key: quoted or bare for key, quoted, bare in ARG_RE.findall(notebook_info.group("args"))}
                line = f'{match.group("indent")}{fence}{language}'

            collapsed = language in NOTEBOOK_LANGUAGES and (
                input_visibility(args.get("input")) or default_input
            ) == "collapsed"
            if collapsed:
                rendered.extend([
                    '<details class="nb-input" markdown="1">',
                    '<summary>Show code</summary>',
                    "",
                ])
            rendered.append(line)

            fence_character = fence[0]
            fence_length = len(fence)

        if collapsed:
            rendered.extend(["", "</details>", ""])
        return rendered


class NotebookFenceExtension(Extension):
    def extendMarkdown(self, md):
        # Run immediately before Python-Markdown's fenced-code preprocessor.
        md.preprocessors.register(
            NotebookFencePreprocessor(md),
            "notebook_fences",
            26,
        )


def makeExtension(**kwargs):
    return NotebookFenceExtension(**kwargs)
