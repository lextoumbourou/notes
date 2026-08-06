"""Render Markdown Notebook fence arguments as normal fenced code blocks."""

import re

from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor


FENCE_RE = re.compile(
    r"^(?P<indent> {0,3})(?P<fence>`{3,}|~{3,})(?P<info>.*)$"
)
NOTEBOOK_INFO_RE = re.compile(
    r"^(?P<language>[A-Za-z0-9_+.-]+)\s+\{[^{}]*\}\s*$"
)


class NotebookFencePreprocessor(Preprocessor):
    def run(self, lines: list[str]) -> list[str]:
        fence_character: str | None = None
        fence_length = 0

        for index, line in enumerate(lines):
            if fence_character is not None:
                closing_fence = re.compile(
                    rf"^ {{0,3}}{re.escape(fence_character)}{{{fence_length},}}\s*$"
                )
                if closing_fence.match(line):
                    fence_character = None
                    fence_length = 0
                continue

            match = FENCE_RE.match(line)
            if not match:
                continue

            fence = match.group("fence")
            info = match.group("info").strip()
            notebook_info = NOTEBOOK_INFO_RE.match(info)
            if notebook_info:
                lines[index] = (
                    f'{match.group("indent")}{fence}'
                    f'{notebook_info.group("language")}'
                )

            fence_character = fence[0]
            fence_length = len(fence)

        return lines


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
