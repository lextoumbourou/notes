"""Skip expensive callout searches in blocks without a callout marker."""

from markdown.extensions import Extension
from markdown_obsidian_callouts.obsidian_callouts import ObsidianCalloutsBlockProcessor


class FastCalloutsBlockProcessor(ObsidianCalloutsBlockProcessor):
    def test(self, parent, block):
        # Every match requires this literal. Leave matching and rendering upstream.
        return "[!" in block and super().test(parent, block)


class FastCalloutsExtension(Extension):
    def extendMarkdown(self, md):
        # Preserve the upstream processor's name and priority before blockquotes.
        md.parser.blockprocessors.register(
            FastCalloutsBlockProcessor(md.parser), "obsidian-callouts", 21.1
        )


makeExtension = FastCalloutsExtension
