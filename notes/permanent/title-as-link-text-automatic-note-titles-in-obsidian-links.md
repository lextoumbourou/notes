---
title: "Title As Link Text: automatic note titles in Obsidian links"
date: 2025-11-01 00:00
modified: 2025-11-01 00:00
summary: "An Obsidian plugin that automatically replaces filenames in your links with the actual note title, supporting both Wikilinks and Markdown links."
tags:
- ObsidianPlugins
---

I've built [Title As Link Text](https://github.com/lextoumbourou/obsidian-title-as-link-text), a new Obsidian plugin that automatically updates your links to use the note's title instead of the filename.

If you use a Zettelkasten-style system or just have notes with slugged filenames, you end up with links that look like this:

```markdown
[[20230408102501]]
[document-name](./complex-topic.md)
```

When what you actually want is:

```markdown
[[20230408102501|My Awesome Note]]
[Understanding Complex Topics](./complex-topic.md)
```

The plugin handles both Wikilinks and Markdown links, and updates them automatically whenever you save or rename a file. It plays nicely with other plugins that make Obsidian play nice with Markdown links, like [Frontmatter Title](https://github.com/snezhig/obsidian-front-matter-title) and [Wikilinks to MDLinks](https://github.com/agathauy/wikilinks-to-mdlinks-obsidian).

### How it finds the title

It checks three things in order:

1. A frontmatter property (default: `title`)
2. The first H1 heading in the file
3. The filename itself as a fallback

So if you have a `title` in your frontmatter, that wins. If not, it looks for a `# Heading`. If neither exists, it just uses the filename as-is.

You can also configure which frontmatter property to use, so if you prefer `name`, `heading`, or something custom, that works too.

### Alias support

It also handles aliases. If a note has aliases defined in frontmatter and a link text matches one of them (even partially or fuzzily), the plugin leaves it alone rather than overwriting it with the title. You can tune the similarity threshold or turn alias matching off entirely if you don't want it.

### Installation

It's available as a Community Plugin.

1. Open **Settings** > **Community Plugins** > **Browse**
2. Search for **Title As Link Text**
3. Click **Install**

There are also two commands available if you want to do a one-off update rather than relying on auto-save: "Update all links" and "Update links for current file".

---

See the [project GitHub](https://github.com/lextoumbourou/obsidian-title-as-link-text) for more details.
