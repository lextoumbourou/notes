---
title: "Obsidian Markdown Notebook: code execution with outputs stored in the file"
date: 2026-04-21 00:00
modified: 2026-04-21 00:00
cover: /_media/obsidian-markdown-notebook-cover.png
hide_cover_in_article: true
summary: "A Jupyter Notebook-style Obsidian plugin that runs code in your notes and stores the outputs directly in the Markdown file."
tags:
- ObsidianPlugins
---

I've built [Obsidian Markdown Notebook](https://github.com/lextoumbourou/obsidian-markdown-notebook), a plugin that lets you execute code in Obsidian with both code and output stored in a single Markdown file.

It allows you to execute code directly in Obsidian, like a [Jupyter Notebook](https://jupyter.org/), but everything, including the outputs, is stored in a single Markdown file. Supported languages are Python, JavaScript, Bash, and R.

Here's an example of plotting Fourier series components:

![_media/obsidian-markdown-notebook-cover.png](_media/obsidian-markdown-notebook-cover.png)

There are quite a few similar plugins already. However, none of them is designed to store outputs directly in the Markdown file itself. **Obsidian Markdown Notebook** scratches an itch I've had for a while.

By default, outputs are rendered as HTML. You can also render as an image by adding `format=image` to the code fence:

````markdown
```python {format=image}
import matplotlib.pyplot as plt
plt.plot([1, 2, 3])
plt.show()
```
<!-- nb-output hash="209f46a0d3ce8ca2" format="image" -->
![](../_media/new-obsidian-plugin-obsidian-markdown-notebook-nb-209f46a0d3ce8ca2.png)
<!-- /nb-output -->
````

Output blocks use HTML comments, so they're invisible in PDF export and any standard Markdown renderer.

Each output block stores a hash of the cell's source code. If the code hasn't changed, the cached output is shown without re-executing. Re-running a cell updates the output in place.

You can also specify document-level defaults in frontmatter, or set project-level defaults in plugin settings.

### Similar Plugins

There is a lot of prior art here, but the major gap is that none of them focuses on storing the output artifact alongside the code.

- [Obsidian Execute Code Plugin](https://github.com/twibiral/obsidian-execute-code) is the closest relative. It does support persistent output since version 2.0.0. However, the output is plain text, and I want rich output (HTML tables, images) to be a first-class citizen from the start.
- [Obsidian Code Emitter](https://github.com/mokeyish/obsidian-code-emitter) is a great plugin that supports 15 different languages without requiring any system dependencies. However, the outputs do not survive vault reload and cannot be rendered to PDF.
- [JupyMD](https://github.com/d-eniz/jupymd) uses [Jupytext](https://github.com/mwouts/jupytext) to pair a Markdown file with a Jupyter notebook, but the outputs are stored in the Jupyter file. I just want something totally native where everything lives in the Markdown file.
- [Jupyter Notebook](https://github.com/jupyter/notebook) was the primary inspiration. There is also a [Markdown-based notebooks proposal](https://github.com/jupyter/enhancement-proposals/pull/103) from 2023 that stalled without consensus. This project takes a pragmatic, Obsidian-native approach rather than waiting for a standard to emerge.

---

The project was developed using a [Research, Plan, Implement Workflow](research-plan-implement-workflow.md). You can see the Markdown files for each in the `.claude` directory.

See the [project Github](https://github.com/lextoumbourou/obsidian-markdown-notebook) for more details.