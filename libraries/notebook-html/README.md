# Notebook HTML

A small Python library for saving generated HTML and displaying it in Obsidian
Markdown Notebook or Jupyter. The notebook uses the vendor's normal SDK; this
library handles the final response. It makes no network requests and needs no
API credentials or third-party Python dependencies.

From the blog repository, install it into the notebook's Python environment:

```bash
uv pip install -e ./libraries/notebook-html anthropic
```

For a downloaded wheel, use `pip install ./notesbylex_notebook_html-0.1.1-py3-none-any.whl anthropic`.
Set `ANTHROPIC_API_KEY` in the notebook process's environment.

```python
from anthropic import Anthropic
from notebook_html import render_anthropic_html

response = Anthropic().messages.create(
    model="claude-opus-5-5",
    max_tokens=20000,
    system="Return a complete, self-contained HTML document.",
    messages=[{"role": "user", "content": "Make a small interactive quiz about caching."}],
)
render_anthropic_html(response, "quiz.html")
```

The last expression returns a rich HTML preview. It also writes `quiz.html` and
`quiz.json`, containing the actual model ID, timestamp, stop reason and token
usage. Thinking blocks and signatures are not saved. Prices are intentionally
not built in, because they vary by model, provider, service tier and date.

`render_anthropic_html` accepts a completed SDK `Message` or its decoded JSON.
It rejects interrupted responses, including `max_tokens` and `tool_use`, before
overwriting an existing page. It accepts a full HTML document, optionally wrapped
in a single Markdown HTML fence. It does not repair incomplete HTML.

For OpenAI's Responses API, install `openai` and version 0.1.2 of this helper,
then set `OPENAI_API_KEY` in the notebook process's environment:

```python
from openai import OpenAI
from notebook_html import render_openai_html

with OpenAI().responses.stream(
    model="gpt-6-sol",
    reasoning={"effort": "high"},
    max_output_tokens=128_000,
    instructions="Return a complete, self-contained HTML document.",
    input="Make a small interactive quiz about caching.",
) as stream:
    response = stream.get_final_response()
render_openai_html(response, "quiz.html")
```

`render_openai_html` accepts a completed SDK `Response` or its decoded JSON.
It saves the model, returned reasoning effort, service tier and usage, including
cache reads, cache writes and reasoning-token counts when reported. It excludes
reasoning content and rejects incomplete or refused responses before overwriting
the saved page. It does not call the API or calculate prices.

For another provider, extract its final text and use the shared renderer:

```python
from notebook_html import render_html
render_html(html_text, "quiz.html")
```

Provider-specific adapters should only extract the provider's completed text
and usage, then use `render_html`. They should not hide or issue model calls.

To display a saved file without paying for another generation:

```python
from notebook_html import load_html
load_html("quiz.html")
```

All renderers accept `height=900` and `title="Generated HTML"`. The preview uses
`srcdoc`, so it survives exporting Markdown without a running server. JavaScript
runs in a sandboxed iframe without same-origin access, storage, forms or popups.
Use self-contained HTML that does not need those features. This is display
isolation, not an HTML sanitizer; standalone HTML runs as a normal web page.

`read_article("note.md")` excludes YAML frontmatter and reads up to the hidden
`<!-- notebook-input-end -->` marker. Put that marker on its own line before the
example section. The visible heading can then be renamed freely.

For notes without a marker, it falls back to `## Hands-on example`, or the
heading supplied with `before="Another heading"`. The marker takes precedence.
A missing boundary raises an error rather than sending the notebook and its
saved outputs back to the model. Reopen the note after upgrading the library
so its persistent Python kernel imports the new version.

Run the focused tests with `python -m unittest discover -s libraries/notebook-html/tests -v`.
