# Notes

You can view the rendered version of these notes at [notesbylex.com](https://notesbylex.com/).

---

This repository contains a collection of literature notes I've been accumulating since 2012 and a [slipbox](https://en.wikipedia.org/wiki/Zettelkasten) of ideas that I've more recently accumulating. The aim is not to provide a collection of information that will be very useful to many people, as it's very tailor to my own interested, but to provide an example of Zettelkasten tailored towards software develpoers.

The notes are taken using [Obsidian](https://obsidian.md/) and a publish version of the notes can be found [here](https://publish.obsidian.md/lex).

## Reference

The [literature](literature) folder contains notes collected mostly from MOOCs but also books, papers and videos.

## Slipbox

The [slipbox](slipbox) folder contains permanent notes which I've collected from reading and thinking. The idea is that each note contains one idea and all ideas are interlinked.

## Generating HTML

This project uses Pelican to generate a HTML version of the notes. You can generate the notes using the following command:

### Create a virtualenv (optional)

```
python3 -m venv .env
source .env/bin/activate
```

### Run setup script

Netlify installs Python before running `setup.sh`. Its `PYTHON_VERSION` is set
to `3.12` in `netlify.toml`, matching `.python-version` and the minimum version
in `pyproject.toml`. Keep these settings aligned when changing Python versions.

```
./setup.sh
```

### Build

To build the blog without running setup, use the `build.sh` script:

```
./build.sh
```

The build checks `.md`, `.rst` and `.txt` notes for forbidden control characters
in one Python process. It allows tabs and normal line endings, reports the file
and first offending line, and stops before Pelican if validation fails. This
replaces the per-file `grep -P` loop, which was slow and unsupported by macOS grep.
The final line reports total elapsed time in whole seconds, including validation,
Pelican and Pagefind. Failed builds report elapsed time and retain their exit code.

Mermaid diagrams use [mermaid-rs-renderer](https://github.com/1jehuang/mermaid-rs-renderer)
v0.3.1. The build installs the official binary into the ignored `.tools/` directory
on its first run, verifies the release archive's SHA-256, and reuses it afterwards.
macOS (Apple Silicon/Intel) and Linux with glibc (arm64/x86_64) are supported.
Netlify's `setup.sh` invokes the same build and installation step.

The native renderer has different diagram layouts and styling from Mermaid CLI.
Chained flowchart edges use the official CLI because mmdr v0.3.1 can misread their
node definitions while still returning valid SVG. Write one edge per line to
avoid this limitation. All 19 diagrams in the current corpus use Rust.
If it fails to render a diagram, the build logs a warning and tries the existing
Mermaid CLI. If both fail, the build exits with an error before search indexing.
The fallback still needs Node and Puppeteer/Chromium. A successful SVG can still
have visual differences, so inspect diagrams when adding new Mermaid features.

To use the previous renderer for comparisons or troubleshooting:

```bash
MERMAID_RENDERER=mmdc ./build.sh
```

`MMDR_BINARY` and `MMDC_BINARY` can override the executable paths. For a direct
Pelican invocation, install the native renderer first with
`uv run python scripts/install_mmdr.py` and use Pelican's `--fatal errors` option.

Obsidian callouts use `markdown_fast_callouts`, a small adapter around the existing
callout extension. Blocks without the required `[!` marker skip the expensive
regex search; matching, nesting, folding and HTML generation use the original
processor. The adapter can be removed when the dependency includes this precheck.

The TOC plugin reuses headings captured during Markdown rendering when possible.
The reader passes them with the rendered source, and link conversion keeps both
in sync. Articles with raw HTML or fenced code, custom heading
rules, unknown later Markdown processors or other intervening HTML edits retain
the HTML scanner. Both paths parse heading fragments with BeautifulSoup to keep
existing labels, anchors and serialization. Nonbreaking-space handling preserves
social previews; malformed or ambiguous markup retains the full-document parser.
Markdown's separate `[TOC]` marker search is disabled because the theme renders
the navigation. Restore its `marker` setting if inline TOCs are needed later.

Executable articles use Obsidian's Markdown Notebook plugin. Python cells and
their saved `nb-output` blocks live together in the `.md` file, with plots stored
in `notes/_media/`. Run cells in Obsidian to refresh their outputs, and give each
generated image descriptive alt text. The `notebook.python` frontmatter setting
selects the interpreter relative to the article's directory.

Model demos can use [Notebook HTML](libraries/notebook-html/README.md) to save a
completed model response and display it as an interactive notebook output. Keep
the vendor SDK call in the article, then call `render_anthropic_html(response,
"../_media/example.html")`. The helper stores HTML and usage, and keeps scripts
inside a sandboxed frame. It does not issue API calls. `load_html(path)` displays
an existing result without generating again. Build its downloadable wheel in a
temporary directory, then copy it into `notes/_media/` (uv puts a `.gitignore`
in its build output directory):

```bash
uv build --wheel libraries/notebook-html --out-dir /tmp/notebook-html-dist
cp /tmp/notebook-html-dist/notesbylex_notebook_html-0.1.1-py3-none-any.whl notes/_media/
```

Pelican's local `markdown_reader` renders these files as ordinary Markdown and
passes captured headings to the TOC plugin. Builds display the saved outputs
without executing code or loading the Jupytext/nbconvert reader. TF-IDF and
CBIS-DDSM no longer need paired `.ipynb` files.

The sidebar plugin caches recent-article lists for each generator during a build,
avoiding repeated filtering and sorting on every page. Tag and category pages
keep their own collections, and equal dates retain posts-first ordering. Only
article data is cached; links and active-page highlighting still render per page.
Each rebuild creates a fresh cache, including development auto-reload.

Pygments is pinned to 2.21.0 for its cached plugin discovery. Pelican 4.12.0
declares `pygments<2.20.0`, so `pyproject.toml` includes an explicit uv override
for the tested version. Use `uv sync` and the build scripts so this override is
applied. The site tests and full output comparison pass; highlighting tokens
can change while code text stays identical. Remove the override when Pelican's
dependency range permits 2.21.0.

Math uses the local `currency_katex` adapter around `pelican-katex`. Inline
`$...$` requires non-whitespace immediately inside both delimiters, and its
closing `$` must not be followed by a digit. Ordinary prices such as `$20 and
$30` remain prose. Write `\$` for an ambiguous literal dollar, and use backticks
for shell variables. Existing `$$...$$` display math and KaTeX options are
preserved. Markdown Notebook code and stored outputs are left untouched. The
adapter changes delimiter recognition, not the renderer.

Rendered KaTeX HTML is stored in Markdown's HTML stash, avoiding XML parsing and
repeated tree walks over generated MathML and SVG. Currency rules, preambles and
renderer options are preserved. Formulas with inline attribute lists retain the
element-tree path so IDs and classes still attach correctly. Extensions that
need to inspect math elements can use `CurrencyKatexExtension(stash_html=False)`.
HTML serialization can differ in attribute order and SVG whitespace while the
rendered formulas and their accessible MathML remain the same.

Run the Markdown extension, TOC, sidebar and build-failure tests with:

```bash
uv run python -m unittest discover -s tests -v
```

### Dev server

To preview the blog locally with auto-reload (serves at http://localhost:8000):

```
./dev.sh
```
