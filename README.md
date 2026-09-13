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
node definitions while still returning valid SVG. On the current corpus, 17
diagrams use Rust and two use the official CLI.
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

The TOC plugin scans for heading fragments and parses only those fragments with
BeautifulSoup. This preserves heading text and anchor rules while avoiding a full
HTML tree and rewrite for every article. HTML outside headings keeps its original
serialization, apart from nonbreaking-space entities needed to preserve generated
social previews. Malformed or ambiguous markup uses the original full-document
parser.

Run the Markdown extension, TOC and build-failure tests with:

```bash
uv run python -m unittest discover -s tests -v
```

### Dev server

To preview the blog locally with auto-reload (serves at http://localhost:8000):

```
./dev.sh
```
