"""Check built pages for notebook code blocks whose stored output is a failed run."""

import argparse
from pathlib import Path
import sys


# The markdown notebook plugin writes this marker above the output of a failed run.
FAILURE_MARKER = 'class="nb-status-error"'


def scan(root):
    """Yield each built HTML page containing a failed run, with the marker's line number."""
    if not root.is_dir():
        raise OSError(f"Not a directory: {root}")
    for path in sorted(root.rglob("*.html")):
        if path.is_symlink() or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        offset = text.find(FAILURE_MARKER)
        if offset != -1:
            yield path, text.count("\n", 0, offset) + 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("output"))
    args = parser.parse_args(argv)
    print("Checking built pages for failed code block runs...")
    found = False
    try:
        for path, line in scan(args.directory):
            print(f"{str(path)!r}:{line}: Execution failed block", file=sys.stderr)
            found = True
    except OSError as error:
        print(f"Cannot complete execution-failure check: {error}", file=sys.stderr)
        return 2
    if found:
        print(
            "Failed code block runs detected. Fix the code, or mark the block "
            "{run=false} and remove its stored output.",
            file=sys.stderr,
        )
        return 1
    print("No failed code block runs found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
