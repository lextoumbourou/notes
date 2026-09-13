"""Check note files for control bytes with one portable process."""

import argparse
import os
from pathlib import Path
import re
import sys


CONTROL_BYTES = re.compile(rb"[\x00-\x08\x0b-\x0c\x0e-\x1f]")
TEXT_SUFFIXES = {".md", ".rst", ".txt"}


def scan(root):
    """Yield the first forbidden byte in each file, without following symlinks."""
    if not root.is_dir():
        raise OSError(f"Not a directory: {root}")

    def raise_walk_error(error):
        raise error

    for directory, subdirectories, filenames in os.walk(root, onerror=raise_walk_error):
        subdirectories.sort()
        for filename in sorted(filenames):
            path = Path(directory) / filename
            if path.suffix not in TEXT_SUFFIXES or path.is_symlink() or not path.is_file():
                continue
            data = path.read_bytes()
            match = CONTROL_BYTES.search(data)
            if match:
                offset = match.start()
                yield path, data.count(b"\n", 0, offset) + 1, data[offset]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("notes"))
    args = parser.parse_args(argv)
    print("Checking for control characters in notes files...")
    found = False
    try:
        for path, line, byte in scan(args.directory):
            print(f"{str(path)!r}:{line}: control character U+{byte:04X}", file=sys.stderr)
            found = True
    except OSError as error:
        print(f"Cannot complete control-character check: {error}", file=sys.stderr)
        return 2
    if found:
        print("Control characters detected. Fix the reported files before building.", file=sys.stderr)
        return 1
    print("No control characters found. Proceeding with build...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
