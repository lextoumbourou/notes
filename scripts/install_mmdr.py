"""Install the pinned native Mermaid renderer for build.sh and dev.sh."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mermaid_renderer import install_native

if __name__ == "__main__":
    install_native()
