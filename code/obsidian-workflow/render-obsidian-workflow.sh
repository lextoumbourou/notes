#!/bin/bash
# Render the Obsidian workflow diagram (HTML) to a transparent retina PNG
# used by notes/permanent/6-months-of-openclaw.md.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$DIR/../../notes/_media/6-months-of-openclaw/obsidian-workflow.png"
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless --disable-gpu --hide-scrollbars \
  --default-background-color=ffffffff \
  --window-size=1000,520 \
  --screenshot="$OUT" \
  "file://$DIR/obsidian-workflow-diagram.html"
echo "Rendered $OUT"
