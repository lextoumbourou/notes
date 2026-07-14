#!/bin/bash
# Render the food-log flow diagram (HTML) to a transparent retina PNG
# used by notes/permanent/6-months-of-openclaw.md.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$DIR/../../notes/_media/6-months-of-openclaw/openclaw-food-log.png"
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless --disable-gpu --hide-scrollbars \
  --default-background-color=00000000 \
  --force-device-scale-factor=2 \
  --window-size=1240,690 \
  --screenshot="$OUT" \
  "file://$DIR/food-log-diagram.html"
echo "Rendered $OUT"
