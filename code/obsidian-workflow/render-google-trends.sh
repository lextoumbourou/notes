#!/bin/bash
# Render the Google Trends embed used by
# notes/permanent/6-months-of-openclaw.md.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$DIR/../../notes/_media/6-months-of-openclaw/openclaw-google-trends.png"
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless --disable-gpu --hide-scrollbars \
  --default-background-color=ffffffff \
  --force-device-scale-factor=2 \
  --run-all-compositor-stages-before-draw \
  --virtual-time-budget=10000 \
  --window-size=795,469 \
  --screenshot="$OUT" \
  "file://$DIR/google-trends.html"
echo "Rendered $OUT"
