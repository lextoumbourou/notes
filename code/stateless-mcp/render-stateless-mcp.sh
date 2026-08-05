#!/bin/bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$DIR/../../notes/_media/stateless-mcp/stateless-mcp-how-it-works.png"

"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless --disable-gpu --hide-scrollbars \
  --default-background-color=ffffffff \
  --force-device-scale-factor=2 \
  --window-size=1200,675 \
  --screenshot="$OUT" \
  "file://$DIR/stateless-mcp-how-it-works.html?step=4"

echo "Rendered $OUT"
