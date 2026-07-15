#!/bin/bash
# Render the agent-agnostic vault diagram used by
# notes/permanent/6-months-of-openclaw.md.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$DIR/../../notes/_media/6-months-of-openclaw/agent-agnostic-vault.png"
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless --disable-gpu --hide-scrollbars \
  --default-background-color=ffffffff \
  --force-device-scale-factor=2 \
  --window-size=1100,620 \
  --screenshot="$OUT" \
  "file://$DIR/agent-agnostic-diagram.html"
echo "Rendered $OUT"
