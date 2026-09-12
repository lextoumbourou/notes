#!/bin/bash

set -euo pipefail

if [ "${MERMAID_RENDERER:-mmdr}" = "mmdr" ]; then
    uv run python scripts/install_mmdr.py
fi

# Serve the blog locally in dev mode: rebuilds on file changes,
# serves at http://localhost:8000
ENV=local PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}." \
  uv run pelican ./notes/ --output=output/ --autoreload --listen --fatal errors
