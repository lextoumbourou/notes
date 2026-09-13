#!/bin/bash

set -euo pipefail

build_started_at=$SECONDS
report_build_time() {
    build_exit_status=$?
    build_elapsed=$((SECONDS - build_started_at))
    if [ "$build_exit_status" -eq 0 ]; then
        printf 'Build completed in %ss.\n' "$build_elapsed"
    else
        printf 'Build failed after %ss (exit %s).\n' "$build_elapsed" "$build_exit_status" >&2
    fi
}
trap report_build_time EXIT

if [ "${MERMAID_RENDERER:-mmdr}" = "mmdr" ]; then
    uv run python scripts/install_mmdr.py
fi

uv run python scripts/check_control_characters.py ./notes/
ENV=local PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}." \
  uv run pelican ./notes/ --output=output/ --fatal errors

echo "Building search index with pagefind..."
npx pagefind --site output --output-path output/pagefind
