#!/bin/bash

# Serve the blog locally in dev mode: rebuilds on file changes,
# serves at http://localhost:8000
ENV=local uv run pelican ./notes/ --output=output/ --autoreload --listen
