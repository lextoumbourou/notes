#!/bin/bash

# Install uv if not already available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Sync Python dependencies using uv
echo "Installing dependencies with uv..."
uv sync

# Clone pelican-plugins repository
git clone --recursive https://github.com/getpelican/pelican-plugins

# Run build script
./build.sh
