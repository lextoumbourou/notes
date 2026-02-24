#!/bin/bash

# Install uv if not already available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Sync Python dependencies using uv
echo "Installing dependencies with uv..."
uv sync

# Install npm dependencies (mermaid-cli, puppeteer)
echo "Installing npm dependencies..."
npm install

# Set Puppeteer to use Chromium from netlify-plugin-chromium
if [ -n "$CHROME_PATH" ]; then
    echo "Setting PUPPETEER_EXECUTABLE_PATH to $CHROME_PATH"
    export PUPPETEER_EXECUTABLE_PATH="$CHROME_PATH"
fi

# Clone pelican-plugins repository
git clone --recursive https://github.com/getpelican/pelican-plugins

# Run build script
./build.sh
