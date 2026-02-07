#!/bin/bash

# Check for control characters in notes files
echo "Checking for control characters in notes files..."
found_control_chars=false

# Find all text files in notes directory
while IFS= read -r -d '' file; do
    # Check for control characters using grep with Perl regex
    # Pattern matches: \x00-\x08, \x0B-\x0C, \x0E-\x1F
    if grep -Pq '[\x00-\x08\x0B-\x0C\x0E-\x1F]' "$file" 2>/dev/null; then
        echo "Warning: Control characters found in: $file"
        found_control_chars=true
    fi
done < <(find ./notes/ -type f \( -name "*.md" -o -name "*.rst" -o -name "*.txt" \) -print0)

if [ "$found_control_chars" = true ]; then
    echo "Control characters detected in one or more files. Please review and clean them before building."
    exit 1
fi

echo "No control characters found. Proceeding with build..."
rm -rf output/
ENV=local uv run pelican ./notes/ --output=output/
