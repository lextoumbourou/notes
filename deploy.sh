#!/bin/bash

# From https://github.com/fgimian/fgimian.github.io/blob/source/deploy.sh (thanks Fots!)

# Create a temporary deploy for the deployment
deploy_dir=$(mktemp -d -t lextoumbourou.github.io)
echo "Using deploy directory ${deploy_dir}"

# Determine the remote origin Git location
git_remote=$(git config --get remote.origin.url)
echo "Determined that Git remote is ${git_remote}"

# Build the site in the deploy directory
echo
echo "Building static site"
pelican ./content --output "$deploy_dir"

# Initialise the deploy directory as a Git repository and add content
cd "$deploy_dir"

echo
echo "Creating a new Git repository and adding content"
git init
git remote add origin "$git_remote"
git checkout --orphan master
git add .
git commit -m "Site updated at $(date -u "+%Y-%m-%d %H:%M:%S") UTC"

echo
echo "Deploying code to the master branch"
git push --force origin master

# Clean up the deploy directory
echo
echo "Cleaning up deploy directory"
rm -rf "$deploy_dir"

# Display a success message
echo "Deployment complete"
