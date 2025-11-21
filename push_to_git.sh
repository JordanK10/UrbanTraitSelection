#!/bin/zsh

# Usage:
#   push_to_git "commit message"

if [ -z "$1" ]; then
    echo "Error: Please provide a commit message."
    echo "Usage: push_to_git \"your commit message\""
    exit 1
fi

COMMIT_MSG="$1"

# Clean previous staging
git reset

# Add everything except unwanted extensions
# -- ':!pattern' is the git pathspec negation syntax
git add . \
    ':!*.csv' \
    ':!*.pkl' \
    ':!*.key' \
    ':!*.pdf' \
    ':!*.RDataTmp' \
    ':!*.shp' \
    ':!*.geojson' \
    ':!*.dbf' \
    ':!*.shx' \
    ':!**/.RDataTmp'

# Manually unstage the specific forbidden files
git restore --staged calculation_scripts/data/census_data1.pkl 2>/dev/null
git restore --staged '**/.RDataTmp' 2>/dev/null
git restore --staged 'finalized_figures/figures14-19_INET' 2>/dev/null

# Commit and push
git commit -m "$COMMIT_MSG"
git push