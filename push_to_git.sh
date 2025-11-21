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
    ':!*.pdf'

# Manually unstage the specific forbidden file
git restore --staged census_data1.pkl 2>/dev/null

# Commit and push
git commit -m "$COMMIT_MSG"
git push