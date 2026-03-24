#!/usr/bin/env bash
set -euo pipefail
# Remove transient files (__pycache__, .pyc, .pyo, .DS_Store) from directories.
# Usage: clean_deploy_artifacts.sh DIR [DIR...]

for dir in "$@"; do
  find "${dir}" -type d -name "__pycache__" -prune -exec rm -rf {} +
  find "${dir}" -type f \( -name "*.pyc" -o -name "*.pyo" -o -name ".DS_Store" \) -delete
done
