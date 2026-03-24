#!/usr/bin/env bash
set -euo pipefail
# Determine whether to create or version a Kaggle dataset.
# Requires env: DATASET_ID, GITHUB_OUTPUT

OWNER="${DATASET_ID%%/*}"
SLUG="${DATASET_ID##*/}"

rm -rf /tmp/kaggle-meta-check
mkdir -p /tmp/kaggle-meta-check

set +e
kaggle datasets metadata -o "${OWNER}" -d "${SLUG}" -p /tmp/kaggle-meta-check >/tmp/kaggle-meta.stdout 2>/tmp/kaggle-meta.stderr
STATUS=$?
set -e

if [ "${STATUS}" -eq 0 ]; then
  echo "mode=version" >> "${GITHUB_OUTPUT}"
  exit 0
fi

if grep -Eqi "404|not found" /tmp/kaggle-meta.stderr; then
  echo "mode=create" >> "${GITHUB_OUTPUT}"
  exit 0
fi

echo "ERROR: could not determine remote dataset state for ${DATASET_ID}" >&2
cat /tmp/kaggle-meta.stderr >&2
exit 1
