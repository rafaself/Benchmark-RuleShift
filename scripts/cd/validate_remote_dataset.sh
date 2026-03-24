#!/usr/bin/env bash
set -euo pipefail
# Validate that the dataset was published to Kaggle.
# Requires env: DATASET_ID

OWNER="${DATASET_ID%%/*}"
SLUG="${DATASET_ID##*/}"

rm -rf /tmp/kaggle-dataset-verify
mkdir -p /tmp/kaggle-dataset-verify

kaggle datasets metadata -o "${OWNER}" -d "${SLUG}" -p /tmp/kaggle-dataset-verify
test -s /tmp/kaggle-dataset-verify/dataset-metadata.json \
  || { echo "ERROR: remote dataset metadata was not downloaded" >&2; exit 1; }
