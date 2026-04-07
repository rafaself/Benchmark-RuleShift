#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"
DATASET_DIR="$ROOT_DIR/kaggle/dataset/private"
METADATA_FILE="$DATASET_DIR/dataset-metadata.json"
ROWS_FILE="$DATASET_DIR/private_leaderboard_rows.json"
KAGGLE_BIN="${KAGGLE_BIN:-kaggle}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing .env file at $ENV_FILE" >&2
  exit 1
fi

if ! command -v "$KAGGLE_BIN" &>/dev/null; then
  echo "Kaggle CLI not found ('$KAGGLE_BIN'). Install it or set KAGGLE_BIN to its path." >&2
  exit 1
fi

set -a
. "$ENV_FILE"
set +a

if [[ -z "${KAGGLE_API_TOKEN:-}" ]]; then
  echo "Missing KAGGLE_API_TOKEN in .env" >&2
  exit 1
fi

if [[ ! -f "$METADATA_FILE" ]]; then
  echo "Missing private dataset metadata at $METADATA_FILE. Run scripts/build_ruleshift_dataset.py first." >&2
  exit 1
fi

if [[ ! -f "$ROWS_FILE" ]]; then
  echo "Missing private leaderboard rows at $ROWS_FILE. Run scripts/build_ruleshift_dataset.py first." >&2
  exit 1
fi

MESSAGE="${1:-Update RuleShift private dataset}"
STAGING_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGING_DIR"' EXIT

cp "$METADATA_FILE" "$STAGING_DIR/dataset-metadata.json"
cp "$ROWS_FILE" "$STAGING_DIR/private_leaderboard_rows.json"

echo "Publishing private dataset from staged payload $STAGING_DIR"
"$KAGGLE_BIN" datasets version -p "$STAGING_DIR" -m "$MESSAGE"
