#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"
DEFAULT_KAGGLE_BIN="$ROOT_DIR/.venv/bin/kaggle"
KAGGLE_BIN="${KAGGLE_BIN:-$DEFAULT_KAGGLE_BIN}"
KAGGLE_TMP_HOME="$(mktemp -d)"
KAGGLE_TMPDIR="$KAGGLE_TMP_HOME/tmp"
DEFAULT_PRIVATE_BUNDLE_DIR="$ROOT_DIR/kaggle/dataset/private"
PRIVATE_BUNDLE_DIR="${COGFLEX_PRIVATE_BUNDLE_DIR:-$DEFAULT_PRIVATE_BUNDLE_DIR}"
ROWS_FILE="private_leaderboard_rows.json"
ANSWER_KEY_FILE="private_answer_key.json"
PREDICTIONS_FILE="private_calibration_predictions.json"
MANIFEST_FILE="private_release_manifest.json"
QUALITY_FILE="private_quality_report.json"
METADATA_FILE="dataset-metadata.json"

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

MESSAGE="${1:-Update CogFlex Suite private dataset}"
STAGING_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "$STAGING_DIR" "$KAGGLE_TMP_HOME"
}

trap cleanup EXIT

if [[ ! -d "$PRIVATE_BUNDLE_DIR" ]]; then
  echo "Private bundle directory does not exist: $PRIVATE_BUNDLE_DIR" >&2
  exit 1
fi

for required_file in "$ROWS_FILE" "$ANSWER_KEY_FILE" "$PREDICTIONS_FILE" "$MANIFEST_FILE" "$QUALITY_FILE"; do
  if [[ ! -f "$PRIVATE_BUNDLE_DIR/$required_file" ]]; then
    echo "Missing required private bundle file: $PRIVATE_BUNDLE_DIR/$required_file" >&2
    exit 1
  fi
done

cp "$PRIVATE_BUNDLE_DIR/$ROWS_FILE" "$STAGING_DIR/$ROWS_FILE"
cp "$PRIVATE_BUNDLE_DIR/$ANSWER_KEY_FILE" "$STAGING_DIR/$ANSWER_KEY_FILE"
cp "$PRIVATE_BUNDLE_DIR/$PREDICTIONS_FILE" "$STAGING_DIR/$PREDICTIONS_FILE"
cp "$PRIVATE_BUNDLE_DIR/$MANIFEST_FILE" "$STAGING_DIR/$MANIFEST_FILE"
cp "$PRIVATE_BUNDLE_DIR/$QUALITY_FILE" "$STAGING_DIR/$QUALITY_FILE"
if [[ -f "$PRIVATE_BUNDLE_DIR/$METADATA_FILE" ]]; then
  cp "$PRIVATE_BUNDLE_DIR/$METADATA_FILE" "$STAGING_DIR/$METADATA_FILE"
else
cat >"$STAGING_DIR/$METADATA_FILE" <<'JSON'
{
  "id": "raptorengineer/cogflex-suite-runtime-private",
  "title": "CogFlex Suite Runtime Private",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}
JSON
fi

mkdir -p "$KAGGLE_TMP_HOME/.kaggle"
mkdir -p "$KAGGLE_TMPDIR"
printf '%s' "$KAGGLE_API_TOKEN" > "$KAGGLE_TMP_HOME/.kaggle/access_token"
chmod 600 "$KAGGLE_TMP_HOME/.kaggle/access_token"

echo "Publishing private dataset from staged payload $STAGING_DIR"
if HOME="$KAGGLE_TMP_HOME" TMPDIR="$KAGGLE_TMPDIR" "$KAGGLE_BIN" datasets version -p "$STAGING_DIR" -m "$MESSAGE"; then
  exit 0
fi

echo "Private dataset version failed; attempting dataset creation instead."
HOME="$KAGGLE_TMP_HOME" TMPDIR="$KAGGLE_TMPDIR" "$KAGGLE_BIN" datasets create -p "$STAGING_DIR"
