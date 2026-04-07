#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"
NOTEBOOK_DIR="$ROOT_DIR/kaggle/notebook"
DEFAULT_KAGGLE_BIN="$ROOT_DIR/.venv/bin/kaggle"
KAGGLE_BIN="${KAGGLE_BIN:-$DEFAULT_KAGGLE_BIN}"
KAGGLE_TMP_HOME="$(mktemp -d)"
KAGGLE_TMPDIR="$KAGGLE_TMP_HOME/tmp"

cleanup() {
  rm -rf "$KAGGLE_TMP_HOME"
}

trap cleanup EXIT

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

mkdir -p "$KAGGLE_TMP_HOME/.kaggle"
mkdir -p "$KAGGLE_TMPDIR"
printf '%s' "$KAGGLE_API_TOKEN" > "$KAGGLE_TMP_HOME/.kaggle/access_token"
chmod 600 "$KAGGLE_TMP_HOME/.kaggle/access_token"

echo "Publishing notebook from $NOTEBOOK_DIR"
HOME="$KAGGLE_TMP_HOME" TMPDIR="$KAGGLE_TMPDIR" "$KAGGLE_BIN" kernels push -p "$NOTEBOOK_DIR"
