#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"
NOTEBOOK_DIR="$ROOT_DIR/kaggle/notebook"
KAGGLE_BIN="$ROOT_DIR/.venv/bin/kaggle"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing .env file at $ENV_FILE" >&2
  exit 1
fi

if [[ ! -x "$KAGGLE_BIN" ]]; then
  echo "Missing Kaggle CLI at $KAGGLE_BIN" >&2
  exit 1
fi

set -a
. "$ENV_FILE"
set +a

if [[ -z "${KAGGLE_API_TOKEN:-}" ]]; then
  echo "Missing KAGGLE_API_TOKEN in .env" >&2
  exit 1
fi

echo "Publishing notebook from $NOTEBOOK_DIR"
"$KAGGLE_BIN" kernels push -p "$NOTEBOOK_DIR"
