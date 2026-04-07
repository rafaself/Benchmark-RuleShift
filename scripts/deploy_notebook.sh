#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"
NOTEBOOK_DIR="$ROOT_DIR/kaggle/notebook"
DEFAULT_KAGGLE_BIN="$ROOT_DIR/.venv/bin/kaggle"
KAGGLE_BIN="${KAGGLE_BIN:-$DEFAULT_KAGGLE_BIN}"
KAGGLE_TMP_HOME="$(mktemp -d)"
KAGGLE_TMPDIR="$KAGGLE_TMP_HOME/tmp"
STAGING_DIR="$(mktemp -d)"
LEGACY_NOTEBOOK_ID="${LEGACY_NOTEBOOK_ID:-raptorengineer/ruleshift-cogflex-notebook-v2}"

cleanup() {
  rm -rf "$KAGGLE_TMP_HOME" "$STAGING_DIR"
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

update_kernel_id() {
  local metadata_path="$1"
  local target_id="$2"
  python3 - "$metadata_path" "$target_id" <<'PY'
import json
import sys
from pathlib import Path

metadata_path = Path(sys.argv[1])
target_id = sys.argv[2]
payload = json.loads(metadata_path.read_text(encoding="utf-8"))
payload["id"] = target_id
payload.pop("id_no", None)
metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

run_kernel_push() {
  local notebook_path="$1"
  local output_path="$2"
  HOME="$KAGGLE_TMP_HOME" TMPDIR="$KAGGLE_TMPDIR" "$KAGGLE_BIN" kernels push -p "$notebook_path" >"$output_path" 2>&1
  if grep -q "Kernel push error:" "$output_path"; then
    return 1
  fi
  return 0
}

echo "Publishing notebook from $NOTEBOOK_DIR"
OUTPUT_LOG="$STAGING_DIR/kernel-push.log"
if run_kernel_push "$NOTEBOOK_DIR" "$OUTPUT_LOG"; then
  cat "$OUTPUT_LOG"
  exit 0
fi

cat "$OUTPUT_LOG" >&2
if ! grep -q "Kernel push error: Notebook not found" "$OUTPUT_LOG"; then
  exit 1
fi

echo "Canonical notebook slug not found. Attempting legacy slug migration from $LEGACY_NOTEBOOK_ID" >&2
cp "$NOTEBOOK_DIR/kernel-metadata.json" "$STAGING_DIR/kernel-metadata.json"
cp "$NOTEBOOK_DIR/ruleshift_notebook_task.ipynb" "$STAGING_DIR/ruleshift_notebook_task.ipynb"

update_kernel_id "$STAGING_DIR/kernel-metadata.json" "$LEGACY_NOTEBOOK_ID"
if ! run_kernel_push "$STAGING_DIR" "$OUTPUT_LOG"; then
  cat "$OUTPUT_LOG" >&2
  exit 1
fi
cat "$OUTPUT_LOG"

echo "Retrying notebook push with canonical slug." >&2
update_kernel_id "$STAGING_DIR/kernel-metadata.json" "raptorengineer/ruleshift-cogflex-notebook"
if ! run_kernel_push "$STAGING_DIR" "$OUTPUT_LOG"; then
  cat "$OUTPUT_LOG" >&2
  exit 1
fi
cat "$OUTPUT_LOG"
