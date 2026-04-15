#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRIVATE_SCRIPT="$ROOT_DIR/scripts/private_local/deploy_private_dataset.sh"

if [[ ! -f "$PRIVATE_SCRIPT" ]]; then
  echo "Missing local private deploy script: $PRIVATE_SCRIPT" >&2
  echo "Keep private deploy assets under scripts/private_local/." >&2
  exit 1
fi

exec "$PRIVATE_SCRIPT" "$@"
