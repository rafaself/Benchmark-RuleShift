#!/usr/bin/env bash
# scripts/release_check.sh
#
# Full release gate: rebuild all artifacts, verify both splits, run tests.
# Writes .release_ok at the repo root on success; removes it on any failure.
# Deploy scripts refuse to publish unless this file exists and matches HEAD.
#
# Usage:
#   ./scripts/release_check.sh
#   make release-check
#
# Override the private bundle output directory:
#   COGFLEX_PRIVATE_BUNDLE_DIR=/abs/path ./scripts/release_check.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SENTINEL="$ROOT_DIR/.release_ok"
PYTHON="${PYTHON:-python3}"
DEFAULT_PRIVATE_BUNDLE_DIR="$ROOT_DIR/kaggle/dataset/private_local"
PRIVATE_BUNDLE_DIR="${COGFLEX_PRIVATE_BUNDLE_DIR:-$DEFAULT_PRIVATE_BUNDLE_DIR}"

# Remove stale sentinel immediately and on any error exit.
# Cancelled on the happy path before the final exit.
rm -f "$SENTINEL"
trap 'echo ""; echo "Release check FAILED. Sentinel cleared." >&2' EXIT

cd "$ROOT_DIR"

echo "=== [1/4] Rebuild public artifacts ==="
"$PYTHON" -m scripts.build_cogflex_dataset

echo ""
echo "=== [2/4] Verify public split ==="
"$PYTHON" -m scripts.verify_cogflex --split public

echo ""
echo "=== [3/4] Build synthetic private bundle ==="
COGFLEX_PRIVATE_BUNDLE_DIR="$PRIVATE_BUNDLE_DIR" \
  "$PYTHON" -m scripts.build_private_cogflex_dataset

echo ""
echo "=== [4/4] Verify private bundle ==="
"$PYTHON" -m scripts.verify_cogflex --split private \
  --private-bundle-dir "$PRIVATE_BUNDLE_DIR"

echo ""
echo "=== [+] Run test suite ==="
"$PYTHON" -m unittest discover -s tests -q

# --- All steps passed. Write sentinel. ---
GIT_HEAD="$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || echo "no-git")"
DIRTY_SUFFIX=""
if ! git -C "$ROOT_DIR" diff --quiet 2>/dev/null \
   || ! git -C "$ROOT_DIR" diff --cached --quiet 2>/dev/null; then
  DIRTY_SUFFIX="+dirty"
fi

printf '%s%s\n' "$GIT_HEAD" "$DIRTY_SUFFIX" > "$SENTINEL"

# Cancel the error trap so the sentinel is preserved.
trap - EXIT

echo ""
echo "============================================================"
echo "Release check PASSED."
echo "Sentinel : $SENTINEL"
echo "HEAD     : ${GIT_HEAD}${DIRTY_SUFFIX}"
echo "============================================================"
echo ""
echo "Publish commands:"
echo "  make deploy-dataset            # public dataset"
echo "  make deploy-private-dataset    # private dataset"
echo "  make deploy-notebook           # notebook"
