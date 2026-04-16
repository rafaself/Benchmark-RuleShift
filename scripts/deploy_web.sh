#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing .env file at $ENV_FILE" >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required to trigger the Cloudflare Pages deploy hook." >&2
  exit 1
fi

set -a
. "$ENV_FILE"
set +a

: "${CLOUDFLARE_PAGES_DEPLOY_HOOK_URL:?Missing CLOUDFLARE_PAGES_DEPLOY_HOOK_URL in .env}"

echo "Triggering Cloudflare Pages deploy hook..."
curl --fail --silent --show-error --request POST "$CLOUDFLARE_PAGES_DEPLOY_HOOK_URL" >/dev/null
echo "Cloudflare Pages deploy hook triggered."
