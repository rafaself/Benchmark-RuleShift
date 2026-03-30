#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ruleshift

python -m pip install --quiet -r requirements-dev.txt
python -m pip install --quiet -e .

exec "$@"
