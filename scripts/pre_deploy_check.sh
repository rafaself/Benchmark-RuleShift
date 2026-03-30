#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
RUNTIME_BUILD_DIR="${RUNTIME_BUILD_DIR:-/tmp/ruleshift-runtime-package-predeploy}"
KERNEL_BUILD_DIR="${KERNEL_BUILD_DIR:-/tmp/ruleshift-kernel-bundle-predeploy}"

echo "=== RuleShift pre-deploy gate ==="

echo "[gate] phase-1 environment sanity..."
"$PYTHON_BIN" --version
"$PYTHON_BIN" -c "from pathlib import Path; import core.kaggle.runner; print(f'cwd={Path.cwd()}'); print(f'core.kaggle.runner={core.kaggle.runner.__file__}')"
echo "[gate] phase-1 environment sanity: ok"

echo "[gate] phase-2 preflight..."
"$PYTHON_BIN" scripts/preflight_kaggle.py
echo "[gate] phase-2 preflight: ok"

echo "[gate] phase-3 targeted schema/runtime tests..."
"$PYTHON_BIN" -m pytest tests/test_kaggle_execution.py -v
echo "[gate] phase-3 targeted schema/runtime tests: ok"

echo "[gate] runtime dataset artifact consistency..."
rm -rf "$RUNTIME_BUILD_DIR"
"$PYTHON_BIN" scripts/build_runtime_dataset_package.py --output-dir "$RUNTIME_BUILD_DIR"
test -f "$RUNTIME_BUILD_DIR/dataset-metadata.json"
test -f "$RUNTIME_BUILD_DIR/packaging/kaggle/frozen_artifacts_manifest.json"
echo "[gate] runtime dataset artifact consistency: ok"

echo "[gate] kernel bundle consistency..."
rm -rf "$KERNEL_BUILD_DIR"
"$PYTHON_BIN" scripts/build_kernel_package.py --output-dir "$KERNEL_BUILD_DIR"
test -f "$KERNEL_BUILD_DIR/kernel-metadata.json"
test -f "$KERNEL_BUILD_DIR/ruleshift_notebook_task.ipynb"
echo "[gate] kernel bundle consistency: ok"

echo "=== Pre-deploy gate passed ==="
