#!/usr/bin/env python3
"""Validate that the published Kaggle kernel matches local metadata.

Requires env: KERNEL_ID, DATASET_ID, DEPLOY_NOTEBOOK_DIR
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

kernel_id = os.environ["KERNEL_ID"]
dataset_id = os.environ["DATASET_ID"]
notebook_dir = Path(os.environ["DEPLOY_NOTEBOOK_DIR"])

verify_dir = Path("/tmp/kaggle-kernel-verify")
if verify_dir.exists():
    shutil.rmtree(verify_dir)
verify_dir.mkdir(parents=True)

subprocess.run(
    ["kaggle", "kernels", "pull", "-p", str(verify_dir), "-k", kernel_id, "-m"],
    check=True,
)

remote_meta_path = verify_dir / "kernel-metadata.json"
if not remote_meta_path.exists() or remote_meta_path.stat().st_size == 0:
    print("ERROR: remote kernel metadata was not downloaded", file=sys.stderr)
    sys.exit(1)

local_meta = json.loads((notebook_dir / "kernel-metadata.json").read_text())
remote_meta = json.loads(remote_meta_path.read_text())

errors = []

if remote_meta.get("id") != local_meta.get("id"):
    errors.append(f"remote id {remote_meta.get('id')!r} != local id {local_meta.get('id')!r}")
if remote_meta.get("code_file") != local_meta.get("code_file"):
    errors.append(
        f"remote code_file {remote_meta.get('code_file')!r} != local code_file {local_meta.get('code_file')!r}"
    )
remote_sources = remote_meta.get("dataset_sources", [])
if dataset_id not in remote_sources:
    errors.append(
        f"expected dataset source {dataset_id!r} not found in remote dataset_sources {remote_sources!r}"
    )

if errors:
    print("Remote kernel validation FAILED:", file=sys.stderr)
    for err in errors:
        print(f"  - {err}", file=sys.stderr)
    sys.exit(1)

print("Remote kernel validation passed.")
