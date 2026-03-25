#!/usr/bin/env python3
"""Verify that remote Kaggle artifacts match the local canonical manifest.

This script downloads the published notebook and dataset from Kaggle and
verifies their SHA-256 hashes against the local manifest.

Requires env:
    KERNEL_ID:  The Kaggle kernel ID (e.g., "user/slug")
    DATASET_ID: The Kaggle dataset ID (e.g., "user/slug")
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

# Ensure we can import core.kaggle
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
from core.kaggle import ArtifactResult, verify_remote_hashes

def get_sha256(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def run_command(cmd: List[str], description: str):
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR during {description}:")
        print(e.stderr)
        sys.exit(1)

def emit_report(results: List[ArtifactResult]):
    print("\n" + "="*80)
    print("KAGGLE REMOTE PUBLICATION DIVERGENCE REPORT")
    print("="*80)
    print(f"{'Artifact Name':<40} | {'Status':<10} | {'Local Hash':<12}... | {'Remote Hash':<12}...")
    print("-" * 80)
    
    failed = False
    for res in results:
        l_hash = res.local_hash[:12] if res.local_hash else "n/a"
        r_hash = res.remote_hash[:12] if res.remote_hash else "n/a"
        print(f"{res.name:<40} | {res.status:<10} | {l_hash:<12} | {r_hash:<12}")
        if res.status != "MATCH":
            failed = True
            
    print("="*80)
    if failed:
        print("RESULT: FAILURE - Remote content differs from local canonical artifacts.")
        sys.exit(1)
    else:
        print("RESULT: SUCCESS - Remote content matches local canonical artifacts.")

def main():
    kernel_id = os.environ.get("KERNEL_ID")
    dataset_id = os.environ.get("DATASET_ID")
    
    if not kernel_id or not dataset_id:
        print("ERROR: KERNEL_ID and DATASET_ID environment variables must be set.")
        sys.exit(1)

    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = repo_root / "packaging" / "kaggle" / "frozen_artifacts_manifest.json"
    tmp_dir = Path("/tmp/kaggle-verify-publication")

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    kernel_dir = tmp_dir / "kernel"
    dataset_dir = tmp_dir / "dataset"
    kernel_dir.mkdir()
    dataset_dir.mkdir()

    # 1. Download notebook and its metadata
    run_command(["kaggle", "kernels", "pull", kernel_id, "-p", str(kernel_dir)], "kernel pull")
    run_command(["kaggle", "kernels", "pull", kernel_id, "-p", str(kernel_dir), "-m"], "kernel metadata pull")

    # 2. Download dataset
    run_command(["kaggle", "datasets", "download", dataset_id, "-p", str(dataset_dir), "--unzip"], "dataset download")

    # 3. Load manifest
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # 4. Verify remote hashes
    results = verify_remote_hashes(manifest, kernel_dir, dataset_dir)
    
    # 5. Add manifest itself to the results (it's not verified by verify_remote_hashes directly)
    manifest_rel_path = Path("packaging/kaggle/frozen_artifacts_manifest.json")
    remote_manifest_path = dataset_dir / manifest_rel_path
    local_manifest_hash = get_sha256(manifest_path)
    remote_manifest_hash = None
    status = "MISSING"
    if remote_manifest_path.is_file():
        remote_manifest_hash = get_sha256(remote_manifest_path)
        status = "MATCH" if remote_manifest_hash == local_manifest_hash else "MISMATCH"
    
    results.append(ArtifactResult("manifest", local_manifest_hash, remote_manifest_hash, status))

    # 6. Emit report
    emit_report(results)

if __name__ == "__main__":
    main()
