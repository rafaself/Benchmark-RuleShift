#!/usr/bin/env python3
"""Build the Kaggle notebook deployment bundle.

Produces a directory ready for `kaggle kernels push` containing:
  - ruleshift_notebook_task.ipynb  (copied verbatim)
  - kernel-metadata.json           (copied verbatim from packaging/kaggle)

The notebook is never modified. kernel-metadata.json is the canonical
file checked into the repository; no fields are generated at build time.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from _package_build import KAGGLE_DIR, REPO_ROOT, build_output_parser, load_required_json, reset_output_dir, sha256

MANIFEST_PATH = KAGGLE_DIR / "frozen_artifacts_manifest.json"
KERNEL_METADATA_PATH = KAGGLE_DIR / "kernel-metadata.json"

REQUIRED_FIELDS = ("id", "title", "code_file", "dataset_sources")


def _verify_notebook_hash(notebook_src: Path, manifest: dict) -> None:
    declared = manifest["entry_points"]["kbench_notebook"]["sha256"]
    actual = sha256(notebook_src)
    if actual != declared:
        raise ValueError(
            f"Notebook hash mismatch - source file may have changed without "
            f"updating frozen_artifacts_manifest.json.\n"
            f"  expected: {declared}\n"
            f"  actual:   {actual}"
        )


def _build(output_dir: Path) -> None:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"frozen_artifacts_manifest.json not found at {MANIFEST_PATH}")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    kernel_metadata = load_required_json(KERNEL_METADATA_PATH, required_fields=REQUIRED_FIELDS)

    notebook_relpath = manifest["entry_points"]["kbench_notebook"]["path"]
    notebook_src = REPO_ROOT / notebook_relpath
    if not notebook_src.is_file():
        raise FileNotFoundError(f"Notebook not found at {notebook_src}")

    _verify_notebook_hash(notebook_src, manifest)

    reset_output_dir(output_dir)

    shutil.copy2(notebook_src, output_dir / notebook_src.name)

    packaged_metadata_path = output_dir / "kernel-metadata.json"
    shutil.copy2(KERNEL_METADATA_PATH, packaged_metadata_path)

    packaged_metadata = json.loads(packaged_metadata_path.read_text(encoding="utf-8"))
    if packaged_metadata != kernel_metadata:
        raise ValueError("Packaged kernel-metadata.json diverged from canonical metadata.")


def main(argv: list[str] | None = None) -> int:
    args = build_output_parser(
        description="Build the Kaggle notebook deployment bundle.",
        help_text="Output directory for the kernel bundle.",
    ).parse_args(argv)
    output_dir = args.output_dir.resolve()
    _build(output_dir)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
