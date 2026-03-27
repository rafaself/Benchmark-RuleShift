#!/usr/bin/env python3
"""Build the Kaggle notebook deployment bundle.

Produces a directory ready for `kaggle kernels push` containing:
  - ruleshift_notebook_task.ipynb  (copied verbatim)
  - kernel-metadata.json           (generated)

The notebook is never modified. kernel-metadata.json is generated fresh
from the canonical values in frozen_artifacts_manifest.json, with
dataset_sources injected from --runtime-dataset-slug.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_KAGGLE_DIR = REPO_ROOT / "packaging" / "kaggle"
_MANIFEST_PATH = _KAGGLE_DIR / "frozen_artifacts_manifest.json"

_KAGGLE_USERNAME = os.environ["KAGGLE_USERNAME"]
_KERNEL_ID = f"{_KAGGLE_USERNAME}/ruleshift-notebook-task"
_KERNEL_TITLE = "RuleShift Notebook Task \u2014 Cognitive Flexibility Benchmark"


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _verify_notebook_hash(notebook_src: Path, manifest: dict) -> None:
    declared = manifest["entry_points"]["kbench_notebook"]["sha256"]
    actual = _sha256(notebook_src)
    if actual != declared:
        raise ValueError(
            f"Notebook hash mismatch — source file may have changed without "
            f"updating frozen_artifacts_manifest.json.\n"
            f"  expected: {declared}\n"
            f"  actual:   {actual}"
        )


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _build(output_dir: Path, runtime_dataset_slug: str) -> None:
    if not _MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"frozen_artifacts_manifest.json not found at {_MANIFEST_PATH}")

    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))

    notebook_relpath = manifest["entry_points"]["kbench_notebook"]["path"]
    notebook_src = REPO_ROOT / notebook_relpath
    if not notebook_src.is_file():
        raise FileNotFoundError(f"Notebook not found at {notebook_src}")

    notebook_filename = notebook_src.name

    _verify_notebook_hash(notebook_src, manifest)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Copy notebook verbatim — no content mutation
    shutil.copy2(notebook_src, output_dir / notebook_filename)

    # Generate kernel-metadata.json
    kernel_metadata = {
        "id": _KERNEL_ID,
        "title": _KERNEL_TITLE,
        "code_file": notebook_filename,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [runtime_dataset_slug],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(kernel_metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the Kaggle notebook deployment bundle.",
    )
    parser.add_argument(
        "--runtime-dataset-slug",
        required=True,
        help=(
            "Fully qualified Kaggle dataset slug for the runtime package "
            "(e.g. KAGGLE_USERNAME/ruleshift-runtime). "
            "Injected into kernel-metadata.json as dataset_sources."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the kernel bundle.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    _build(output_dir, args.runtime_dataset_slug)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
