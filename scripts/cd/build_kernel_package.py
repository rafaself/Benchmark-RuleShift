#!/usr/bin/env python3
"""Build the Kaggle notebook deployment bundle.

Produces a directory ready for `kaggle kernels push` containing:
  - ruleshift_notebook_task.ipynb  (copied verbatim)
  - kernel-metadata.json           (copied verbatim from packaging/kaggle)

The notebook is never modified. kernel-metadata.json is the canonical
file checked into the repository; no fields are generated at build time.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from _kaggle_build_audit import (
    assert_kernel_metadata_matches_canonical,
    sha256,
    write_audit_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
_KAGGLE_DIR = REPO_ROOT / "packaging" / "kaggle"
_MANIFEST_PATH = _KAGGLE_DIR / "frozen_artifacts_manifest.json"
_KERNEL_METADATA_PATH = _KAGGLE_DIR / "kernel-metadata.json"

_REQUIRED_FIELDS = ("id", "title", "code_file", "dataset_sources")


# ---------------------------------------------------------------------------
def _verify_notebook_hash(notebook_src: Path, manifest: dict) -> None:
    declared = manifest["entry_points"]["kbench_notebook"]["sha256"]
    actual = sha256(notebook_src)
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

def _build(output_dir: Path, audit_manifest_path: Path | None = None) -> None:
    if not _MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"frozen_artifacts_manifest.json not found at {_MANIFEST_PATH}")
    if not _KERNEL_METADATA_PATH.is_file():
        raise FileNotFoundError(f"kernel-metadata.json not found at {_KERNEL_METADATA_PATH}")

    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    kernel_metadata = json.loads(_KERNEL_METADATA_PATH.read_text(encoding="utf-8"))

    missing = [f for f in _REQUIRED_FIELDS if not kernel_metadata.get(f)]
    if missing:
        raise ValueError(f"kernel-metadata.json is missing required fields: {missing}")

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

    # Copy canonical kernel-metadata.json verbatim
    packaged_metadata_path = output_dir / "kernel-metadata.json"
    shutil.copy2(_KERNEL_METADATA_PATH, packaged_metadata_path)

    packaged_metadata = json.loads(packaged_metadata_path.read_text(encoding="utf-8"))
    assert_kernel_metadata_matches_canonical(
        canonical=kernel_metadata,
        packaged=packaged_metadata,
    )

    if audit_manifest_path is not None:
        write_audit_manifest(
            manifest_path=audit_manifest_path,
            artifact_kind="kernel_bundle",
            canonical_metadata_path=_KERNEL_METADATA_PATH,
            packaged_metadata_path=packaged_metadata_path,
            metadata_contract="byte_identical_metadata_copy",
            output_dir=output_dir,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the Kaggle notebook deployment bundle.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the kernel bundle.",
    )
    parser.add_argument(
        "--audit-manifest-path",
        type=Path,
        help="Optional path for a JSON audit manifest describing the built bundle.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    audit_manifest_path = (
        args.audit_manifest_path.resolve() if args.audit_manifest_path is not None else None
    )
    _build(output_dir, audit_manifest_path=audit_manifest_path)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
