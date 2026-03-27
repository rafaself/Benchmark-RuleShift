#!/usr/bin/env python3
"""Build the public Kaggle runtime dataset package.

Produces a clean directory ready for Kaggle dataset upload containing:
  - src/           (benchmark logic + frozen public split manifests)
  - packaging/kaggle/frozen_artifacts_manifest.json
  - dataset-metadata.json  (generated)

Private artifacts are explicitly forbidden. The script aborts if any
forbidden file is detected in the source tree before copying.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_DATASET_ID = "raptorengineer/ruleshift-runtime"
_DATASET_TITLE = "RuleShift Runtime"
_LICENSE = "CC0-1.0"

_FORBIDDEN_FILENAMES = frozenset(
    (
        "private_leaderboard.json",
        "private_episodes.json",
    )
)
_IGNORED_DIRS = frozenset((".git", ".venv", ".pytest_cache", "__pycache__"))


# ---------------------------------------------------------------------------
# Safety gate
# ---------------------------------------------------------------------------

def _check_no_private_artifacts(src_dir: Path, frozen_manifests_path: Path) -> None:
    """Abort if any private artifact is present in the source tree."""
    errors: list[str] = []

    for search_root in (src_dir,):
        for name in _FORBIDDEN_FILENAMES:
            for hit in search_root.rglob(name):
                if any(part in _IGNORED_DIRS for part in hit.parts):
                    continue
                errors.append(f"forbidden private artifact in src: {hit.relative_to(REPO_ROOT)}")

    manifest = json.loads(frozen_manifests_path.read_text(encoding="utf-8"))
    if "private_leaderboard" in manifest.get("frozen_split_manifests", {}):
        errors.append(
            "frozen_artifacts_manifest.json references private_leaderboard — "
            "must not be included in the public package"
        )

    if errors:
        for msg in errors:
            print(f"ERROR: {msg}", file=sys.stderr)
        raise SystemExit("Private data detected — aborting build.")


# ---------------------------------------------------------------------------
# Integrity verification
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _verify_split_manifest_hashes(
    manifest: dict, output_dir: Path
) -> None:
    """Verify copied frozen split manifests match hashes declared in frozen_artifacts_manifest."""
    for partition, entry in manifest.get("frozen_split_manifests", {}).items():
        dest = output_dir / entry["path"]
        if not dest.is_file():
            raise FileNotFoundError(f"Expected split manifest not found in output: {dest}")
        actual = _sha256(dest)
        expected = entry["sha256"]
        if actual != expected:
            raise ValueError(
                f"Hash mismatch for {entry['path']} ({partition}): "
                f"expected {expected}, got {actual}"
            )


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _build(output_dir: Path) -> None:
    src_dir = REPO_ROOT / "src"
    frozen_manifests_path = REPO_ROOT / "packaging" / "kaggle" / "frozen_artifacts_manifest.json"

    if not src_dir.is_dir():
        raise FileNotFoundError(f"src/ not found at {src_dir}")
    if not frozen_manifests_path.is_file():
        raise FileNotFoundError(f"frozen_artifacts_manifest.json not found at {frozen_manifests_path}")

    _check_no_private_artifacts(src_dir, frozen_manifests_path)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Copy src/, excluding compiled bytecode caches
    shutil.copytree(
        src_dir,
        output_dir / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )

    # Copy frozen_artifacts_manifest.json into packaging/kaggle/ mirror
    manifest_dest_dir = output_dir / "packaging" / "kaggle"
    manifest_dest_dir.mkdir(parents=True)
    shutil.copy2(frozen_manifests_path, manifest_dest_dir / "frozen_artifacts_manifest.json")

    # Verify split manifest hashes against what was just copied
    manifest = json.loads(frozen_manifests_path.read_text(encoding="utf-8"))
    _verify_split_manifest_hashes(manifest, output_dir)

    # Generate dataset-metadata.json
    dataset_metadata = {
        "id": _DATASET_ID,
        "title": _DATASET_TITLE,
        "licenses": [{"name": _LICENSE}],
        "resources": [
            {"path": str(p.relative_to(output_dir))}
            for p in sorted(output_dir.rglob("*"))
            if p.is_file() and p.name != "dataset-metadata.json"
        ],
    }
    (output_dir / "dataset-metadata.json").write_text(
        json.dumps(dataset_metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the public Kaggle runtime dataset package.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the runtime package.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    _build(output_dir)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
