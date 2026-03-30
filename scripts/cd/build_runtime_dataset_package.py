#!/usr/bin/env python3
"""Build the public Kaggle runtime dataset package.

Produces a clean directory ready for Kaggle dataset upload containing:
  - src/           (runtime-only benchmark logic + frozen public split manifests)
  - packaging/kaggle/frozen_artifacts_manifest.json
  - dataset-metadata.json  (from packaging/kaggle, augmented with resources)

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
_KAGGLE_DIR = REPO_ROOT / "packaging" / "kaggle"
_DATASET_METADATA_PATH = _KAGGLE_DIR / "dataset-metadata.json"

_REQUIRED_FIELDS = ("id", "title", "licenses")

_FORBIDDEN_FILENAMES = frozenset(
    (
        "private_leaderboard.json",
        "private_episodes.json",
    )
)
_IGNORED_DIRS = frozenset((".git", ".venv", ".pytest_cache", "__pycache__"))
_RUNTIME_SOURCE_RELPATHS = (
    "src/core/__init__.py",
    "src/core/kaggle/__init__.py",
    "src/core/kaggle/diagnostics_summary.py",
    "src/core/kaggle/episode_ledger.py",
    "src/core/kaggle/execution.py",
    "src/core/kaggle/execution_artifacts.py",
    "src/core/kaggle/failure_categories.py",
    "src/core/kaggle/manifest.py",
    "src/core/kaggle/notebook_status.py",
    "src/core/kaggle/payload.py",
    "src/core/kaggle/run_context.py",
    "src/core/kaggle/run_log_io.py",
    "src/core/kaggle/run_logging.py",
    "src/core/kaggle/run_manifest.py",
    "src/core/kaggle/types.py",
    "src/core/metrics.py",
    "src/core/parser.py",
    "src/core/private_split.py",
    "src/core/slices.py",
    "src/core/splits.py",
    "src/frozen_splits/dev.json",
    "src/frozen_splits/public_leaderboard.json",
    "src/tasks/__init__.py",
    "src/tasks/ruleshift_benchmark/__init__.py",
    "src/tasks/ruleshift_benchmark/generator.py",
    "src/tasks/ruleshift_benchmark/protocol.py",
    "src/tasks/ruleshift_benchmark/render.py",
    "src/tasks/ruleshift_benchmark/rules.py",
    "src/tasks/ruleshift_benchmark/schema.py",
    "src/tasks/ruleshift_benchmark/schema_derivations.py",
)


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


def _copy_runtime_source_subset(output_dir: Path) -> None:
    missing_sources = [
        relpath for relpath in _RUNTIME_SOURCE_RELPATHS
        if not (REPO_ROOT / relpath).is_file()
    ]
    if missing_sources:
        raise FileNotFoundError(
            "Missing runtime source files: " + ", ".join(missing_sources)
        )

    for relpath in _RUNTIME_SOURCE_RELPATHS:
        source = REPO_ROOT / relpath
        destination = output_dir / relpath
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _assert_dataset_metadata_matches_canonical(
    canonical: dict[str, object],
    packaged: dict[str, object],
) -> None:
    expected_keys = set(canonical) | {"resources"}
    actual_keys = set(packaged)

    if actual_keys != expected_keys:
        raise ValueError("Packaged dataset-metadata.json has unexpected keys")
    for key, value in canonical.items():
        if packaged.get(key) != value:
            raise ValueError(f"Packaged dataset-metadata.json drifted at {key!r}")

    resources = packaged.get("resources")
    if not isinstance(resources, list) or not resources:
        raise ValueError("resources must be a non-empty list")


def _assert_sorted_unique_resource_paths(packaged: dict[str, object]) -> None:
    resources = packaged.get("resources")
    if not isinstance(resources, list):
        raise ValueError("resources must be a list")

    paths = [entry.get("path") for entry in resources if isinstance(entry, dict)]
    if len(paths) != len(resources):
        raise ValueError("resources entries must be mappings")
    if any(not isinstance(path, str) or not path for path in paths):
        raise ValueError("resources entries must contain non-empty string paths")
    if paths != sorted(paths):
        raise ValueError("resources paths must be sorted lexicographically")
    if len(paths) != len(set(paths)):
        raise ValueError("resources paths must be unique")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _build(output_dir: Path) -> None:
    src_dir = REPO_ROOT / "src"
    frozen_manifests_path = _KAGGLE_DIR / "frozen_artifacts_manifest.json"

    if not src_dir.is_dir():
        raise FileNotFoundError(f"src/ not found at {src_dir}")
    if not frozen_manifests_path.is_file():
        raise FileNotFoundError(f"frozen_artifacts_manifest.json not found at {frozen_manifests_path}")
    if not _DATASET_METADATA_PATH.is_file():
        raise FileNotFoundError(f"dataset-metadata.json not found at {_DATASET_METADATA_PATH}")

    dataset_metadata = json.loads(_DATASET_METADATA_PATH.read_text(encoding="utf-8"))
    missing = [f for f in _REQUIRED_FIELDS if not dataset_metadata.get(f)]
    if missing:
        raise ValueError(f"dataset-metadata.json is missing required fields: {missing}")

    _check_no_private_artifacts(src_dir, frozen_manifests_path)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    _copy_runtime_source_subset(output_dir)

    # Copy frozen_artifacts_manifest.json into packaging/kaggle/ mirror
    manifest_dest_dir = output_dir / "packaging" / "kaggle"
    manifest_dest_dir.mkdir(parents=True)
    shutil.copy2(frozen_manifests_path, manifest_dest_dir / "frozen_artifacts_manifest.json")

    # Verify split manifest hashes against what was just copied
    manifest = json.loads(frozen_manifests_path.read_text(encoding="utf-8"))
    _verify_split_manifest_hashes(manifest, output_dir)

    # Write dataset-metadata.json: canonical fields + resources list
    dataset_metadata["resources"] = [
        {"path": str(p.relative_to(output_dir))}
        for p in sorted(output_dir.rglob("*"))
        if p.is_file() and p.name != "dataset-metadata.json"
    ]
    packaged_metadata_path = output_dir / "dataset-metadata.json"
    packaged_metadata_path.write_text(
        json.dumps(dataset_metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    packaged_metadata = json.loads(packaged_metadata_path.read_text(encoding="utf-8"))
    _assert_dataset_metadata_matches_canonical(
        canonical=json.loads(_DATASET_METADATA_PATH.read_text(encoding="utf-8")),
        packaged=packaged_metadata,
    )
    _assert_sorted_unique_resource_paths(packaged_metadata)


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
