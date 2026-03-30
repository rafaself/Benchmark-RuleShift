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

import json
import shutil
import sys
from pathlib import Path

from _package_build import KAGGLE_DIR, REPO_ROOT, build_output_parser, load_required_json, reset_output_dir, sha256

DATASET_METADATA_PATH = KAGGLE_DIR / "dataset-metadata.json"

REQUIRED_FIELDS = ("id", "title", "licenses")

FORBIDDEN_FILENAMES = frozenset(
    (
        "private_leaderboard.json",
        "private_episodes.json",
    )
)
IGNORED_DIRS = frozenset((".git", ".venv", ".pytest_cache", "__pycache__"))
RUNTIME_SOURCE_RELPATHS = (
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


def _check_no_private_artifacts(src_dir: Path, frozen_manifests_path: Path) -> None:
    """Abort if any private artifact is present in the source tree."""
    errors: list[str] = []

    for search_root in (src_dir,):
        for name in FORBIDDEN_FILENAMES:
            for hit in search_root.rglob(name):
                if any(part in IGNORED_DIRS for part in hit.parts):
                    continue
                errors.append(f"forbidden private artifact in src: {hit.relative_to(REPO_ROOT)}")

    manifest = json.loads(frozen_manifests_path.read_text(encoding="utf-8"))
    if "private_leaderboard" in manifest.get("frozen_split_manifests", {}):
        errors.append(
            "frozen_artifacts_manifest.json references private_leaderboard - "
            "must not be included in the public package"
        )

    if errors:
        for msg in errors:
            print(f"ERROR: {msg}", file=sys.stderr)
        raise SystemExit("Private data detected - aborting build.")


def _verify_split_manifest_hashes(manifest: dict, output_dir: Path) -> None:
    """Verify copied frozen split manifests match hashes declared in frozen_artifacts_manifest."""
    for partition, entry in manifest.get("frozen_split_manifests", {}).items():
        dest = output_dir / entry["path"]
        if not dest.is_file():
            raise FileNotFoundError(f"Expected split manifest not found in output: {dest}")
        actual = sha256(dest)
        expected = entry["sha256"]
        if actual != expected:
            raise ValueError(
                f"Hash mismatch for {entry['path']} ({partition}): "
                f"expected {expected}, got {actual}"
            )


def _copy_runtime_source_subset(output_dir: Path) -> None:
    missing_sources = [relpath for relpath in RUNTIME_SOURCE_RELPATHS if not (REPO_ROOT / relpath).is_file()]
    if missing_sources:
        raise FileNotFoundError("Missing runtime source files: " + ", ".join(missing_sources))

    for relpath in RUNTIME_SOURCE_RELPATHS:
        source = REPO_ROOT / relpath
        destination = output_dir / relpath
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


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


def _build(output_dir: Path) -> None:
    src_dir = REPO_ROOT / "src"
    frozen_manifests_path = KAGGLE_DIR / "frozen_artifacts_manifest.json"

    if not src_dir.is_dir():
        raise FileNotFoundError(f"src/ not found at {src_dir}")
    if not frozen_manifests_path.is_file():
        raise FileNotFoundError(f"frozen_artifacts_manifest.json not found at {frozen_manifests_path}")

    dataset_metadata = load_required_json(DATASET_METADATA_PATH, required_fields=REQUIRED_FIELDS)

    _check_no_private_artifacts(src_dir, frozen_manifests_path)

    reset_output_dir(output_dir)

    _copy_runtime_source_subset(output_dir)

    manifest_dest_dir = output_dir / "packaging" / "kaggle"
    manifest_dest_dir.mkdir(parents=True)
    shutil.copy2(frozen_manifests_path, manifest_dest_dir / "frozen_artifacts_manifest.json")

    manifest = json.loads(frozen_manifests_path.read_text(encoding="utf-8"))
    _verify_split_manifest_hashes(manifest, output_dir)

    dataset_metadata["resources"] = [
        {"path": str(path.relative_to(output_dir))}
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "dataset-metadata.json"
    ]
    packaged_metadata_path = output_dir / "dataset-metadata.json"
    packaged_metadata_path.write_text(
        json.dumps(dataset_metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    packaged_metadata = json.loads(packaged_metadata_path.read_text(encoding="utf-8"))
    canonical_dataset_metadata = load_required_json(DATASET_METADATA_PATH, required_fields=REQUIRED_FIELDS)
    _assert_dataset_metadata_matches_canonical(
        canonical=canonical_dataset_metadata,
        packaged=packaged_metadata,
    )
    _assert_sorted_unique_resource_paths(packaged_metadata)


def main(argv: list[str] | None = None) -> int:
    args = build_output_parser(
        description="Build the public Kaggle runtime dataset package.",
        help_text="Output directory for the runtime package.",
    ).parse_args(argv)
    output_dir = args.output_dir.resolve()
    _build(output_dir)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
