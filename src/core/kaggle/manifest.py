from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from core.splits import MANIFEST_VERSION, load_split_manifest
from tasks.ruleshift_benchmark.schema import (
    DIFFICULTY_VERSION,
    GENERATOR_VERSION,
    SPEC_VERSION,
    TEMPLATE_SET_VERSION,
)

__all__ = [
    "KAGGLE_STAGING_MANIFEST_PATH",
    "RUN_MANIFEST_FILENAME",
    "build_run_manifest",
    "load_kaggle_staging_manifest",
    "resolve_kaggle_artifact_path",
    "validate_kaggle_staging_manifest",
    "write_run_manifest",
]

RUN_MANIFEST_FILENAME = "run_manifest.json"
_ARTIFACT_GROUPS: Final[tuple[str, ...]] = ("entry_points", "frozen_split_manifests")
_RUNTIME_ENTRY_POINTS: Final[tuple[str, ...]] = ("kbench_notebook", "kernel_metadata")
_MANIFEST_PARTITIONS: Final[tuple[str, ...]] = ("dev", "public_leaderboard")
_EXPECTED_BENCHMARK_VERSIONS: Final[dict[str, str]] = {
    "manifest_version": MANIFEST_VERSION,
    "spec_version": SPEC_VERSION,
    "generator_version": GENERATOR_VERSION,
    "template_set_version": TEMPLATE_SET_VERSION,
    "difficulty_version": DIFFICULTY_VERSION,
}


def _repo_root(repo_root: Path | str | None = None) -> Path:
    if repo_root is None:
        return Path(__file__).resolve().parents[3]
    return Path(repo_root).resolve()


def _manifest_path(repo_root: Path | str | None = None) -> Path:
    return _repo_root(repo_root) / "packaging" / "kaggle" / "frozen_artifacts_manifest.json"


KAGGLE_STAGING_MANIFEST_PATH: Final[Path] = _manifest_path()


def load_kaggle_staging_manifest(repo_root: Path | str | None = None) -> dict[str, object]:
    return json.loads(_manifest_path(repo_root).read_text(encoding="utf-8"))


def resolve_kaggle_artifact_path(
    relative_path: str,
    *,
    repo_root: Path | str | None = None,
) -> Path:
    return _repo_root(repo_root) / relative_path


def validate_kaggle_staging_manifest(repo_root: Path | str | None = None) -> None:
    manifest = load_kaggle_staging_manifest(repo_root)
    benchmark_versions = manifest.get("benchmark_versions")
    if benchmark_versions != _EXPECTED_BENCHMARK_VERSIONS:
        raise ValueError("benchmark_versions must match the canonical split and schema versions")

    frozen_split_manifests = _require_mapping(manifest, "frozen_split_manifests")
    entry_points = _require_mapping(manifest, "entry_points")
    if tuple(entry_points) != _RUNTIME_ENTRY_POINTS:
        raise ValueError("entry_points must contain only the official runtime submission paths")
    if tuple(frozen_split_manifests) != _MANIFEST_PARTITIONS:
        raise ValueError("frozen_split_manifests must follow the canonical partition order")

    for partition in _MANIFEST_PARTITIONS:
        artifact = _require_mapping(frozen_split_manifests, partition)
        split_manifest = load_split_manifest(partition)
        if artifact.get("manifest_version") != split_manifest.manifest_version:
            raise ValueError(f"{partition} manifest_version does not match the frozen split")
        if artifact.get("seed_bank_version") != split_manifest.seed_bank_version:
            raise ValueError(f"{partition} seed_bank_version does not match the frozen split")
        if artifact.get("episode_split") != split_manifest.episode_split.value:
            raise ValueError(f"{partition} episode_split does not match the frozen split")

    for group_name in _ARTIFACT_GROUPS:
        artifact_group = _require_mapping(manifest, group_name)
        for label in artifact_group:
            artifact_map = _require_mapping(artifact_group, label)
            relative_path = artifact_map.get("path")
            if not isinstance(relative_path, str) or not relative_path:
                raise ValueError(f"{group_name}.{label} must define a non-empty path")
            sha256 = artifact_map.get("sha256")
            if not isinstance(sha256, str) or not sha256:
                raise ValueError(f"{group_name}.{label} must define a non-empty sha256")

            artifact_path = resolve_kaggle_artifact_path(relative_path, repo_root=repo_root)
            if not artifact_path.is_file():
                raise FileNotFoundError(
                    f"{group_name}.{label} points to a missing file: {artifact_path}"
                )
            actual_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            if actual_sha256 != sha256:
                raise ValueError(
                    f"{group_name}.{label} sha256 mismatch: expected {sha256}, got {actual_sha256}"
                )


def build_run_manifest(
    *,
    run_id: str,
    repo_root: Path | str,
    provider: str = "unknown",
    model: str = "unknown",
    started_at: str | None = None,
    finished_at: str | None = None,
) -> dict[str, Any]:
    root = _repo_root(repo_root)
    kaggle_manifest = load_kaggle_staging_manifest(root)
    benchmark_versions = kaggle_manifest.get("benchmark_versions")
    if not isinstance(benchmark_versions, dict):
        benchmark_versions = {}

    runtime_dataset_metadata = _read_json_file(root / "packaging" / "kaggle" / "dataset-metadata.json")
    runtime_kernel_metadata = _read_json_file(root / "packaging" / "kaggle" / "kernel-metadata.json")
    notebook_path = _resolve_notebook_path(kaggle_manifest, repo_root=root)
    return {
        "run_id": run_id,
        "git_commit": _resolve_git_commit(root),
        "benchmark_version": _string_or_none(benchmark_versions.get("manifest_version")),
        "parser_version": "v2",
        "metrics_version": "v1",
        "notebook_bundle_hash": _sha256_file(notebook_path) if notebook_path is not None else None,
        "runtime_dataset_id": _resolve_runtime_dataset_id(
            dataset_metadata=runtime_dataset_metadata,
            kernel_metadata=runtime_kernel_metadata,
        ),
        "runtime_dataset_version": None,
        "provider": provider,
        "model": model,
        "started_at": started_at or _utc_now_isoformat(),
        "finished_at": finished_at or _utc_now_isoformat(),
    }


def write_run_manifest(
    *,
    run_id: str,
    output_dir: Path | str,
    repo_root: Path | str,
    provider: str = "unknown",
    model: str = "unknown",
    started_at: str | None = None,
    finished_at: str | None = None,
) -> Path:
    manifest = build_run_manifest(
        run_id=run_id,
        repo_root=repo_root,
        provider=provider,
        model=model,
        started_at=started_at,
        finished_at=finished_at,
    )
    manifest_path = Path(output_dir).resolve() / RUN_MANIFEST_FILENAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _resolve_git_commit(repo_root: Path) -> str | None:
    github_sha = _string_or_none(os.environ.get("GITHUB_SHA"))
    if github_sha is not None:
        return github_sha

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    value = result.stdout.strip()
    return value or None


def _resolve_notebook_path(
    kaggle_manifest: dict[str, object],
    *,
    repo_root: Path,
) -> Path | None:
    entry_points = kaggle_manifest.get("entry_points")
    if not isinstance(entry_points, dict):
        return None
    notebook_entry = entry_points.get("kbench_notebook")
    if not isinstance(notebook_entry, dict):
        return None
    relative_path = notebook_entry.get("path")
    if not isinstance(relative_path, str) or not relative_path:
        return None
    path = resolve_kaggle_artifact_path(relative_path, repo_root=repo_root)
    if not path.is_file():
        return None
    return path


def _resolve_runtime_dataset_id(
    *,
    dataset_metadata: dict[str, Any],
    kernel_metadata: dict[str, Any],
) -> str | None:
    dataset_id = _string_or_none(dataset_metadata.get("id"))
    if dataset_id is not None:
        return dataset_id
    dataset_sources = kernel_metadata.get("dataset_sources")
    if isinstance(dataset_sources, list) and dataset_sources:
        return _string_or_none(dataset_sources[0])
    return None


def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _string_or_none(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _utc_now_isoformat() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_mapping(mapping: dict[str, object], key: str) -> dict[str, object]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be a mapping")
    return value
