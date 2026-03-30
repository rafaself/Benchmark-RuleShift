from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from core.kaggle.manifest import (
    load_kaggle_staging_manifest,
    resolve_kaggle_artifact_path,
)
from core.kaggle.run_log_io import read_jsonl_records
from core.kaggle.run_logging import (
    BENCHMARK_LOG_FILENAME,
    BenchmarkRunContext,
)
from core.metrics import METRIC_VERSION
from core.parser import PARSER_VERSION

__all__ = [
    "RUN_MANIFEST_FILENAME",
    "build_run_manifest",
    "write_run_manifest",
]

RUN_MANIFEST_FILENAME = "run_manifest.json"


def build_run_manifest(
    *,
    context: BenchmarkRunContext,
    repo_root: Path | str,
) -> dict[str, Any]:
    """Build a compact immutable provenance record for a finalized run."""
    root = Path(repo_root).resolve()
    kaggle_manifest = load_kaggle_staging_manifest(root)
    benchmark_versions = kaggle_manifest.get("benchmark_versions")
    if not isinstance(benchmark_versions, dict):
        benchmark_versions = {}

    runtime_dataset_metadata = _read_json_file(root / "packaging" / "kaggle" / "dataset-metadata.json")
    runtime_kernel_metadata = _read_json_file(root / "packaging" / "kaggle" / "kernel-metadata.json")
    notebook_path = _resolve_notebook_path(kaggle_manifest, repo_root=root)
    log_records = read_jsonl_records(context.output_dir / BENCHMARK_LOG_FILENAME)

    return {
        "run_id": context.run_id,
        "git_commit": _resolve_git_commit(root),
        "benchmark_version": _string_or_none(benchmark_versions.get("manifest_version")),
        "parser_version": PARSER_VERSION,
        "metrics_version": METRIC_VERSION,
        "notebook_bundle_hash": _sha256_file(notebook_path) if notebook_path is not None else None,
        "runtime_dataset_id": _resolve_runtime_dataset_id(
            dataset_metadata=runtime_dataset_metadata,
            kernel_metadata=runtime_kernel_metadata,
        ),
        # Kaggle runtime dataset version is not currently carried in the canonical
        # checked-in metadata or run context, so keep the key stable with null.
        "runtime_dataset_version": None,
        "provider": context.provider,
        "model": context.model,
        "started_at": _find_event_timestamp(log_records, event="run_started"),
        "finished_at": _find_event_timestamp(log_records, event="run_finished", last=True),
    }


def write_run_manifest(
    *,
    context: BenchmarkRunContext,
    repo_root: Path | str,
) -> Path:
    manifest = build_run_manifest(context=context, repo_root=repo_root)
    manifest_path = context.output_dir / RUN_MANIFEST_FILENAME
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest_path


def _resolve_git_commit(repo_root: Path) -> str | None:
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


def _find_event_timestamp(
    log_records: list[dict[str, Any]],
    *,
    event: str,
    last: bool = False,
) -> str | None:
    selected_timestamp: str | None = None
    for record in log_records:
        if record.get("event") != event:
            continue
        timestamp = record.get("timestamp")
        if isinstance(timestamp, str) and timestamp:
            selected_timestamp = timestamp
            if not last:
                return selected_timestamp
    return selected_timestamp


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _string_or_none(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None
