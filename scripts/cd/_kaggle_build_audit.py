from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def assert_dataset_metadata_matches_canonical(
    canonical: dict[str, Any],
    packaged: dict[str, Any],
) -> None:
    expected_keys = set(canonical) | {"resources"}
    actual_keys = set(packaged)

    missing = sorted(set(canonical) - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    drifted = sorted(
        key for key, value in canonical.items()
        if packaged.get(key) != value
    )
    resources = packaged.get("resources")

    errors: list[str] = []
    if missing:
        errors.append(f"missing canonical keys: {missing}")
    if unexpected:
        errors.append(f"unexpected packaged keys: {unexpected}")
    if drifted:
        errors.append(f"drifted canonical fields: {drifted}")
    if not isinstance(resources, list) or not resources:
        errors.append("resources must be a non-empty list")

    if errors:
        raise ValueError(
            "Packaged dataset-metadata.json diverged from canonical metadata: "
            + "; ".join(errors)
        )


def assert_kernel_metadata_matches_canonical(
    canonical: dict[str, Any],
    packaged: dict[str, Any],
) -> None:
    if packaged != canonical:
        raise ValueError(
            "Packaged kernel-metadata.json diverged from canonical metadata."
        )


def assert_sorted_unique_resource_paths(packaged: dict[str, Any]) -> None:
    resources = packaged.get("resources")
    if not isinstance(resources, list):
        raise ValueError("resources must be a list")

    paths = [entry.get("path") for entry in resources]
    if any(not isinstance(path, str) or not path for path in paths):
        raise ValueError("resources entries must contain non-empty string paths")
    if paths != sorted(paths):
        raise ValueError("resources paths must be sorted lexicographically")
    if len(paths) != len(set(paths)):
        raise ValueError("resources paths must be unique")


def build_file_inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256(path),
            "size": path.stat().st_size,
        }
        for path in sorted(p for p in root.rglob("*") if p.is_file())
    ]


def write_audit_manifest(
    *,
    manifest_path: Path,
    artifact_kind: str,
    canonical_metadata_path: Path,
    packaged_metadata_path: Path,
    metadata_contract: str,
    output_dir: Path,
) -> None:
    payload = {
        "artifact_kind": artifact_kind,
        "metadata_contract": metadata_contract,
        "canonical_metadata_path": canonical_metadata_path.as_posix(),
        "packaged_metadata_path": packaged_metadata_path.as_posix(),
        "canonical_metadata_sha256": sha256(canonical_metadata_path),
        "packaged_metadata_sha256": sha256(packaged_metadata_path),
        "files": build_file_inventory(output_dir),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
