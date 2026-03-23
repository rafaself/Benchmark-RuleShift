from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Final

from core.parser import ParseStatus, parse_binary_output, parse_narrative_output
from core.splits import MANIFEST_VERSION, PARTITIONS, load_split_manifest
from tasks.iron_find_electric.protocol import PROBE_COUNT, InteractionLabel, parse_label
from tasks.iron_find_electric.schema import (
    DIFFICULTY_VERSION,
    GENERATOR_VERSION,
    SPEC_VERSION,
    TEMPLATE_SET_VERSION,
)

__all__ = [
    "Label",
    "BinaryResponse",
    "KAGGLE_STAGING_MANIFEST_PATH",
    "load_kaggle_staging_manifest",
    "normalize_binary_response",
    "normalize_narrative_response",
    "resolve_kaggle_artifact_path",
    "score_episode",
    "validate_kaggle_staging_manifest",
]

_ARTIFACT_GROUPS: Final[tuple[str, ...]] = (
    "entry_points",
    "frozen_split_manifests",
)
_RUNTIME_ENTRY_POINTS: Final[tuple[str, ...]] = (
    "kbench_notebook",
    "kernel_metadata",
)
_EXPECTED_BENCHMARK_VERSIONS: Final[dict[str, str]] = {
    "manifest_version": MANIFEST_VERSION,
    "spec_version": SPEC_VERSION,
    "generator_version": GENERATOR_VERSION,
    "template_set_version": TEMPLATE_SET_VERSION,
    "difficulty_version": DIFFICULTY_VERSION,
}


class Label(str, Enum):
    attract = "attract"
    repel = "repel"


@dataclass(frozen=True, slots=True)
class BinaryResponse:
    probe_6: Label
    probe_7: Label
    probe_8: Label
    probe_9: Label

    def as_tuple(self) -> tuple[str, str, str, str]:
        return (
            self.probe_6.value,
            self.probe_7.value,
            self.probe_8.value,
            self.probe_9.value,
        )


def _repo_root(repo_root: Path | str | None = None) -> Path:
    if repo_root is None:
        return Path(__file__).resolve().parents[2]
    return Path(repo_root).resolve()


def _manifest_path(repo_root: Path | str | None = None) -> Path:
    return _repo_root(repo_root) / "packaging" / "kaggle" / "frozen_artifacts_manifest.json"


KAGGLE_STAGING_MANIFEST_PATH: Final[Path] = _manifest_path()


def load_kaggle_staging_manifest(
    repo_root: Path | str | None = None,
) -> dict[str, object]:
    return json.loads(_manifest_path(repo_root).read_text(encoding="utf-8"))


def resolve_kaggle_artifact_path(
    relative_path: str,
    *,
    repo_root: Path | str | None = None,
) -> Path:
    return _repo_root(repo_root) / relative_path


def validate_kaggle_staging_manifest(
    repo_root: Path | str | None = None,
) -> None:
    manifest = load_kaggle_staging_manifest(repo_root)

    if manifest.get("bundle_version") != "R16":
        raise ValueError("bundle_version must equal R16")
    if manifest.get("task_id") != "iron_find_electric_v1":
        raise ValueError("task_id must equal iron_find_electric_v1")
    if manifest.get("task_name") != "Iron Find Electric v1":
        raise ValueError("task_name must equal Iron Find Electric v1")

    benchmark_versions = manifest.get("benchmark_versions")
    if benchmark_versions != _EXPECTED_BENCHMARK_VERSIONS:
        raise ValueError(
            "benchmark_versions must match the canonical split and schema versions"
        )

    if manifest.get("current_emitted_difficulty_labels") != ["easy", "medium"]:
        raise ValueError(
            "current_emitted_difficulty_labels must equal ['easy', 'medium']"
        )
    if manifest.get("reserved_difficulty_labels") != ["hard"]:
        raise ValueError("reserved_difficulty_labels must equal ['hard']")

    frozen_split_manifests = _require_mapping(
        manifest,
        "frozen_split_manifests",
    )
    entry_points = _require_mapping(
        manifest,
        "entry_points",
    )
    if tuple(entry_points) != _RUNTIME_ENTRY_POINTS:
        raise ValueError("entry_points must contain only the official runtime submission paths")
    if tuple(frozen_split_manifests) != PARTITIONS:
        raise ValueError("frozen_split_manifests must follow the canonical partition order")

    for partition in PARTITIONS:
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
        for label, artifact in artifact_group.items():
            artifact_map = _require_mapping(artifact_group, label)
            relative_path = artifact_map.get("path")
            if not isinstance(relative_path, str) or not relative_path:
                raise ValueError(f"{group_name}.{label} must define a non-empty path")
            sha256 = artifact_map.get("sha256")
            if not isinstance(sha256, str) or not sha256:
                raise ValueError(f"{group_name}.{label} must define a non-empty sha256")

            artifact_path = resolve_kaggle_artifact_path(
                relative_path,
                repo_root=repo_root,
            )
            if not artifact_path.is_file():
                raise FileNotFoundError(
                    f"{group_name}.{label} points to a missing file: {artifact_path}"
                )

            actual_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            if actual_sha256 != sha256:
                raise ValueError(
                    f"{group_name}.{label} sha256 mismatch: expected {sha256}, got {actual_sha256}"
                )


def normalize_binary_response(response: object) -> tuple[str, ...] | None:
    if isinstance(response, BinaryResponse):
        return response.as_tuple()

    if isinstance(response, str):
        parsed = parse_binary_output(response)
        if parsed.status is ParseStatus.VALID:
            return tuple(label.value for label in parsed.labels)

    return None


def normalize_narrative_response(response: object) -> tuple[str, ...] | None:
    if isinstance(response, str):
        parsed = parse_narrative_output(response)
        if parsed.status is ParseStatus.VALID:
            return tuple(label.value for label in parsed.labels)

    return None


def score_episode(
    predictions: tuple[str, ...] | tuple[InteractionLabel, ...] | None,
    probe_targets: tuple[str, ...] | tuple[InteractionLabel, ...],
) -> tuple[int, int]:
    normalized_targets = _normalize_labels(probe_targets)
    if normalized_targets is None:
        raise ValueError(f"probe_targets must contain exactly {PROBE_COUNT} valid labels")

    normalized_predictions = _normalize_labels(predictions)
    if normalized_predictions is None:
        return (0, PROBE_COUNT)

    num_correct = sum(
        prediction is target
        for prediction, target in zip(normalized_predictions, normalized_targets)
    )
    return (num_correct, PROBE_COUNT)


def _require_mapping(
    mapping: dict[str, object],
    key: str,
) -> dict[str, object]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be a mapping")
    return value


def _normalize_labels(
    labels: tuple[str, ...] | tuple[InteractionLabel, ...] | None,
) -> tuple[InteractionLabel, ...] | None:
    if labels is None:
        return None

    normalized_labels = tuple(parse_label(label) for label in labels)
    if len(normalized_labels) != PROBE_COUNT:
        return None
    return normalized_labels
