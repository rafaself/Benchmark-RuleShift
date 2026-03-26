"""Private evaluation data loader.

Loads the private leaderboard split from a materialized episodes file
(`private_episodes.json`) rather than reconstructing it from seeds.

The private dataset is kept separate from the public runtime package and must
be attached explicitly in the private evaluation environment.

Entry points:
    resolve_private_dataset_root(...)
    load_private_split(...)
    load_private_split_manifest_info(...)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import hashlib
from typing import Final

from core.splits import FrozenSplitEpisode, MANIFEST_VERSION
from tasks.ruleshift_benchmark.schema import (
    DifficultyFactors,
    Episode,
    EpisodeItem,
    ProbeMetadata,
    derive_difficulty_factors,
    derive_difficulty_profile,
)

__all__ = [
    "PRIVATE_EPISODES_FILENAME",
    "PRIVATE_DATASET_ROOT_ENV_VAR",
    "PRIVATE_SPLIT_ARTIFACT_SCHEMA_VERSION",
    "build_private_split_artifact",
    "write_private_split_artifact",
    "load_private_split",
    "load_private_split_manifest_info",
    "resolve_private_dataset_root",
]

PRIVATE_EPISODES_FILENAME: Final[str] = "private_episodes.json"
PRIVATE_DATASET_ROOT_ENV_VAR: Final[str] = "RULESHIFT_PRIVATE_DATASET_ROOT"
PRIVATE_SPLIT_ARTIFACT_SCHEMA_VERSION: Final[str] = "private_split_artifact.v1"
_EXPECTED_PARTITION: Final[str] = "private_leaderboard"
_EXPECTED_EPISODE_SPLIT: Final[str] = "private"
_KAGGLE_PRIVATE_SEARCH_ROOTS: Final[tuple[Path, ...]] = (Path("/kaggle/input"),)


def build_private_split_artifact(
    *,
    benchmark_version: str,
    seeds: tuple[int, ...] | list[int],
) -> dict[str, object]:
    """Build the canonical private split artifact payload."""
    normalized_seeds = tuple(seeds)
    if benchmark_version != MANIFEST_VERSION:
        raise ValueError(
            f"benchmark_version must equal {MANIFEST_VERSION!r}, got {benchmark_version!r}"
        )
    if not normalized_seeds:
        raise ValueError("seeds must not be empty")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in normalized_seeds):
        raise ValueError("seeds must contain only integer values")
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("seeds must contain unique values")

    episodes = [
        {
            "seed": seed,
            "episode": _normalize_generated_private_episode(seed),
        }
        for seed in normalized_seeds
    ]
    payload = {
        "partition": _EXPECTED_PARTITION,
        "episode_split": _EXPECTED_EPISODE_SPLIT,
        "benchmark_version": benchmark_version,
        "schema_version": PRIVATE_SPLIT_ARTIFACT_SCHEMA_VERSION,
        "episodes": episodes,
    }
    return {
        **payload,
        "artifact_checksum": _compute_private_artifact_checksum(payload),
    }


def write_private_split_artifact(
    output_path: Path | str,
    *,
    benchmark_version: str,
    seeds: tuple[int, ...] | list[int],
) -> Path:
    """Materialize a private split artifact to disk."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = build_private_split_artifact(
        benchmark_version=benchmark_version,
        seeds=seeds,
    )
    destination.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def load_private_split(
    private_dataset_root: Path | str | None = None,
) -> tuple[FrozenSplitEpisode, ...]:
    """Load the private evaluation split from a materialized episodes file.

    Args:
        private_dataset_root: Directory containing ``private_episodes.json``.
            When omitted, the loader resolves the private dataset from the
            private evaluation environment.

    Returns:
        A tuple of :class:`~core.splits.FrozenSplitEpisode` records, one per
        private episode, in seed order.

    Raises:
        FileNotFoundError: If ``private_episodes.json`` is not found.
        ValueError: If the file is structurally invalid or partition metadata
            does not match expectations.
    """
    episodes_path = _resolve_private_episodes_path(private_dataset_root)
    payload = json.loads(episodes_path.read_text(encoding="utf-8"))
    return _parse_private_episodes(payload)


def load_private_split_manifest_info(
    private_dataset_root: Path | str | None = None,
) -> dict[str, object]:
    """Return manifest-equivalent metadata from ``private_episodes.json``."""
    episodes_path = _resolve_private_episodes_path(private_dataset_root)
    payload = json.loads(episodes_path.read_text(encoding="utf-8"))
    records = _parse_private_episodes(payload)

    return {
        "benchmark_version": payload.get("benchmark_version"),
        "schema_version": payload.get("schema_version"),
        "artifact_checksum": payload.get("artifact_checksum"),
        "episode_split": payload.get("episode_split", _EXPECTED_EPISODE_SPLIT),
        "seeds": tuple(record.seed for record in records),
    }


def resolve_private_dataset_root(
    private_dataset_root: Path | str | None = None,
) -> Path:
    """Resolve the mounted private dataset directory.

    Resolution order:
      1. Explicit ``private_dataset_root`` argument
      2. ``RULESHIFT_PRIVATE_DATASET_ROOT`` environment variable
      3. Kaggle input mounts under ``/kaggle/input``
    """
    if private_dataset_root is not None:
        return _validate_private_dataset_root(
            Path(private_dataset_root),
            context="explicit private_dataset_root",
        )

    env_value = os.environ.get(PRIVATE_DATASET_ROOT_ENV_VAR)
    if env_value:
        return _validate_private_dataset_root(
            Path(env_value),
            context=f"{PRIVATE_DATASET_ROOT_ENV_VAR}={env_value}",
        )

    for search_root in _KAGGLE_PRIVATE_SEARCH_ROOTS:
        if not search_root.exists():
            continue
        for episodes_path in search_root.rglob(PRIVATE_EPISODES_FILENAME):
            return episodes_path.parent

    raise FileNotFoundError(
        "Private evaluation dataset is not attached. "
        "Attach the authorized private dataset mount or set "
        f"{PRIVATE_DATASET_ROOT_ENV_VAR} to the mounted dataset root."
    )


def _resolve_private_episodes_path(
    private_dataset_root: Path | str | None,
) -> Path:
    return resolve_private_dataset_root(private_dataset_root) / PRIVATE_EPISODES_FILENAME


def _validate_private_dataset_root(root: Path, *, context: str) -> Path:
    candidate = root / PRIVATE_EPISODES_FILENAME
    if not candidate.is_file():
        raise FileNotFoundError(
            f"private_episodes.json not found for {context} at {candidate}. "
            "Attach the authorized private dataset mount before running private evaluation."
        )
    return root


def _parse_private_episodes(payload: object) -> tuple[FrozenSplitEpisode, ...]:
    if not isinstance(payload, dict):
        raise ValueError("private_episodes.json must contain a JSON object")

    partition = payload.get("partition")
    if partition != _EXPECTED_PARTITION:
        raise ValueError(
            f"partition must equal {_EXPECTED_PARTITION!r}, got {partition!r}"
        )

    episode_split = payload.get("episode_split")
    if episode_split != _EXPECTED_EPISODE_SPLIT:
        raise ValueError(
            f"episode_split must equal {_EXPECTED_EPISODE_SPLIT!r}, got {episode_split!r}"
        )

    benchmark_version = payload.get("benchmark_version")
    if benchmark_version != MANIFEST_VERSION:
        raise ValueError(
            f"benchmark_version must equal {MANIFEST_VERSION!r}, got {benchmark_version!r}"
        )

    schema_version = payload.get("schema_version")
    if schema_version != PRIVATE_SPLIT_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            "schema_version must equal "
            f"{PRIVATE_SPLIT_ARTIFACT_SCHEMA_VERSION!r}, got {schema_version!r}"
        )

    artifact_checksum = payload.get("artifact_checksum")
    if not isinstance(artifact_checksum, str) or not artifact_checksum:
        raise ValueError("artifact_checksum must be a non-empty string")

    episodes_raw = payload.get("episodes")
    if not isinstance(episodes_raw, list) or not episodes_raw:
        raise ValueError("episodes must be a non-empty list")

    if artifact_checksum != _compute_private_artifact_checksum(
        {
            "partition": partition,
            "episode_split": episode_split,
            "benchmark_version": benchmark_version,
            "schema_version": schema_version,
            "episodes": episodes_raw,
        }
    ):
        raise ValueError("artifact_checksum does not match the private artifact payload")

    return tuple(
        _parse_episode_row(row, benchmark_version, artifact_checksum)
        for row in episodes_raw
    )


def _normalize_generated_private_episode(seed: int) -> dict[str, object]:
    from core.validate import normalize_episode_payload
    from tasks.ruleshift_benchmark.generator import generate_episode
    from tasks.ruleshift_benchmark.protocol import Split

    return normalize_episode_payload(generate_episode(seed, split=Split.PRIVATE))


def _compute_private_artifact_checksum(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_episode_row(
    row: object,
    manifest_version: str,
    seed_bank_version: str,
) -> FrozenSplitEpisode:
    if not isinstance(row, dict):
        raise ValueError("each episode row must be a JSON object")

    seed = row.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("episode row 'seed' must be an int")

    episode_payload = row.get("episode")
    if not isinstance(episode_payload, dict):
        raise ValueError("episode row 'episode' must be a JSON object")

    episode = _build_episode(episode_payload)

    return FrozenSplitEpisode(
        partition=_EXPECTED_PARTITION,
        seed=seed,
        manifest_version=manifest_version,
        seed_bank_version=seed_bank_version,
        episode=episode,
    )


def _build_episode(ep: dict) -> Episode:  # type: ignore[type-arg]
    """Reconstruct a typed Episode from the normalized episode payload dict."""
    items = tuple(
        EpisodeItem(
            position=item["position"],
            phase=item["phase"],
            kind=item["kind"],
            q1=item["q1"],
            q2=item["q2"],
            label=item.get("label"),
        )
        for item in ep["items"]
    )
    probe_targets = tuple(ep["probe_targets"])
    probe_label_counts = tuple(
        (pair[0], pair[1]) for pair in ep["probe_label_counts"]
    )
    probe_sign_pattern_counts = tuple(
        (pair[0], pair[1]) for pair in ep["probe_sign_pattern_counts"]
    )
    probe_metadata = tuple(
        ProbeMetadata(
            position=pm["position"],
            is_disagreement_probe=pm["is_disagreement_probe"],
            old_rule_label=pm["old_rule_label"],
            new_rule_label=pm["new_rule_label"],
        )
        for pm in ep["probe_metadata"]
    )
    pre_count = ep["pre_count"]
    difficulty_factors = derive_difficulty_factors(items, pre_count)
    raw_difficulty_factors = ep.get("difficulty_factors", difficulty_factors)
    if isinstance(raw_difficulty_factors, dict):
        raw_difficulty_factors = DifficultyFactors(**raw_difficulty_factors)
    derived_difficulty, difficulty_profile_id = derive_difficulty_profile(
        difficulty_factors
    )
    return Episode(
        episode_id=ep["episode_id"],
        split=ep["split"],
        difficulty=ep.get("difficulty", derived_difficulty.value),
        template_id=ep["template_id"],
        template_family=ep.get("template_family", "canonical"),
        rule_A=ep["rule_A"],
        rule_B=ep["rule_B"],
        transition=ep["transition"],
        pre_count=pre_count,
        post_labeled_count=ep["post_labeled_count"],
        shift_after_position=ep["shift_after_position"],
        contradiction_count_post=ep["contradiction_count_post"],
        difficulty_profile_id=ep.get(
            "difficulty_profile_id",
            difficulty_profile_id.value,
        ),
        difficulty_factors=raw_difficulty_factors,
        items=items,
        probe_targets=probe_targets,
        probe_label_counts=probe_label_counts,
        probe_sign_pattern_counts=probe_sign_pattern_counts,
        probe_metadata=probe_metadata,
        difficulty_version=ep.get("difficulty_version", "R13"),
        spec_version=ep.get("spec_version", "v1"),
        generator_version=ep.get("generator_version", "R13"),
        template_set_version=ep.get("template_set_version", "v2"),
    )
