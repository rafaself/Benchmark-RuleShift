"""Private evaluation data loader.

Loads the private leaderboard split from a materialized episodes file
(`private_episodes.json`) rather than reconstructing it from seeds.

The private dataset is kept separate from the public runtime package.
On Kaggle it is attached as a second input dataset.  Locally it lives at
`packaging/kaggle/private/private_episodes.json` inside the repo.

Entry point for callers:
    load_private_split(private_dataset_root) -> tuple[FrozenSplitEpisode, ...]

`private_dataset_root` is the directory that contains `private_episodes.json`.
On Kaggle this will be the dataset root (e.g.
`/kaggle/input/ruleshift-private-evaluation/`).  Locally it defaults to
`packaging/kaggle/private/` inside the repo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from core.splits import FrozenSplitEpisode, MANIFEST_VERSION
from tasks.ruleshift_benchmark.schema import (
    Episode,
    EpisodeItem,
    ProbeMetadata,
)

__all__ = [
    "PRIVATE_EPISODES_FILENAME",
    "load_private_split",
]

PRIVATE_EPISODES_FILENAME: Final[str] = "private_episodes.json"
_EXPECTED_PARTITION: Final[str] = "private_leaderboard"


def load_private_split(
    private_dataset_root: Path | str | None = None,
) -> tuple[FrozenSplitEpisode, ...]:
    """Load the private evaluation split from a materialized episodes file.

    Args:
        private_dataset_root: Directory containing ``private_episodes.json``.
            Defaults to ``packaging/kaggle/private/`` relative to the repo root.

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


def _resolve_private_episodes_path(
    private_dataset_root: Path | str | None,
) -> Path:
    if private_dataset_root is not None:
        root = Path(private_dataset_root)
        candidate = root / PRIVATE_EPISODES_FILENAME
        if not candidate.is_file():
            raise FileNotFoundError(
                f"private_episodes.json not found at {candidate}"
            )
        return candidate

    # Default: repo-relative location
    repo_root = Path(__file__).resolve().parents[2]
    default_path = (
        repo_root / "packaging" / "kaggle" / "private" / PRIVATE_EPISODES_FILENAME
    )
    if not default_path.is_file():
        raise FileNotFoundError(
            f"private_episodes.json not found at default path {default_path}. "
            "Pass private_dataset_root explicitly or attach the private dataset."
        )
    return default_path


def _parse_private_episodes(payload: object) -> tuple[FrozenSplitEpisode, ...]:
    if not isinstance(payload, dict):
        raise ValueError("private_episodes.json must contain a JSON object")

    partition = payload.get("partition")
    if partition != _EXPECTED_PARTITION:
        raise ValueError(
            f"partition must equal {_EXPECTED_PARTITION!r}, got {partition!r}"
        )

    manifest_version = payload.get("manifest_version")
    if manifest_version != MANIFEST_VERSION:
        raise ValueError(
            f"manifest_version must equal {MANIFEST_VERSION!r}, got {manifest_version!r}"
        )

    seed_bank_version = payload.get("seed_bank_version")
    if not isinstance(seed_bank_version, str) or not seed_bank_version:
        raise ValueError("seed_bank_version must be a non-empty string")

    episodes_raw = payload.get("episodes")
    if not isinstance(episodes_raw, list) or not episodes_raw:
        raise ValueError("episodes must be a non-empty list")

    return tuple(
        _parse_episode_row(row, manifest_version, seed_bank_version)
        for row in episodes_raw
    )


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
    return Episode(
        episode_id=ep["episode_id"],
        split=ep["split"],
        difficulty=ep["difficulty"],
        template_id=ep["template_id"],
        rule_A=ep["rule_A"],
        rule_B=ep["rule_B"],
        transition=ep["transition"],
        pre_count=ep["pre_count"],
        post_labeled_count=ep["post_labeled_count"],
        shift_after_position=ep["shift_after_position"],
        contradiction_count_post=ep["contradiction_count_post"],
        items=items,
        probe_targets=probe_targets,
        probe_label_counts=probe_label_counts,
        probe_sign_pattern_counts=probe_sign_pattern_counts,
        probe_metadata=probe_metadata,
        difficulty_version=ep.get("difficulty_version", "R12"),
        spec_version=ep.get("spec_version", "v1"),
        generator_version=ep.get("generator_version", "R12"),
        template_set_version=ep.get("template_set_version", "v1"),
    )
