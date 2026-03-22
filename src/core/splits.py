from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Final

from core.validate import (
    DatasetDistributionSummary,
    ValidationIssue,
    normalize_episode_payload,
    validate_dataset,
)
from tasks.iron_find_electric.generator import generate_episode
from tasks.iron_find_electric.protocol import Split, parse_split
from tasks.iron_find_electric.schema import (
    DIFFICULTY_VERSION,
    GENERATOR_VERSION,
    SPEC_VERSION,
    TEMPLATE_SET_VERSION,
    Episode,
)

__all__ = [
    "PARTITIONS",
    "FrozenSplitManifest",
    "FrozenSplitEpisode",
    "SplitDistributionSummary",
    "DistributionGap",
    "CrossSplitComparisonSummary",
    "FrozenSplitAudit",
    "load_split_manifest",
    "generate_frozen_split",
    "load_frozen_split",
    "load_all_frozen_splits",
    "audit_frozen_splits",
    "assert_no_partition_overlap",
]

PARTITIONS: Final[tuple[str, ...]] = (
    "dev",
    "public_leaderboard",
    "private_leaderboard",
)
MANIFEST_VERSION: Final[str] = "R8"
_MANIFEST_FIELD_ORDER: Final[tuple[str, ...]] = (
    "partition",
    "episode_split",
    "manifest_version",
    "seed_bank_version",
    "spec_version",
    "generator_version",
    "template_set_version",
    "difficulty_version",
    "seeds",
)
_MANIFEST_DIR: Final[Path] = Path(__file__).resolve().parents[1] / "frozen_splits"
_PARTITION_TO_EPISODE_SPLIT: Final[dict[str, Split]] = {
    "dev": Split.DEV,
    "public_leaderboard": Split.PUBLIC,
    "private_leaderboard": Split.PRIVATE,
}
_WITHIN_SPLIT_DIFFICULTY_GAP_THRESHOLD: Final[float] = 0.25
_CROSS_SPLIT_GAP_THRESHOLDS: Final[dict[str, float]] = {
    "template": 0.25,
    "transition": 0.25,
    "difficulty": 0.25,
    "probe_label": 0.125,
}


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value)


@dataclass(frozen=True, slots=True)
class FrozenSplitManifest:
    partition: str
    episode_split: Split
    manifest_version: str
    seed_bank_version: str
    spec_version: str
    generator_version: str
    template_set_version: str
    difficulty_version: str
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.partition not in PARTITIONS:
            raise ValueError(f"unknown partition: {self.partition}")

        object.__setattr__(self, "episode_split", parse_split(self.episode_split))
        expected_split = _PARTITION_TO_EPISODE_SPLIT[self.partition]
        if self.episode_split is not expected_split:
            raise ValueError(
                "episode_split does not match the canonical partition mapping"
            )

        if self.manifest_version != MANIFEST_VERSION:
            raise ValueError(f"manifest_version must equal {MANIFEST_VERSION}")
        if not _is_nonempty_string(self.seed_bank_version):
            raise ValueError("seed_bank_version must be a non-empty string")
        if self.spec_version != SPEC_VERSION:
            raise ValueError(f"spec_version must equal {SPEC_VERSION}")
        if self.generator_version != GENERATOR_VERSION:
            raise ValueError(f"generator_version must equal {GENERATOR_VERSION}")
        if self.template_set_version != TEMPLATE_SET_VERSION:
            raise ValueError(
                f"template_set_version must equal {TEMPLATE_SET_VERSION}"
            )
        if self.difficulty_version != DIFFICULTY_VERSION:
            raise ValueError(
                f"difficulty_version must equal {DIFFICULTY_VERSION}"
            )

        normalized_seeds = tuple(self.seeds)
        if not normalized_seeds:
            raise ValueError("seeds must not be empty")
        if any(not _is_plain_int(seed) for seed in normalized_seeds):
            raise TypeError("seeds must contain only int values")
        if len(set(normalized_seeds)) != len(normalized_seeds):
            raise ValueError("seeds must contain unique values")
        object.__setattr__(self, "seeds", normalized_seeds)


@dataclass(frozen=True, slots=True)
class FrozenSplitEpisode:
    partition: str
    seed: int
    manifest_version: str
    seed_bank_version: str
    episode: Episode

    def __post_init__(self) -> None:
        if self.partition not in PARTITIONS:
            raise ValueError(f"unknown partition: {self.partition}")
        if not _is_plain_int(self.seed):
            raise TypeError("seed must be an int")
        if self.manifest_version != MANIFEST_VERSION:
            raise ValueError(f"manifest_version must equal {MANIFEST_VERSION}")
        if not _is_nonempty_string(self.seed_bank_version):
            raise ValueError("seed_bank_version must be a non-empty string")
        if not isinstance(self.episode, Episode):
            raise TypeError("episode must be an Episode")
        expected_split = _PARTITION_TO_EPISODE_SPLIT[self.partition]
        if self.episode.split is not expected_split:
            raise ValueError("episode split does not match partition mapping")


@dataclass(frozen=True, slots=True)
class SplitDistributionSummary:
    dataset_summary: DatasetDistributionSummary
    difficulty_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class DistributionGap:
    label: str
    min_share: float
    max_share: float
    gap: float


@dataclass(frozen=True, slots=True)
class CrossSplitComparisonSummary:
    template: tuple[DistributionGap, ...]
    transition: tuple[DistributionGap, ...]
    probe_label: tuple[DistributionGap, ...]
    difficulty: tuple[DistributionGap, ...]


@dataclass(frozen=True, slots=True)
class FrozenSplitAudit:
    per_partition: tuple[tuple[str, SplitDistributionSummary], ...]
    cross_partition: CrossSplitComparisonSummary
    issues: tuple[ValidationIssue, ...]


def load_split_manifest(partition: str) -> FrozenSplitManifest:
    if partition not in PARTITIONS:
        raise ValueError(f"unknown partition: {partition}")

    manifest_path = _MANIFEST_DIR / f"{partition}.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_fields = tuple(payload)
    if actual_fields != _MANIFEST_FIELD_ORDER:
        raise ValueError(
            "manifest fields must exactly match the canonical order: "
            + ", ".join(_MANIFEST_FIELD_ORDER)
        )

    return FrozenSplitManifest(
        partition=payload["partition"],
        episode_split=payload["episode_split"],
        manifest_version=payload["manifest_version"],
        seed_bank_version=payload["seed_bank_version"],
        spec_version=payload["spec_version"],
        generator_version=payload["generator_version"],
        template_set_version=payload["template_set_version"],
        difficulty_version=payload["difficulty_version"],
        seeds=tuple(payload["seeds"]),
    )


def generate_frozen_split(
    manifest: FrozenSplitManifest,
) -> tuple[FrozenSplitEpisode, ...]:
    return tuple(
        FrozenSplitEpisode(
            partition=manifest.partition,
            seed=seed,
            manifest_version=manifest.manifest_version,
            seed_bank_version=manifest.seed_bank_version,
            episode=generate_episode(seed, split=manifest.episode_split),
        )
        for seed in manifest.seeds
    )


def load_frozen_split(partition: str) -> tuple[FrozenSplitEpisode, ...]:
    return generate_frozen_split(load_split_manifest(partition))


def load_all_frozen_splits() -> dict[str, tuple[FrozenSplitEpisode, ...]]:
    return {
        partition: load_frozen_split(partition)
        for partition in PARTITIONS
    }


def audit_frozen_splits(
    splits: dict[str, tuple[FrozenSplitEpisode, ...]] | None = None,
) -> FrozenSplitAudit:
    normalized_splits = load_all_frozen_splits() if splits is None else dict(splits)
    issues: list[ValidationIssue] = []
    per_partition: list[tuple[str, SplitDistributionSummary]] = []

    for partition in PARTITIONS:
        episodes = tuple(
            record.episode for record in normalized_splits.get(partition, ())
        )
        validation_result = validate_dataset(episodes)
        difficulty_counts = _count_difficulty_counts(episodes)
        per_partition.append(
            (
                partition,
                SplitDistributionSummary(
                    dataset_summary=validation_result.summary,
                    difficulty_counts=difficulty_counts,
                ),
            )
        )
        for issue in validation_result.issues:
            issues.append(
                ValidationIssue(
                    code=issue.code,
                    message=f"{partition}: {issue.message}",
                )
            )

        difficulty_gap = _share_gap_from_counts(difficulty_counts)
        if difficulty_gap > _WITHIN_SPLIT_DIFFICULTY_GAP_THRESHOLD:
            issues.append(
                ValidationIssue(
                    code="difficulty_balance",
                    message=(
                        f"{partition}: difficulty share gap {difficulty_gap:.6f} "
                        f"exceeds {_WITHIN_SPLIT_DIFFICULTY_GAP_THRESHOLD:.6f}: "
                        f"{difficulty_counts}"
                    ),
                )
            )

    overlap_issues = _collect_overlap_issues(normalized_splits)
    issues.extend(overlap_issues)

    cross_partition = _build_cross_partition_summary(per_partition)
    issues.extend(_collect_cross_partition_gap_issues(cross_partition))

    return FrozenSplitAudit(
        per_partition=tuple(per_partition),
        cross_partition=cross_partition,
        issues=tuple(issues),
    )


def assert_no_partition_overlap(
    splits: dict[str, tuple[FrozenSplitEpisode, ...]] | None = None,
) -> None:
    normalized_splits = load_all_frozen_splits() if splits is None else dict(splits)
    issues = _collect_overlap_issues(normalized_splits)
    if not issues:
        return
    raise ValueError("; ".join(f"{issue.code}: {issue.message}" for issue in issues))


def _count_difficulty_counts(
    episodes: tuple[Episode, ...],
) -> tuple[tuple[str, int], ...]:
    counts: dict[str, int] = {}
    for episode in episodes:
        label = episode.difficulty.value
        counts[label] = counts.get(label, 0) + 1
    return tuple((label, counts[label]) for label in sorted(counts))


def _share_gap_from_counts(counts: tuple[tuple[str, int], ...]) -> float:
    total = sum(count for _, count in counts)
    if total == 0 or len(counts) <= 1:
        return 0.0
    shares = tuple(count / total for _, count in counts)
    return max(shares) - min(shares)


def _collect_overlap_issues(
    splits: dict[str, tuple[FrozenSplitEpisode, ...]],
) -> tuple[ValidationIssue, ...]:
    seed_groups: dict[int, list[str]] = {}
    episode_id_groups: dict[str, list[str]] = {}
    payload_groups: dict[str, list[str]] = {}

    for partition in PARTITIONS:
        for record in splits.get(partition, ()):
            seed_groups.setdefault(record.seed, []).append(partition)
            episode_id_groups.setdefault(record.episode.episode_id, []).append(partition)
            payload_groups.setdefault(
                _payload_fingerprint(record.episode),
                [],
            ).append(partition)

    issues: list[ValidationIssue] = []
    issues.extend(
        _build_overlap_issues(
            code="overlap_seed",
            groups=seed_groups,
            formatter=lambda key: f"seed {key}",
        )
    )
    issues.extend(
        _build_overlap_issues(
            code="overlap_episode_id",
            groups=episode_id_groups,
            formatter=lambda key: f"episode_id {key}",
        )
    )
    issues.extend(
        _build_overlap_issues(
            code="overlap_payload",
            groups=payload_groups,
            formatter=lambda key: f"payload {key}",
        )
    )
    return tuple(issues)


def _build_overlap_issues(
    *,
    code: str,
    groups: dict[object, list[str]],
    formatter,
) -> tuple[ValidationIssue, ...]:
    duplicates = []
    for key, partitions in groups.items():
        if len(set(partitions)) > 1:
            duplicates.append(
                ValidationIssue(
                    code=code,
                    message=(
                        f"{formatter(key)} appears in partitions: "
                        + ", ".join(sorted(set(partitions)))
                    ),
                )
            )
    duplicates.sort(key=lambda issue: issue.message)
    return tuple(duplicates)


def _payload_fingerprint(episode: Episode) -> str:
    payload = normalize_episode_payload(episode)
    payload = {
        key: value
        for key, value in payload.items()
        if key not in {"episode_id", "split"}
    }
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _build_cross_partition_summary(
    per_partition: list[tuple[str, SplitDistributionSummary]],
) -> CrossSplitComparisonSummary:
    partition_summaries = dict(per_partition)
    difficulty_labels = tuple(
        sorted(
            {
                label
                for summary in partition_summaries.values()
                for label, _ in summary.difficulty_counts
            }
        )
    )
    return CrossSplitComparisonSummary(
        template=_distribution_gaps(
            per_partition,
            lambda summary: summary.dataset_summary.template_counts,
            tuple(label for label, _ in per_partition[0][1].dataset_summary.template_counts),
        ),
        transition=_distribution_gaps(
            per_partition,
            lambda summary: summary.dataset_summary.transition_counts,
            tuple(
                label
                for label, _ in per_partition[0][1].dataset_summary.transition_counts
            ),
        ),
        probe_label=_distribution_gaps(
            per_partition,
            lambda summary: summary.dataset_summary.probe_label_counts,
            tuple(
                label
                for label, _ in per_partition[0][1].dataset_summary.probe_label_counts
            ),
        ),
        difficulty=_distribution_gaps(
            per_partition,
            lambda summary: summary.difficulty_counts,
            difficulty_labels,
        ),
    )


def _distribution_gaps(
    per_partition: list[tuple[str, SplitDistributionSummary]],
    counts_getter,
    labels: tuple[str, ...],
) -> tuple[DistributionGap, ...]:
    gaps: list[DistributionGap] = []
    for label in labels:
        shares = []
        for _, summary in per_partition:
            counts = dict(counts_getter(summary))
            total = sum(counts.values())
            share = 0.0 if total == 0 else counts.get(label, 0) / total
            shares.append(share)
        gaps.append(
            DistributionGap(
                label=label,
                min_share=min(shares),
                max_share=max(shares),
                gap=max(shares) - min(shares),
            )
        )
    return tuple(gaps)


def _collect_cross_partition_gap_issues(
    summary: CrossSplitComparisonSummary,
) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    family_to_gaps = {
        "template": summary.template,
        "transition": summary.transition,
        "difficulty": summary.difficulty,
        "probe_label": summary.probe_label,
    }
    for family, gaps in family_to_gaps.items():
        threshold = _CROSS_SPLIT_GAP_THRESHOLDS[family]
        exceeded = tuple(gap for gap in gaps if gap.gap > threshold)
        if not exceeded:
            continue
        issues.append(
            ValidationIssue(
                code=f"cross_partition_{family}_balance",
                message=(
                    f"cross-partition {family} share gaps exceed {threshold:.6f}: "
                    + ", ".join(
                        f"{gap.label}={gap.gap:.6f}"
                        for gap in exceeded
                    )
                ),
            )
        )
    return tuple(issues)
