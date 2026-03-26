from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Sequence

from core.parser import NarrativeParsedResult, NarrativeParseStatus, ParseStatus, ParsedPrediction
from tasks.ruleshift_benchmark.protocol import PROBE_COUNT, InteractionLabel
from tasks.ruleshift_benchmark.schema import Episode

__all__ = [
    "SLICE_DIMENSIONS",
    "ErrorType",
    "SliceAccuracy",
    "EpisodeSliceData",
    "SliceReport",
    "classify_binary_error_type",
    "compute_episode_slice_data",
    "build_slice_report",
]

# Canonical order of required slice dimensions.
SLICE_DIMENSIONS: tuple[str, ...] = (
    "template",
    "template_family",
    "difficulty",
    "shift_position",
    "transition_type",
    "error_type",
)

_DIFFICULTY_ORDER: tuple[str, ...] = ("easy", "medium", "hard")
_TRANSITION_ORDER: tuple[str, ...] = ("R_std_to_R_inv", "R_inv_to_R_std")
_TEMPLATE_ORDER: tuple[str, ...] = ("T1", "T2")
_TEMPLATE_FAMILY_ORDER: tuple[str, ...] = ("canonical", "observation_log")


class ErrorType(StrEnum):
    """Canonical error taxonomy for binary episode classification."""

    OLD_RULE_PERSISTENCE = "old_rule_persistence"
    PREMATURE_SWITCH = "premature_switch"
    RECENCY_OVERWEIGHT = "recency_overweight"
    INVALID_NARRATIVE = "invalid_narrative"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SliceAccuracy:
    """Aggregate accuracy for a single slice value (e.g. template T1)."""

    episode_count: int
    correct_probes: int
    total_probes: int

    @property
    def accuracy(self) -> float:
        if self.total_probes == 0:
            return 0.0
        return self.correct_probes / self.total_probes

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_count": self.episode_count,
            "correct_probes": self.correct_probes,
            "total_probes": self.total_probes,
            "accuracy": self.accuracy,
        }


@dataclass(frozen=True, slots=True)
class EpisodeSliceData:
    """Per-episode metadata and error classification used to build a SliceReport."""

    episode_id: str
    template: str           # "T1" or "T2"
    template_family: str    # "canonical" or "observation_log"
    difficulty: str         # "easy" or "medium"
    shift_position: str     # str(shift_after_position): "2" or "3"
    transition_type: str    # "R_std_to_R_inv" or "R_inv_to_R_std"
    error_type: ErrorType
    correct_probes: int     # 0–4
    total_probes: int       # always PROBE_COUNT (4)


@dataclass(frozen=True, slots=True)
class SliceReport:
    """Machine-readable accuracy slices across all required benchmark dimensions.

    All five slice dimensions are always present and use stable string keys,
    so the structure is safe to compare across runs and store as JSON.
    """

    template: tuple[tuple[str, SliceAccuracy], ...]
    template_family: tuple[tuple[str, SliceAccuracy], ...]
    difficulty: tuple[tuple[str, SliceAccuracy], ...]
    shift_position: tuple[tuple[str, SliceAccuracy], ...]
    transition_type: tuple[tuple[str, SliceAccuracy], ...]
    # error_type maps each ErrorType value to the number of failing episodes
    # that were classified under that category.
    error_type: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "template": {k: v.to_dict() for k, v in self.template},
            "template_family": {k: v.to_dict() for k, v in self.template_family},
            "difficulty": {k: v.to_dict() for k, v in self.difficulty},
            "shift_position": {k: v.to_dict() for k, v in self.shift_position},
            "transition_type": {k: v.to_dict() for k, v in self.transition_type},
            "error_type": dict(self.error_type),
        }


def classify_binary_error_type(
    *,
    prediction: ParsedPrediction,
    targets: tuple[InteractionLabel, ...],
    probe_metadata: tuple,  # tuple of ProbeMetadata (duck-typed to avoid circular import)
    narrative_result: NarrativeParsedResult | None = None,
) -> ErrorType:
    """Classify the dominant error type for a single binary episode result.

    Classification hierarchy (first match wins):

    1. ``invalid_narrative`` — narrative result failed schema/format validation.
    2. ``old_rule_persistence`` — every error probe predicted the pre-shift rule label.
    3. ``recency_overweight`` — every error probe predicted the post-shift rule label
       (unexpected given targets already match new_rule_label; arises on episodes
       where probe_targets and new_rule_labels diverge).
    4. ``premature_switch`` — mixed error pattern: some probes match old rule and/or
       some match new rule, but neither dominates exclusively.
    5. ``unknown`` — unclassifiable: provider failure, binary parse failure, all
       correct, or no probe-metadata match.
    """
    # Narrative schema failure takes priority.
    if (
        narrative_result is not None
        and narrative_result.status is not NarrativeParseStatus.VALID
        and narrative_result.status is not NarrativeParseStatus.SKIPPED_PROVIDER_FAILURE
    ):
        return ErrorType.INVALID_NARRATIVE

    # Provider failure or binary parse error → unclassifiable.
    if prediction.status is not ParseStatus.VALID:
        return ErrorType.UNKNOWN

    if len(prediction.labels) != PROBE_COUNT:
        return ErrorType.UNKNOWN

    correct_count = sum(
        predicted is target
        for predicted, target in zip(prediction.labels, targets)
    )
    if correct_count == PROBE_COUNT:
        # All correct — not an error episode.
        return ErrorType.UNKNOWN

    # Classify by comparing error probes to old/new rule labels.
    error_count = 0
    old_rule_error_count = 0
    recency_error_count = 0
    for predicted, target, metadata in zip(prediction.labels, targets, probe_metadata):
        if predicted is target:
            continue
        error_count += 1
        if predicted is metadata.old_rule_label:
            old_rule_error_count += 1
        elif predicted is metadata.new_rule_label:
            recency_error_count += 1

    if error_count == 0:
        return ErrorType.UNKNOWN

    if old_rule_error_count == error_count:
        return ErrorType.OLD_RULE_PERSISTENCE

    if recency_error_count == error_count:
        return ErrorType.RECENCY_OVERWEIGHT

    # Mixed pattern: some probes match old rule, some match new, or neither.
    if old_rule_error_count > 0 or recency_error_count > 0:
        return ErrorType.PREMATURE_SWITCH

    return ErrorType.UNKNOWN


def compute_episode_slice_data(
    *,
    episode: Episode,
    prediction: ParsedPrediction,
    narrative_result: NarrativeParsedResult | None = None,
) -> EpisodeSliceData:
    """Compute slice metadata and error classification for one binary episode result."""
    if prediction.status is ParseStatus.VALID and len(prediction.labels) == PROBE_COUNT:
        correct_probes = sum(
            predicted is target
            for predicted, target in zip(prediction.labels, episode.probe_targets)
        )
    else:
        correct_probes = 0

    error_type = classify_binary_error_type(
        prediction=prediction,
        targets=episode.probe_targets,
        probe_metadata=episode.probe_metadata,
        narrative_result=narrative_result,
    )

    return EpisodeSliceData(
        episode_id=episode.episode_id,
        template=episode.template_id.value,
        template_family=episode.template_family.value,
        difficulty=episode.difficulty.value,
        shift_position=str(episode.shift_after_position),
        transition_type=episode.transition.value,
        error_type=error_type,
        correct_probes=correct_probes,
        total_probes=PROBE_COUNT,
    )


def build_slice_report(
    episode_slices: Sequence[EpisodeSliceData],
) -> SliceReport:
    """Aggregate per-episode slice data into a SliceReport."""
    return SliceReport(
        template=_aggregate_accuracy(
            episode_slices, lambda s: s.template, _TEMPLATE_ORDER
        ),
        template_family=_aggregate_accuracy(
            episode_slices, lambda s: s.template_family, _TEMPLATE_FAMILY_ORDER
        ),
        difficulty=_aggregate_accuracy(
            episode_slices, lambda s: s.difficulty, _DIFFICULTY_ORDER
        ),
        shift_position=_aggregate_accuracy(
            episode_slices, lambda s: s.shift_position
        ),
        transition_type=_aggregate_accuracy(
            episode_slices, lambda s: s.transition_type, _TRANSITION_ORDER
        ),
        error_type=_aggregate_error_type(episode_slices),
    )


def _aggregate_accuracy(
    slices: Sequence[EpisodeSliceData],
    key_fn: Callable[[EpisodeSliceData], str],
    order: tuple[str, ...] | None = None,
) -> tuple[tuple[str, SliceAccuracy], ...]:
    groups: dict[str, list[EpisodeSliceData]] = {}
    for s in slices:
        key = key_fn(s)
        groups.setdefault(key, []).append(s)

    if order is not None:
        sorted_keys = [k for k in order if k in groups]
        sorted_keys += sorted(k for k in groups if k not in order)
    else:
        sorted_keys = sorted(groups)

    return tuple(
        (
            key,
            SliceAccuracy(
                episode_count=len(groups[key]),
                correct_probes=sum(s.correct_probes for s in groups[key]),
                total_probes=sum(s.total_probes for s in groups[key]),
            ),
        )
        for key in sorted_keys
    )


def _aggregate_error_type(
    slices: Sequence[EpisodeSliceData],
) -> tuple[tuple[str, int], ...]:
    """Count failing episodes per ErrorType, in canonical enum order."""
    counts: dict[str, int] = {et.value: 0 for et in ErrorType}
    for s in slices:
        if s.correct_probes < s.total_probes:
            counts[s.error_type.value] += 1
    return tuple((et.value, counts[et.value]) for et in ErrorType)
