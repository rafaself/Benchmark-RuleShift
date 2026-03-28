from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from tasks.ruleshift_benchmark.protocol import (
    LABELED_ITEM_COUNT,
    InteractionLabel,
    TEMPLATES,
    TemplateFamily,
    TemplateId,
    Transition,
)
from tasks.ruleshift_benchmark.schema import Episode

from core.validate.episode import (
    ValidationIssue,
    EpisodeValidationResult,
    normalize_episode_payload,
    validate_episode,
)

__all__ = [
    "DatasetDistributionSummary",
    "DatasetValidationResult",
    "validate_dataset",
]

_TEMPLATE_ORDER = tuple(template_id.value for template_id in TemplateId)
_TEMPLATE_FAMILY_ORDER = tuple(
    template_family.value for template_family in TemplateFamily
)
_TRANSITION_ORDER = (
    Transition.R_STD_TO_R_INV.value,
    Transition.R_INV_TO_R_STD.value,
)
_SHIFT_POSITION_ORDER = tuple(
    str(TEMPLATES[template_id].shift_after_position) for template_id in TemplateId
)
_PROBE_LABEL_ORDER = (
    InteractionLabel.ATTRACT.value,
    InteractionLabel.REPEL.value,
)
_PROBE_SIGN_PATTERN_ORDER = ("++", "--", "+-", "-+")
_VERSION_FIELD_ORDER = (
    "spec_version",
    "generator_version",
    "template_set_version",
    "difficulty_version",
)


@dataclass(frozen=True, slots=True)
class DatasetDistributionSummary:
    template_counts: tuple[tuple[str, int], ...]
    template_family_counts: tuple[tuple[str, int], ...]
    transition_counts: tuple[tuple[str, int], ...]
    shift_position_counts: tuple[tuple[str, int], ...]
    probe_label_counts: tuple[tuple[str, int], ...]
    sign_pattern_counts: tuple[tuple[str, int], ...]
    version_values: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True, slots=True)
class DatasetValidationResult:
    ok: bool
    episode_results: tuple[EpisodeValidationResult, ...]
    issues: tuple[ValidationIssue, ...]
    summary: DatasetDistributionSummary


def validate_dataset(episodes: Iterable[Episode]) -> DatasetValidationResult:
    normalized_episodes = tuple(episodes)
    episode_results = tuple(validate_episode(episode) for episode in normalized_episodes)
    issues: list[ValidationIssue] = []

    invalid_episode_ids = tuple(
        result.episode_id for result in episode_results if not result.ok
    )
    if invalid_episode_ids:
        issues.append(
            ValidationIssue(
                code="invalid_episode",
                message="invalid episodes: " + ", ".join(invalid_episode_ids),
            )
        )

    duplicate_ids = tuple(
        episode_id
        for episode_id, count in _count_values(
            str(getattr(episode, "episode_id", "")) for episode in normalized_episodes
        )
        if count > 1
    )
    if duplicate_ids:
        issues.append(
            ValidationIssue(
                code="duplicate_episode_id",
                message="duplicate episode_id values: " + ", ".join(duplicate_ids),
            )
        )

    payload_groups = _group_payload_episode_ids(normalized_episodes)
    duplicate_payload_groups = tuple(
        episode_ids for episode_ids in payload_groups if len(episode_ids) > 1
    )
    if duplicate_payload_groups:
        issues.append(
            ValidationIssue(
                code="duplicate_episode_payload",
                message="duplicate episode payloads: "
                + "; ".join(", ".join(group) for group in duplicate_payload_groups),
            )
        )

    summary = DatasetDistributionSummary(
        template_counts=_count_in_canonical_order(
            (_safe_value(getattr(episode, "template_id", None)) for episode in normalized_episodes),
            _TEMPLATE_ORDER,
        ),
        template_family_counts=_count_in_canonical_order(
            (
                _safe_value(getattr(episode, "template_family", None))
                for episode in normalized_episodes
            ),
            _TEMPLATE_FAMILY_ORDER,
        ),
        transition_counts=_count_in_canonical_order(
            (_safe_value(getattr(episode, "transition", None)) for episode in normalized_episodes),
            _TRANSITION_ORDER,
        ),
        shift_position_counts=_count_in_canonical_order(
            (
                str(getattr(episode, "shift_after_position", None))
                for episode in normalized_episodes
            ),
            _SHIFT_POSITION_ORDER,
        ),
        probe_label_counts=_count_in_canonical_order(
            (
                _safe_value(label_value)
                for episode in normalized_episodes
                for label_value in getattr(episode, "probe_targets", ())
            ),
            _PROBE_LABEL_ORDER,
        ),
        sign_pattern_counts=_count_probe_sign_patterns(normalized_episodes),
        version_values=tuple(
            (
                field_name,
                tuple(
                    sorted(
                        {
                            str(getattr(episode, field_name, None))
                            for episode in normalized_episodes
                        }
                    )
                ),
            )
            for field_name in _VERSION_FIELD_ORDER
        ),
    )

    _append_balance_issue(
        issues,
        code="template_balance",
        field_name="template usage",
        counts=summary.template_counts,
        threshold=max(1, math.ceil(0.2 * len(normalized_episodes))),
    )
    _append_balance_issue(
        issues,
        code="template_family_balance",
        field_name="template family usage",
        counts=summary.template_family_counts,
        threshold=max(1, math.ceil(0.2 * len(normalized_episodes))),
    )
    _append_balance_issue(
        issues,
        code="transition_balance",
        field_name="transition usage",
        counts=summary.transition_counts,
        threshold=max(1, math.ceil(0.2 * len(normalized_episodes))),
    )
    _append_balance_issue(
        issues,
        code="shift_position_balance",
        field_name="shift-position usage",
        counts=summary.shift_position_counts,
        threshold=max(1, math.ceil(0.2 * len(normalized_episodes))),
    )
    if summary.probe_label_counts:
        probe_label_gap = abs(
            summary.probe_label_counts[0][1] - summary.probe_label_counts[1][1]
        )
        probe_label_threshold = max(
            2,
            math.ceil(
                0.2 * sum(count for _, count in summary.probe_label_counts)
            ),
        )
        if probe_label_gap > probe_label_threshold:
            issues.append(
                ValidationIssue(
                    code="probe_label_balance",
                    message=(
                        f"probe label counts differ by {probe_label_gap}, "
                        f"which exceeds the allowed threshold of {probe_label_threshold}"
                    ),
                )
            )
    missing_sign_patterns = tuple(
        pattern for pattern, count in summary.sign_pattern_counts if count == 0
    )
    if missing_sign_patterns:
        issues.append(
            ValidationIssue(
                code="sign_pattern_coverage",
                message="missing probe sign-pattern coverage for: "
                + ", ".join(missing_sign_patterns),
            )
        )

    inconsistent_version_fields = tuple(
        field_name for field_name, values in summary.version_values if len(values) > 1
    )
    if inconsistent_version_fields:
        issues.append(
            ValidationIssue(
                code="version_consistency",
                message="version fields vary across dataset: "
                + ", ".join(inconsistent_version_fields),
            )
        )

    return DatasetValidationResult(
        ok=not issues,
        episode_results=episode_results,
        issues=tuple(issues),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _group_payload_episode_ids(
    episodes: tuple[Episode, ...],
) -> tuple[tuple[str, ...], ...]:
    import json

    groups: dict[str, list[str]] = {}
    for episode in episodes:
        normalized_payload = normalize_episode_payload(episode)
        normalized_payload = {
            key: value for key, value in normalized_payload.items() if key != "episode_id"
        }
        payload_key = json.dumps(
            normalized_payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        groups.setdefault(payload_key, []).append(str(getattr(episode, "episode_id", "")))
    return tuple(
        tuple(episode_ids)
        for _, episode_ids in sorted(groups.items(), key=lambda item: item[0])
    )


def _count_values(values: Iterable[str]) -> tuple[tuple[str, int], ...]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return tuple(sorted(counts.items()))


def _count_in_canonical_order(
    values: Iterable[str],
    order: tuple[str, ...],
) -> tuple[tuple[str, int], ...]:
    counts = {key: 0 for key in order}
    for value in values:
        if value in counts:
            counts[value] += 1
    return tuple((key, counts[key]) for key in order)


def _count_probe_sign_patterns(
    episodes: Iterable[Episode],
) -> tuple[tuple[str, int], ...]:
    counts = {pattern: 0 for pattern in _PROBE_SIGN_PATTERN_ORDER}
    for episode in episodes:
        for item in tuple(getattr(episode, "items", ()))[LABELED_ITEM_COUNT:]:
            pattern = _probe_sign_pattern(getattr(item, "q1", 0), getattr(item, "q2", 0))
            counts[pattern] += 1
    return tuple((pattern, counts[pattern]) for pattern in _PROBE_SIGN_PATTERN_ORDER)


def _append_balance_issue(
    issues: list[ValidationIssue],
    *,
    code: str,
    field_name: str,
    counts: tuple[tuple[str, int], ...],
    threshold: int,
) -> None:
    if not counts:
        return
    count_values = tuple(count for _, count in counts)
    if max(count_values) - min(count_values) > threshold:
        issues.append(
            ValidationIssue(
                code=code,
                message=(
                    f"{field_name} differs by more than the allowed threshold of "
                    f"{threshold}: {counts}"
                ),
            )
        )


def _probe_sign_pattern(q1: int, q2: int) -> str:
    if q1 > 0 and q2 > 0:
        return "++"
    if q1 < 0 and q2 < 0:
        return "--"
    if q1 > 0 and q2 < 0:
        return "+-"
    return "-+"


def _safe_value(value: object) -> str:
    if hasattr(value, "value"):
        return str(getattr(value, "value"))
    return str(value)
