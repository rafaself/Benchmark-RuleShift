from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from core.parser import ParseStatus, ParsedPrediction
from tasks.iron_find_electric.protocol import (
    PROBE_COUNT,
    InteractionLabel,
    parse_label,
)

__all__ = [
    "MetricSummary",
    "compute_post_shift_probe_accuracy",
    "compute_metrics",
]


@dataclass(frozen=True, slots=True)
class MetricSummary:
    post_shift_probe_accuracy: float
    parse_valid_rate: float
    binary_accuracy: float
    narrative_accuracy: float


def compute_post_shift_probe_accuracy(
    predictions: Iterable[ParsedPrediction],
    targets: Iterable[tuple[InteractionLabel, ...]],
) -> float:
    normalized_predictions = tuple(predictions)
    normalized_targets = _normalize_targets(targets)
    _validate_prediction_target_lengths(normalized_predictions, normalized_targets)

    total_probes = len(normalized_targets) * PROBE_COUNT
    if total_probes == 0:
        return 0.0

    correct_predictions = sum(
        _count_correct_probes(prediction, target)
        for prediction, target in zip(normalized_predictions, normalized_targets)
    )
    return correct_predictions / total_probes


def compute_metrics(
    binary_predictions: Iterable[ParsedPrediction],
    binary_targets: Iterable[tuple[InteractionLabel, ...]],
    narrative_predictions: Iterable[ParsedPrediction],
    narrative_targets: Iterable[tuple[InteractionLabel, ...]],
) -> MetricSummary:
    normalized_binary_predictions = tuple(binary_predictions)
    normalized_binary_targets = _normalize_targets(binary_targets)
    _validate_prediction_target_lengths(
        normalized_binary_predictions,
        normalized_binary_targets,
    )

    normalized_narrative_predictions = tuple(narrative_predictions)
    normalized_narrative_targets = _normalize_targets(narrative_targets)
    _validate_prediction_target_lengths(
        normalized_narrative_predictions,
        normalized_narrative_targets,
    )

    all_predictions = normalized_binary_predictions + normalized_narrative_predictions
    total_predictions = len(all_predictions)
    valid_predictions = sum(
        prediction.status is ParseStatus.VALID for prediction in all_predictions
    )

    return MetricSummary(
        post_shift_probe_accuracy=compute_post_shift_probe_accuracy(
            all_predictions,
            normalized_binary_targets + normalized_narrative_targets,
        ),
        parse_valid_rate=(
            valid_predictions / total_predictions if total_predictions else 0.0
        ),
        binary_accuracy=compute_post_shift_probe_accuracy(
            normalized_binary_predictions,
            normalized_binary_targets,
        ),
        narrative_accuracy=compute_post_shift_probe_accuracy(
            normalized_narrative_predictions,
            normalized_narrative_targets,
        ),
    )


def _count_correct_probes(
    prediction: ParsedPrediction,
    target: tuple[InteractionLabel, ...],
) -> int:
    if prediction.status is not ParseStatus.VALID:
        return 0
    if len(prediction.labels) != PROBE_COUNT:
        return 0

    return sum(
        predicted_label is target_label
        for predicted_label, target_label in zip(prediction.labels, target)
    )


def _normalize_targets(
    targets: Iterable[tuple[InteractionLabel, ...]],
) -> tuple[tuple[InteractionLabel, ...], ...]:
    return tuple(_normalize_target(target) for target in targets)


def _normalize_target(
    target: tuple[InteractionLabel, ...],
) -> tuple[InteractionLabel, ...]:
    normalized_target = tuple(parse_label(label) for label in target)
    if len(normalized_target) != PROBE_COUNT:
        raise ValueError(f"targets must contain exactly {PROBE_COUNT} labels per row")
    return normalized_target


def _validate_prediction_target_lengths(
    predictions: tuple[ParsedPrediction, ...],
    targets: tuple[tuple[InteractionLabel, ...], ...],
) -> None:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must contain the same number of rows")
