from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, TYPE_CHECKING

from core.parser import NarrativeParsedResult, NarrativeParseStatus, ParseStatus, ParsedPrediction
from tasks.ruleshift_benchmark.protocol import (
    PROBE_COUNT,
    InteractionLabel,
    parse_label,
)

if TYPE_CHECKING:
    from core.invariance import InvarianceReport
    from core.slices import SliceReport

__all__ = [
    "METRIC_VERSION",
    "MetricSummary",
    "compute_post_shift_probe_accuracy",
    "compute_metrics",
]

METRIC_VERSION = "v1"


@dataclass(frozen=True, slots=True)
class MetricSummary:
    # Leaderboard headline — Binary only.
    post_shift_probe_accuracy: float
    # Binary parse success rate (provider failures excluded from denominator).
    binary_parse_valid_rate: float
    # Narrative schema validation rate (provider failures excluded).
    narrative_schema_valid_rate: float
    # Number of narrative parse/validation failures (excludes provider failures).
    narrative_parse_failure_count: int
    # Mandatory evaluation slices across all required dimensions.
    # None only when episodes were not supplied to compute_metrics; populated
    # by run_model_benchmark which has access to the Episode objects.
    slice_report: "SliceReport | None" = None
    # Optional invariance report: per-perturbation-class accuracy on perturbed
    # prompts.  None when run_model_benchmark was called with run_invariance=False
    # (the default).  Diagnostic only — does not affect the leaderboard metric.
    invariance_report: "InvarianceReport | None" = None


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
    narrative_results: Iterable[NarrativeParsedResult],
) -> MetricSummary:
    normalized_binary_predictions = tuple(binary_predictions)
    normalized_binary_targets = _normalize_targets(binary_targets)
    _validate_prediction_target_lengths(
        normalized_binary_predictions,
        normalized_binary_targets,
    )

    normalized_narrative_results = tuple(narrative_results)

    # Binary scoring — the only leaderboard metric.
    binary_accuracy = compute_post_shift_probe_accuracy(
        normalized_binary_predictions,
        normalized_binary_targets,
    )

    # Binary parse valid rate (provider failures excluded).
    binary_attempted = [
        p for p in normalized_binary_predictions
        if p.status is not ParseStatus.SKIPPED_PROVIDER_FAILURE
    ]
    binary_valid = sum(p.status is ParseStatus.VALID for p in binary_attempted)
    binary_parse_valid_rate = binary_valid / len(binary_attempted) if binary_attempted else 0.0

    # Narrative schema validity (provider failures excluded from denominator).
    narrative_attempted = [
        r for r in normalized_narrative_results
        if r.status is not NarrativeParseStatus.SKIPPED_PROVIDER_FAILURE
    ]
    narrative_valid = sum(r.status is NarrativeParseStatus.VALID for r in narrative_attempted)
    narrative_schema_valid_rate = (
        narrative_valid / len(narrative_attempted) if narrative_attempted else 0.0
    )
    narrative_parse_failure_count = len(narrative_attempted) - narrative_valid

    return MetricSummary(
        post_shift_probe_accuracy=binary_accuracy,
        binary_parse_valid_rate=binary_parse_valid_rate,
        narrative_schema_valid_rate=narrative_schema_valid_rate,
        narrative_parse_failure_count=narrative_parse_failure_count,
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
