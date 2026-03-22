from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable, Literal

from core.parser import ParseStatus, ParsedPrediction
from tasks.iron_find_electric.baselines import BaselineRunResult
from tasks.iron_find_electric.protocol import PROBE_COUNT
from tasks.iron_find_electric.schema import Episode

__all__ = [
    "AuditSource",
    "AuditSliceSummary",
    "HeuristicAlignmentSummary",
    "AuditSourceSummary",
    "BaselineComparisonSummary",
    "ModeComparisonSummary",
    "AuditReport",
    "run_audit",
]

TaskMode = Literal["Binary", "Narrative"]

_TASK_MODE_ORDER: Final[tuple[TaskMode, ...]] = ("Binary", "Narrative")
_TEMPLATE_ORDER: Final[tuple[str, ...]] = ("T1", "T2")
_DIFFICULTY_ORDER: Final[tuple[str, ...]] = ("easy", "medium", "hard")
_FAILURE_PATTERN_REFERENCE_NAMES: Final[tuple[tuple[str, str], ...]] = (
    ("persistence-like", "never_update"),
    ("recency / last-evidence-like", "last_evidence"),
    ("physics-prior", "physics_prior"),
    ("template-position", "template_position"),
)


def _is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _normalize_task_mode(task_mode: TaskMode | str | None) -> TaskMode | None:
    if task_mode is None:
        return None
    if task_mode == "Binary":
        return "Binary"
    if task_mode == "Narrative":
        return "Narrative"
    if isinstance(task_mode, str):
        normalized = task_mode.strip().lower()
        if normalized == "binary":
            return "Binary"
        if normalized == "narrative":
            return "Narrative"
    raise ValueError("task_mode must be Binary, Narrative, or None")


@dataclass(frozen=True, slots=True)
class AuditSource:
    name: str
    predictions: tuple[ParsedPrediction, ...]
    task_mode: TaskMode | None = None
    is_baseline: bool = False

    def __post_init__(self) -> None:
        if not _is_nonempty_string(self.name):
            raise ValueError("name must be a non-empty string")
        normalized_predictions = tuple(self.predictions)
        if not all(isinstance(prediction, ParsedPrediction) for prediction in normalized_predictions):
            raise TypeError("predictions must contain ParsedPrediction values")
        object.__setattr__(self, "predictions", normalized_predictions)
        object.__setattr__(self, "task_mode", _normalize_task_mode(self.task_mode))
        if not isinstance(self.is_baseline, bool):
            raise TypeError("is_baseline must be a bool")

    @classmethod
    def from_parsed_predictions(
        cls,
        name: str,
        predictions: Iterable[ParsedPrediction],
        *,
        task_mode: TaskMode | str | None = None,
        is_baseline: bool = False,
    ) -> "AuditSource":
        return cls(
            name=name,
            predictions=tuple(predictions),
            task_mode=task_mode,
            is_baseline=is_baseline,
        )

    @classmethod
    def from_baseline_run(
        cls,
        baseline_run: BaselineRunResult,
        *,
        task_mode: TaskMode | str | None = None,
    ) -> "AuditSource":
        if not isinstance(baseline_run, BaselineRunResult):
            raise TypeError("baseline_run must be a BaselineRunResult")
        return cls(
            name=baseline_run.baseline_name,
            predictions=tuple(row.parsed_prediction for row in baseline_run.rows),
            task_mode=task_mode,
            is_baseline=True,
        )


@dataclass(frozen=True, slots=True)
class AuditSliceSummary:
    episode_count: int
    correct_probe_count: int
    total_probe_count: int
    accuracy: float
    valid_prediction_count: int
    parse_valid_rate: float


@dataclass(frozen=True, slots=True)
class HeuristicAlignmentSummary:
    pattern_name: str
    reference_source_name: str
    matching_probe_count: int
    total_probe_count: int
    probe_agreement_rate: float
    matching_error_probe_count: int
    total_error_probes: int
    error_agreement_rate: float
    matching_episode_count: int
    episode_count: int
    episode_agreement_rate: float


@dataclass(frozen=True, slots=True)
class AuditSourceSummary:
    name: str
    task_mode: TaskMode | None
    is_baseline: bool
    overall: AuditSliceSummary
    by_template: tuple[tuple[str, AuditSliceSummary], ...]
    by_difficulty: tuple[tuple[str, AuditSliceSummary], ...]
    failure_patterns: tuple[HeuristicAlignmentSummary, ...]


@dataclass(frozen=True, slots=True)
class BaselineComparisonSummary:
    accuracy_ranking: tuple[tuple[str, float], ...]
    best_baseline_name: str | None
    best_baseline_accuracy: float | None


@dataclass(frozen=True, slots=True)
class ModeComparisonSummary:
    binary_accuracy: float
    narrative_accuracy: float
    accuracy_gap: float
    binary_parse_valid_rate: float
    narrative_parse_valid_rate: float
    parse_valid_rate_gap: float


@dataclass(frozen=True, slots=True)
class AuditReport:
    episode_count: int
    difficulty_labels_present: tuple[str, ...]
    difficulty_labels_missing: tuple[str, ...]
    task_mode_summaries: tuple[tuple[TaskMode, AuditSliceSummary], ...]
    source_summaries: tuple[AuditSourceSummary, ...]
    baseline_comparison: BaselineComparisonSummary
    mode_comparison: ModeComparisonSummary | None
    limitations: tuple[str, ...]


def run_audit(
    episodes: Iterable[Episode],
    sources: Iterable[AuditSource],
) -> AuditReport:
    normalized_episodes = tuple(episodes)
    normalized_sources = tuple(sources)

    if not all(isinstance(episode, Episode) for episode in normalized_episodes):
        raise TypeError("episodes must contain Episode values")
    if not all(isinstance(source, AuditSource) for source in normalized_sources):
        raise TypeError("sources must contain AuditSource values")

    episode_count = len(normalized_episodes)
    episode_targets = tuple(episode.probe_targets for episode in normalized_episodes)
    for source in normalized_sources:
        if len(source.predictions) != episode_count:
            raise ValueError(
                f"source {source.name!r} must contain exactly {episode_count} predictions"
            )

    difficulty_labels_present = tuple(
        difficulty
        for difficulty in _DIFFICULTY_ORDER
        if any(episode.difficulty.value == difficulty for episode in normalized_episodes)
    )
    difficulty_labels_missing = tuple(
        difficulty
        for difficulty in _DIFFICULTY_ORDER
        if difficulty not in difficulty_labels_present
    )
    limitations = []
    if "hard" in difficulty_labels_missing:
        limitations.append("No emitted hard episodes in supplied set; hard slice omitted.")

    heuristic_sources = {
        source.name: source
        for source in normalized_sources
        if source.is_baseline
        and source.name in {reference_name for _, reference_name in _FAILURE_PATTERN_REFERENCE_NAMES}
    }

    source_summaries = tuple(
        _build_source_summary(
            source=source,
            episodes=normalized_episodes,
            targets=episode_targets,
            heuristic_sources=heuristic_sources,
        )
        for source in normalized_sources
    )
    task_mode_summaries = _build_task_mode_summaries(
        episodes=normalized_episodes,
        sources=normalized_sources,
    )
    mode_comparison = _build_mode_comparison(task_mode_summaries)

    return AuditReport(
        episode_count=episode_count,
        difficulty_labels_present=difficulty_labels_present,
        difficulty_labels_missing=difficulty_labels_missing,
        task_mode_summaries=task_mode_summaries,
        source_summaries=source_summaries,
        baseline_comparison=_build_baseline_comparison(source_summaries),
        mode_comparison=mode_comparison,
        limitations=tuple(limitations),
    )


def _build_source_summary(
    *,
    source: AuditSource,
    episodes: tuple[Episode, ...],
    targets: tuple[tuple[object, ...], ...],
    heuristic_sources: dict[str, AuditSource],
) -> AuditSourceSummary:
    return AuditSourceSummary(
        name=source.name,
        task_mode=source.task_mode,
        is_baseline=source.is_baseline,
        overall=_build_slice_summary(source.predictions, targets),
        by_template=_build_episode_slices(
            episodes=episodes,
            predictions=source.predictions,
            labels=_TEMPLATE_ORDER,
            key_fn=lambda episode: episode.template_id.value,
        ),
        by_difficulty=_build_episode_slices(
            episodes=episodes,
            predictions=source.predictions,
            labels=tuple(
                difficulty
                for difficulty in _DIFFICULTY_ORDER
                if any(episode.difficulty.value == difficulty for episode in episodes)
            ),
            key_fn=lambda episode: episode.difficulty.value,
        ),
        failure_patterns=_build_failure_patterns(
            source=source,
            reference_sources=heuristic_sources,
            targets=targets,
        ),
    )


def _build_episode_slices(
    *,
    episodes: tuple[Episode, ...],
    predictions: tuple[ParsedPrediction, ...],
    labels: tuple[str, ...],
    key_fn,
) -> tuple[tuple[str, AuditSliceSummary], ...]:
    slice_summaries: list[tuple[str, AuditSliceSummary]] = []
    for label in labels:
        slice_indices = tuple(
            index for index, episode in enumerate(episodes) if key_fn(episode) == label
        )
        if not slice_indices:
            continue
        slice_summaries.append(
            (
                label,
                _build_slice_summary(
                    tuple(predictions[index] for index in slice_indices),
                    tuple(episodes[index].probe_targets for index in slice_indices),
                ),
            )
        )
    return tuple(slice_summaries)


def _build_failure_patterns(
    *,
    source: AuditSource,
    reference_sources: dict[str, AuditSource],
    targets: tuple[tuple[object, ...], ...],
) -> tuple[HeuristicAlignmentSummary, ...]:
    failure_patterns: list[HeuristicAlignmentSummary] = []
    for pattern_name, reference_name in _FAILURE_PATTERN_REFERENCE_NAMES:
        if source.name == reference_name:
            continue
        reference_source = reference_sources.get(reference_name)
        if reference_source is None:
            continue
        failure_patterns.append(
            _build_heuristic_alignment_summary(
                pattern_name=pattern_name,
                reference_source_name=reference_name,
                source_predictions=source.predictions,
                reference_predictions=reference_source.predictions,
                targets=targets,
            )
        )
    return tuple(failure_patterns)


def _build_heuristic_alignment_summary(
    *,
    pattern_name: str,
    reference_source_name: str,
    source_predictions: tuple[ParsedPrediction, ...],
    reference_predictions: tuple[ParsedPrediction, ...],
    targets: tuple[tuple[object, ...], ...],
) -> HeuristicAlignmentSummary:
    matching_probe_count = 0
    matching_error_probe_count = 0
    matching_episode_count = 0
    total_error_probes = 0
    episode_count = len(source_predictions)
    total_probe_count = episode_count * PROBE_COUNT

    for source_prediction, reference_prediction, target in zip(
        source_predictions,
        reference_predictions,
        targets,
    ):
        source_labels = _prediction_labels(source_prediction)
        reference_labels = _prediction_labels(reference_prediction)
        if source_labels is not None and reference_labels is not None:
            matching_probe_count += sum(
                source_label is reference_label
                for source_label, reference_label in zip(source_labels, reference_labels)
            )
            if source_labels == reference_labels:
                matching_episode_count += 1

        if source_labels is None:
            total_error_probes += PROBE_COUNT
            continue

        for index, (source_label, target_label) in enumerate(zip(source_labels, target)):
            if source_label is target_label:
                continue
            total_error_probes += 1
            if reference_labels is not None and source_label is reference_labels[index]:
                matching_error_probe_count += 1

    return HeuristicAlignmentSummary(
        pattern_name=pattern_name,
        reference_source_name=reference_source_name,
        matching_probe_count=matching_probe_count,
        total_probe_count=total_probe_count,
        probe_agreement_rate=(
            matching_probe_count / total_probe_count if total_probe_count else 0.0
        ),
        matching_error_probe_count=matching_error_probe_count,
        total_error_probes=total_error_probes,
        error_agreement_rate=(
            matching_error_probe_count / total_error_probes
            if total_error_probes
            else 0.0
        ),
        matching_episode_count=matching_episode_count,
        episode_count=episode_count,
        episode_agreement_rate=(
            matching_episode_count / episode_count if episode_count else 0.0
        ),
    )


def _build_task_mode_summaries(
    *,
    episodes: tuple[Episode, ...],
    sources: tuple[AuditSource, ...],
) -> tuple[tuple[TaskMode, AuditSliceSummary], ...]:
    task_mode_summaries: list[tuple[TaskMode, AuditSliceSummary]] = []
    for task_mode in _TASK_MODE_ORDER:
        mode_sources = tuple(source for source in sources if source.task_mode == task_mode)
        if not mode_sources:
            continue
        pooled_predictions = tuple(
            prediction
            for source in mode_sources
            for prediction in source.predictions
        )
        pooled_targets = tuple(
            episode.probe_targets
            for source in mode_sources
            for episode in episodes
        )
        task_mode_summaries.append(
            (
                task_mode,
                _build_slice_summary(pooled_predictions, pooled_targets),
            )
        )
    return tuple(task_mode_summaries)


def _build_mode_comparison(
    task_mode_summaries: tuple[tuple[TaskMode, AuditSliceSummary], ...],
) -> ModeComparisonSummary | None:
    task_mode_map = dict(task_mode_summaries)
    binary_summary = task_mode_map.get("Binary")
    narrative_summary = task_mode_map.get("Narrative")
    if binary_summary is None or narrative_summary is None:
        return None
    return ModeComparisonSummary(
        binary_accuracy=binary_summary.accuracy,
        narrative_accuracy=narrative_summary.accuracy,
        accuracy_gap=binary_summary.accuracy - narrative_summary.accuracy,
        binary_parse_valid_rate=binary_summary.parse_valid_rate,
        narrative_parse_valid_rate=narrative_summary.parse_valid_rate,
        parse_valid_rate_gap=(
            binary_summary.parse_valid_rate - narrative_summary.parse_valid_rate
        ),
    )


def _build_baseline_comparison(
    source_summaries: tuple[AuditSourceSummary, ...],
) -> BaselineComparisonSummary:
    baseline_scores = tuple(
        (summary.name, summary.overall.accuracy)
        for summary in source_summaries
        if summary.is_baseline
    )
    accuracy_ranking = tuple(
        sorted(
            baseline_scores,
            key=lambda item: (-item[1], item[0]),
        )
    )
    if not accuracy_ranking:
        return BaselineComparisonSummary(
            accuracy_ranking=(),
            best_baseline_name=None,
            best_baseline_accuracy=None,
        )
    best_baseline_name, best_baseline_accuracy = accuracy_ranking[0]
    return BaselineComparisonSummary(
        accuracy_ranking=accuracy_ranking,
        best_baseline_name=best_baseline_name,
        best_baseline_accuracy=best_baseline_accuracy,
    )


def _build_slice_summary(
    predictions: tuple[ParsedPrediction, ...],
    targets: tuple[tuple[object, ...], ...],
) -> AuditSliceSummary:
    episode_count = len(predictions)
    total_probe_count = episode_count * PROBE_COUNT
    correct_probe_count = sum(
        _count_correct_probes(prediction, target)
        for prediction, target in zip(predictions, targets)
    )
    valid_prediction_count = sum(
        prediction.status is ParseStatus.VALID for prediction in predictions
    )
    return AuditSliceSummary(
        episode_count=episode_count,
        correct_probe_count=correct_probe_count,
        total_probe_count=total_probe_count,
        accuracy=correct_probe_count / total_probe_count if total_probe_count else 0.0,
        valid_prediction_count=valid_prediction_count,
        parse_valid_rate=(
            valid_prediction_count / episode_count if episode_count else 0.0
        ),
    )


def _count_correct_probes(
    prediction: ParsedPrediction,
    target: tuple[object, ...],
) -> int:
    labels = _prediction_labels(prediction)
    if labels is None:
        return 0
    return sum(predicted_label is target_label for predicted_label, target_label in zip(labels, target))


def _prediction_labels(
    prediction: ParsedPrediction,
) -> tuple[object, ...] | None:
    if prediction.status is not ParseStatus.VALID:
        return None
    if len(prediction.labels) != PROBE_COUNT:
        return None
    return prediction.labels
