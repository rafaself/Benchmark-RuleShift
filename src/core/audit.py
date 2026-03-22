from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable, Literal, Mapping

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
    "ReleaseAuditSourceSummary",
    "MatchedModeComparisonSummary",
    "ReleaseAuditReport",
    "AuditReport",
    "run_audit",
    "run_release_r15_reaudit",
    "serialize_release_r15_reaudit_report",
]

TaskMode = Literal["Binary", "Narrative"]

_TASK_MODE_ORDER: Final[tuple[TaskMode, ...]] = ("Binary", "Narrative")
_TEMPLATE_ORDER: Final[tuple[str, ...]] = ("T1", "T2")
_DIFFICULTY_ORDER: Final[tuple[str, ...]] = ("easy", "medium", "hard")
_RELEASE_SPLIT_ORDER: Final[tuple[str, ...]] = (
    "dev",
    "public_leaderboard",
    "private_leaderboard",
)
_FAILURE_PATTERN_REFERENCE_NAMES: Final[tuple[tuple[str, str], ...]] = (
    ("persistence-like", "never_update"),
    ("recency / last-evidence-like", "last_evidence"),
    ("physics-prior", "physics_prior"),
    ("template-position", "template_position"),
)
_R15_BASELINE_ORDER: Final[tuple[str, ...]] = (
    "random",
    "never_update",
    "last_evidence",
    "physics_prior",
    "template_position",
)
_R15_RANDOM_BASELINE_SEED: Final[int] = 11
_CRITICAL_HEURISTIC_BASELINES: Final[tuple[str, ...]] = (
    "never_update",
    "last_evidence",
    "physics_prior",
    "template_position",
)
_R15_MIN_SUBSET_GAP: Final[float] = 0.1
_R15_SHORTCUT_BOUND: Final[float] = 0.55


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
    source_family: str | None = None
    is_real_model: bool = False

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
        if self.source_family is not None and not _is_nonempty_string(self.source_family):
            raise ValueError("source_family must be a non-empty string or None")
        if not isinstance(self.is_real_model, bool):
            raise TypeError("is_real_model must be a bool")

    @classmethod
    def from_parsed_predictions(
        cls,
        name: str,
        predictions: Iterable[ParsedPrediction],
        *,
        task_mode: TaskMode | str | None = None,
        is_baseline: bool = False,
        source_family: str | None = None,
        is_real_model: bool = False,
    ) -> "AuditSource":
        return cls(
            name=name,
            predictions=tuple(predictions),
            task_mode=task_mode,
            is_baseline=is_baseline,
            source_family=source_family,
            is_real_model=is_real_model,
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
class ReleaseAuditSourceSummary:
    name: str
    task_mode: TaskMode | None
    is_baseline: bool
    source_family: str | None
    is_real_model: bool
    covered_splits: tuple[str, ...]
    overall: AuditSliceSummary
    by_split: tuple[tuple[str, AuditSliceSummary], ...]
    by_template: tuple[tuple[str, AuditSliceSummary], ...]
    by_difficulty: tuple[tuple[str, AuditSliceSummary], ...]
    failure_patterns: tuple[HeuristicAlignmentSummary, ...]


@dataclass(frozen=True, slots=True)
class MatchedModeComparisonSummary:
    source_family: str
    binary_source_name: str
    narrative_source_name: str
    covered_splits: tuple[str, ...]
    overall: ModeComparisonSummary
    by_template: tuple[tuple[str, ModeComparisonSummary], ...]
    by_difficulty: tuple[tuple[str, ModeComparisonSummary], ...]


@dataclass(frozen=True, slots=True)
class ReleaseAuditReport:
    release_id: str
    random_baseline_seed: int
    split_episode_counts: tuple[tuple[str, int], ...]
    difficulty_labels_present: tuple[str, ...]
    difficulty_labels_missing: tuple[str, ...]
    baseline_summaries: tuple[ReleaseAuditSourceSummary, ...]
    model_summaries: tuple[ReleaseAuditSourceSummary, ...]
    baseline_comparison: BaselineComparisonSummary
    baseline_comparison_by_split: tuple[tuple[str, BaselineComparisonSummary], ...]
    matched_mode_comparisons: tuple[MatchedModeComparisonSummary, ...]
    audit_note: str
    limitations: tuple[str, ...]


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


def run_release_r15_reaudit(
    episodes_by_split: Mapping[str, Iterable[Episode]]
    | tuple[tuple[str, tuple[Episode, ...]], ...]
    | None = None,
    *,
    model_sources_by_split: Mapping[str, Iterable[AuditSource]]
    | tuple[tuple[str, tuple[AuditSource, ...]], ...]
    | None = None,
    random_baseline_seed: int = _R15_RANDOM_BASELINE_SEED,
    release_id: str = "R15",
) -> ReleaseAuditReport:
    normalized_episodes_by_split = _normalize_release_episodes_by_split(episodes_by_split)
    normalized_model_sources_by_split = _normalize_release_sources_by_split(
        model_sources_by_split,
        split_names=tuple(split_name for split_name, _ in normalized_episodes_by_split),
    )
    baseline_sources_by_split = _build_release_baseline_sources_by_split(
        normalized_episodes_by_split,
        random_baseline_seed=random_baseline_seed,
    )
    split_audit_reports = tuple(
        (
            split_name,
            run_audit(
                episodes,
                tuple(baseline_sources_by_split[split_name].values()),
            ),
        )
        for split_name, episodes in normalized_episodes_by_split
    )
    split_source_maps = {
        split_name: {summary.name: summary for summary in report.source_summaries}
        for split_name, report in split_audit_reports
    }
    split_episode_counts = tuple(
        (split_name, len(episodes))
        for split_name, episodes in normalized_episodes_by_split
    )

    baseline_summaries = tuple(
        _build_release_source_summary(
            source_name=baseline_name,
            task_mode=None,
            is_baseline=True,
            source_family=None,
            is_real_model=False,
            source_entries=tuple(
                (
                    split_name,
                    baseline_sources_by_split[split_name][baseline_name],
                    episodes,
                )
                for split_name, episodes in normalized_episodes_by_split
            ),
            baseline_sources_by_split=baseline_sources_by_split,
        )
        for baseline_name in _R15_BASELINE_ORDER
    )
    model_summaries = _build_release_model_summaries(
        normalized_episodes_by_split,
        normalized_model_sources_by_split,
        baseline_sources_by_split=baseline_sources_by_split,
    )
    difficulty_labels_present = tuple(
        difficulty
        for difficulty in _DIFFICULTY_ORDER
        if any(
            episode.difficulty.value == difficulty
            for _, episodes in normalized_episodes_by_split
            for episode in episodes
        )
    )
    difficulty_labels_missing = tuple(
        difficulty
        for difficulty in _DIFFICULTY_ORDER
        if difficulty not in difficulty_labels_present
    )
    baseline_comparison = _build_release_baseline_comparison(baseline_summaries)
    baseline_comparison_by_split = tuple(
        (
            split_name,
            _build_release_baseline_comparison_by_split(
                split_name=split_name,
                split_source_map=split_source_maps[split_name],
            ),
        )
        for split_name, _ in normalized_episodes_by_split
    )
    matched_mode_comparisons = _build_release_mode_comparisons(
        normalized_episodes_by_split,
        normalized_model_sources_by_split,
    )
    limitations = _build_release_limitations(
        difficulty_labels_missing=difficulty_labels_missing,
        model_summaries=model_summaries,
        matched_mode_comparisons=matched_mode_comparisons,
        expected_splits=tuple(split_name for split_name, _ in normalized_episodes_by_split),
    )
    audit_note = _build_release_audit_note(
        split_source_maps=split_source_maps,
        difficulty_labels_missing=difficulty_labels_missing,
        model_summaries=model_summaries,
        matched_mode_comparisons=matched_mode_comparisons,
    )

    return ReleaseAuditReport(
        release_id=release_id,
        random_baseline_seed=random_baseline_seed,
        split_episode_counts=split_episode_counts,
        difficulty_labels_present=difficulty_labels_present,
        difficulty_labels_missing=difficulty_labels_missing,
        baseline_summaries=baseline_summaries,
        model_summaries=model_summaries,
        baseline_comparison=baseline_comparison,
        baseline_comparison_by_split=baseline_comparison_by_split,
        matched_mode_comparisons=matched_mode_comparisons,
        audit_note=audit_note,
        limitations=limitations,
    )


def serialize_release_r15_reaudit_report(
    report: ReleaseAuditReport,
) -> dict[str, object]:
    return {
        "release_id": report.release_id,
        "random_baseline_seed": report.random_baseline_seed,
        "split_episode_counts": [
            {"split": split_name, "episode_count": episode_count}
            for split_name, episode_count in report.split_episode_counts
        ],
        "difficulty_labels_present": list(report.difficulty_labels_present),
        "difficulty_labels_missing": list(report.difficulty_labels_missing),
        "baseline_summaries": [
            _serialize_release_source_summary(summary)
            for summary in report.baseline_summaries
        ],
        "model_summaries": [
            _serialize_release_source_summary(summary)
            for summary in report.model_summaries
        ],
        "baseline_comparison": _serialize_baseline_comparison(report.baseline_comparison),
        "baseline_comparison_by_split": [
            {
                "split_name": split_name,
                **_serialize_baseline_comparison(summary),
            }
            for split_name, summary in report.baseline_comparison_by_split
        ],
        "matched_mode_comparisons": [
            {
                "source_family": summary.source_family,
                "binary_source_name": summary.binary_source_name,
                "narrative_source_name": summary.narrative_source_name,
                "covered_splits": list(summary.covered_splits),
                "overall": _serialize_mode_comparison(summary.overall),
                "by_template": [
                    {"label": label, **_serialize_mode_comparison(slice_summary)}
                    for label, slice_summary in summary.by_template
                ],
                "by_difficulty": [
                    {"label": label, **_serialize_mode_comparison(slice_summary)}
                    for label, slice_summary in summary.by_difficulty
                ],
            }
            for summary in report.matched_mode_comparisons
        ],
        "audit_note": report.audit_note,
        "limitations": list(report.limitations),
    }


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


def _normalize_release_episodes_by_split(
    episodes_by_split: Mapping[str, Iterable[Episode]]
    | tuple[tuple[str, tuple[Episode, ...]], ...]
    | None,
) -> tuple[tuple[str, tuple[Episode, ...]], ...]:
    if episodes_by_split is None:
        from core.splits import load_frozen_split

        return tuple(
            (
                split_name,
                tuple(record.episode for record in load_frozen_split(split_name)),
            )
            for split_name in _RELEASE_SPLIT_ORDER
        )

    provided = dict(episodes_by_split)
    normalized: list[tuple[str, tuple[Episode, ...]]] = []
    for split_name in _ordered_release_split_names(tuple(provided)):
        episodes = tuple(provided[split_name])
        if not all(isinstance(episode, Episode) for episode in episodes):
            raise TypeError("episodes_by_split must contain Episode values")
        normalized.append((split_name, episodes))
    return tuple(normalized)


def _normalize_release_sources_by_split(
    model_sources_by_split: Mapping[str, Iterable[AuditSource]]
    | tuple[tuple[str, tuple[AuditSource, ...]], ...]
    | None,
    *,
    split_names: tuple[str, ...],
) -> tuple[tuple[str, tuple[AuditSource, ...]], ...]:
    if model_sources_by_split is None:
        return tuple((split_name, ()) for split_name in split_names)

    provided = dict(model_sources_by_split)
    normalized: list[tuple[str, tuple[AuditSource, ...]]] = []
    for split_name in split_names:
        sources = tuple(provided.get(split_name, ()))
        if not all(isinstance(source, AuditSource) for source in sources):
            raise TypeError("model_sources_by_split must contain AuditSource values")
        if any(source.is_baseline for source in sources):
            raise ValueError("model_sources_by_split must not contain baseline sources")
        normalized.append((split_name, sources))
    return tuple(normalized)


def _ordered_release_split_names(
    split_names: tuple[str, ...],
) -> tuple[str, ...]:
    canonical = tuple(name for name in _RELEASE_SPLIT_ORDER if name in split_names)
    extras = tuple(sorted(name for name in split_names if name not in canonical))
    return canonical + extras


def _build_release_baseline_sources_by_split(
    episodes_by_split: tuple[tuple[str, tuple[Episode, ...]], ...],
    *,
    random_baseline_seed: int,
) -> dict[str, dict[str, AuditSource]]:
    from tasks.iron_find_electric.baselines import (
        last_evidence_baseline,
        never_update_baseline,
        physics_prior_baseline,
        random_baseline,
        run_baselines,
        template_position_baseline,
    )

    baseline_fns = (
        ("random", lambda episode: random_baseline(episode, seed=random_baseline_seed)),
        ("never_update", never_update_baseline),
        ("last_evidence", last_evidence_baseline),
        ("physics_prior", physics_prior_baseline),
        ("template_position", template_position_baseline),
    )
    baseline_sources_by_split: dict[str, dict[str, AuditSource]] = {}
    for split_name, episodes in episodes_by_split:
        runs = run_baselines(episodes, baseline_fns)
        baseline_sources_by_split[split_name] = {
            source.name: source
            for source in (
                AuditSource.from_baseline_run(run)
                for run in runs
            )
        }
    return baseline_sources_by_split


def _build_release_model_summaries(
    episodes_by_split: tuple[tuple[str, tuple[Episode, ...]], ...],
    model_sources_by_split: tuple[tuple[str, tuple[AuditSource, ...]], ...],
    *,
    baseline_sources_by_split: Mapping[str, Mapping[str, AuditSource]],
) -> tuple[ReleaseAuditSourceSummary, ...]:
    grouped_entries: dict[
        tuple[str, TaskMode | None, str | None, bool],
        list[tuple[str, AuditSource, tuple[Episode, ...]]],
    ] = {}
    for split_name, sources in model_sources_by_split:
        episodes = dict(episodes_by_split)[split_name]
        for source in sources:
            key = (
                source.name,
                source.task_mode,
                source.source_family,
                source.is_real_model,
            )
            grouped_entries.setdefault(key, []).append((split_name, source, episodes))

    return tuple(
        _build_release_source_summary(
            source_name=source_name,
            task_mode=task_mode,
            is_baseline=False,
            source_family=source_family,
            is_real_model=is_real_model,
            source_entries=tuple(
                _sort_source_entries(entries)
            ),
            baseline_sources_by_split=baseline_sources_by_split,
        )
        for (source_name, task_mode, source_family, is_real_model), entries in grouped_entries.items()
    )


def _sort_source_entries(
    entries: list[tuple[str, AuditSource, tuple[Episode, ...]]],
) -> tuple[tuple[str, AuditSource, tuple[Episode, ...]], ...]:
    split_order = _ordered_release_split_names(tuple(split_name for split_name, _, _ in entries))
    split_rank = {split_name: index for index, split_name in enumerate(split_order)}
    return tuple(
        sorted(entries, key=lambda entry: split_rank[entry[0]])
    )


def _build_release_source_summary(
    *,
    source_name: str,
    task_mode: TaskMode | None,
    is_baseline: bool,
    source_family: str | None,
    is_real_model: bool,
    source_entries: tuple[tuple[str, AuditSource, tuple[Episode, ...]], ...],
    baseline_sources_by_split: Mapping[str, Mapping[str, AuditSource]],
) -> ReleaseAuditSourceSummary:
    covered_splits: list[str] = []
    combined_episodes: list[Episode] = []
    combined_predictions: list[ParsedPrediction] = []
    by_split: list[tuple[str, AuditSliceSummary]] = []

    for split_name, source, episodes in source_entries:
        _validate_source_lengths(source=source, episodes=episodes)
        covered_splits.append(split_name)
        combined_episodes.extend(episodes)
        combined_predictions.extend(source.predictions)
        by_split.append(
            (
                split_name,
                _build_slice_summary(
                    source.predictions,
                    tuple(episode.probe_targets for episode in episodes),
                ),
            )
        )

    combined_source = AuditSource.from_parsed_predictions(
        source_name,
        combined_predictions,
        task_mode=task_mode,
        is_baseline=is_baseline,
        source_family=source_family,
        is_real_model=is_real_model,
    )
    heuristic_sources = {
        reference_name: AuditSource.from_parsed_predictions(
            reference_name,
            tuple(
                prediction
                for split_name in covered_splits
                for prediction in baseline_sources_by_split[split_name][reference_name].predictions
            ),
            is_baseline=True,
        )
        for _, reference_name in _FAILURE_PATTERN_REFERENCE_NAMES
        if all(reference_name in baseline_sources_by_split[split_name] for split_name in covered_splits)
    }
    generic_summary = _build_source_summary(
        source=combined_source,
        episodes=tuple(combined_episodes),
        targets=tuple(episode.probe_targets for episode in combined_episodes),
        heuristic_sources=heuristic_sources,
    )
    return ReleaseAuditSourceSummary(
        name=source_name,
        task_mode=task_mode,
        is_baseline=is_baseline,
        source_family=source_family,
        is_real_model=is_real_model,
        covered_splits=tuple(covered_splits),
        overall=generic_summary.overall,
        by_split=tuple(by_split),
        by_template=generic_summary.by_template,
        by_difficulty=generic_summary.by_difficulty,
        failure_patterns=generic_summary.failure_patterns,
    )


def _validate_source_lengths(
    *,
    source: AuditSource,
    episodes: tuple[Episode, ...],
) -> None:
    if len(source.predictions) != len(episodes):
        raise ValueError(
            f"source {source.name!r} must contain exactly {len(episodes)} predictions"
        )


def _build_release_baseline_comparison(
    baseline_summaries: tuple[ReleaseAuditSourceSummary, ...],
) -> BaselineComparisonSummary:
    return _build_baseline_comparison_from_scores(
        tuple((summary.name, summary.overall.accuracy) for summary in baseline_summaries)
    )


def _build_release_baseline_comparison_by_split(
    *,
    split_name: str,
    split_source_map: Mapping[str, AuditSourceSummary],
) -> BaselineComparisonSummary:
    del split_name
    return _build_baseline_comparison_from_scores(
        tuple(
            (baseline_name, split_source_map[baseline_name].overall.accuracy)
            for baseline_name in _R15_BASELINE_ORDER
        )
    )


def _build_baseline_comparison_from_scores(
    baseline_scores: tuple[tuple[str, float], ...],
) -> BaselineComparisonSummary:
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


def _build_release_mode_comparisons(
    episodes_by_split: tuple[tuple[str, tuple[Episode, ...]], ...],
    model_sources_by_split: tuple[tuple[str, tuple[AuditSource, ...]], ...],
) -> tuple[MatchedModeComparisonSummary, ...]:
    family_mode_sources: dict[
        str,
        dict[TaskMode, dict[str, AuditSource]],
    ] = {}
    source_names_by_family_mode: dict[tuple[str, TaskMode], set[str]] = {}

    for split_name, sources in model_sources_by_split:
        for source in sources:
            if source.task_mode not in _TASK_MODE_ORDER:
                continue
            family = source.source_family or source.name
            split_sources = family_mode_sources.setdefault(
                family,
                {"Binary": {}, "Narrative": {}},
            )[source.task_mode]
            if split_name in split_sources:
                raise ValueError(
                    f"duplicate {source.task_mode} source for family {family!r} on split {split_name!r}"
                )
            split_sources[split_name] = source
            source_names_by_family_mode.setdefault((family, source.task_mode), set()).add(source.name)

    episode_map = dict(episodes_by_split)
    comparisons: list[MatchedModeComparisonSummary] = []
    for family, mode_map in family_mode_sources.items():
        binary_sources = mode_map["Binary"]
        narrative_sources = mode_map["Narrative"]
        matched_splits = tuple(
            split_name
            for split_name in _ordered_release_split_names(tuple(episode_map))
            if split_name in binary_sources and split_name in narrative_sources
        )
        if not matched_splits:
            continue
        binary_source_name = _resolve_single_mode_source_name(
            family=family,
            task_mode="Binary",
            source_names=source_names_by_family_mode[(family, "Binary")],
        )
        narrative_source_name = _resolve_single_mode_source_name(
            family=family,
            task_mode="Narrative",
            source_names=source_names_by_family_mode[(family, "Narrative")],
        )
        combined_episodes = tuple(
            episode
            for split_name in matched_splits
            for episode in episode_map[split_name]
        )
        combined_binary_predictions = tuple(
            prediction
            for split_name in matched_splits
            for prediction in binary_sources[split_name].predictions
        )
        combined_narrative_predictions = tuple(
            prediction
            for split_name in matched_splits
            for prediction in narrative_sources[split_name].predictions
        )
        _validate_prediction_target_lengths_for_modes(
            binary_predictions=combined_binary_predictions,
            narrative_predictions=combined_narrative_predictions,
            episodes=combined_episodes,
            binary_source_name=binary_source_name,
            narrative_source_name=narrative_source_name,
        )
        comparisons.append(
            MatchedModeComparisonSummary(
                source_family=family,
                binary_source_name=binary_source_name,
                narrative_source_name=narrative_source_name,
                covered_splits=matched_splits,
                overall=_build_mode_comparison_from_predictions(
                    binary_predictions=combined_binary_predictions,
                    narrative_predictions=combined_narrative_predictions,
                    targets=tuple(episode.probe_targets for episode in combined_episodes),
                ),
                by_template=_build_mode_slice_comparisons(
                    episodes=combined_episodes,
                    binary_predictions=combined_binary_predictions,
                    narrative_predictions=combined_narrative_predictions,
                    labels=_TEMPLATE_ORDER,
                    key_fn=lambda episode: episode.template_id.value,
                ),
                by_difficulty=_build_mode_slice_comparisons(
                    episodes=combined_episodes,
                    binary_predictions=combined_binary_predictions,
                    narrative_predictions=combined_narrative_predictions,
                    labels=tuple(
                        difficulty
                        for difficulty in _DIFFICULTY_ORDER
                        if any(
                            episode.difficulty.value == difficulty
                            for episode in combined_episodes
                        )
                    ),
                    key_fn=lambda episode: episode.difficulty.value,
                ),
            )
        )
    return tuple(comparisons)


def _resolve_single_mode_source_name(
    *,
    family: str,
    task_mode: TaskMode,
    source_names: set[str],
) -> str:
    if len(source_names) != 1:
        raise ValueError(
            f"matched mode comparison requires a single {task_mode} source name for family {family!r}"
        )
    return next(iter(source_names))


def _validate_prediction_target_lengths_for_modes(
    *,
    binary_predictions: tuple[ParsedPrediction, ...],
    narrative_predictions: tuple[ParsedPrediction, ...],
    episodes: tuple[Episode, ...],
    binary_source_name: str,
    narrative_source_name: str,
) -> None:
    episode_count = len(episodes)
    if len(binary_predictions) != episode_count:
        raise ValueError(
            f"binary source {binary_source_name!r} must contain exactly {episode_count} predictions"
        )
    if len(narrative_predictions) != episode_count:
        raise ValueError(
            f"narrative source {narrative_source_name!r} must contain exactly {episode_count} predictions"
        )


def _build_mode_slice_comparisons(
    *,
    episodes: tuple[Episode, ...],
    binary_predictions: tuple[ParsedPrediction, ...],
    narrative_predictions: tuple[ParsedPrediction, ...],
    labels: tuple[str, ...],
    key_fn,
) -> tuple[tuple[str, ModeComparisonSummary], ...]:
    comparisons: list[tuple[str, ModeComparisonSummary]] = []
    for label in labels:
        slice_indices = tuple(
            index for index, episode in enumerate(episodes) if key_fn(episode) == label
        )
        if not slice_indices:
            continue
        comparisons.append(
            (
                label,
                _build_mode_comparison_from_predictions(
                    binary_predictions=tuple(binary_predictions[index] for index in slice_indices),
                    narrative_predictions=tuple(
                        narrative_predictions[index] for index in slice_indices
                    ),
                    targets=tuple(episodes[index].probe_targets for index in slice_indices),
                ),
            )
        )
    return tuple(comparisons)


def _build_mode_comparison_from_predictions(
    *,
    binary_predictions: tuple[ParsedPrediction, ...],
    narrative_predictions: tuple[ParsedPrediction, ...],
    targets: tuple[tuple[object, ...], ...],
) -> ModeComparisonSummary:
    binary_summary = _build_slice_summary(binary_predictions, targets)
    narrative_summary = _build_slice_summary(narrative_predictions, targets)
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


def _build_release_limitations(
    *,
    difficulty_labels_missing: tuple[str, ...],
    model_summaries: tuple[ReleaseAuditSourceSummary, ...],
    matched_mode_comparisons: tuple[MatchedModeComparisonSummary, ...],
    expected_splits: tuple[str, ...],
) -> tuple[str, ...]:
    limitations: list[str] = []
    if "hard" in difficulty_labels_missing:
        limitations.append("No emitted hard episodes in supplied set; hard slice omitted.")
    if not model_summaries:
        limitations.append(
            "No structured model runs supplied; frozen R15 re-audit covers deterministic baselines only."
        )
    elif not any(summary.is_real_model for summary in model_summaries):
        limitations.append(
            "No real-model runs supplied; model coverage is limited to structured prediction fixtures."
        )
    for summary in model_summaries:
        missing_splits = tuple(split_name for split_name in expected_splits if split_name not in summary.covered_splits)
        if missing_splits:
            limitations.append(
                f"Model source {summary.name} only covers {', '.join(summary.covered_splits)}; missing {', '.join(missing_splits)}."
            )
    if not matched_mode_comparisons:
        limitations.append(
            "No matched Binary/Narrative model runs supplied; Binary vs Narrative comparison is unavailable."
        )
    limitations.append(
        "Narrative remains a robustness companion and does not replace the primary Binary post-shift probe audit."
    )
    return tuple(limitations)


def _build_release_audit_note(
    *,
    split_source_maps: Mapping[str, Mapping[str, AuditSourceSummary]],
    difficulty_labels_missing: tuple[str, ...],
    model_summaries: tuple[ReleaseAuditSourceSummary, ...],
    matched_mode_comparisons: tuple[MatchedModeComparisonSummary, ...],
) -> str:
    recency_worst_split_name, recency_worst_accuracy = _worst_split_score(
        split_source_maps,
        baseline_name="last_evidence",
    )
    critical_baseline_name, critical_split_name, critical_peak_accuracy = max(
        (
            (baseline_name, split_name, split_source_maps[split_name][baseline_name].overall.accuracy)
            for baseline_name in _CRITICAL_HEURISTIC_BASELINES
            for split_name in split_source_maps
        ),
        key=lambda item: (item[2], item[0], item[1]),
    )
    private_split_name = (
        "private_leaderboard"
        if "private_leaderboard" in split_source_maps
        else next(reversed(tuple(split_source_maps)))
    )
    best_template_gap = max(
        _score_gap(split_source_maps[private_split_name][baseline_name].by_template)
        for baseline_name in _CRITICAL_HEURISTIC_BASELINES
    )
    best_difficulty_gap = max(
        _score_gap(split_source_maps[private_split_name][baseline_name].by_difficulty)
        for baseline_name in _CRITICAL_HEURISTIC_BASELINES
    )

    model_clause = _build_release_model_clause(
        model_summaries=model_summaries,
        matched_mode_comparisons=matched_mode_comparisons,
        critical_peak_accuracy=critical_peak_accuracy,
    )
    blockers: list[str] = []
    if best_template_gap < _R15_MIN_SUBSET_GAP or best_difficulty_gap < _R15_MIN_SUBSET_GAP:
        blockers.append(
            f"private-leaderboard slice separation is still weak (best template gap {best_template_gap:.6f}; best emitted-difficulty gap {best_difficulty_gap:.6f})"
        )
    if not any(summary.is_real_model for summary in model_summaries):
        blockers.append("real-model coverage is still absent in-repo")
    if "hard" in difficulty_labels_missing:
        blockers.append("hard remains reserved and un-emitted")

    return (
        "R15 re-audited the refreshed R14 frozen splits with deterministic baselines "
        "and any supplied structured model runs. "
        + (
            f"`last_evidence` peaks at {recency_worst_accuracy:.6f} on {recency_worst_split_name}, "
            "so the recency shortcut no longer looks like the dominant failure mode. "
            if recency_worst_accuracy <= _R15_SHORTCUT_BOUND
            else f"`last_evidence` peaks at {recency_worst_accuracy:.6f} on {recency_worst_split_name}, "
            "so recency still looks like a serious shortcut threat. "
        )
        + (
            f"The strongest critical heuristic is {critical_baseline_name}="
            f"{critical_peak_accuracy:.6f} on {critical_split_name}, which keeps the named shortcut "
            "baselines materially bounded."
            if critical_peak_accuracy <= _R15_SHORTCUT_BOUND
            else f"The strongest critical heuristic is {critical_baseline_name}="
            f"{critical_peak_accuracy:.6f} on {critical_split_name}, so the named shortcut "
            "baselines are still too strong."
        )
        + " "
        + model_clause
        + (
            " Packaging is still blocked because " + "; ".join(blockers) + "."
            if blockers
            else " No packaging blocker remains in the re-audit surface."
        )
    )


def _build_release_model_clause(
    *,
    model_summaries: tuple[ReleaseAuditSourceSummary, ...],
    matched_mode_comparisons: tuple[MatchedModeComparisonSummary, ...],
    critical_peak_accuracy: float,
) -> str:
    real_model_summaries = tuple(summary for summary in model_summaries if summary.is_real_model)
    if not real_model_summaries:
        return "No real-model runs are bundled in-repo, so meaningful model-vs-heuristic separation is still unverified."

    best_real_model = max(
        real_model_summaries,
        key=lambda summary: (summary.overall.accuracy, summary.name),
    )
    separation_gap = best_real_model.overall.accuracy - critical_peak_accuracy
    mode_clause = (
        f" Matched Binary/Narrative comparisons are available for {len(matched_mode_comparisons)} source family"
        + ("" if len(matched_mode_comparisons) == 1 else "ies")
        + "."
        if matched_mode_comparisons
        else " No matched Binary/Narrative real-model comparison is available."
    )
    if separation_gap > 0:
        return (
            f"The best supplied real-model run is {best_real_model.name}="
            f"{best_real_model.overall.accuracy:.6f}, above the strongest critical heuristic by "
            f"{separation_gap:.6f}.{mode_clause}"
        )
    return (
        f"The best supplied real-model run is {best_real_model.name}="
        f"{best_real_model.overall.accuracy:.6f}, which does not separate above the strongest "
        f"critical heuristic ({critical_peak_accuracy:.6f}).{mode_clause}"
    )


def _worst_split_score(
    split_source_maps: Mapping[str, Mapping[str, AuditSourceSummary]],
    *,
    baseline_name: str,
) -> tuple[str, float]:
    return max(
        (
            (split_name, split_source_maps[split_name][baseline_name].overall.accuracy)
            for split_name in split_source_maps
        ),
        key=lambda item: (item[1], item[0]),
    )


def _serialize_release_source_summary(
    summary: ReleaseAuditSourceSummary,
) -> dict[str, object]:
    return {
        "name": summary.name,
        "task_mode": summary.task_mode,
        "is_baseline": summary.is_baseline,
        "source_family": summary.source_family,
        "is_real_model": summary.is_real_model,
        "covered_splits": list(summary.covered_splits),
        "overall": _serialize_slice_summary(summary.overall),
        "by_split": [
            {"split_name": split_name, **_serialize_slice_summary(slice_summary)}
            for split_name, slice_summary in summary.by_split
        ],
        "by_template": [
            {"label": label, **_serialize_slice_summary(slice_summary)}
            for label, slice_summary in summary.by_template
        ],
        "by_difficulty": [
            {"label": label, **_serialize_slice_summary(slice_summary)}
            for label, slice_summary in summary.by_difficulty
        ],
        "failure_patterns": [
            {
                "pattern_name": pattern.pattern_name,
                "reference_source_name": pattern.reference_source_name,
                "matching_probe_count": pattern.matching_probe_count,
                "total_probe_count": pattern.total_probe_count,
                "probe_agreement_rate": pattern.probe_agreement_rate,
                "matching_error_probe_count": pattern.matching_error_probe_count,
                "total_error_probes": pattern.total_error_probes,
                "error_agreement_rate": pattern.error_agreement_rate,
                "matching_episode_count": pattern.matching_episode_count,
                "episode_count": pattern.episode_count,
                "episode_agreement_rate": pattern.episode_agreement_rate,
            }
            for pattern in summary.failure_patterns
        ],
    }


def _serialize_slice_summary(
    summary: AuditSliceSummary,
) -> dict[str, object]:
    return {
        "episode_count": summary.episode_count,
        "correct_probe_count": summary.correct_probe_count,
        "total_probe_count": summary.total_probe_count,
        "accuracy": summary.accuracy,
        "valid_prediction_count": summary.valid_prediction_count,
        "parse_valid_rate": summary.parse_valid_rate,
    }


def _serialize_baseline_comparison(
    summary: BaselineComparisonSummary,
) -> dict[str, object]:
    return {
        "accuracy_ranking": [
            {"name": name, "accuracy": accuracy}
            for name, accuracy in summary.accuracy_ranking
        ],
        "best_baseline_name": summary.best_baseline_name,
        "best_baseline_accuracy": summary.best_baseline_accuracy,
    }


def _serialize_mode_comparison(
    summary: ModeComparisonSummary,
) -> dict[str, object]:
    return {
        "binary_accuracy": summary.binary_accuracy,
        "narrative_accuracy": summary.narrative_accuracy,
        "accuracy_gap": summary.accuracy_gap,
        "binary_parse_valid_rate": summary.binary_parse_valid_rate,
        "narrative_parse_valid_rate": summary.narrative_parse_valid_rate,
        "parse_valid_rate_gap": summary.parse_valid_rate_gap,
    }


def _score_gap(
    scores: tuple[tuple[str, AuditSliceSummary], ...],
) -> float:
    if len(scores) <= 1:
        return 0.0
    values = tuple(summary.accuracy for _, summary in scores)
    return max(values) - min(values)


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
