from __future__ import annotations

from typing import Final, Iterable, Mapping

from maintainer.audit.core import (
    AuditSliceSummary,
    AuditSource,
    AuditSourceSummary,
    BaselineComparisonSummary,
    MatchedModeComparisonSummary,
    ModeComparisonSummary,
    ReleaseAuditReport,
    ReleaseAuditSourceSummary,
    TaskMode,
    build_baseline_comparison_from_scores,
    build_mode_comparison_from_predictions,
    build_slice_summary,
    build_source_summary,
    score_gap,
    run_audit,
    DIFFICULTY_ORDER,
    FAILURE_PATTERN_REFERENCE_NAMES,
    RELEASE_SPLIT_ORDER,
    TASK_MODE_ORDER,
    TEMPLATE_ORDER,
    TEMPLATE_FAMILY_ORDER,
)
from tasks.ruleshift_benchmark.schema import Episode

__all__ = [
    "run_release_r15_reaudit",
]

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
        for difficulty in DIFFICULTY_ORDER
        if any(
            episode.difficulty.value == difficulty
            for _, episodes in normalized_episodes_by_split
            for episode in episodes
        )
    )
    difficulty_labels_missing = tuple(
        difficulty
        for difficulty in DIFFICULTY_ORDER
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
            for split_name in RELEASE_SPLIT_ORDER
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
    canonical = tuple(name for name in RELEASE_SPLIT_ORDER if name in split_names)
    extras = tuple(sorted(name for name in split_names if name not in canonical))
    return canonical + extras


def _build_release_baseline_sources_by_split(
    episodes_by_split: tuple[tuple[str, tuple[Episode, ...]], ...],
    *,
    random_baseline_seed: int,
) -> dict[str, dict[str, AuditSource]]:
    from maintainer.baselines import (
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
    combined_predictions: list = []
    by_split: list[tuple[str, AuditSliceSummary]] = []

    for split_name, source, episodes in source_entries:
        _validate_source_lengths(source=source, episodes=episodes)
        covered_splits.append(split_name)
        combined_episodes.extend(episodes)
        combined_predictions.extend(source.predictions)
        by_split.append(
            (
                split_name,
                build_slice_summary(
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
        for _, reference_name in FAILURE_PATTERN_REFERENCE_NAMES
        if all(reference_name in baseline_sources_by_split[split_name] for split_name in covered_splits)
    }
    generic_summary = build_source_summary(
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
        by_template_family=generic_summary.by_template_family,
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
    return build_baseline_comparison_from_scores(
        tuple((summary.name, summary.overall.accuracy) for summary in baseline_summaries)
    )


def _build_release_baseline_comparison_by_split(
    *,
    split_name: str,
    split_source_map: Mapping[str, AuditSourceSummary],
) -> BaselineComparisonSummary:
    del split_name
    return build_baseline_comparison_from_scores(
        tuple(
            (baseline_name, split_source_map[baseline_name].overall.accuracy)
            for baseline_name in _R15_BASELINE_ORDER
        )
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
            if source.task_mode not in TASK_MODE_ORDER:
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
                overall=build_mode_comparison_from_predictions(
                    binary_predictions=combined_binary_predictions,
                    narrative_predictions=combined_narrative_predictions,
                    targets=tuple(episode.probe_targets for episode in combined_episodes),
                ),
                by_split=tuple(
                    (
                        split_name,
                        build_mode_comparison_from_predictions(
                            binary_predictions=binary_sources[split_name].predictions,
                            narrative_predictions=narrative_sources[split_name].predictions,
                            targets=tuple(
                                episode.probe_targets for episode in episode_map[split_name]
                            ),
                        ),
                    )
                    for split_name in matched_splits
                ),
                by_template=_build_mode_slice_comparisons(
                    episodes=combined_episodes,
                    binary_predictions=combined_binary_predictions,
                    narrative_predictions=combined_narrative_predictions,
                    labels=TEMPLATE_ORDER,
                    key_fn=lambda episode: episode.template_id.value,
                ),
                by_template_family=_build_mode_slice_comparisons(
                    episodes=combined_episodes,
                    binary_predictions=combined_binary_predictions,
                    narrative_predictions=combined_narrative_predictions,
                    labels=TEMPLATE_FAMILY_ORDER,
                    key_fn=lambda episode: episode.template_family.value,
                ),
                by_difficulty=_build_mode_slice_comparisons(
                    episodes=combined_episodes,
                    binary_predictions=combined_binary_predictions,
                    narrative_predictions=combined_narrative_predictions,
                    labels=DIFFICULTY_ORDER,
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
    binary_predictions: tuple,
    narrative_predictions: tuple,
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
    binary_predictions: tuple,
    narrative_predictions: tuple,
    labels: tuple[str, ...],
    key_fn,
) -> tuple[tuple[str, ModeComparisonSummary], ...]:
    comparisons: list[tuple[str, ModeComparisonSummary]] = []
    for label in labels:
        slice_indices = tuple(
            index for index, episode in enumerate(episodes) if key_fn(episode) == label
        )
        comparisons.append(
            (
                label,
                build_mode_comparison_from_predictions(
                    binary_predictions=tuple(binary_predictions[index] for index in slice_indices),
                    narrative_predictions=tuple(
                        narrative_predictions[index] for index in slice_indices
                    ),
                    targets=tuple(episodes[index].probe_targets for index in slice_indices),
                ),
            )
        )
    return tuple(comparisons)


def _build_release_limitations(
    *,
    difficulty_labels_missing: tuple[str, ...],
    model_summaries: tuple[ReleaseAuditSourceSummary, ...],
    matched_mode_comparisons: tuple[MatchedModeComparisonSummary, ...],
    expected_splits: tuple[str, ...],
) -> tuple[str, ...]:
    limitations: list[str] = []
    if "hard" in difficulty_labels_missing:
        limitations.append(
            "Supplied episodes do not cover the full emitted difficulty set."
        )
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
        "Narrative remains required non-leaderboard robustness evidence on the same frozen episodes and probe targets as Binary; only the final four labels are scored, and it does not replace the primary Binary post-shift probe audit."
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
        score_gap(split_source_maps[private_split_name][baseline_name].by_template)
        for baseline_name in _CRITICAL_HEURISTIC_BASELINES
    )
    best_difficulty_gap = max(
        score_gap(split_source_maps[private_split_name][baseline_name].by_difficulty)
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
        blockers.append("some supplied episodes miss part of the emitted difficulty set")

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
