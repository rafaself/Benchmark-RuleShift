from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable, Mapping

from core.audit import AuditSource, AuditSourceSummary, run_audit
from tasks.ruleshift_benchmark.baselines import (
    BaselineFn,
    last_evidence_baseline,
    never_update_baseline,
    physics_prior_baseline,
    random_baseline,
    run_baselines,
    template_position_baseline,
)
from tasks.ruleshift_benchmark.schema import Episode

__all__ = [
    "SplitBaselineAccuracySummary",
    "BaselineAccuracySummary",
    "ShortcutUpperBoundRule",
    "DominantHeuristicRule",
    "SubsetSeparationRule",
    "ValidityGateConfig",
    "ValidityGateCheck",
    "BenchmarkValidityReport",
    "R13_VALIDITY_GATE",
    "run_benchmark_validity_report",
    "evaluate_benchmark_validity_gate",
    "serialize_benchmark_validity_report",
    "validate_benchmark_validity",
]

_VALIDITY_REPORT_SPLITS: Final[tuple[str, ...]] = (
    "dev",
    "public_leaderboard",
    "private_leaderboard",
)
_VALIDITY_GATE_SPLITS: Final[tuple[str, ...]] = (
    "public_leaderboard",
    "private_leaderboard",
)
_VALIDITY_AUDIT_SPLIT: Final[str] = "private_leaderboard"
_VALIDITY_BASELINE_ORDER: Final[tuple[str, ...]] = (
    "random",
    "never_update",
    "last_evidence",
    "physics_prior",
    "template_position",
)
_CRITICAL_SHORTCUT_BASELINES: Final[tuple[str, ...]] = (
    "never_update",
    "last_evidence",
    "physics_prior",
    "template_position",
)
_R13_RANDOM_BASELINE_SEED: Final[int] = 11
_SHORTCUT_ACCURACY_CAP: Final[float] = 0.55
_MIN_TEMPLATE_GAP: Final[float] = 0.1
_MIN_DIFFICULTY_GAP: Final[float] = 0.1


@dataclass(frozen=True, slots=True)
class SplitBaselineAccuracySummary:
    split_name: str
    accuracy: float
    by_template: tuple[tuple[str, float], ...]
    by_template_family: tuple[tuple[str, float], ...]
    by_difficulty: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class BaselineAccuracySummary:
    baseline_name: str
    overall_accuracy: float
    by_split: tuple[SplitBaselineAccuracySummary, ...]
    by_template: tuple[tuple[str, float], ...]
    by_template_family: tuple[tuple[str, float], ...]
    by_difficulty: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class ShortcutUpperBoundRule:
    baseline_name: str
    max_accuracy: float
    splits: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DominantHeuristicRule:
    baseline_names: tuple[str, ...]
    max_accuracy: float
    splits: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SubsetSeparationRule:
    baseline_names: tuple[str, ...]
    split_name: str
    min_template_gap: float
    min_difficulty_gap: float


@dataclass(frozen=True, slots=True)
class ValidityGateConfig:
    release_id: str
    random_baseline_seed: int
    report_splits: tuple[str, ...]
    gate_splits: tuple[str, ...]
    audit_split: str
    baseline_order: tuple[str, ...]
    critical_baselines: tuple[str, ...]
    shortcut_upper_bounds: tuple[ShortcutUpperBoundRule, ...]
    dominance_rule: DominantHeuristicRule
    subset_separation_rule: SubsetSeparationRule


@dataclass(frozen=True, slots=True)
class ValidityGateCheck:
    code: str
    passed: bool
    message: str


@dataclass(frozen=True, slots=True)
class BenchmarkValidityReport:
    release_id: str
    random_baseline_seed: int
    report_splits: tuple[str, ...]
    gate_splits: tuple[str, ...]
    audit_split: str
    split_episode_counts: tuple[tuple[str, int], ...]
    difficulty_labels_present: tuple[str, ...]
    difficulty_labels_missing: tuple[str, ...]
    critical_baselines: tuple[str, ...]
    baseline_summaries: tuple[BaselineAccuracySummary, ...]
    checks: tuple[ValidityGateCheck, ...]
    passed: bool
    comparison_summary: str
    validity_note: str
    limitations: tuple[str, ...]


R13_VALIDITY_GATE: Final[ValidityGateConfig] = ValidityGateConfig(
    release_id="R13",
    random_baseline_seed=_R13_RANDOM_BASELINE_SEED,
    report_splits=_VALIDITY_REPORT_SPLITS,
    gate_splits=_VALIDITY_GATE_SPLITS,
    audit_split=_VALIDITY_AUDIT_SPLIT,
    baseline_order=_VALIDITY_BASELINE_ORDER,
    critical_baselines=_CRITICAL_SHORTCUT_BASELINES,
    shortcut_upper_bounds=(
        ShortcutUpperBoundRule(
            baseline_name="never_update",
            max_accuracy=_SHORTCUT_ACCURACY_CAP,
            splits=_VALIDITY_GATE_SPLITS,
        ),
        ShortcutUpperBoundRule(
            baseline_name="last_evidence",
            max_accuracy=_SHORTCUT_ACCURACY_CAP,
            splits=_VALIDITY_GATE_SPLITS,
        ),
        ShortcutUpperBoundRule(
            baseline_name="physics_prior",
            max_accuracy=_SHORTCUT_ACCURACY_CAP,
            splits=_VALIDITY_GATE_SPLITS,
        ),
        ShortcutUpperBoundRule(
            baseline_name="template_position",
            max_accuracy=_SHORTCUT_ACCURACY_CAP,
            splits=_VALIDITY_GATE_SPLITS,
        ),
    ),
    dominance_rule=DominantHeuristicRule(
        baseline_names=_CRITICAL_SHORTCUT_BASELINES,
        max_accuracy=_SHORTCUT_ACCURACY_CAP,
        splits=_VALIDITY_GATE_SPLITS,
    ),
    subset_separation_rule=SubsetSeparationRule(
        baseline_names=_CRITICAL_SHORTCUT_BASELINES,
        split_name=_VALIDITY_AUDIT_SPLIT,
        min_template_gap=_MIN_TEMPLATE_GAP,
        min_difficulty_gap=_MIN_DIFFICULTY_GAP,
    ),
)


def run_benchmark_validity_report(
    episodes_by_split: Mapping[str, Iterable[Episode]] | None = None,
    *,
    gate: ValidityGateConfig = R13_VALIDITY_GATE,
) -> BenchmarkValidityReport:
    return validate_benchmark_validity(episodes_by_split=episodes_by_split, gate=gate)


def evaluate_benchmark_validity_gate(
    report: BenchmarkValidityReport,
    *,
    gate: ValidityGateConfig = R13_VALIDITY_GATE,
) -> tuple[ValidityGateCheck, ...]:
    baseline_map = {
        summary.baseline_name: summary for summary in report.baseline_summaries
    }
    checks = [
        _evaluate_shortcut_upper_bound(rule, baseline_map)
        for rule in gate.shortcut_upper_bounds
    ]
    checks.append(_evaluate_dominance_rule(gate.dominance_rule, baseline_map))
    checks.append(_evaluate_subset_separation_rule(gate.subset_separation_rule, baseline_map))
    return tuple(checks)


def serialize_benchmark_validity_report(
    report: BenchmarkValidityReport,
) -> dict[str, object]:
    return {
        "release_id": report.release_id,
        "random_baseline_seed": report.random_baseline_seed,
        "report_splits": list(report.report_splits),
        "gate_splits": list(report.gate_splits),
        "audit_split": report.audit_split,
        "split_episode_counts": [
            {"split": split_name, "episode_count": episode_count}
            for split_name, episode_count in report.split_episode_counts
        ],
        "difficulty_labels_present": list(report.difficulty_labels_present),
        "difficulty_labels_missing": list(report.difficulty_labels_missing),
        "critical_baselines": list(report.critical_baselines),
        "baseline_summaries": [
            {
                "baseline_name": summary.baseline_name,
                "overall_accuracy": summary.overall_accuracy,
                "by_split": [
                    {
                        "split_name": split_summary.split_name,
                        "accuracy": split_summary.accuracy,
                        "by_template": [
                            {"label": label, "accuracy": accuracy}
                            for label, accuracy in split_summary.by_template
                        ],
                        "by_template_family": [
                            {"label": label, "accuracy": accuracy}
                            for label, accuracy in split_summary.by_template_family
                        ],
                        "by_difficulty": [
                            {"label": label, "accuracy": accuracy}
                            for label, accuracy in split_summary.by_difficulty
                        ],
                    }
                    for split_summary in summary.by_split
                ],
                "by_template": [
                    {"label": label, "accuracy": accuracy}
                    for label, accuracy in summary.by_template
                ],
                "by_template_family": [
                    {"label": label, "accuracy": accuracy}
                    for label, accuracy in summary.by_template_family
                ],
                "by_difficulty": [
                    {"label": label, "accuracy": accuracy}
                    for label, accuracy in summary.by_difficulty
                ],
            }
            for summary in report.baseline_summaries
        ],
        "checks": [
            {
                "code": check.code,
                "passed": check.passed,
                "message": check.message,
            }
            for check in report.checks
        ],
        "passed": report.passed,
        "comparison_summary": report.comparison_summary,
        "validity_note": report.validity_note,
        "limitations": list(report.limitations),
    }


def validate_benchmark_validity(
    episodes_by_split: Mapping[str, Iterable[Episode]]
    | tuple[tuple[str, tuple[Episode, ...]], ...]
    | None = None,
    *,
    gate: ValidityGateConfig = R13_VALIDITY_GATE,
    report: BenchmarkValidityReport | None = None,
) -> BenchmarkValidityReport:
    base_report = (
        _coerce_benchmark_validity_report(report)
        if report is not None
        else _build_base_benchmark_validity_report(episodes_by_split, gate=gate)
    )
    checks = evaluate_benchmark_validity_gate(base_report, gate=gate)
    passed = all(check.passed for check in checks)
    return BenchmarkValidityReport(
        release_id=base_report.release_id,
        random_baseline_seed=base_report.random_baseline_seed,
        report_splits=base_report.report_splits,
        gate_splits=base_report.gate_splits,
        audit_split=base_report.audit_split,
        split_episode_counts=base_report.split_episode_counts,
        difficulty_labels_present=base_report.difficulty_labels_present,
        difficulty_labels_missing=base_report.difficulty_labels_missing,
        critical_baselines=base_report.critical_baselines,
        baseline_summaries=base_report.baseline_summaries,
        checks=checks,
        passed=passed,
        comparison_summary=_build_gate_comparison_summary(
            base_report,
            checks=checks,
            gate=gate,
        ),
        validity_note=_build_validity_note(
            base_report,
            checks=checks,
            passed=passed,
            gate=gate,
        ),
        limitations=base_report.limitations,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_base_benchmark_validity_report(
    episodes_by_split: Mapping[str, Iterable[Episode]]
    | tuple[tuple[str, tuple[Episode, ...]], ...]
    | None,
    *,
    gate: ValidityGateConfig,
) -> BenchmarkValidityReport:
    normalized_episodes_by_split = _normalize_validity_episodes_by_split(
        episodes_by_split,
        gate=gate,
    )
    baseline_fns = _build_validity_baselines(gate.random_baseline_seed)
    split_audit_source_maps: dict[str, dict[str, AuditSourceSummary]] = {}
    split_episode_counts = tuple(
        (split_name, len(split_episodes))
        for split_name, split_episodes in normalized_episodes_by_split
    )

    for split_name, split_episodes in normalized_episodes_by_split:
        split_runs = run_baselines(
            split_episodes,
            tuple(
                (baseline_name, baseline_fns[baseline_name])
                for baseline_name in gate.baseline_order
            ),
        )
        split_report = run_audit(
            split_episodes,
            tuple(AuditSource.from_baseline_run(run) for run in split_runs),
        )
        split_audit_source_maps[split_name] = {
            summary.name: summary for summary in split_report.source_summaries
        }

    all_episodes = tuple(
        episode
        for _, split_episodes in normalized_episodes_by_split
        for episode in split_episodes
    )
    overall_runs = run_baselines(
        all_episodes,
        tuple(
            (baseline_name, baseline_fns[baseline_name])
            for baseline_name in gate.baseline_order
        ),
    )
    overall_report = run_audit(
        all_episodes,
        tuple(AuditSource.from_baseline_run(run) for run in overall_runs),
    )
    overall_source_map = {
        summary.name: summary for summary in overall_report.source_summaries
    }
    baseline_summaries = tuple(
        _build_baseline_accuracy_summary(
            baseline_name=baseline_name,
            overall_summary=overall_source_map[baseline_name],
            split_source_maps=split_audit_source_maps,
            split_order=gate.report_splits,
        )
        for baseline_name in gate.baseline_order
    )
    return BenchmarkValidityReport(
        release_id=gate.release_id,
        random_baseline_seed=gate.random_baseline_seed,
        report_splits=gate.report_splits,
        gate_splits=gate.gate_splits,
        audit_split=gate.audit_split,
        split_episode_counts=split_episode_counts,
        difficulty_labels_present=overall_report.difficulty_labels_present,
        difficulty_labels_missing=overall_report.difficulty_labels_missing,
        critical_baselines=gate.critical_baselines,
        baseline_summaries=baseline_summaries,
        checks=(),
        passed=False,
        comparison_summary=_build_baseline_comparison_summary(
            baseline_summaries,
            critical_baselines=gate.critical_baselines,
        ),
        validity_note="",
        limitations=overall_report.limitations,
    )


def _build_validity_baselines(
    random_seed: int,
) -> dict[str, BaselineFn]:
    return {
        "random": lambda episode: random_baseline(episode, seed=random_seed),
        "never_update": never_update_baseline,
        "last_evidence": last_evidence_baseline,
        "physics_prior": physics_prior_baseline,
        "template_position": template_position_baseline,
    }


def _normalize_validity_episodes_by_split(
    episodes_by_split: Mapping[str, Iterable[Episode]]
    | tuple[tuple[str, tuple[Episode, ...]], ...]
    | None,
    *,
    gate: ValidityGateConfig,
) -> tuple[tuple[str, tuple[Episode, ...]], ...]:
    if episodes_by_split is None:
        from core.splits import load_frozen_split

        return tuple(
            (
                split_name,
                tuple(record.episode for record in load_frozen_split(split_name)),
            )
            for split_name in gate.report_splits
        )

    if isinstance(episodes_by_split, tuple):
        provided = dict(episodes_by_split)
    else:
        provided = dict(episodes_by_split)  # type: ignore[arg-type]

    normalized: list[tuple[str, tuple[Episode, ...]]] = []
    for split_name in gate.report_splits:
        if split_name not in provided:
            raise ValueError(f"missing benchmark validity split: {split_name}")
        split_episodes = tuple(provided[split_name])
        if not all(isinstance(episode, Episode) for episode in split_episodes):
            raise TypeError("episodes_by_split must contain Episode values")
        normalized.append((split_name, split_episodes))
    return tuple(normalized)


def _build_baseline_accuracy_summary(
    *,
    baseline_name: str,
    overall_summary: AuditSourceSummary,
    split_source_maps: Mapping[str, Mapping[str, AuditSourceSummary]],
    split_order: tuple[str, ...],
) -> BaselineAccuracySummary:
    return BaselineAccuracySummary(
        baseline_name=baseline_name,
        overall_accuracy=overall_summary.overall.accuracy,
        by_split=tuple(
            SplitBaselineAccuracySummary(
                split_name=split_name,
                accuracy=split_source_maps[split_name][baseline_name].overall.accuracy,
                by_template=tuple(
                    (label, summary.accuracy)
                    for label, summary in split_source_maps[split_name][baseline_name].by_template
                ),
                by_template_family=tuple(
                    (label, summary.accuracy)
                    for label, summary in split_source_maps[split_name][baseline_name].by_template_family
                ),
                by_difficulty=tuple(
                    (label, summary.accuracy)
                    for label, summary in split_source_maps[split_name][baseline_name].by_difficulty
                ),
            )
            for split_name in split_order
        ),
        by_template=tuple(
            (label, summary.accuracy) for label, summary in overall_summary.by_template
        ),
        by_template_family=tuple(
            (label, summary.accuracy)
            for label, summary in overall_summary.by_template_family
        ),
        by_difficulty=tuple(
            (label, summary.accuracy) for label, summary in overall_summary.by_difficulty
        ),
    )


def _evaluate_shortcut_upper_bound(
    rule: ShortcutUpperBoundRule,
    baseline_map: Mapping[str, BaselineAccuracySummary],
) -> ValidityGateCheck:
    summary = baseline_map[rule.baseline_name]
    split_scores = tuple(
        (split_summary.split_name, split_summary.accuracy)
        for split_summary in summary.by_split
        if split_summary.split_name in rule.splits
    )
    worst_split_name, worst_accuracy = max(split_scores, key=lambda item: item[1])
    passed = all(accuracy <= rule.max_accuracy for _, accuracy in split_scores)
    return ValidityGateCheck(
        code=f"{rule.baseline_name}_bounded",
        passed=passed,
        message=(
            f"{rule.baseline_name} must stay at or below {rule.max_accuracy:.6f} "
            f"on {', '.join(rule.splits)}; observed worst split "
            f"{worst_split_name}={worst_accuracy:.6f}."
        ),
    )


def _evaluate_dominance_rule(
    rule: DominantHeuristicRule,
    baseline_map: Mapping[str, BaselineAccuracySummary],
) -> ValidityGateCheck:
    candidates = [
        (summary.baseline_name, split_summary.split_name, split_summary.accuracy)
        for summary in baseline_map.values()
        if summary.baseline_name in rule.baseline_names
        for split_summary in summary.by_split
        if split_summary.split_name in rule.splits
    ]
    dominant_baseline, dominant_split, dominant_accuracy = max(
        candidates,
        key=lambda item: (item[2], item[0], item[1]),
    )
    return ValidityGateCheck(
        code="no_dominant_trivial_heuristic",
        passed=dominant_accuracy <= rule.max_accuracy,
        message=(
            "No critical shortcut baseline may dominate the evaluation splits above "
            f"{rule.max_accuracy:.6f}; observed {dominant_baseline}="
            f"{dominant_accuracy:.6f} on {dominant_split}."
        ),
    )


def _evaluate_subset_separation_rule(
    rule: SubsetSeparationRule,
    baseline_map: Mapping[str, BaselineAccuracySummary],
) -> ValidityGateCheck:
    template_candidates: list[tuple[str, float]] = []
    difficulty_candidates: list[tuple[str, float]] = []

    for baseline_name in rule.baseline_names:
        split_summary = _find_split_summary(
            baseline_map[baseline_name],
            split_name=rule.split_name,
        )
        template_candidates.append(
            (baseline_name, _score_gap(split_summary.by_template))
        )
        difficulty_candidates.append(
            (baseline_name, _score_gap(split_summary.by_difficulty))
        )

    best_template_baseline, best_template_gap = max(
        template_candidates,
        key=lambda item: (item[1], item[0]),
    )
    best_difficulty_baseline, best_difficulty_gap = max(
        difficulty_candidates,
        key=lambda item: (item[1], item[0]),
    )
    passed = (
        best_template_gap >= rule.min_template_gap
        and best_difficulty_gap >= rule.min_difficulty_gap
    )
    return ValidityGateCheck(
        code="heuristic_subset_separation",
        passed=passed,
        message=(
            f"Critical baselines must show at least {rule.min_template_gap:.6f} "
            f"template gap and {rule.min_difficulty_gap:.6f} difficulty gap on "
            f"{rule.split_name}; observed best template gap "
            f"{best_template_baseline}={best_template_gap:.6f}, best difficulty gap "
            f"{best_difficulty_baseline}={best_difficulty_gap:.6f}."
        ),
    )


def _build_baseline_comparison_summary(
    baseline_summaries: tuple[BaselineAccuracySummary, ...],
    *,
    critical_baselines: tuple[str, ...],
) -> str:
    overall_best = max(
        baseline_summaries,
        key=lambda summary: (summary.overall_accuracy, summary.baseline_name),
    )
    split_candidates = [
        (summary.baseline_name, split_summary.split_name, split_summary.accuracy)
        for summary in baseline_summaries
        for split_summary in summary.by_split
        if summary.baseline_name in critical_baselines
    ]
    critical_best_name, critical_best_split, critical_best_accuracy = max(
        split_candidates,
        key=lambda item: (item[2], item[0], item[1]),
    )
    return (
        f"Best overall baseline is {overall_best.baseline_name}="
        f"{overall_best.overall_accuracy:.6f}; strongest critical shortcut result is "
        f"{critical_best_name}={critical_best_accuracy:.6f} on {critical_best_split}."
    )


def _build_gate_comparison_summary(
    report: BenchmarkValidityReport,
    *,
    checks: tuple[ValidityGateCheck, ...],
    gate: ValidityGateConfig,
) -> str:
    failed_checks = tuple(check for check in checks if not check.passed)
    if not failed_checks:
        best_baseline_name, best_split_name, best_accuracy = max(
            (
                (summary.baseline_name, split_summary.split_name, split_summary.accuracy)
                for summary in report.baseline_summaries
                if summary.baseline_name in gate.critical_baselines
                for split_summary in summary.by_split
                if split_summary.split_name in gate.gate_splits
            ),
            key=lambda item: (item[2], item[0], item[1]),
        )
        return (
            "The repaired benchmark clears the R13 anti-shortcut gate: "
            f"{best_baseline_name} peaks at {best_accuracy:.6f} on {best_split_name}."
        )
    return (
        "The repaired benchmark still collapses to shortcut behavior under the R13 "
        f"gate: {failed_checks[0].message}"
    )


def _build_validity_note(
    report: BenchmarkValidityReport,
    *,
    checks: tuple[ValidityGateCheck, ...],
    passed: bool,
    gate: ValidityGateConfig,
) -> str:
    shortcut_cap = gate.dominance_rule.max_accuracy
    subset_rule = gate.subset_separation_rule
    failed_checks = tuple(check for check in checks if not check.passed)
    blocking_clause = (
        "No release blocker remains."
        if passed
        else "Blocked by " + "; ".join(check.message for check in failed_checks)
    )
    return (
        "Critical shortcut threats: "
        + ", ".join(gate.critical_baselines)
        + ". Pass requires each named shortcut heuristic to stay at or below "
        + f"{shortcut_cap:.6f} on {', '.join(gate.gate_splits)}, the best "
        + "critical heuristic to stay under the same cap, and the audit split to show "
        + f"at least {subset_rule.min_template_gap:.6f} template gap plus "
        + f"{subset_rule.min_difficulty_gap:.6f} difficulty gap across emitted labels. "
        + f"Current status: {'PASS' if passed else 'FAIL'}. "
        + blocking_clause
        + (
            " Some supplied episodes do not cover the full emitted difficulty set."
            if "hard" in report.difficulty_labels_missing
            else ""
        )
    )


def _find_split_summary(
    summary: BaselineAccuracySummary,
    *,
    split_name: str,
) -> SplitBaselineAccuracySummary:
    for split_summary in summary.by_split:
        if split_summary.split_name == split_name:
            return split_summary
    raise ValueError(f"missing split summary for {summary.baseline_name}: {split_name}")


def _score_gap(scores: tuple[tuple[str, float], ...]) -> float:
    if len(scores) <= 1:
        return 0.0
    values = tuple(score for _, score in scores)
    return max(values) - min(values)


def _coerce_benchmark_validity_report(
    report: BenchmarkValidityReport,
) -> BenchmarkValidityReport:
    if not isinstance(report, BenchmarkValidityReport):
        raise TypeError("report must be a BenchmarkValidityReport")
    return report
