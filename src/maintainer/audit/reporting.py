from __future__ import annotations

from maintainer.audit.core import (
    AuditSliceSummary,
    BaselineComparisonSummary,
    ModeComparisonSummary,
    ReleaseAuditReport,
    ReleaseAuditSourceSummary,
)

__all__ = [
    "serialize_release_r15_reaudit_report",
]


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
                "by_split": [
                    {"label": label, **_serialize_mode_comparison(slice_summary)}
                    for label, slice_summary in summary.by_split
                ],
                "by_template": [
                    {"label": label, **_serialize_mode_comparison(slice_summary)}
                    for label, slice_summary in summary.by_template
                ],
                "by_template_family": [
                    {"label": label, **_serialize_mode_comparison(slice_summary)}
                    for label, slice_summary in summary.by_template_family
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
        "by_template_family": [
            {"label": label, **_serialize_slice_summary(slice_summary)}
            for label, slice_summary in summary.by_template_family
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
        "exact_agreement_count": summary.exact_agreement_count,
        "exact_agreement_denominator": summary.exact_agreement_denominator,
        "exact_agreement_rate": summary.exact_agreement_rate,
    }
