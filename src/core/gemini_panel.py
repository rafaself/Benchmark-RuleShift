from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import sys

from core.audit import (
    AuditSource,
    MatchedModeComparisonSummary,
    ModeComparisonSummary,
    ReleaseAuditReport,
    ReleaseAuditSourceSummary,
    run_release_r15_reaudit,
)
from core.model_execution import ModelMode, ModelRunConfig
from core.model_runner import (
    BenchmarkModeRunRow,
    BenchmarkRunResult,
    run_model_benchmark,
)
from core.parser import ParseStatus
from core.providers.gemini import GeminiAdapter
from core.providers.registry import (
    ProviderExecutionSurface,
    get_provider_spec,
    resolve_provider_model_name,
)
from core.report_outputs import (
    build_latest_report_path,
    current_report_timestamp,
    write_text_with_timestamped_snapshot,
)
from core.splits import PARTITIONS, load_frozen_split
from tasks.iron_find_electric.baselines import (
    last_evidence_baseline,
    never_update_baseline,
)
from tasks.iron_find_electric.schema import Episode

__all__ = [
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_GEMINI_FIRST_PANEL_REPORT_PATH",
    "default_gemini_first_panel_report_path",
    "GeminiFirstPanelArtifacts",
    "run_gemini_first_panel",
    "render_gemini_first_panel_markdown",
]

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
_GEMINI_PROVIDER_SPEC = get_provider_spec("gemini")
DEFAULT_GEMINI_MODEL = _GEMINI_PROVIDER_SPEC.default_benchmark_model or DEFAULT_GEMINI_MODEL
DEFAULT_GEMINI_FIRST_PANEL_REPORT_PATH = build_latest_report_path(
    "live",
    "gemini-first-panel",
    "binary-only",
    filename="report.md",
)
_DEFAULT_PANEL_MODES: tuple[ModelMode, ...] = (ModelMode.BINARY,)
_DEFAULT_PANEL_CONFIG = ModelRunConfig(
    timeout_seconds=60.0,
    temperature=0.0,
    thinking_budget=0,
)
_BASELINE_ORDER: tuple[str, ...] = (
    "random",
    "never_update",
    "last_evidence",
    "physics_prior",
    "template_position",
)
_TASK_MODE_LABELS = {
    ModelMode.BINARY: "Binary",
    ModelMode.NARRATIVE: "Narrative",
}


@dataclass(frozen=True, slots=True)
class GeminiFirstPanelArtifacts:
    provider_name: str
    model_name: str
    prompt_modes: tuple[ModelMode, ...]
    release_report: ReleaseAuditReport
    report_markdown: str
    report_path: Path
    artifact_payload: dict[str, object] | None = None
    artifact_path: Path | None = None
    snapshot_report_path: Path | None = None
    snapshot_artifact_path: Path | None = None


def run_gemini_first_panel(
    *,
    model_name: str = DEFAULT_GEMINI_MODEL,
    report_path: Path | None = None,
    modes: tuple[ModelMode, ...] = _DEFAULT_PANEL_MODES,
    config: ModelRunConfig = _DEFAULT_PANEL_CONFIG,
    adapter: GeminiAdapter | None = None,
) -> GeminiFirstPanelArtifacts:
    normalized_modes = tuple(ModelMode(mode) for mode in modes)
    if not normalized_modes:
        raise ValueError("modes must not be empty")
    if len(set(normalized_modes)) != len(normalized_modes):
        raise ValueError("modes must not contain duplicates")

    resolved_model_name = resolve_provider_model_name(
        "gemini",
        surface=ProviderExecutionSurface.LOCAL_BENCHMARK,
        model_name=model_name,
    )
    active_adapter = GeminiAdapter.from_env() if adapter is None else adapter
    episodes_by_split: dict[str, tuple[Episode, ...]] = {}
    benchmark_results_by_split: dict[str, BenchmarkRunResult] = {}
    model_sources_by_split: dict[str, tuple[AuditSource, ...]] = {}
    provider_name = _GEMINI_PROVIDER_SPEC.provider_name

    for split_name in PARTITIONS:
        episodes = tuple(record.episode for record in load_frozen_split(split_name))
        benchmark_result = run_model_benchmark(
            episodes,
            active_adapter,
            provider_name=provider_name,
            model_name=resolved_model_name,
            config=config,
            modes=normalized_modes,
            progress_callback=_build_panel_progress_callback(split_name),
        )
        episodes_by_split[split_name] = episodes
        benchmark_results_by_split[split_name] = benchmark_result
        model_sources_by_split[split_name] = tuple(
            AuditSource.from_parsed_predictions(
                f"{resolved_model_name} {_TASK_MODE_LABELS[mode_result.mode]}",
                tuple(row.parsed_prediction for row in mode_result.rows),
                task_mode=_TASK_MODE_LABELS[mode_result.mode],
                source_family=resolved_model_name,
                is_real_model=True,
            )
            for mode_result in benchmark_result.mode_results
        )

    release_report = run_release_r15_reaudit(
        episodes_by_split=episodes_by_split,
        model_sources_by_split=model_sources_by_split,
        release_id="R18",
    )
    artifact_payload = _build_panel_artifact(
        provider_name=provider_name,
        model_name=resolved_model_name,
        prompt_modes=normalized_modes,
        release_report=release_report,
        episodes_by_split=episodes_by_split,
        benchmark_results_by_split=benchmark_results_by_split,
    )
    report_markdown = render_gemini_first_panel_markdown(
        release_report,
        model_name=resolved_model_name,
        provider_name=provider_name,
        prompt_modes=normalized_modes,
        artifact_payload=artifact_payload,
    )
    resolved_report_path = (
        DEFAULT_GEMINI_FIRST_PANEL_REPORT_PATH if report_path is None else report_path
    )
    resolved_artifact_path = _artifact_path_for_report(resolved_report_path)
    report_timestamp = current_report_timestamp()
    _, snapshot_report_path = write_text_with_timestamped_snapshot(
        resolved_report_path,
        report_markdown,
        timestamp=report_timestamp,
    )
    _, snapshot_artifact_path = write_text_with_timestamped_snapshot(
        resolved_artifact_path,
        json.dumps(artifact_payload, indent=2) + "\n",
        timestamp=report_timestamp,
    )

    return GeminiFirstPanelArtifacts(
        provider_name=provider_name,
        model_name=resolved_model_name,
        prompt_modes=normalized_modes,
        release_report=release_report,
        report_markdown=report_markdown,
        report_path=resolved_report_path,
        artifact_payload=artifact_payload,
        artifact_path=resolved_artifact_path,
        snapshot_report_path=snapshot_report_path,
        snapshot_artifact_path=snapshot_artifact_path,
    )


def default_gemini_first_panel_report_path(*, include_narrative: bool) -> Path:
    target = "binary-vs-narrative" if include_narrative else "binary-only"
    return build_latest_report_path(
        "live",
        "gemini-first-panel",
        target,
        filename="report.md",
    )


def _build_panel_progress_callback(split_name: str):
    def _report_progress(mode: ModelMode, index: int, total: int, episode_id: str) -> None:
        print(
            f"[gemini-first-panel] split={split_name} mode={mode.value} "
            f"episode={index}/{total} id={episode_id}",
            file=sys.stderr,
            flush=True,
        )

    return _report_progress


def render_gemini_first_panel_markdown(
    release_report: ReleaseAuditReport,
    *,
    model_name: str,
    provider_name: str,
    prompt_modes: tuple[ModelMode, ...],
    artifact_payload: dict[str, object] | None = None,
) -> str:
    model_summaries = tuple(
        summary
        for summary in release_report.model_summaries
        if summary.source_family == model_name and summary.is_real_model
    )
    if not model_summaries:
        raise ValueError(f"no real-model summaries found for {model_name!r}")

    lines = [
        "# Gemini First Panel Report",
        "",
        f"- Release: {release_report.release_id}",
        f"- Provider: {provider_name}",
        f"- Model: {model_name}",
        f"- Prompt modes run: {', '.join(mode.value for mode in prompt_modes)}",
        f"- Covered splits: {', '.join(split_name for split_name, _ in release_report.split_episode_counts)}",
        "",
    ]

    primary_summary = _pick_primary_summary(model_summaries, prompt_modes)
    binary_summary = _pick_mode_summary(model_summaries, task_mode="Binary")
    narrative_summary = _pick_mode_summary(model_summaries, task_mode="Narrative")
    matched_mode_comparison = _pick_matched_mode_comparison(
        release_report,
        source_family=model_name,
    )
    lines.extend(
        [
            "## Headline",
            "",
            f"- Binary-only headline metric: {primary_summary.name} = {primary_summary.overall.accuracy:.6f}",
            f"- Binary parse-valid rate: {primary_summary.overall.parse_valid_rate:.6f}",
            f"- Best baseline: {release_report.baseline_comparison.best_baseline_name} "
            f"({(release_report.baseline_comparison.best_baseline_accuracy or 0.0):.6f})",
            "",
            "| Source | Accuracy | Gap vs model |",
            "| --- | ---: | ---: |",
            f"| {primary_summary.name} | {primary_summary.overall.accuracy:.6f} | 0.000000 |",
        ]
    )
    for baseline_summary in release_report.baseline_summaries:
        lines.append(
            f"| {baseline_summary.name} | {baseline_summary.overall.accuracy:.6f} | "
            f"{primary_summary.overall.accuracy - baseline_summary.overall.accuracy:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Headline By Split",
            "",
            "| Split | Model | random | never-update | last-evidence | physics-prior | template-position |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    model_by_split = dict(primary_summary.by_split)
    baselines_by_name = {summary.name: summary for summary in release_report.baseline_summaries}
    for split_name, _episode_count in release_report.split_episode_counts:
        lines.append(
            f"| {split_name} | {model_by_split[split_name].accuracy:.6f} | "
            f"{dict(baselines_by_name['random'].by_split)[split_name].accuracy:.6f} | "
            f"{dict(baselines_by_name['never_update'].by_split)[split_name].accuracy:.6f} | "
            f"{dict(baselines_by_name['last_evidence'].by_split)[split_name].accuracy:.6f} | "
            f"{dict(baselines_by_name['physics_prior'].by_split)[split_name].accuracy:.6f} | "
            f"{dict(baselines_by_name['template_position'].by_split)[split_name].accuracy:.6f} |"
        )

    if matched_mode_comparison is not None and binary_summary is not None and narrative_summary is not None:
        lines.extend(
            [
                "",
                "## Paired Robustness",
                "",
                "| Scope | Binary accuracy | Narrative accuracy | Delta | Binary parse-valid | Narrative parse-valid |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
                _render_mode_comparison_row("overall", matched_mode_comparison.overall),
            ]
        )
        binary_by_split = dict(binary_summary.by_split)
        narrative_by_split = dict(narrative_summary.by_split)
        for split_name, _episode_count in release_report.split_episode_counts:
            if split_name not in binary_by_split or split_name not in narrative_by_split:
                continue
            lines.append(
                _render_mode_comparison_row(
                    split_name,
                    _build_mode_comparison_from_split_summaries(
                        binary_by_split[split_name],
                        narrative_by_split[split_name],
                    ),
                )
            )

        lines.extend(
            [
                "",
                "## Diagnostic Slices",
                "",
                "| Slice type | Label | Binary accuracy | Narrative accuracy | Delta | Binary parse-valid | Narrative parse-valid |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for label, comparison in matched_mode_comparison.by_template:
            lines.append(_render_mode_slice_row("template", label, comparison))
        for label, comparison in matched_mode_comparison.by_difficulty:
            lines.append(_render_mode_slice_row("difficulty", label, comparison))

        if artifact_payload is not None:
            taxonomy_rows = tuple(
                artifact_payload.get("failure_taxonomy", ())
            )
            if taxonomy_rows:
                lines.extend(
                    [
                        "",
                        "## Failure Taxonomy",
                        "",
                        "| Scope | Mode | Provider/runtime error rate | Parse/format failure rate | Adaptation failure rate | Possible old-rule persistence rate | Possible recency overshoot rate |",
                        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
                    ]
                )
                for row in taxonomy_rows:
                    if not isinstance(row, dict):
                        continue
                    lines.append(
                        "| {scope} | {mode} | {runtime_error_rate:.6f} | {parse_failure_rate:.6f} | "
                        "{adaptation_failure_rate:.6f} | {possible_old_rule_persistence_rate:.6f} | "
                        "{possible_recency_overshoot_rate:.6f} |".format(
                            scope=row["scope"],
                            mode=row["mode"],
                            runtime_error_rate=row["runtime_error_rate"],
                            parse_failure_rate=row["parse_failure_rate"],
                            adaptation_failure_rate=row["adaptation_failure_rate"],
                            possible_old_rule_persistence_rate=row[
                                "possible_old_rule_persistence_rate"
                            ],
                            possible_recency_overshoot_rate=row[
                                "possible_recency_overshoot_rate"
                            ],
                        )
                    )
                lines.extend(
                    [
                        "",
                        "Taxonomy rates are episode-level over scored outputs. Persistence and recency tags are diagnostic-only exact-match comparisons against `never_update` and `last_evidence`.",
                    ]
                )
            lines.extend(_build_live_execution_notes(artifact_payload))
    elif len(model_summaries) > 1:
        lines.extend(
            [
                "",
                "## Additional Prompt Modes",
                "",
                "| Source | Accuracy | Parse-valid rate | Covered splits |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        for summary in model_summaries:
            if summary.name == primary_summary.name:
                continue
            lines.append(
                f"| {summary.name} | {summary.overall.accuracy:.6f} | "
                f"{summary.overall.parse_valid_rate:.6f} | {', '.join(summary.covered_splits)} |"
            )

    lines.extend(
        [
            "",
            "## Binary Diagnostic Slices",
            "",
            "| Template | Accuracy | Parse-valid rate |",
            "| --- | ---: | ---: |",
        ]
    )
    for label, slice_summary in primary_summary.by_template:
        lines.append(
            f"| {label} | {slice_summary.accuracy:.6f} | {slice_summary.parse_valid_rate:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Binary Difficulty Slices",
            "",
            "| Difficulty | Accuracy | Parse-valid rate |",
            "| --- | ---: | ---: |",
        ]
    )
    for label, slice_summary in primary_summary.by_difficulty:
        lines.append(
            f"| {label} | {slice_summary.accuracy:.6f} | {slice_summary.parse_valid_rate:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `hard` remains reserved and is not emitted in the current frozen repaired benchmark, so no hard slice is reported.",
        ]
    )
    if ModelMode.NARRATIVE not in prompt_modes:
        lines.append(
            "- Narrative mode was not run in this first real-model panel, so Binary vs Narrative comparison is unavailable."
        )
    for limitation in release_report.limitations:
        if "hard slice omitted" in limitation:
            continue
        if "No matched Binary/Narrative model runs supplied" in limitation and ModelMode.NARRATIVE not in prompt_modes:
            continue
        lines.append(f"- {limitation}")

    return "\n".join(lines) + "\n"


def _pick_primary_summary(
    model_summaries: tuple[ReleaseAuditSourceSummary, ...],
    prompt_modes: tuple[ModelMode, ...],
) -> ReleaseAuditSourceSummary:
    if ModelMode.BINARY in prompt_modes:
        for summary in model_summaries:
            if summary.task_mode == "Binary":
                return summary
    return model_summaries[0]


def _artifact_path_for_report(report_path: Path) -> Path:
    if report_path.name == "report.md" and report_path.parent.name == "latest":
        return report_path.with_name("artifact.json")
    return report_path.with_suffix(".json")


def _build_panel_artifact(
    *,
    provider_name: str,
    model_name: str,
    prompt_modes: tuple[ModelMode, ...],
    release_report: ReleaseAuditReport,
    episodes_by_split: dict[str, tuple[Episode, ...]],
    benchmark_results_by_split: dict[str, BenchmarkRunResult],
) -> dict[str, object]:
    split_payloads = [
        _build_split_artifact(
            split_name=split_name,
            episodes=episodes_by_split[split_name],
            benchmark_result=benchmark_results_by_split[split_name],
            prompt_modes=prompt_modes,
        )
        for split_name, _episode_count in release_report.split_episode_counts
    ]
    return {
        "release_id": release_report.release_id,
        "provider_name": provider_name,
        "model_name": model_name,
        "prompt_modes": [mode.value for mode in prompt_modes],
        "splits": split_payloads,
        "failure_taxonomy": _build_failure_taxonomy(split_payloads, prompt_modes),
    }


def _build_split_artifact(
    *,
    split_name: str,
    episodes: tuple[Episode, ...],
    benchmark_result: BenchmarkRunResult,
    prompt_modes: tuple[ModelMode, ...],
) -> dict[str, object]:
    mode_rows = {mode_result.mode: mode_result.rows for mode_result in benchmark_result.mode_results}
    for mode in prompt_modes:
        if mode not in mode_rows:
            continue
        if len(mode_rows[mode]) != len(episodes):
            raise ValueError(
                f"{mode.value} rows must contain exactly {len(episodes)} entries for split {split_name!r}"
            )

    rows: list[dict[str, object]] = []
    for index, episode in enumerate(episodes):
        mode_payloads: dict[str, object] = {}
        paired_episode_id: str | None = None
        paired_targets: tuple[object, ...] | None = None
        for mode in prompt_modes:
            if mode not in mode_rows:
                continue
            row = mode_rows[mode][index]
            if row.episode_id != episode.episode_id:
                raise ValueError(
                    f"{mode.value} row order must match frozen episode order on split {split_name!r}"
                )
            if tuple(row.target) != tuple(episode.probe_targets):
                raise ValueError(
                    f"{mode.value} targets must match frozen probe targets on split {split_name!r}"
                )
            if paired_episode_id is None:
                paired_episode_id = row.episode_id
            elif row.episode_id != paired_episode_id:
                raise ValueError(
                    f"paired Binary/Narrative rows must reference the same episode on split {split_name!r}"
                )
            if paired_targets is None:
                paired_targets = tuple(row.target)
            elif tuple(row.target) != paired_targets:
                raise ValueError(
                    f"paired Binary/Narrative rows must share the same probe targets on split {split_name!r}"
                )
            mode_payloads[mode.value] = _build_mode_row_payload(row=row, episode=episode)

        rows.append(
            {
                "episode_id": episode.episode_id,
                "template_id": episode.template_id.value,
                "difficulty": episode.difficulty.value,
                "transition": episode.transition.value,
                "probe_targets": [label.value for label in episode.probe_targets],
                "modes": mode_payloads,
            }
        )

    return {
        "split_name": split_name,
        "episode_count": len(episodes),
        "pairing_checks": {
            "same_episode_order": True,
            "same_probe_targets": True,
        },
        "rows": rows,
    }


def _build_mode_row_payload(
    *,
    row: BenchmarkModeRunRow,
    episode: Episode,
) -> dict[str, object]:
    predicted_labels = tuple(row.parsed_prediction.labels)
    correct_probe_count = _count_correct_labels(row=row)
    has_runtime_error = row.execution.raw_result.error_type is not None
    is_parse_failure = (
        not has_runtime_error and row.parsed_prediction.status is ParseStatus.INVALID
    )
    is_adaptation_failure = (
        row.parsed_prediction.status is ParseStatus.VALID
        and correct_probe_count < len(episode.probe_targets)
    )
    possible_old_rule_persistence = (
        is_adaptation_failure and predicted_labels == never_update_baseline(episode)
    )
    possible_recency_overshoot = (
        is_adaptation_failure and predicted_labels == last_evidence_baseline(episode)
    )
    if has_runtime_error:
        failure_bucket = "runtime_error"
    elif is_parse_failure:
        failure_bucket = "parse_failure"
    elif is_adaptation_failure:
        failure_bucket = "adaptation_failure"
    else:
        failure_bucket = "correct"

    return {
        "parse_status": row.parsed_prediction.status.value,
        "predicted_labels": [label.value for label in predicted_labels],
        "correct_probe_count": correct_probe_count,
        "failure_bucket": failure_bucket,
        "possible_old_rule_persistence": possible_old_rule_persistence,
        "possible_recency_overshoot": possible_recency_overshoot,
        "error_type": row.execution.raw_result.error_type,
        "error_message": row.execution.raw_result.error_message,
        "response_text": row.execution.raw_result.response_text,
        "finish_reason": row.execution.raw_result.finish_reason,
    }


def _count_correct_labels(*, row: BenchmarkModeRunRow) -> int:
    if row.parsed_prediction.status is not ParseStatus.VALID:
        return 0
    return sum(
        predicted_label is target_label
        for predicted_label, target_label in zip(row.parsed_prediction.labels, row.target)
    )


def _build_failure_taxonomy(
    split_payloads: list[dict[str, object]],
    prompt_modes: tuple[ModelMode, ...],
) -> list[dict[str, object]]:
    taxonomy_rows = _build_failure_taxonomy_for_scope(
        scope_name="overall",
        scoped_split_payloads=split_payloads,
        prompt_modes=prompt_modes,
    )
    for split_payload in split_payloads:
        taxonomy_rows.extend(
            _build_failure_taxonomy_for_scope(
                scope_name=str(split_payload["split_name"]),
                scoped_split_payloads=[split_payload],
                prompt_modes=prompt_modes,
            )
        )
    return taxonomy_rows


def _build_failure_taxonomy_for_scope(
    *,
    scope_name: str,
    scoped_split_payloads: list[dict[str, object]],
    prompt_modes: tuple[ModelMode, ...],
) -> list[dict[str, object]]:
    scope_rows: list[dict[str, object]] = []
    for mode in prompt_modes:
        mode_name = _TASK_MODE_LABELS[mode]
        mode_rows = [
            mode_payload
            for split_payload in scoped_split_payloads
            for row in split_payload["rows"]
            for mode_key, mode_payload in row["modes"].items()
            if mode_key == mode.value
        ]
        episode_count = len(mode_rows)
        if episode_count == 0:
            continue
        runtime_error_count = sum(
            mode_row["failure_bucket"] == "runtime_error" for mode_row in mode_rows
        )
        parse_failure_count = sum(
            mode_row["failure_bucket"] == "parse_failure" for mode_row in mode_rows
        )
        adaptation_failure_count = sum(
            mode_row["failure_bucket"] == "adaptation_failure" for mode_row in mode_rows
        )
        possible_old_rule_persistence_count = sum(
            bool(mode_row["possible_old_rule_persistence"]) for mode_row in mode_rows
        )
        possible_recency_overshoot_count = sum(
            bool(mode_row["possible_recency_overshoot"]) for mode_row in mode_rows
        )
        scope_rows.append(
            {
                "scope": scope_name,
                "mode": mode_name,
                "episode_count": episode_count,
                "runtime_error_rate": runtime_error_count / episode_count,
                "parse_failure_rate": parse_failure_count / episode_count,
                "adaptation_failure_rate": adaptation_failure_count / episode_count,
                "possible_old_rule_persistence_rate": (
                    possible_old_rule_persistence_count / episode_count
                ),
                "possible_recency_overshoot_rate": (
                    possible_recency_overshoot_count / episode_count
                ),
            }
        )
    return scope_rows


def _build_live_execution_notes(artifact_payload: dict[str, object]) -> list[str]:
    taxonomy_rows = tuple(artifact_payload.get("failure_taxonomy", ()))
    if not taxonomy_rows:
        return []

    overall_rows = [
        row
        for row in taxonomy_rows
        if isinstance(row, dict) and row.get("scope") == "overall"
    ]
    if not overall_rows:
        return []

    notes: list[str] = []
    if any(float(row.get("runtime_error_rate", 0.0)) > 0.0 for row in overall_rows):
        notes.extend(
            [
                "",
                "## Live Execution Review",
                "",
            ]
        )
        if any(float(row.get("runtime_error_rate", 0.0)) >= 1.0 for row in overall_rows):
            notes.append(
                "All outputs in at least one prompt mode failed at the provider/runtime stage, so this run is not interpretable as a robustness finding and requires a rerun."
            )
        else:
            notes.append(
                "Provider/runtime failures were observed in the live run. Review them separately from true parse/format failures before drawing benchmark conclusions."
            )
    return notes


def _pick_mode_summary(
    model_summaries: tuple[ReleaseAuditSourceSummary, ...],
    *,
    task_mode: str,
) -> ReleaseAuditSourceSummary | None:
    for summary in model_summaries:
        if summary.task_mode == task_mode:
            return summary
    return None


def _pick_matched_mode_comparison(
    release_report: ReleaseAuditReport,
    *,
    source_family: str,
) -> MatchedModeComparisonSummary | None:
    for comparison in release_report.matched_mode_comparisons:
        if comparison.source_family == source_family:
            return comparison
    return None


def _build_mode_comparison_from_split_summaries(
    binary_split_summary,
    narrative_split_summary,
) -> ModeComparisonSummary:
    return ModeComparisonSummary(
        binary_accuracy=binary_split_summary.accuracy,
        narrative_accuracy=narrative_split_summary.accuracy,
        accuracy_gap=binary_split_summary.accuracy - narrative_split_summary.accuracy,
        binary_parse_valid_rate=binary_split_summary.parse_valid_rate,
        narrative_parse_valid_rate=narrative_split_summary.parse_valid_rate,
        parse_valid_rate_gap=(
            binary_split_summary.parse_valid_rate
            - narrative_split_summary.parse_valid_rate
        ),
    )


def _render_mode_comparison_row(
    scope: str,
    comparison: ModeComparisonSummary,
) -> str:
    return (
        f"| {scope} | {comparison.binary_accuracy:.6f} | "
        f"{comparison.narrative_accuracy:.6f} | {comparison.accuracy_gap:.6f} | "
        f"{comparison.binary_parse_valid_rate:.6f} | "
        f"{comparison.narrative_parse_valid_rate:.6f} |"
    )


def _render_mode_slice_row(
    slice_type: str,
    label: str,
    comparison: ModeComparisonSummary,
) -> str:
    return (
        f"| {slice_type} | {label} | {comparison.binary_accuracy:.6f} | "
        f"{comparison.narrative_accuracy:.6f} | {comparison.accuracy_gap:.6f} | "
        f"{comparison.binary_parse_valid_rate:.6f} | "
        f"{comparison.narrative_parse_valid_rate:.6f} |"
    )
