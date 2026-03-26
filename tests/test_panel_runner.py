from __future__ import annotations

from core.audit import AuditSource, run_release_r15_reaudit
from core.metrics import compute_metrics
from core.model_execution import (
    ModelExecutionOutcome,
    ModelExecutionRecord,
    ModelMode,
    ModelRawResult,
    ModelRequest,
    ModelRunConfig,
)
from core.model_runner import (
    BenchmarkModeRunResult,
    BenchmarkModeRunRow,
    BenchmarkRunResult,
)
from core.panel_runner import (
    build_panel_artifact,
    build_panel_run_metadata,
    build_panel_raw_capture,
    render_panel_markdown,
)
from tasks.ruleshift_benchmark.generator import generate_episode
from core.parser import NarrativeParsedResult, NarrativeParseStatus, ParseStatus, ParsedPrediction
from tasks.ruleshift_benchmark.protocol import Split


def _row_from_prediction(
    episode,
    *,
    mode: ModelMode,
    provider_name: str,
    model_name: str,
    parsed_prediction: ParsedPrediction,
    response_text: str | None = None,
    execution_outcome: ModelExecutionOutcome = ModelExecutionOutcome.COMPLETED,
    error_type: str | None = None,
    error_message: str | None = None,
) -> BenchmarkModeRunRow:
    request = ModelRequest(
        provider_name=provider_name,
        model_name=model_name,
        prompt_text=f"{episode.episode_id}:{mode.value}",
        mode=mode,
    )
    raw_result = ModelRawResult.from_request(
        request,
        execution_outcome=execution_outcome,
        response_text=response_text,
        error_type=error_type,
        error_message=error_message,
        finish_reason="stop" if execution_outcome is ModelExecutionOutcome.COMPLETED else None,
    )
    return BenchmarkModeRunRow(
        episode_id=episode.episode_id,
        execution=ModelExecutionRecord(
            request=request,
            config=ModelRunConfig(),
            raw_result=raw_result,
        ),
        parsed_prediction=parsed_prediction,
        target=episode.probe_targets,
        narrative_result=None,
    )


def _old_rule_labels(episode):
    return tuple(metadata.old_rule_label for metadata in episode.probe_metadata)


def _new_rule_labels(episode):
    return tuple(metadata.new_rule_label for metadata in episode.probe_metadata)


def _mixed_labels(episode):
    labels = list(episode.probe_targets)
    old_flipped = False
    recency_flipped = False
    for index, (target, metadata) in enumerate(
        zip(episode.probe_targets, episode.probe_metadata)
    ):
        if target is metadata.new_rule_label and not old_flipped:
            labels[index] = metadata.old_rule_label
            old_flipped = True
            continue
        if target is metadata.old_rule_label and not recency_flipped:
            labels[index] = metadata.new_rule_label
            recency_flipped = True
    return tuple(labels)


def _old_rule_only_labels(episode):
    labels = list(episode.probe_targets)
    for index, (target, metadata) in enumerate(
        zip(episode.probe_targets, episode.probe_metadata)
    ):
        if target is metadata.new_rule_label:
            labels[index] = metadata.old_rule_label
            break
    return tuple(labels)


def test_panel_runner_builds_direct_diagnostic_summary_and_report_labels():
    provider_name = "test-provider"
    model_name = "diagnostic-model"
    episodes = (
        generate_episode(0, split=Split.DEV),
        generate_episode(1, split=Split.DEV),
        generate_episode(2, split=Split.DEV),
    )

    binary_rows = (
        _row_from_prediction(
            episodes[0],
            mode=ModelMode.BINARY,
            provider_name=provider_name,
            model_name=model_name,
            parsed_prediction=ParsedPrediction(
                labels=_new_rule_labels(episodes[0]),
                status=ParseStatus.VALID,
            ),
            response_text=", ".join(label.value for label in _new_rule_labels(episodes[0])),
        ),
        _row_from_prediction(
            episodes[1],
            mode=ModelMode.BINARY,
            provider_name=provider_name,
            model_name=model_name,
            parsed_prediction=ParsedPrediction(
                labels=_mixed_labels(episodes[1]),
                status=ParseStatus.VALID,
            ),
            response_text=", ".join(label.value for label in _mixed_labels(episodes[1])),
        ),
        _row_from_prediction(
            episodes[2],
            mode=ModelMode.BINARY,
            provider_name=provider_name,
            model_name=model_name,
            parsed_prediction=ParsedPrediction(
                labels=episodes[2].probe_targets,
                status=ParseStatus.VALID,
            ),
            response_text=", ".join(label.value for label in episodes[2].probe_targets),
        ),
    )
    narrative_rows = (
        _row_from_prediction(
            episodes[0],
            mode=ModelMode.NARRATIVE,
            provider_name=provider_name,
            model_name=model_name,
            parsed_prediction=ParsedPrediction(labels=(), status=ParseStatus.INVALID),
            response_text="invalid narrative output",
        ),
        _row_from_prediction(
            episodes[1],
            mode=ModelMode.NARRATIVE,
            provider_name=provider_name,
            model_name=model_name,
            parsed_prediction=ParsedPrediction.skipped_provider_failure(),
            execution_outcome=ModelExecutionOutcome.PROVIDER_FAILURE,
            error_type="TimeoutError",
            error_message="timed out",
        ),
        _row_from_prediction(
            episodes[2],
            mode=ModelMode.NARRATIVE,
            provider_name=provider_name,
            model_name=model_name,
            parsed_prediction=ParsedPrediction(
                labels=_old_rule_only_labels(episodes[2]),
                status=ParseStatus.VALID,
            ),
            response_text=", ".join(label.value for label in _old_rule_only_labels(episodes[2])),
        ),
    )

    benchmark_result = BenchmarkRunResult(
        provider_name=provider_name,
        model_name=model_name,
        config=ModelRunConfig(),
        mode_results=(
            BenchmarkModeRunResult(mode=ModelMode.BINARY, rows=binary_rows),
            BenchmarkModeRunResult(
                mode=ModelMode.NARRATIVE,
                rows=narrative_rows,
            ),
        ),
        metrics=compute_metrics(
            binary_predictions=tuple(row.parsed_prediction for row in binary_rows),
            binary_targets=tuple(row.target for row in binary_rows),
            narrative_results=(),
        ),
    )

    release_report = run_release_r15_reaudit(
        episodes_by_split=(("dev", episodes),),
        model_sources_by_split=(
            (
                "dev",
                (
                    AuditSource.from_parsed_predictions(
                        f"{model_name} Binary",
                        tuple(row.parsed_prediction for row in binary_rows),
                        task_mode="Binary",
                        source_family=model_name,
                        is_real_model=True,
                    ),
                    AuditSource.from_parsed_predictions(
                        f"{model_name} Narrative",
                        tuple(row.parsed_prediction for row in narrative_rows),
                        task_mode="Narrative",
                        source_family=model_name,
                        is_real_model=True,
                    ),
                ),
            ),
        ),
        release_id="R18",
    )

    artifact = build_panel_artifact(
        provider_name=provider_name,
        model_name=model_name,
        prompt_modes=(ModelMode.BINARY, ModelMode.NARRATIVE),
        release_report=release_report,
        episodes_by_split={"dev": episodes},
        benchmark_results_by_split={"dev": benchmark_result},
    )

    assert artifact["artifact_schema_version"] == "v1.1"
    assert "execution_summary" in artifact

    binary_overall = next(
        row
        for row in artifact["diagnostic_summary"]
        if row["scope_type"] == "overall" and row["mode"] == "Binary"
    )
    narrative_overall = next(
        row
        for row in artifact["diagnostic_summary"]
        if row["scope_type"] == "overall" and row["mode"] == "Narrative"
    )

    assert binary_overall["episode_count"] == 3
    assert binary_overall["parse_valid_count"] == 3
    assert binary_overall["adaptation_failure_count"] == 2
    assert binary_overall["exact_global_recency_overshoot_count"] == 1
    assert binary_overall["old_rule_only_count"] == 0
    assert binary_overall["recency_overshoot_only_count"] == 1
    assert binary_overall["mixed_disagreement_count"] == 1
    assert binary_overall["error_probe_count"] == 4
    assert binary_overall["old_rule_error_probe_count"] == 1
    assert binary_overall["recency_overshoot_error_probe_count"] == 3

    assert narrative_overall["runtime_error_count"] == 1
    assert narrative_overall["parse_failure_count"] == 1
    assert narrative_overall["parse_valid_count"] == 1
    assert narrative_overall["adaptation_failure_count"] == 1
    assert narrative_overall["adaptation_failure_rate_among_parse_valid"] == 1.0
    assert narrative_overall["old_rule_only_count"] == 1
    assert narrative_overall["old_rule_error_probe_count"] == 1
    assert narrative_overall["error_probe_count"] == 1

    first_binary_mode = artifact["splits"][0]["rows"][0]["modes"]["binary"]
    assert "response_text" not in first_binary_mode
    assert "error_message" not in first_binary_mode

    execution_overall = next(
        row
        for row in artifact["execution_summary"]
        if row["scope_type"] == "overall" and row["mode"] == "Binary"
    )
    assert execution_overall["provider_failure_count"] == 0
    assert execution_overall["completed_count"] == 3

    drilldown_rows = artifact["diagnostic_episode_rows"]
    assert any(
        row["mode"] == "Binary"
        and row["disagreement_profile"] == "mixed"
        for row in drilldown_rows
    )
    assert any(
        row["mode"] == "Narrative"
        and row["failure_bucket"] == "runtime_error"
        for row in drilldown_rows
    )

    markdown = render_panel_markdown(
        release_report,
        model_name=model_name,
        provider_name=provider_name,
        prompt_modes=(ModelMode.BINARY, ModelMode.NARRATIVE),
        artifact_payload=artifact,
        report_title="Diagnostic Panel Report",
    )

    assert "Binary-only headline metric: diagnostic-model Binary =" in markdown
    assert "## Execution Provenance (diagnostic-only)" in markdown
    assert "## Failure Decomposition (diagnostic-only)" in markdown
    assert "## Direct Disagreement Diagnostics (diagnostic-only)" in markdown
    assert "## Diagnostic Failure Slices (diagnostic-only)" in markdown
    assert "## Failure Taxonomy (diagnostic-only)" in markdown
    assert "diagnostic-only" in markdown
    assert (
        "Parse/format failures dominate at least one prompt mode in this run."
        in markdown
    )
    assert "new benchmark scoring" in markdown

    raw_capture = build_panel_raw_capture(
        provider_name=provider_name,
        model_name=model_name,
        prompt_modes=(ModelMode.BINARY, ModelMode.NARRATIVE),
        release_report=release_report,
        episodes_by_split={"dev": episodes},
        benchmark_results_by_split={"dev": benchmark_result},
    )

    assert raw_capture["capture_schema_version"] == "v1.1"
    assert (
        raw_capture["splits"][0]["rows"][0]["modes"]["binary"]["response_text"]
        is not None
    )

    metadata = build_panel_run_metadata(
        provider_name=provider_name,
        requested_model_name=model_name,
        prompt_modes=(ModelMode.BINARY, ModelMode.NARRATIVE),
        release_report=release_report,
        benchmark_results_by_split={"dev": benchmark_result},
        execution_timestamp="20260323_120000",
        invocation_surface="cli",
        invocation_command=("ife", "diagnostic-panel"),
    )

    assert metadata["run_metadata_schema_version"] == "v1"
    assert metadata["requested_model_id"] == model_name
    assert metadata["served_model_id"] is None
    assert metadata["invocation"]["surface"] == "cli"
    assert metadata["benchmark_versions"]["generator_version"] == "R13"
    assert metadata["frozen_artifacts"]["split_manifests"][0]["split_name"] == "dev"
