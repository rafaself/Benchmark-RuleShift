from __future__ import annotations

from pathlib import Path

from core.audit import (
    AuditSource,
    run_release_r15_reaudit,
)
from core.model_execution import ModelMode, ModelRunConfig
from core.model_runner import run_model_benchmark
from core.panel_runner import (
    DEFAULT_PANEL_CONFIG,
    DEFAULT_PANEL_MODES,
    PanelArtifacts,
    TASK_MODE_LABELS,
    build_panel_artifact,
    build_panel_run_metadata,
    build_panel_raw_capture,
    build_panel_progress_callback,
    render_panel_markdown,
)
from core.providers.anthropic import AnthropicAdapter
from core.providers.registry import (
    ProviderExecutionSurface,
    get_provider_spec,
    resolve_provider_model_name,
)
from core.report_outputs import (
    build_latest_report_path,
    current_report_timestamp,
    write_canonical_run_outputs,
)
from core.splits import PARTITIONS, load_frozen_split
from tasks.ruleshift_benchmark.schema import Episode

__all__ = [
    "DEFAULT_ANTHROPIC_MODEL",
    "AnthropicPanelArtifacts",
    "default_anthropic_panel_report_path",
    "run_anthropic_panel",
]

_ANTHROPIC_PROVIDER_SPEC = get_provider_spec("anthropic")
DEFAULT_ANTHROPIC_MODEL = (
    _ANTHROPIC_PROVIDER_SPEC.default_benchmark_model
    or "claude-3-5-haiku-20241022"
)
_REPORT_TITLE = "Anthropic Panel Report"

AnthropicPanelArtifacts = PanelArtifacts


def default_anthropic_panel_report_path(*, include_narrative: bool) -> Path:
    target = "binary-vs-narrative" if include_narrative else "binary-only"
    return build_latest_report_path(
        "live",
        "anthropic-panel",
        target,
        filename="report.md",
    )


def run_anthropic_panel(
    *,
    model_name: str = DEFAULT_ANTHROPIC_MODEL,
    report_path: Path | None = None,
    modes: tuple[ModelMode, ...] = DEFAULT_PANEL_MODES,
    config: ModelRunConfig = DEFAULT_PANEL_CONFIG,
    adapter: AnthropicAdapter | None = None,
    invocation_surface: str = "python-api",
    invocation_command: tuple[str, ...] | None = None,
) -> AnthropicPanelArtifacts:
    normalized_modes = tuple(ModelMode(mode) for mode in modes)
    if not normalized_modes:
        raise ValueError("modes must not be empty")
    if len(set(normalized_modes)) != len(normalized_modes):
        raise ValueError("modes must not contain duplicates")

    resolved_model_name = resolve_provider_model_name(
        "anthropic",
        surface=ProviderExecutionSurface.LOCAL_BENCHMARK,
        model_name=model_name,
    )
    active_adapter = (
        AnthropicAdapter.from_env() if adapter is None else adapter
    )
    episodes_by_split: dict[str, tuple[Episode, ...]] = {}
    benchmark_results_by_split: dict[str, object] = {}
    model_sources_by_split: dict[str, tuple[AuditSource, ...]] = {}
    provider_name = _ANTHROPIC_PROVIDER_SPEC.provider_name

    for split_name in PARTITIONS:
        episodes = tuple(
            record.episode for record in load_frozen_split(split_name)
        )
        benchmark_result = run_model_benchmark(
            episodes,
            active_adapter,
            provider_name=provider_name,
            model_name=resolved_model_name,
            config=config,
            modes=normalized_modes,
            progress_callback=build_panel_progress_callback(
                split_name, panel_label="anthropic-panel"
            ),
        )
        episodes_by_split[split_name] = episodes
        benchmark_results_by_split[split_name] = benchmark_result
        model_sources_by_split[split_name] = tuple(
            AuditSource.from_parsed_predictions(
                f"{resolved_model_name} {TASK_MODE_LABELS[mode_result.mode]}",
                tuple(
                    row.parsed_prediction for row in mode_result.rows
                ),
                task_mode=TASK_MODE_LABELS[mode_result.mode],
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
    artifact_payload = build_panel_artifact(
        provider_name=provider_name,
        model_name=resolved_model_name,
        prompt_modes=normalized_modes,
        release_report=release_report,
        episodes_by_split=episodes_by_split,
        benchmark_results_by_split=benchmark_results_by_split,
    )
    raw_capture_payload = build_panel_raw_capture(
        provider_name=provider_name,
        model_name=resolved_model_name,
        prompt_modes=normalized_modes,
        release_report=release_report,
        episodes_by_split=episodes_by_split,
        benchmark_results_by_split=benchmark_results_by_split,
    )
    report_markdown = render_panel_markdown(
        release_report,
        model_name=resolved_model_name,
        provider_name=provider_name,
        prompt_modes=normalized_modes,
        artifact_payload=artifact_payload,
        report_title=_REPORT_TITLE,
    )
    resolved_report_path = (
        report_path
        if report_path is not None
        else default_anthropic_panel_report_path(
            include_narrative=ModelMode.NARRATIVE in normalized_modes
        )
    )
    report_timestamp = current_report_timestamp()
    metadata_payload = build_panel_run_metadata(
        provider_name=provider_name,
        requested_model_name=resolved_model_name,
        prompt_modes=normalized_modes,
        release_report=release_report,
        benchmark_results_by_split=benchmark_results_by_split,
        execution_timestamp=report_timestamp,
        invocation_surface=invocation_surface,
        invocation_command=invocation_command,
    )
    write_result = write_canonical_run_outputs(
        report_path=resolved_report_path,
        report_markdown=report_markdown,
        artifact_payload=artifact_payload,
        raw_capture_payload=raw_capture_payload,
        metadata_payload=metadata_payload,
        timestamp=report_timestamp,
    )

    return AnthropicPanelArtifacts(
        provider_name=provider_name,
        model_name=resolved_model_name,
        prompt_modes=normalized_modes,
        release_report=release_report,
        report_markdown=report_markdown,
        report_path=write_result.report_path,
        artifact_payload=artifact_payload,
        artifact_path=write_result.artifact_path,
        metadata_payload=metadata_payload,
        metadata_path=write_result.metadata_path,
        snapshot_report_path=write_result.snapshot_report_path,
        snapshot_artifact_path=write_result.snapshot_artifact_path,
        snapshot_metadata_path=write_result.snapshot_metadata_path,
        sample_path=write_result.sample_path,
    )
