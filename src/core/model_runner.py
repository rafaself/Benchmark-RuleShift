from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace as _dc_replace

from core.invariance import InvarianceCase, build_invariance_report, generate_invariance_cases
from core.metrics import MetricSummary, compute_metrics
from core.model_execution import (
    ModelAdapter,
    ModelExecutionRecord,
    ModelExecutionOutcome,
    ModelMode,
    ModelRawResult,
    ModelRequest,
    ModelRunConfig,
)
from core.parser import (
    NarrativeParsedResult,
    NarrativeParseStatus,
    ParseStatus,
    ParsedPrediction,
    parse_binary_output,
    parse_narrative_audit_output,
)
from core.slices import build_slice_report, compute_episode_slice_data
from tasks.ruleshift_benchmark.protocol import InteractionLabel
from tasks.ruleshift_benchmark.render import render_binary_prompt, render_narrative_prompt
from tasks.ruleshift_benchmark.schema import Episode

__all__ = [
    "BenchmarkModeRunRow",
    "BenchmarkModeRunResult",
    "BenchmarkRunResult",
    "run_model_benchmark",
]

_MODE_ORDER: tuple[ModelMode, ...] = (ModelMode.BINARY, ModelMode.NARRATIVE)
RunProgressCallback = Callable[[ModelMode, int, int, str], None]


@dataclass(frozen=True, slots=True)
class BenchmarkModeRunRow:
    episode_id: str
    execution: ModelExecutionRecord
    parsed_prediction: ParsedPrediction
    target: tuple[InteractionLabel, ...]
    narrative_result: NarrativeParsedResult | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkModeRunResult:
    mode: ModelMode
    rows: tuple[BenchmarkModeRunRow, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkRunResult:
    provider_name: str
    model_name: str
    config: ModelRunConfig
    mode_results: tuple[BenchmarkModeRunResult, ...]
    metrics: MetricSummary


def run_model_benchmark(
    episodes: Iterable[Episode],
    adapter: ModelAdapter,
    *,
    provider_name: str,
    model_name: str,
    config: ModelRunConfig | None = None,
    modes: Iterable[ModelMode] = _MODE_ORDER,
    progress_callback: RunProgressCallback | None = None,
    run_invariance: bool = False,
) -> BenchmarkRunResult:
    normalized_episodes = tuple(episodes)
    normalized_modes = _normalize_modes(modes)
    normalized_config = config if config is not None else ModelRunConfig()

    if not normalized_episodes:
        raise ValueError("episodes must not be empty")
    if not all(isinstance(episode, Episode) for episode in normalized_episodes):
        raise TypeError("episodes must contain Episode values")

    renderers = {
        ModelMode.BINARY: render_binary_prompt,
        ModelMode.NARRATIVE: render_narrative_prompt,
    }

    mode_results = tuple(
        _run_mode(
            episodes=normalized_episodes,
            adapter=adapter,
            provider_name=provider_name,
            model_name=model_name,
            config=normalized_config,
            mode=mode,
            renderer=renderers[mode],
            progress_callback=progress_callback,
        )
        for mode in normalized_modes
    )
    rows_by_mode = {result.mode: result.rows for result in mode_results}

    binary_rows = rows_by_mode.get(ModelMode.BINARY, ())
    narrative_rows = rows_by_mode.get(ModelMode.NARRATIVE, ())
    narrative_results = tuple(
        row.narrative_result
        for row in narrative_rows
        if row.narrative_result is not None
    )

    base_metrics = compute_metrics(
        binary_predictions=tuple(row.parsed_prediction for row in binary_rows),
        binary_targets=tuple(row.target for row in binary_rows),
        narrative_results=narrative_results,
    )

    # Build per-episode slice data pairing binary rows with Episode objects.
    # Narrative results are matched by episode_id so error classification can
    # use NarrativeParsedResult when the narrative mode was also run.
    narrative_result_by_episode: dict[str, NarrativeParsedResult] = {
        row.episode_id: row.narrative_result
        for row in narrative_rows
        if row.narrative_result is not None
    }
    episode_slices = [
        compute_episode_slice_data(
            episode=episode,
            prediction=binary_row.parsed_prediction,
            narrative_result=narrative_result_by_episode.get(episode.episode_id),
        )
        for episode, binary_row in zip(normalized_episodes, binary_rows)
    ]
    slice_report = build_slice_report(episode_slices)
    metrics = _dc_replace(base_metrics, slice_report=slice_report)

    if run_invariance:
        inv_cases = generate_invariance_cases(normalized_episodes)
        inv_predictions = _run_invariance_cases(
            cases=inv_cases,
            adapter=adapter,
            provider_name=provider_name,
            model_name=model_name,
            config=normalized_config,
        )
        inv_report = build_invariance_report(inv_predictions)
        metrics = _dc_replace(metrics, invariance_report=inv_report)

    return BenchmarkRunResult(
        provider_name=provider_name,
        model_name=model_name,
        config=normalized_config,
        mode_results=mode_results,
        metrics=metrics,
    )


def _run_invariance_cases(
    *,
    cases: list[InvarianceCase],
    adapter: ModelAdapter,
    provider_name: str,
    model_name: str,
    config: ModelRunConfig,
) -> list[tuple[InvarianceCase, ParsedPrediction]]:
    """Run each invariance case's perturbed prompt through the adapter in Binary mode."""
    results: list[tuple[InvarianceCase, ParsedPrediction]] = []
    for case in cases:
        request = ModelRequest(
            provider_name=provider_name,
            model_name=model_name,
            prompt_text=case.perturbed_prompt,
            mode=ModelMode.BINARY,
        )
        try:
            raw_result = adapter.generate(request, config)
        except Exception as exc:
            raw_result = ModelRawResult.from_request(
                request,
                execution_outcome=ModelExecutionOutcome.PROVIDER_FAILURE,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

        if raw_result.execution_outcome is ModelExecutionOutcome.COMPLETED:
            prediction = parse_binary_output(raw_result.response_text or "")
        else:
            prediction = ParsedPrediction.skipped_provider_failure()

        results.append((case, prediction))
    return results


def _normalize_modes(modes: Iterable[ModelMode]) -> tuple[ModelMode, ...]:
    normalized_modes = tuple(ModelMode(mode) for mode in modes)
    if not normalized_modes:
        raise ValueError("modes must not be empty")
    if len(set(normalized_modes)) != len(normalized_modes):
        raise ValueError("modes must not contain duplicates")
    return normalized_modes


def _run_mode(
    *,
    episodes: tuple[Episode, ...],
    adapter: ModelAdapter,
    provider_name: str,
    model_name: str,
    config: ModelRunConfig,
    mode: ModelMode,
    renderer: Callable[[Episode], str],
    progress_callback: RunProgressCallback | None,
) -> BenchmarkModeRunResult:
    total = len(episodes)
    rows: list[BenchmarkModeRunRow] = []
    for index, episode in enumerate(episodes, start=1):
        row = _run_episode(
            episode=episode,
            adapter=adapter,
            provider_name=provider_name,
            model_name=model_name,
            config=config,
            mode=mode,
            renderer=renderer,
        )
        rows.append(row)
        if progress_callback is not None:
            progress_callback(mode, index, total, row.episode_id)
    return BenchmarkModeRunResult(
        mode=mode,
        rows=tuple(rows),
    )


def _run_episode(
    *,
    episode: Episode,
    adapter: ModelAdapter,
    provider_name: str,
    model_name: str,
    config: ModelRunConfig,
    mode: ModelMode,
    renderer: Callable[[Episode], str],
) -> BenchmarkModeRunRow:
    request = ModelRequest(
        provider_name=provider_name,
        model_name=model_name,
        prompt_text=renderer(episode),
        mode=mode,
    )
    try:
        raw_result = adapter.generate(request, config)
    except Exception as exc:
        raw_result = ModelRawResult.from_request(
            request,
            execution_outcome=ModelExecutionOutcome.PROVIDER_FAILURE,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

    _validate_raw_result(request, raw_result)

    narrative_result: NarrativeParsedResult | None = None

    if mode is ModelMode.NARRATIVE:
        if raw_result.execution_outcome is ModelExecutionOutcome.COMPLETED:
            narrative_result = parse_narrative_audit_output(raw_result.response_text or "")
        else:
            narrative_result = NarrativeParsedResult.skipped_provider_failure()
        parsed_prediction = _narrative_to_parsed_prediction(narrative_result)
    else:
        if raw_result.execution_outcome is ModelExecutionOutcome.COMPLETED:
            parsed_prediction = parse_binary_output(raw_result.response_text or "")
        else:
            parsed_prediction = ParsedPrediction.skipped_provider_failure()

    return BenchmarkModeRunRow(
        episode_id=episode.episode_id,
        execution=ModelExecutionRecord(
            request=request,
            config=config,
            raw_result=raw_result,
        ),
        parsed_prediction=parsed_prediction,
        target=episode.probe_targets,
        narrative_result=narrative_result,
    )


def _narrative_to_parsed_prediction(result: NarrativeParsedResult) -> ParsedPrediction:
    """Derive a ParsedPrediction from a NarrativeParsedResult for audit compatibility."""
    if result.status is NarrativeParseStatus.VALID:
        assert result.output is not None
        return ParsedPrediction(
            labels=result.output.final_binary_answer,
            status=ParseStatus.VALID,
        )
    if result.status is NarrativeParseStatus.SKIPPED_PROVIDER_FAILURE:
        return ParsedPrediction.skipped_provider_failure()
    return ParsedPrediction(labels=(), status=ParseStatus.INVALID)


def _validate_raw_result(request: ModelRequest, raw_result: ModelRawResult) -> None:
    if raw_result.provider_name != request.provider_name:
        raise ValueError("raw_result.provider_name must match request.provider_name")
    if raw_result.model_name != request.model_name:
        raise ValueError("raw_result.model_name must match request.model_name")
    if raw_result.mode is not request.mode:
        raise ValueError("raw_result.mode must match request.mode")
