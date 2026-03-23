from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

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
from core.parser import ParsedPrediction, parse_binary_output, parse_narrative_output
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
    parsers = {
        ModelMode.BINARY: parse_binary_output,
        ModelMode.NARRATIVE: parse_narrative_output,
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
            parser=parsers[mode],
            progress_callback=progress_callback,
        )
        for mode in normalized_modes
    )
    rows_by_mode = {result.mode: result.rows for result in mode_results}

    return BenchmarkRunResult(
        provider_name=provider_name,
        model_name=model_name,
        config=normalized_config,
        mode_results=mode_results,
        metrics=compute_metrics(
            binary_predictions=tuple(
                row.parsed_prediction for row in rows_by_mode.get(ModelMode.BINARY, ())
            ),
            binary_targets=tuple(row.target for row in rows_by_mode.get(ModelMode.BINARY, ())),
            narrative_predictions=tuple(
                row.parsed_prediction for row in rows_by_mode.get(ModelMode.NARRATIVE, ())
            ),
            narrative_targets=tuple(
                row.target for row in rows_by_mode.get(ModelMode.NARRATIVE, ())
            ),
        ),
    )


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
    parser: Callable[[str], ParsedPrediction],
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
            parser=parser,
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
    parser: Callable[[str], ParsedPrediction],
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
    if raw_result.execution_outcome is ModelExecutionOutcome.COMPLETED:
        parsed_prediction = parser(raw_result.response_text or "")
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
    )


def _validate_raw_result(request: ModelRequest, raw_result: ModelRawResult) -> None:
    if raw_result.provider_name != request.provider_name:
        raise ValueError("raw_result.provider_name must match request.provider_name")
    if raw_result.model_name != request.model_name:
        raise ValueError("raw_result.model_name must match request.model_name")
    if raw_result.mode is not request.mode:
        raise ValueError("raw_result.mode must match request.mode")
