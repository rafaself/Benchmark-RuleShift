from __future__ import annotations

from dataclasses import dataclass

from core.metrics import MetricSummary
from core.model_execution import (
    ModelAdapter,
    ModelExecutionOutcome,
    ModelMode,
    ModelRawResult,
    ModelRequest,
    ModelRunConfig,
)
from core.model_runner import run_model_benchmark
from generator import generate_episode
from parser import ParseStatus
from render import render_binary_prompt, render_narrative_prompt


def _binary_labels_for(episode) -> str:
    return ", ".join(label.value for label in episode.probe_targets)


def _narrative_labels_for(episode) -> str:
    return "Reasoning.\n" + _binary_labels_for(episode)


@dataclass
class FakeModelAdapter:
    responses: dict[tuple[ModelMode, str], ModelRawResult | Exception]
    calls: list[tuple[ModelRequest, ModelRunConfig]]

    def __init__(
        self,
        responses: dict[tuple[ModelMode, str], ModelRawResult | Exception],
    ) -> None:
        self.responses = dict(responses)
        self.calls = []

    def generate(self, request: ModelRequest, config: ModelRunConfig) -> ModelRawResult:
        self.calls.append((request, config))
        response = self.responses[(request.mode, request.prompt_text)]
        if isinstance(response, Exception):
            raise response
        return response


def test_fake_adapter_matches_runtime_protocol_shape():
    assert isinstance(FakeModelAdapter({}), ModelAdapter)


def test_runner_is_deterministic_for_fixed_fake_adapter_outputs():
    episodes = (generate_episode(0), generate_episode(1))
    config = ModelRunConfig(timeout_seconds=3.0)
    adapter = FakeModelAdapter(
        {
            (ModelMode.BINARY, render_binary_prompt(episode)): ModelRawResult.from_request(
                ModelRequest(
                    provider_name="fake-provider",
                    model_name="fake-model",
                    prompt_text=render_binary_prompt(episode),
                    mode=ModelMode.BINARY,
                ),
                response_text=_binary_labels_for(episode),
                duration_seconds=0.25,
            )
            for episode in episodes
        }
        | {
            (
                ModelMode.NARRATIVE,
                render_narrative_prompt(episode),
            ): ModelRawResult.from_request(
                ModelRequest(
                    provider_name="fake-provider",
                    model_name="fake-model",
                    prompt_text=render_narrative_prompt(episode),
                    mode=ModelMode.NARRATIVE,
                ),
                response_text=_narrative_labels_for(episode),
                duration_seconds=0.5,
            )
            for episode in episodes
        }
    )

    first_result = run_model_benchmark(
        episodes,
        adapter,
        provider_name="fake-provider",
        model_name="fake-model",
        config=config,
    )
    second_result = run_model_benchmark(
        episodes,
        adapter,
        provider_name="fake-provider",
        model_name="fake-model",
        config=config,
    )

    assert first_result == second_result
    assert first_result.metrics == MetricSummary(
        post_shift_probe_accuracy=1.0,
        parse_valid_rate=1.0,
        binary_accuracy=1.0,
        narrative_accuracy=1.0,
    )


def test_runner_flows_renderer_to_adapter_to_parser_to_metrics():
    episodes = (generate_episode(0),)
    episode = episodes[0]
    config = ModelRunConfig(timeout_seconds=2.5)
    binary_prompt = render_binary_prompt(episode)
    narrative_prompt = render_narrative_prompt(episode)
    adapter = FakeModelAdapter(
        {
            (ModelMode.BINARY, binary_prompt): ModelRawResult.from_request(
                ModelRequest(
                    provider_name="fake-provider",
                    model_name="fake-model",
                    prompt_text=binary_prompt,
                    mode=ModelMode.BINARY,
                ),
                response_text=_binary_labels_for(episode),
                duration_seconds=0.1,
            ),
            (ModelMode.NARRATIVE, narrative_prompt): ModelRawResult.from_request(
                ModelRequest(
                    provider_name="fake-provider",
                    model_name="fake-model",
                    prompt_text=narrative_prompt,
                    mode=ModelMode.NARRATIVE,
                ),
                response_text=_narrative_labels_for(episode),
                duration_seconds=0.2,
            ),
        }
    )

    result = run_model_benchmark(
        episodes,
        adapter,
        provider_name="fake-provider",
        model_name="fake-model",
        config=config,
    )
    binary_mode, narrative_mode = result.mode_results
    binary_row = binary_mode.rows[0]
    narrative_row = narrative_mode.rows[0]

    assert [request.prompt_text for request, _ in adapter.calls] == [
        binary_prompt,
        narrative_prompt,
    ]
    assert all(call_config == config for _, call_config in adapter.calls)

    assert binary_row.execution.request.prompt_text == binary_prompt
    assert narrative_row.execution.request.prompt_text == narrative_prompt
    assert binary_row.parsed_prediction.status is ParseStatus.VALID
    assert narrative_row.parsed_prediction.status is ParseStatus.VALID
    assert binary_row.target == episode.probe_targets
    assert narrative_row.target == episode.probe_targets


def test_runner_marks_malformed_raw_outputs_as_invalid_without_changing_scoring_rules():
    episode = generate_episode(0)
    binary_prompt = render_binary_prompt(episode)
    narrative_prompt = render_narrative_prompt(episode)
    adapter = FakeModelAdapter(
        {
            (ModelMode.BINARY, binary_prompt): ModelRawResult.from_request(
                ModelRequest(
                    provider_name="fake-provider",
                    model_name="fake-model",
                    prompt_text=binary_prompt,
                    mode=ModelMode.BINARY,
                ),
                response_text=_binary_labels_for(episode),
            ),
            (ModelMode.NARRATIVE, narrative_prompt): ModelRawResult.from_request(
                ModelRequest(
                    provider_name="fake-provider",
                    model_name="fake-model",
                    prompt_text=narrative_prompt,
                    mode=ModelMode.NARRATIVE,
                ),
                response_text="Reasoning without a final labels line.",
            ),
        }
    )

    result = run_model_benchmark(
        (episode,),
        adapter,
        provider_name="fake-provider",
        model_name="fake-model",
    )
    binary_mode, narrative_mode = result.mode_results

    assert binary_mode.rows[0].parsed_prediction.status is ParseStatus.VALID
    assert narrative_mode.rows[0].parsed_prediction.status is ParseStatus.INVALID
    assert result.metrics == MetricSummary(
        post_shift_probe_accuracy=1.0,
        parse_valid_rate=0.5,
        binary_accuracy=1.0,
        narrative_accuracy=0.0,
    )


def test_runner_captures_adapter_failures_as_invalid_rows():
    episode = generate_episode(1)
    binary_prompt = render_binary_prompt(episode)
    narrative_prompt = render_narrative_prompt(episode)
    adapter = FakeModelAdapter(
        {
            (ModelMode.BINARY, binary_prompt): ModelRawResult.from_request(
                ModelRequest(
                    provider_name="fake-provider",
                    model_name="fake-model",
                    prompt_text=binary_prompt,
                    mode=ModelMode.BINARY,
                ),
                response_text=_binary_labels_for(episode),
            ),
            (ModelMode.NARRATIVE, narrative_prompt): TimeoutError("timed out"),
        }
    )

    result = run_model_benchmark(
        (episode,),
        adapter,
        provider_name="fake-provider",
        model_name="fake-model",
    )
    narrative_row = result.mode_results[1].rows[0]

    assert narrative_row.parsed_prediction.status is ParseStatus.SKIPPED_PROVIDER_FAILURE
    assert (
        narrative_row.execution.raw_result.execution_outcome
        is ModelExecutionOutcome.PROVIDER_FAILURE
    )
    assert narrative_row.execution.raw_result.error_type == "TimeoutError"
    assert narrative_row.execution.raw_result.error_message == "timed out"
    assert narrative_row.execution.raw_result.response_text is None
    assert result.metrics == MetricSummary(
        post_shift_probe_accuracy=1.0,
        parse_valid_rate=1.0,
        binary_accuracy=1.0,
        narrative_accuracy=0.0,
    )


def test_runner_reports_progress_after_each_completed_episode():
    episodes = (generate_episode(0), generate_episode(1))
    progress_events: list[tuple[ModelMode, int, int, str]] = []
    adapter = FakeModelAdapter(
        {
            (ModelMode.BINARY, render_binary_prompt(episode)): ModelRawResult.from_request(
                ModelRequest(
                    provider_name="fake-provider",
                    model_name="fake-model",
                    prompt_text=render_binary_prompt(episode),
                    mode=ModelMode.BINARY,
                ),
                response_text=_binary_labels_for(episode),
            )
            for episode in episodes
        }
    )

    run_model_benchmark(
        episodes,
        adapter,
        provider_name="fake-provider",
        model_name="fake-model",
        modes=(ModelMode.BINARY,),
        progress_callback=lambda mode, index, total, episode_id: progress_events.append(
            (mode, index, total, episode_id)
        ),
    )

    assert progress_events == [
        (ModelMode.BINARY, 1, 2, episodes[0].episode_id),
        (ModelMode.BINARY, 2, 2, episodes[1].episode_id),
    ]
