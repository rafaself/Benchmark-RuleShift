from __future__ import annotations

from pathlib import Path

import pytest

from core.model_execution import ModelExecutionOutcome, ModelMode, ModelRequest, ModelRunConfig
from core.providers.openai import (
    OPENAI_API_KEY_ENV_VAR,
    MissingOpenAIApiKeyError,
    MissingOpenAISdkError,
    OpenAIAdapter,
)


class _FakeUsage:
    input_tokens = 14
    output_tokens = 6
    total_tokens = 20


class _FakeResponse:
    def __init__(
        self,
        *,
        output_parsed: object = None,
        output_text: str | None = None,
        finish_reason: str = "stop",
        response_id: str = "resp-openai-123",
        model: str = "gpt-5-mini-2025-08-07",
    ) -> None:
        self.output_parsed = output_parsed
        self.output_text = output_text
        self.finish_reason = finish_reason
        self.id = response_id
        self.model = model
        self.usage = _FakeUsage()


class _FakeResponses:
    def __init__(
        self,
        *,
        response: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class _FakeOpenAIClient:
    def __init__(self, *, responses: _FakeResponses) -> None:
        self.responses = responses


def test_openai_adapter_maps_binary_structured_output_to_labels():
    responses = _FakeResponses(
        response=_FakeResponse(
            output_parsed={"labels": ["attract", "repel", "repel", "attract"]}
        )
    )
    client = _FakeOpenAIClient(responses=responses)
    adapter = OpenAIAdapter(api_key="test-key", client=client)

    request = ModelRequest(
        provider_name="openai",
        model_name="gpt-5-mini-2025-08-07",
        prompt_text="Benchmark prompt",
        mode=ModelMode.BINARY,
    )
    result = adapter.generate(
        request,
        ModelRunConfig(timeout_seconds=12.0, temperature=0.0, thinking_budget=0),
    )

    assert result.succeeded is True
    assert result.execution_outcome is ModelExecutionOutcome.COMPLETED
    assert result.response_text == "attract, repel, repel, attract"
    assert result.usage is not None
    assert result.usage.input_tokens == 14
    assert result.usage.output_tokens == 6
    assert result.usage.total_tokens == 20
    assert result.response_id == "resp-openai-123"
    assert result.provider_model_version == "gpt-5-mini-2025-08-07"
    assert result.finish_reason == "stop"

    assert len(responses.calls) == 1
    call = responses.calls[0]
    assert call["model"] == "gpt-5-mini-2025-08-07"
    assert "Benchmark prompt" in str(call["input"])
    assert '"labels"' in str(call["input"])
    assert call["text"] == {
        "format": {
            "type": "json_schema",
            "name": "ife_binary_labels",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "labels": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["attract", "repel"],
                        },
                        "minItems": 4,
                        "maxItems": 4,
                    }
                },
                "required": ["labels"],
                "additionalProperties": False,
            },
        }
    }


def test_openai_adapter_returns_raw_text_for_narrative_mode():
    responses = _FakeResponses(
        response=_FakeResponse(
            output_text="Reasoning.\nattract, repel, repel, attract"
        )
    )
    client = _FakeOpenAIClient(responses=responses)
    adapter = OpenAIAdapter(api_key="test-key", client=client)

    request = ModelRequest(
        provider_name="openai",
        model_name="gpt-5-mini-2025-08-07",
        prompt_text="Narrative prompt",
        mode=ModelMode.NARRATIVE,
    )
    result = adapter.generate(request, ModelRunConfig())

    assert result.succeeded is True
    assert result.response_text == "Reasoning.\nattract, repel, repel, attract"

    call = responses.calls[0]
    assert call["input"] == "Narrative prompt"
    assert "text" not in call


def test_openai_adapter_returns_provider_failure_on_exception():
    responses = _FakeResponses(error=RuntimeError("provider exploded"))
    client = _FakeOpenAIClient(responses=responses)
    adapter = OpenAIAdapter(api_key="test-key", client=client)

    request = ModelRequest(
        provider_name="openai",
        model_name="gpt-5-mini-2025-08-07",
        prompt_text="Benchmark prompt",
        mode=ModelMode.BINARY,
    )
    result = adapter.generate(request, ModelRunConfig())

    assert result.succeeded is False
    assert result.execution_outcome is ModelExecutionOutcome.PROVIDER_FAILURE
    assert result.error_type == "RuntimeError"
    assert result.error_message == "provider exploded"
    assert result.response_text is None


def test_openai_adapter_fails_clearly_when_api_key_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.delenv(OPENAI_API_KEY_ENV_VAR, raising=False)
    monkeypatch.setattr("core.providers.openai._repo_root", lambda: tmp_path)

    with pytest.raises(MissingOpenAIApiKeyError) as excinfo:
        OpenAIAdapter.from_env(env={})

    assert OPENAI_API_KEY_ENV_VAR in str(excinfo.value)


def test_openai_adapter_reads_api_key_from_env(
    monkeypatch: pytest.MonkeyPatch,
):
    responses = _FakeResponses(
        response=_FakeResponse(
            output_parsed={"labels": ["attract", "repel", "repel", "attract"]}
        )
    )
    client = _FakeOpenAIClient(responses=responses)
    monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "env-key")

    adapter = OpenAIAdapter.from_env(client=client)
    result = adapter.generate(
        ModelRequest(
            provider_name="openai",
            model_name="gpt-5-mini-2025-08-07",
            prompt_text="Benchmark prompt",
            mode=ModelMode.BINARY,
        ),
        ModelRunConfig(),
    )

    assert result.succeeded is True
    assert adapter._api_key == "env-key"


def test_openai_adapter_reads_api_key_from_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    responses = _FakeResponses(
        response=_FakeResponse(
            output_parsed={"labels": ["attract", "repel", "repel", "attract"]}
        )
    )
    client = _FakeOpenAIClient(responses=responses)
    monkeypatch.delenv(OPENAI_API_KEY_ENV_VAR, raising=False)
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=dotenv-key\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("core.providers.openai._repo_root", lambda: tmp_path)

    adapter = OpenAIAdapter.from_env(client=client)
    result = adapter.generate(
        ModelRequest(
            provider_name="openai",
            model_name="gpt-5-mini-2025-08-07",
            prompt_text="Benchmark prompt",
            mode=ModelMode.BINARY,
        ),
        ModelRunConfig(),
    )

    assert result.succeeded is True
    assert adapter._api_key == "dotenv-key"


def test_openai_adapter_raises_clear_error_when_sdk_is_missing():
    adapter = OpenAIAdapter(
        api_key="test-key",
        client_factory=lambda _api_key: (_ for _ in ()).throw(
            MissingOpenAISdkError("openai is not installed")
        ),
    )

    result = adapter.generate(
        ModelRequest(
            provider_name="openai",
            model_name="gpt-5-mini-2025-08-07",
            prompt_text="Benchmark prompt",
            mode=ModelMode.BINARY,
        ),
        ModelRunConfig(),
    )

    assert result.succeeded is False
    assert result.execution_outcome is ModelExecutionOutcome.PROVIDER_FAILURE
    assert result.error_type == "MissingOpenAISdkError"
    assert result.error_message == "openai is not installed"
