from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from core.model_execution import ModelMode, ModelRequest, ModelRunConfig
from core.providers.gemini import (
    GEMINI_API_KEY_ENV_VAR,
    GeminiAdapter,
    MissingGeminiApiKeyError,
    MissingGeminiSdkError,
)


class _FakeUsageMetadata:
    prompt_token_count = 11
    candidates_token_count = 7
    total_token_count = 18


class _FakeCandidate:
    finish_reason = "STOP"


class _FakeResponse:
    def __init__(
        self,
        *,
        parsed: object = None,
        text: str | None = None,
    ) -> None:
        self.parsed = parsed
        self.text = text
        self.usage_metadata = _FakeUsageMetadata()
        self.response_id = "resp-123"
        self.model_version = "gemini-2.5-flash-001"
        self.candidates = (_FakeCandidate(),)


class _FakeModels:
    def __init__(
        self,
        *,
        response: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class _FakeClient:
    def __init__(self, *, api_key: str, models: _FakeModels) -> None:
        self.api_key = api_key
        self.models = models


def _install_fake_google_genai(
    monkeypatch: pytest.MonkeyPatch,
    *,
    models: _FakeModels,
) -> None:
    genai_module = types.ModuleType("google.genai")
    genai_module.Client = lambda *, api_key: _FakeClient(api_key=api_key, models=models)

    google_module = types.ModuleType("google")
    google_module.genai = genai_module

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)


def test_gemini_adapter_reads_api_key_from_environment_and_maps_structured_binary_output(
    monkeypatch: pytest.MonkeyPatch,
):
    models = _FakeModels(
        response=_FakeResponse(parsed={"labels": ["attract", "repel", "repel", "attract"]})
    )
    _install_fake_google_genai(monkeypatch, models=models)

    adapter = GeminiAdapter.from_env(env={GEMINI_API_KEY_ENV_VAR: "test-key"})
    request = ModelRequest(
        provider_name="gemini",
        model_name="gemini-2.5-flash",
        prompt_text="Benchmark prompt",
        mode=ModelMode.BINARY,
    )
    result = adapter.generate(
        request,
        ModelRunConfig(timeout_seconds=12.0, temperature=0.0, thinking_budget=0),
    )

    assert result.succeeded is True
    assert result.response_text == "attract, repel, repel, attract"
    assert result.usage is not None
    assert result.usage.input_tokens == 11
    assert result.usage.output_tokens == 7
    assert result.usage.total_tokens == 18
    assert result.response_id == "resp-123"
    assert result.provider_model_version == "gemini-2.5-flash-001"
    assert result.finish_reason == "STOP"

    assert len(models.calls) == 1
    call = models.calls[0]
    assert call["model"] == "gemini-2.5-flash"
    assert "Benchmark prompt" in str(call["contents"])
    assert '"labels"' in str(call["contents"])
    assert call["config"] == {
        "temperature": 0.0,
        "thinking_config": {"thinking_budget": 0},
        "http_options": {"timeout": 12000},
        "response_mime_type": "application/json",
        "response_json_schema": {
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
        },
    }


def test_gemini_adapter_reads_api_key_from_process_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    models = _FakeModels(
        response=_FakeResponse(parsed={"labels": ["attract", "repel", "repel", "attract"]})
    )
    _install_fake_google_genai(monkeypatch, models=models)
    monkeypatch.setenv(GEMINI_API_KEY_ENV_VAR, "process-key")

    adapter = GeminiAdapter.from_env()
    result = adapter.generate(
        ModelRequest(
            provider_name="gemini",
            model_name="gemini-2.5-flash",
            prompt_text="Benchmark prompt",
            mode=ModelMode.BINARY,
        ),
        ModelRunConfig(),
    )

    assert result.succeeded is True
    assert len(models.calls) == 1


def test_gemini_adapter_reads_api_key_from_repo_root_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    models = _FakeModels(
        response=_FakeResponse(parsed={"labels": ["attract", "repel", "repel", "attract"]})
    )
    _install_fake_google_genai(monkeypatch, models=models)
    monkeypatch.delenv(GEMINI_API_KEY_ENV_VAR, raising=False)
    (tmp_path / ".env").write_text(
        "GEMINI_API_KEY=dotenv-key\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("core.providers.gemini._repo_root", lambda: tmp_path)

    adapter = GeminiAdapter.from_env()
    result = adapter.generate(
        ModelRequest(
            provider_name="gemini",
            model_name="gemini-2.5-flash",
            prompt_text="Benchmark prompt",
            mode=ModelMode.BINARY,
        ),
        ModelRunConfig(),
    )

    assert result.succeeded is True
    assert len(models.calls) == 1


def test_process_environment_overrides_repo_root_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    models = _FakeModels(
        response=_FakeResponse(parsed={"labels": ["attract", "repel", "repel", "attract"]})
    )
    _install_fake_google_genai(monkeypatch, models=models)
    (tmp_path / ".env").write_text(
        "GEMINI_API_KEY=dotenv-key\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("core.providers.gemini._repo_root", lambda: tmp_path)
    monkeypatch.setenv(GEMINI_API_KEY_ENV_VAR, "process-key")

    adapter = GeminiAdapter.from_env()
    result = adapter.generate(
        ModelRequest(
            provider_name="gemini",
            model_name="gemini-2.5-flash",
            prompt_text="Benchmark prompt",
            mode=ModelMode.BINARY,
        ),
        ModelRunConfig(),
    )

    assert result.succeeded is True
    assert len(models.calls) == 1
    assert adapter._api_key == "process-key"


def test_gemini_adapter_uses_json_text_fallback_for_binary_mapping(
    monkeypatch: pytest.MonkeyPatch,
):
    models = _FakeModels(
        response=_FakeResponse(text='{"labels":["repel","repel","attract","attract"]}')
    )
    _install_fake_google_genai(monkeypatch, models=models)

    adapter = GeminiAdapter.from_env(env={GEMINI_API_KEY_ENV_VAR: "test-key"})
    request = ModelRequest(
        provider_name="gemini",
        model_name="gemini-2.5-flash",
        prompt_text="Benchmark prompt",
        mode=ModelMode.BINARY,
    )

    result = adapter.generate(request, ModelRunConfig())

    assert result.succeeded is True
    assert result.response_text == "repel, repel, attract, attract"


def test_gemini_adapter_returns_canonical_error_result_for_provider_failures(
    monkeypatch: pytest.MonkeyPatch,
):
    models = _FakeModels(error=RuntimeError("provider exploded"))
    _install_fake_google_genai(monkeypatch, models=models)

    adapter = GeminiAdapter.from_env(env={GEMINI_API_KEY_ENV_VAR: "test-key"})
    request = ModelRequest(
        provider_name="gemini",
        model_name="gemini-2.5-flash",
        prompt_text="Benchmark prompt",
        mode=ModelMode.BINARY,
    )

    result = adapter.generate(request, ModelRunConfig())

    assert result.succeeded is False
    assert result.error_type == "RuntimeError"
    assert result.error_message == "provider exploded"
    assert result.response_text is None


def test_gemini_adapter_fails_clearly_when_api_key_is_missing():
    with pytest.raises(MissingGeminiApiKeyError) as excinfo:
        GeminiAdapter.from_env(env={})

    assert GEMINI_API_KEY_ENV_VAR in str(excinfo.value)


def test_gemini_adapter_fails_clearly_when_sdk_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delitem(sys.modules, "google", raising=False)
    monkeypatch.delitem(sys.modules, "google.genai", raising=False)

    result = GeminiAdapter.from_env(
        env={GEMINI_API_KEY_ENV_VAR: "test-key"}
    ).generate(
        ModelRequest(
            provider_name="gemini",
            model_name="gemini-2.5-flash",
            prompt_text="Benchmark prompt",
            mode=ModelMode.BINARY,
        ),
        ModelRunConfig(),
    )

    assert result.succeeded is False
    assert result.error_type == "MissingGeminiSdkError"
    assert "google-genai" in (result.error_message or "")
