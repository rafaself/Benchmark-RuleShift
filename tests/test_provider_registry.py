from __future__ import annotations

import pytest

from core.providers.registry import (
    ProviderExecutionSurface,
    ProviderSelectionError,
    build_provider_spec,
    get_provider_spec,
    is_pinned_benchmark_model_id,
    provider_allowed_on_surface,
    resolve_provider_model_name,
)


def test_registered_gemini_provider_exposes_expected_metadata():
    spec = get_provider_spec("gemini")

    assert spec.provider_name == "gemini"
    assert spec.kaggle_supported is True
    assert spec.local_only is False
    assert spec.default_benchmark_model == "gemini-2.5-flash-001"
    assert spec.capabilities.supports_json_schema_structured_output is True
    assert spec.sdk_extra == "gemini"


def test_pinned_model_validation_rejects_floating_gemini_aliases():
    spec = get_provider_spec("gemini")

    assert is_pinned_benchmark_model_id(spec, "gemini-2.5-flash-001") is True
    assert is_pinned_benchmark_model_id(spec, "gemini-2.5-flash") is False


def test_resolve_provider_model_name_uses_pinned_default():
    assert resolve_provider_model_name(
        "gemini",
        surface=ProviderExecutionSurface.LOCAL_BENCHMARK,
        model_name=None,
    ) == "gemini-2.5-flash-001"


def test_resolve_provider_model_name_rejects_floating_aliases():
    with pytest.raises(ProviderSelectionError):
        resolve_provider_model_name(
            "gemini",
            surface=ProviderExecutionSurface.LOCAL_BENCHMARK,
            model_name="gemini-2.5-flash",
        )


def test_local_only_provider_is_blocked_on_kaggle_surface():
    local_only_spec = build_provider_spec(
        provider_name="openai",
        kaggle_supported=False,
        local_only=True,
        default_benchmark_model="gpt-5-mini-2025-08-07",
        supports_json_schema_structured_output=True,
        pinned_model_pattern=r"^gpt-[a-z0-9.-]+-\d{4}-\d{2}-\d{2}$",
        sdk_extra="openai",
    )

    assert provider_allowed_on_surface(
        local_only_spec,
        ProviderExecutionSurface.LOCAL_BENCHMARK,
    ) is True
    assert provider_allowed_on_surface(
        local_only_spec,
        ProviderExecutionSurface.KAGGLE_BENCHMARK,
    ) is False
