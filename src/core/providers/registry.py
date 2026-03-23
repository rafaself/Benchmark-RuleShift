from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re
from typing import Final

__all__ = [
    "ProviderCapabilitySet",
    "ProviderExecutionSurface",
    "ProviderSelectionError",
    "ProviderSpec",
    "build_provider_spec",
    "get_provider_spec",
    "is_pinned_benchmark_model_id",
    "provider_allowed_on_surface",
    "resolve_provider_model_name",
]


class ProviderExecutionSurface(StrEnum):
    LOCAL_BENCHMARK = "local_benchmark"
    KAGGLE_BENCHMARK = "kaggle_benchmark"


@dataclass(frozen=True, slots=True)
class ProviderCapabilitySet:
    supports_json_schema_structured_output: bool = False


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    provider_name: str
    kaggle_supported: bool
    local_only: bool
    default_benchmark_model: str | None
    capabilities: ProviderCapabilitySet
    pinned_model_pattern: re.Pattern[str] | None = None
    sdk_extra: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.provider_name, str) or not self.provider_name.strip():
            raise ValueError("provider_name must be a non-empty string")
        if self.kaggle_supported and self.local_only:
            raise ValueError("provider spec cannot be both kaggle_supported and local_only")
        if self.default_benchmark_model is not None and not self.default_benchmark_model.strip():
            raise ValueError("default_benchmark_model must be a non-empty string or None")


class ProviderSelectionError(ValueError):
    """Raised when provider selection violates benchmark execution policy."""


def build_provider_spec(
    *,
    provider_name: str,
    kaggle_supported: bool,
    local_only: bool,
    default_benchmark_model: str | None,
    supports_json_schema_structured_output: bool = False,
    pinned_model_pattern: str | None = None,
    sdk_extra: str | None = None,
) -> ProviderSpec:
    return ProviderSpec(
        provider_name=provider_name,
        kaggle_supported=kaggle_supported,
        local_only=local_only,
        default_benchmark_model=default_benchmark_model,
        capabilities=ProviderCapabilitySet(
            supports_json_schema_structured_output=supports_json_schema_structured_output,
        ),
        pinned_model_pattern=(
            None if pinned_model_pattern is None else re.compile(pinned_model_pattern)
        ),
        sdk_extra=sdk_extra,
    )


_REGISTERED_PROVIDERS: Final[dict[str, ProviderSpec]] = {
    "gemini": build_provider_spec(
        provider_name="gemini",
        kaggle_supported=True,
        local_only=False,
        default_benchmark_model="gemini-2.5-flash-001",
        supports_json_schema_structured_output=True,
        pinned_model_pattern=r"^gemini-[a-z0-9.-]+-\d{3}$",
        sdk_extra="gemini",
    ),
}


def get_provider_spec(provider_name: str) -> ProviderSpec:
    try:
        return _REGISTERED_PROVIDERS[provider_name]
    except KeyError as exc:
        raise ProviderSelectionError(f"unknown provider: {provider_name!r}") from exc


def provider_allowed_on_surface(
    provider_spec: ProviderSpec,
    surface: ProviderExecutionSurface,
) -> bool:
    if surface is ProviderExecutionSurface.LOCAL_BENCHMARK:
        return True
    if surface is ProviderExecutionSurface.KAGGLE_BENCHMARK:
        return provider_spec.kaggle_supported and not provider_spec.local_only
    return False


def is_pinned_benchmark_model_id(provider_spec: ProviderSpec, model_name: str) -> bool:
    if not isinstance(model_name, str) or not model_name.strip():
        return False
    if provider_spec.pinned_model_pattern is None:
        return True
    return provider_spec.pinned_model_pattern.fullmatch(model_name) is not None


def resolve_provider_model_name(
    provider_name: str,
    *,
    surface: ProviderExecutionSurface,
    model_name: str | None,
) -> str:
    provider_spec = get_provider_spec(provider_name)
    if not provider_allowed_on_surface(provider_spec, surface):
        raise ProviderSelectionError(
            f"provider {provider_name!r} is not allowed on {surface.value}"
        )

    resolved_model_name = (
        provider_spec.default_benchmark_model if model_name is None else model_name
    )
    if resolved_model_name is None:
        raise ProviderSelectionError(
            f"provider {provider_name!r} does not define a default benchmark model"
        )
    if not is_pinned_benchmark_model_id(provider_spec, resolved_model_name):
        raise ProviderSelectionError(
            f"benchmark runs require a pinned model ID for provider {provider_name!r}; "
            f"got {resolved_model_name!r}"
        )
    return resolved_model_name
