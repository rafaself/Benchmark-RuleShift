from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

__all__ = [
    "ModelExecutionOutcome",
    "ModelMode",
    "ModelRunConfig",
    "ModelUsage",
    "ModelRequest",
    "ModelRawResult",
    "ModelExecutionRecord",
    "ModelAdapter",
]


def _is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class ModelExecutionOutcome(StrEnum):
    COMPLETED = "completed"
    PROVIDER_FAILURE = "provider_failure"


class ModelMode(StrEnum):
    BINARY = "binary"
    NARRATIVE = "narrative"


@dataclass(frozen=True, slots=True)
class ModelRunConfig:
    timeout_seconds: float | None = None
    temperature: float | None = None
    thinking_budget: int | None = None

    def __post_init__(self) -> None:
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive or None")
        if self.temperature is not None and self.temperature < 0:
            raise ValueError("temperature must be non-negative or None")
        if self.thinking_budget is not None:
            if not _is_plain_int(self.thinking_budget):
                raise TypeError("thinking_budget must be an int or None")
            if self.thinking_budget < -1:
                raise ValueError("thinking_budget must be at least -1 or None")


@dataclass(frozen=True, slots=True)
class ModelUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be non-negative or None")


@dataclass(frozen=True, slots=True)
class ModelRequest:
    provider_name: str
    model_name: str
    prompt_text: str
    mode: ModelMode

    def __post_init__(self) -> None:
        if not _is_nonempty_string(self.provider_name):
            raise ValueError("provider_name must be a non-empty string")
        if not _is_nonempty_string(self.model_name):
            raise ValueError("model_name must be a non-empty string")
        if not _is_nonempty_string(self.prompt_text):
            raise ValueError("prompt_text must be a non-empty string")
        object.__setattr__(self, "mode", ModelMode(self.mode))


@dataclass(frozen=True, slots=True)
class ModelRawResult:
    provider_name: str
    model_name: str
    mode: ModelMode
    execution_outcome: ModelExecutionOutcome = ModelExecutionOutcome.COMPLETED
    response_text: str | None = None
    duration_seconds: float | None = None
    error_type: str | None = None
    error_message: str | None = None
    usage: ModelUsage | None = None
    response_id: str | None = None
    provider_model_version: str | None = None
    finish_reason: str | None = None

    def __post_init__(self) -> None:
        if not _is_nonempty_string(self.provider_name):
            raise ValueError("provider_name must be a non-empty string")
        if not _is_nonempty_string(self.model_name):
            raise ValueError("model_name must be a non-empty string")
        object.__setattr__(self, "mode", ModelMode(self.mode))
        object.__setattr__(
            self,
            "execution_outcome",
            ModelExecutionOutcome(self.execution_outcome),
        )
        if self.response_text is not None and not isinstance(self.response_text, str):
            raise TypeError("response_text must be a string or None")
        if self.duration_seconds is not None and self.duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative or None")
        if self.error_type is not None and not _is_nonempty_string(self.error_type):
            raise ValueError("error_type must be a non-empty string or None")
        if self.error_message is not None and not isinstance(self.error_message, str):
            raise TypeError("error_message must be a string or None")
        if self.usage is not None and not isinstance(self.usage, ModelUsage):
            raise TypeError("usage must be a ModelUsage or None")
        if self.response_id is not None and not _is_nonempty_string(self.response_id):
            raise ValueError("response_id must be a non-empty string or None")
        if self.provider_model_version is not None and not _is_nonempty_string(
            self.provider_model_version
        ):
            raise ValueError("provider_model_version must be a non-empty string or None")
        if self.finish_reason is not None and not _is_nonempty_string(self.finish_reason):
            raise ValueError("finish_reason must be a non-empty string or None")
        if self.execution_outcome is ModelExecutionOutcome.COMPLETED:
            if self.error_type is not None or self.error_message is not None:
                raise ValueError(
                    "completed model results must not define error_type or error_message"
                )
        else:
            if self.error_type is None:
                raise ValueError("provider-failure results must define error_type")
            if self.response_text is not None:
                raise ValueError(
                    "provider-failure results must not define response_text"
                )

    @property
    def succeeded(self) -> bool:
        return self.execution_outcome is ModelExecutionOutcome.COMPLETED

    @classmethod
    def from_request(
        cls,
        request: ModelRequest,
        *,
        execution_outcome: ModelExecutionOutcome = ModelExecutionOutcome.COMPLETED,
        response_text: str | None = None,
        duration_seconds: float | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        usage: ModelUsage | None = None,
        response_id: str | None = None,
        provider_model_version: str | None = None,
        finish_reason: str | None = None,
    ) -> "ModelRawResult":
        return cls(
            provider_name=request.provider_name,
            model_name=request.model_name,
            mode=request.mode,
            execution_outcome=execution_outcome,
            response_text=response_text,
            duration_seconds=duration_seconds,
            error_type=error_type,
            error_message=error_message,
            usage=usage,
            response_id=response_id,
            provider_model_version=provider_model_version,
            finish_reason=finish_reason,
        )


@dataclass(frozen=True, slots=True)
class ModelExecutionRecord:
    request: ModelRequest
    config: ModelRunConfig
    raw_result: ModelRawResult

    def __post_init__(self) -> None:
        if not isinstance(self.request, ModelRequest):
            raise TypeError("request must be a ModelRequest")
        if not isinstance(self.config, ModelRunConfig):
            raise TypeError("config must be a ModelRunConfig")
        if not isinstance(self.raw_result, ModelRawResult):
            raise TypeError("raw_result must be a ModelRawResult")


@runtime_checkable
class ModelAdapter(Protocol):
    def generate(
        self,
        request: ModelRequest,
        config: ModelRunConfig,
    ) -> ModelRawResult: ...
