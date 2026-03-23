from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import os
from pathlib import Path
import time

from core.model_execution import (
    ModelExecutionOutcome,
    ModelMode,
    ModelRawResult,
    ModelRequest,
    ModelRunConfig,
    ModelUsage,
)

__all__ = [
    "OPENAI_API_KEY_ENV_VAR",
    "OpenAIAdapter",
    "OpenAIConfigurationError",
    "MissingOpenAIApiKeyError",
    "MissingOpenAISdkError",
]

OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"
_DEFAULT_TEMPERATURE = 0.0
_BINARY_RESPONSE_SCHEMA = {
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
}
_BINARY_JSON_SUFFIX = (
    '\n\nReturn the final answer as JSON with one key named "labels". '
    'Its value must be an array of 4 strings in probe order. '
    'Each string must be either "attract" or "repel".'
)


def _is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


class OpenAIConfigurationError(RuntimeError):
    """Raised when OpenAI execution is not configured correctly."""


class MissingOpenAIApiKeyError(OpenAIConfigurationError):
    """Raised when OPENAI_API_KEY is not available."""


class MissingOpenAISdkError(OpenAIConfigurationError):
    """Raised when openai is not installed."""


class OpenAIAdapter:
    def __init__(
        self,
        *,
        api_key: str,
        client: object | None = None,
        client_factory: Callable[[str], object] | None = None,
    ) -> None:
        if not _is_nonempty_string(api_key):
            raise MissingOpenAIApiKeyError(
                f"{OPENAI_API_KEY_ENV_VAR} must be set to run OpenAI benchmark panels."
            )
        self._api_key = api_key
        self._client = client
        self._client_factory = client_factory

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
        client: object | None = None,
        client_factory: Callable[[str], object] | None = None,
    ) -> "OpenAIAdapter":
        normalized_env = _build_env_mapping(env)
        api_key = normalized_env.get(OPENAI_API_KEY_ENV_VAR, "").strip()
        if not api_key:
            raise MissingOpenAIApiKeyError(
                f"{OPENAI_API_KEY_ENV_VAR} is not set. Export {OPENAI_API_KEY_ENV_VAR} or add "
                "it to the repo-root `.env`, then rerun `ife openai-panel`."
            )
        return cls(
            api_key=api_key,
            client=client,
            client_factory=client_factory,
        )

    def generate(
        self,
        request: ModelRequest,
        config: ModelRunConfig,
    ) -> ModelRawResult:
        started_at = time.perf_counter()
        try:
            create_kwargs: dict[str, object] = {
                "model": request.model_name,
                "input": self._render_input(request),
                "temperature": (
                    _DEFAULT_TEMPERATURE
                    if config.temperature is None
                    else config.temperature
                ),
            }
            text_config = self._build_text_config(request)
            if text_config is not None:
                create_kwargs["text"] = text_config
            if config.timeout_seconds is not None:
                create_kwargs["timeout"] = config.timeout_seconds
            response = self._client_instance().responses.create(**create_kwargs)
        except Exception as exc:
            return ModelRawResult.from_request(
                request,
                execution_outcome=ModelExecutionOutcome.PROVIDER_FAILURE,
                duration_seconds=time.perf_counter() - started_at,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

        return ModelRawResult.from_request(
            request,
            response_text=self._extract_response_text(request, response),
            duration_seconds=time.perf_counter() - started_at,
            usage=self._extract_usage(response),
            response_id=self._extract_string_attr(response, "id"),
            provider_model_version=self._extract_string_attr(response, "model"),
            finish_reason=self._extract_finish_reason(response),
        )

    def _client_instance(self) -> object:
        if self._client is None:
            factory = self._client_factory or _default_client_factory
            self._client = factory(self._api_key)
        return self._client

    def _render_input(self, request: ModelRequest) -> str:
        if request.mode is ModelMode.BINARY:
            return request.prompt_text + _BINARY_JSON_SUFFIX
        return request.prompt_text

    def _build_text_config(self, request: ModelRequest) -> dict[str, object] | None:
        if request.mode is not ModelMode.BINARY:
            return None
        return {
            "format": {
                "type": "json_schema",
                "name": "ife_binary_labels",
                "strict": True,
                "schema": _BINARY_RESPONSE_SCHEMA,
            }
        }

    def _extract_response_text(
        self,
        request: ModelRequest,
        response: object,
    ) -> str | None:
        if request.mode is ModelMode.BINARY:
            labels = _extract_binary_labels(
                getattr(response, "output_parsed", None)
            )
            if labels is None:
                labels = _extract_binary_labels(
                    getattr(response, "parsed", None)
                )
            if labels is None:
                text = _extract_output_text(response)
                labels = _extract_binary_labels_from_text(text)
                if labels is None:
                    return text
            return ", ".join(labels)
        return _extract_output_text(response)

    def _extract_usage(self, response: object) -> ModelUsage | None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        return ModelUsage(
            input_tokens=_read_int_attr(usage, "input_tokens"),
            output_tokens=_read_int_attr(usage, "output_tokens"),
            total_tokens=_read_int_attr(usage, "total_tokens"),
        )

    def _extract_string_attr(self, response: object, attr_name: str) -> str | None:
        value = getattr(response, attr_name, None)
        if _is_nonempty_string(value):
            return value.strip()
        return None

    def _extract_finish_reason(self, response: object) -> str | None:
        value = getattr(response, "finish_reason", None)
        if _is_nonempty_string(value):
            return value.strip()

        output = getattr(response, "output", None)
        if not isinstance(output, (list, tuple)) or not output:
            return None
        for item in output:
            item_reason = getattr(item, "finish_reason", None)
            if _is_nonempty_string(item_reason):
                return item_reason.strip()
        return None


def _default_client_factory(api_key: str) -> object:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise MissingOpenAISdkError(
            "openai is not installed. Install project dependencies before running "
            "the OpenAI benchmark panel."
        ) from exc
    return OpenAI(api_key=api_key)


def _build_env_mapping(env: Mapping[str, str] | None) -> dict[str, str]:
    normalized_env = _load_repo_root_dotenv()
    normalized_env.update(os.environ)
    if env is not None:
        normalized_env.update(env)
    return normalized_env


def _load_repo_root_dotenv() -> dict[str, str]:
    dotenv_path = _repo_root() / ".env"
    if not dotenv_path.is_file():
        return {}

    parsed: dict[str, str] = {}
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        normalized_key = key.strip()
        if not normalized_key:
            continue
        parsed[normalized_key] = _parse_dotenv_value(value)
    return parsed


def _parse_dotenv_value(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
        return stripped[1:-1]
    return stripped


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_int_attr(value: object, attr_name: str) -> int | None:
    candidate = getattr(value, attr_name, None)
    if isinstance(candidate, int) and not isinstance(candidate, bool):
        return candidate
    return None


def _extract_output_text(response: object) -> str | None:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text

    output = getattr(response, "output", None)
    if not isinstance(output, (list, tuple)):
        return None

    text_parts: list[str] = []
    for item in output:
        content = getattr(item, "content", None)
        if not isinstance(content, (list, tuple)):
            continue
        for part in content:
            part_type = getattr(part, "type", None)
            if part_type not in {"output_text", "text"}:
                continue
            text = getattr(part, "text", None)
            if _is_nonempty_string(text):
                text_parts.append(text.strip())
    if not text_parts:
        return None
    return "\n".join(text_parts)


def _extract_binary_labels_from_text(text: str | None) -> tuple[str, ...] | None:
    if not isinstance(text, str):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return _extract_binary_labels(payload)


def _extract_binary_labels(payload: object) -> tuple[str, ...] | None:
    if not isinstance(payload, Mapping):
        return None
    labels = payload.get("labels")
    if not isinstance(labels, (list, tuple)):
        return None
    normalized = tuple(str(label).strip().lower() for label in labels)
    if not normalized:
        return None
    return normalized
