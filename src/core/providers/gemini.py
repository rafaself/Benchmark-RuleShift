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
    "GEMINI_API_KEY_ENV_VAR",
    "GeminiConfigurationError",
    "MissingGeminiApiKeyError",
    "MissingGeminiSdkError",
    "GeminiAdapter",
]

GEMINI_API_KEY_ENV_VAR = "GEMINI_API_KEY"
_DEFAULT_TEMPERATURE = 0.0
_DEFAULT_THINKING_BUDGET = 0
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
}
_BINARY_JSON_SUFFIX = (
    '\n\nReturn the final answer as JSON with one key named "labels". '
    'Its value must be an array of 4 strings in probe order. '
    'Each string must be either "attract" or "repel".'
)


def _is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


class GeminiConfigurationError(RuntimeError):
    """Raised when Gemini execution is not configured correctly."""


class MissingGeminiApiKeyError(GeminiConfigurationError):
    """Raised when GEMINI_API_KEY is not available."""


class MissingGeminiSdkError(GeminiConfigurationError):
    """Raised when google-genai is not installed."""


class GeminiAdapter:
    def __init__(
        self,
        *,
        api_key: str,
        client: object | None = None,
        client_factory: Callable[[str], object] | None = None,
    ) -> None:
        if not _is_nonempty_string(api_key):
            raise MissingGeminiApiKeyError(
                f"{GEMINI_API_KEY_ENV_VAR} must be set to run Gemini benchmark panels."
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
    ) -> "GeminiAdapter":
        normalized_env = _build_env_mapping(env)
        api_key = normalized_env.get(GEMINI_API_KEY_ENV_VAR, "").strip()
        if not api_key:
            raise MissingGeminiApiKeyError(
                f"{GEMINI_API_KEY_ENV_VAR} is not set. Export {GEMINI_API_KEY_ENV_VAR} or add it "
                "to the repo-root `.env`, then rerun `ife gemini-first-panel`."
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
            response = self._client_instance().models.generate_content(
                model=request.model_name,
                contents=self._render_contents(request),
                config=self._build_generation_config(request, config),
            )
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
            response_id=self._read_string_attr(response, "response_id"),
            provider_model_version=self._read_string_attr(response, "model_version"),
            finish_reason=self._extract_finish_reason(response),
        )

    def _client_instance(self) -> object:
        if self._client is None:
            factory = self._client_factory or _default_client_factory
            self._client = factory(self._api_key)
        return self._client

    def _render_contents(self, request: ModelRequest) -> str:
        if request.mode is ModelMode.BINARY:
            return request.prompt_text + _BINARY_JSON_SUFFIX
        return request.prompt_text

    def _build_generation_config(
        self,
        request: ModelRequest,
        config: ModelRunConfig,
    ) -> dict[str, object]:
        generation_config: dict[str, object] = {
            "temperature": (
                _DEFAULT_TEMPERATURE if config.temperature is None else config.temperature
            ),
        }
        thinking_budget = (
            _DEFAULT_THINKING_BUDGET
            if config.thinking_budget is None
            else config.thinking_budget
        )
        generation_config["thinking_config"] = {
            "thinking_budget": thinking_budget,
        }
        if config.timeout_seconds is not None:
            generation_config["http_options"] = {
                "timeout": int(config.timeout_seconds * 1000),
            }
        if request.mode is ModelMode.BINARY:
            generation_config["response_mime_type"] = "application/json"
            generation_config["response_json_schema"] = _BINARY_RESPONSE_SCHEMA
        return generation_config

    def _extract_response_text(self, request: ModelRequest, response: object) -> str | None:
        if request.mode is ModelMode.BINARY:
            labels = _extract_binary_labels(getattr(response, "parsed", None))
            if labels is None:
                text = self._read_text_attr(response, "text")
                labels = _extract_binary_labels_from_text(text)
                if labels is None:
                    return text
            return ", ".join(labels)
        return self._read_text_attr(response, "text")

    def _extract_usage(self, response: object) -> ModelUsage | None:
        usage_metadata = getattr(response, "usage_metadata", None)
        if usage_metadata is None:
            return None
        return ModelUsage(
            input_tokens=_read_int_attr(usage_metadata, "prompt_token_count"),
            output_tokens=_read_int_attr(usage_metadata, "candidates_token_count"),
            total_tokens=_read_int_attr(usage_metadata, "total_token_count"),
        )

    def _extract_finish_reason(self, response: object) -> str | None:
        candidates = getattr(response, "candidates", None)
        if not isinstance(candidates, (list, tuple)) or not candidates:
            return None
        finish_reason = getattr(candidates[0], "finish_reason", None)
        if finish_reason is None:
            return None
        text = str(finish_reason).strip()
        return text or None

    def _read_text_attr(self, response: object, attr_name: str) -> str | None:
        value = getattr(response, attr_name, None)
        if value is not None:
            return str(value)
        if attr_name != "text":
            return None
        return _extract_text_from_candidates(getattr(response, "candidates", None))

    def _read_string_attr(self, response: object, attr_name: str) -> str | None:
        value = getattr(response, attr_name, None)
        if not _is_nonempty_string(value):
            return None
        return value.strip()


def _default_client_factory(api_key: str) -> object:
    try:
        from google import genai
    except ImportError as exc:
        raise MissingGeminiSdkError(
            "google-genai is not installed. Install project dependencies before running "
            "the Gemini benchmark panel."
        ) from exc
    return genai.Client(api_key=api_key)


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


def _extract_binary_labels_from_text(text: str | None) -> tuple[str, ...] | None:
    if text is None:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return _extract_binary_labels(payload)


def _extract_binary_labels(payload: object) -> tuple[str, ...] | None:
    labels: object
    if isinstance(payload, Mapping):
        labels = payload.get("labels")
    else:
        labels = getattr(payload, "labels", None)

    if not isinstance(labels, (list, tuple)):
        return None

    normalized = tuple(str(label).strip().lower() for label in labels)
    if not normalized:
        return None
    return normalized


def _extract_text_from_candidates(candidates: object) -> str | None:
    if not isinstance(candidates, (list, tuple)):
        return None

    text_parts: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", ()) or ():
            text = getattr(part, "text", None)
            if _is_nonempty_string(text):
                text_parts.append(text.strip())

    if not text_parts:
        return None
    return "\n".join(text_parts)
