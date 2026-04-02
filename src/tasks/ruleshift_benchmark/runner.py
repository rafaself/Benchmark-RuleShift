from dataclasses import dataclass
from enum import Enum

from tasks.ruleshift_benchmark.protocol import (
    PROBE_COUNT,
    InteractionLabel,
    format_public_label,
    parse_public_label,
)

__all__ = [
    "KaggleExecutionError",
    "Label",
    "BinaryResponse",
    "normalize_binary_response",
    "run_binary_task",
    "score_episode",
]


class Label(str, Enum):
    type_a = "type_a"
    type_b = "type_b"


@dataclass(frozen=True)
class BinaryResponse:
    probe_6: Label
    probe_7: Label
    probe_8: Label
    probe_9: Label

    def as_tuple(self) -> tuple[str, str, str, str]:
        return (
            _coerce_binary_label(self.probe_6, field_name="probe_6"),
            _coerce_binary_label(self.probe_7, field_name="probe_7"),
            _coerce_binary_label(self.probe_8, field_name="probe_8"),
            _coerce_binary_label(self.probe_9, field_name="probe_9"),
        )


class KaggleExecutionError(RuntimeError):
    pass


def run_binary_task(
    *,
    llm: object,
    prompt_binary: str,
    probe_targets: tuple[str, ...] | tuple[InteractionLabel, ...],
) -> tuple[int, int]:
    try:
        response = llm.prompt(prompt_binary, schema=BinaryResponse)
    except Exception as exc:
        raise KaggleExecutionError(
            "llm.prompt failed before producing a scoreable response"
        ) from exc

    try:
        normalized_response = normalize_binary_response(response)
    except ValueError as exc:
        raise KaggleExecutionError(
            f"llm.prompt returned an invalid binary response: {exc}"
        ) from exc
    if normalized_response is None:
        raise KaggleExecutionError(
            f"llm.prompt returned an unscoreable response of type {type(response).__name__}"
        )
    return score_episode(normalized_response, probe_targets)


def normalize_binary_response(response: object) -> tuple[str, ...] | None:
    if response is None:
        return None
    if isinstance(response, BinaryResponse):
        return tuple(
            _coerce_binary_label(getattr(response, field_name), field_name=field_name)
            for field_name in ("probe_6", "probe_7", "probe_8", "probe_9")
        )
    if isinstance(response, str):
        return _parse_binary_output(response)

    binary_response = _try_coerce_to_binary_response(response)
    if binary_response is not None:
        return binary_response.as_tuple()
    return None


def score_episode(
    predictions: tuple[str, ...] | tuple[InteractionLabel, ...] | None,
    probe_targets: tuple[str, ...] | tuple[InteractionLabel, ...],
) -> tuple[int, int]:
    normalized_targets = _normalize_labels(probe_targets)
    if normalized_targets is None:
        raise ValueError(f"probe_targets must contain exactly {PROBE_COUNT} valid labels")

    normalized_predictions = _normalize_labels(predictions)
    if normalized_predictions is None:
        return (0, PROBE_COUNT)

    num_correct = sum(
        prediction is target
        for prediction, target in zip(normalized_predictions, normalized_targets)
    )
    return (num_correct, PROBE_COUNT)


def _parse_binary_output(text: str) -> tuple[str, ...] | None:
    normalized_text = text.strip().strip("`").replace("\n", ",")
    tokens = tuple(token.strip().lower() for token in normalized_text.split(",") if token.strip())
    if len(tokens) != PROBE_COUNT:
        return None
    try:
        return tuple(format_public_label(parse_public_label(token)) for token in tokens)
    except ValueError:
        return None


def _try_coerce_to_binary_response(response: object) -> BinaryResponse | None:
    values: dict[str, object]

    if isinstance(response, dict):
        values = response
    elif hasattr(response, "__getitem__") and hasattr(response, "keys"):
        try:
            values = {
                field: response[field]
                for field in ("probe_6", "probe_7", "probe_8", "probe_9")
            }
        except (KeyError, TypeError):
            return None
    elif all(hasattr(response, field) for field in ("probe_6", "probe_7", "probe_8", "probe_9")):
        values = {
            field: getattr(response, field)
            for field in ("probe_6", "probe_7", "probe_8", "probe_9")
        }
    else:
        return None

    try:
        labels = tuple(
            Label(_coerce_binary_label(values[field], field_name=field))
            for field in ("probe_6", "probe_7", "probe_8", "probe_9")
        )
    except (KeyError, TypeError):
        return None
    return BinaryResponse(*labels)


def _coerce_binary_label(value: object, *, field_name: str) -> str:
    raw_value = _extract_label_value(value)
    try:
        return format_public_label(parse_public_label(raw_value))
    except ValueError as exc:
        raise ValueError(f"invalid binary response field {field_name}: {raw_value!r}") from exc


def _extract_label_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _normalize_labels(
    labels: tuple[str, ...] | tuple[InteractionLabel, ...] | None,
) -> tuple[InteractionLabel, ...] | None:
    if labels is None:
        return None
    try:
        normalized_labels = tuple(
            label if isinstance(label, InteractionLabel) else parse_public_label(label)
            for label in labels
        )
    except ValueError:
        return None
    if len(normalized_labels) != PROBE_COUNT:
        return None
    return normalized_labels
