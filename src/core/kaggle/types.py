from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import random
from typing import Sequence

from core.parser import (
    NarrativeParsedResult,
    NarrativeParseStatus,
    ParsedPrediction,
    ParseStatus,
    parse_binary_output,
    parse_narrative_audit_output,
)
from tasks.ruleshift_benchmark.protocol import PROBE_COUNT, InteractionLabel, parse_label

__all__ = [
    "Label",
    "BinaryResponse",
    "ConfidenceInterval",
    "parse_binary_response",
    "parse_narrative_response",
    "normalize_binary_response",
    "normalize_narrative_response",
    "score_episode",
    "compute_bootstrap_confidence_interval",
]


class Label(str, Enum):
    attract = "attract"
    repel = "repel"


@dataclass(frozen=True, slots=True)
class BinaryResponse:
    probe_6: Label
    probe_7: Label
    probe_8: Label
    probe_9: Label

    def as_tuple(self) -> tuple[str, str, str, str]:
        return (
            self.probe_6.value,
            self.probe_7.value,
            self.probe_8.value,
            self.probe_9.value,
        )


def normalize_binary_response(response: object) -> tuple[str, ...] | None:
    parsed = parse_binary_response(response)
    if parsed.status is ParseStatus.VALID:
        return tuple(label.value for label in parsed.labels)
    return None


def normalize_narrative_response(response: object) -> tuple[str, ...] | None:
    parsed = parse_narrative_response(response)
    if parsed.status is NarrativeParseStatus.VALID and parsed.output is not None:
        return tuple(label.value for label in parsed.output.final_decision)
    return None


def parse_binary_response(response: object) -> ParsedPrediction:
    if response is None:
        return ParsedPrediction.skipped_provider_failure()

    if isinstance(response, BinaryResponse):
        return ParsedPrediction(
            labels=tuple(parse_label(label) for label in response.as_tuple()),
            status=ParseStatus.VALID,
        )

    if isinstance(response, str):
        return parse_binary_output(response)

    # Dict/mapping path: LLM SDKs (including Kaggle Benchmarks) commonly
    # return structured schema responses as plain dicts or mapping-like
    # objects rather than the exact dataclass type.
    binary_response = _try_coerce_to_binary_response(response)
    if binary_response is not None:
        return ParsedPrediction(
            labels=tuple(parse_label(label) for label in binary_response.as_tuple()),
            status=ParseStatus.VALID,
        )

    return ParsedPrediction(labels=(), status=ParseStatus.INVALID)


_BINARY_RESPONSE_FIELDS: tuple[str, ...] = (
    "probe_6", "probe_7", "probe_8", "probe_9",
)


def _try_coerce_to_binary_response(response: object) -> BinaryResponse | None:
    """Attempt to construct a BinaryResponse from a dict, mapping, or
    attribute-bearing object that matches the expected schema fields.

    Returns None if the response cannot be coerced.
    """
    values: dict[str, object] = {}

    if isinstance(response, dict):
        values = response
    elif hasattr(response, "__getitem__") and hasattr(response, "keys"):
        # Mapping-like wrapper (e.g. MappingProxyType, Pydantic model dict view)
        try:
            values = {k: response[k] for k in _BINARY_RESPONSE_FIELDS}
        except (KeyError, TypeError):
            return None
    elif all(hasattr(response, field) for field in _BINARY_RESPONSE_FIELDS):
        # Attribute-bearing wrapper (e.g. Pydantic model, named tuple)
        values = {field: getattr(response, field) for field in _BINARY_RESPONSE_FIELDS}
    else:
        return None

    try:
        labels = tuple(
            Label(_extract_label_value(values[field]))
            for field in _BINARY_RESPONSE_FIELDS
        )
        return BinaryResponse(*labels)
    except (KeyError, TypeError, ValueError):
        return None


def _extract_label_value(value: object) -> str:
    """Extract a string label value from a field that may be a str, Enum, or
    object with a .value attribute."""
    if isinstance(value, str):
        return value
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def parse_narrative_response(response: object) -> NarrativeParsedResult:
    if response is None:
        return NarrativeParsedResult.skipped_provider_failure()

    if isinstance(response, str):
        return parse_narrative_audit_output(response)

    # Some SDKs wrap even plain-text responses in an object with .text or
    # ["text"].  Extract the text and parse normally.
    text = _try_extract_text(response)
    if text is not None:
        return parse_narrative_audit_output(text)

    return NarrativeParsedResult(
        output=None,
        status=NarrativeParseStatus.INVALID_FORMAT,
        failure_detail=f"unsupported response type: {type(response).__name__}",
    )


def _try_extract_text(response: object) -> str | None:
    """Attempt to extract a text string from a wrapper object or dict."""
    # Dict with a "text" or "content" key
    if isinstance(response, dict):
        for key in ("text", "content"):
            value = response.get(key)
            if isinstance(value, str):
                return value
        return None

    # Object with a .text or .content attribute
    for attr in ("text", "content"):
        value = getattr(response, attr, None)
        if isinstance(value, str):
            return value

    # If it has a __str__ that returns something non-trivial, use it
    # (but only for non-builtin types to avoid "{'key': 'value'}" strings)
    if not isinstance(response, (dict, list, tuple, set, frozenset)):
        text = str(response)
        if text and text != repr(response):
            return text

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


def _normalize_labels(
    labels: tuple[str, ...] | tuple[InteractionLabel, ...] | None,
) -> tuple[InteractionLabel, ...] | None:
    if labels is None:
        return None

    normalized_labels = tuple(parse_label(label) for label in labels)
    if len(normalized_labels) != PROBE_COUNT:
        return None
    return normalized_labels


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    mean: float
    lower: float
    upper: float
    level: float
    margin: float


def compute_bootstrap_confidence_interval(
    num_correct: Sequence[int],
    total: Sequence[int],
    *,
    level: float = 0.95,
    n_bootstraps: int = 1000,
    seed: int = 2025,
) -> ConfidenceInterval:
    if len(num_correct) != len(total):
        raise ValueError("num_correct and total must have the same length")

    n = len(num_correct)
    if n == 0:
        return ConfidenceInterval(0.0, 0.0, 0.0, level, 0.0)

    nc = list(num_correct)
    tot = list(total)
    grand_total_probes = sum(tot)

    if grand_total_probes == 0:
        return ConfidenceInterval(0.0, 0.0, 0.0, level, 0.0)

    grand_mean = sum(nc) / grand_total_probes

    rng = random.Random(seed)
    indices = range(n)
    means: list[float] = []

    for _ in range(n_bootstraps):
        sample_indices = rng.choices(indices, k=n)
        s_c = sum(nc[i] for i in sample_indices)
        s_t = sum(tot[i] for i in sample_indices)
        if s_t > 0:
            means.append(s_c / s_t)
        else:
            means.append(0.0)

    means.sort()

    alpha = 1.0 - level
    lower_idx = int(n_bootstraps * (alpha / 2))
    upper_idx = int(n_bootstraps * (1 - (alpha / 2)))

    lower_idx = max(0, min(lower_idx, n_bootstraps - 1))
    upper_idx = max(0, min(upper_idx, n_bootstraps - 1))

    lower = means[lower_idx]
    upper = means[upper_idx]

    margin = max(grand_mean - lower, upper - grand_mean)

    return ConfidenceInterval(
        mean=grand_mean,
        lower=lower,
        upper=upper,
        level=level,
        margin=margin,
    )
